import io
import os

import keras
import matplotlib as mpl
import numpy as np
import scipy.ndimage
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageDraw, ImageOps

from gate import (
    GATE_BLOCK_THRESHOLD,
    GATE_PASS_THRESHOLD,
    build_clip_bundle,
    build_face_mesh,
    clip_oral_probability,
    detect_face_and_mouth,
    evaluate_gate,
)

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass
try:
    import pillow_avif  # noqa: F401
except ImportError:
    pass

# Cap decoded pixels well below Pillow's default decompression-bomb limit;
# clinical photographs above 50 megapixels are not expected.
Image.MAX_IMAGE_PIXELS = 50_000_000

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("DYSPOLNET_MODEL", os.path.join(HERE, "..", "DysPOLNet.hdf5"))
IMG_SIZE = (300, 300)
OPERATING_THRESHOLD = 0.10

SUPPORTED_UPLOAD_TYPES = [
    "jpg", "jpeg", "png", "bmp",
    "tif", "tiff", "webp",
    "heic", "heif", "avif",
]

# Bounding-box rendering for suspicious regions.
# Orange chosen over the colonoscopy-standard lime green because dysplasia
# is a "warning" finding, not a "go" signal.
BOX_COLOR = (255, 140, 0)
BOX_PERCENTILE = 85
BOX_RELATIVE_FLOOR = 0.2
BOX_MIN_AREA_FRAC = 0.01
BOX_MAX = 3
HEATMAP_ALPHA_WITH_BOXES = 0.25

# Platt scaling coefficients from the original `lr` pickle
# (LogisticRegression: coef_=4.14699105, intercept_=-2.96500325).
PLATT_COEF = 4.14699105
PLATT_INTERCEPT = -2.96500325


def platt_calibrate(raw_score):
    return 1.0 / (1.0 + np.exp(-(PLATT_COEF * raw_score + PLATT_INTERCEPT)))


class UnsupportedImageError(ValueError):
    pass


def decode_uploaded_image(file_bytes):
    # Magic-byte rejections before handing to Pillow, so an extension-renamed
    # file (foo.pdf -> foo.jpg) gets a useful error instead of a decode crash.
    if file_bytes[:5] == b"%PDF-":
        raise UnsupportedImageError(
            "PDF files are not supported. Please export a single page as JPG or PNG."
        )
    if b"<svg" in file_bytes[:256].lower():
        raise UnsupportedImageError(
            "SVG files are not supported. Please use a raster image format."
        )
    if len(file_bytes) > 132 and file_bytes[128:132] == b"DICM":
        raise UnsupportedImageError(
            "DICOM files are not supported in this version. "
            "Please export the image as JPG or PNG."
        )

    try:
        img = Image.open(io.BytesIO(file_bytes))
        img.load()
    except Image.DecompressionBombError:
        raise UnsupportedImageError(
            "Image exceeds the 50-megapixel limit. Please resize before uploading."
        )
    except Exception as exc:
        raise UnsupportedImageError(
            "Could not decode this file as a supported image format."
        ) from exc

    if img.format == "GIF" and getattr(img, "is_animated", False):
        raise UnsupportedImageError(
            "Animated GIF files are not supported. Please use a still image."
        )

    multi_page_warning = None
    n_frames = getattr(img, "n_frames", 1)
    if n_frames > 1 and img.format == "TIFF":
        img.seek(0)
        multi_page_warning = (
            f"Multi-page TIFF detected ({n_frames} pages). Using page 1 only."
        )

    # Bake EXIF orientation here so every downstream consumer (gate, model,
    # cropping, display) works in a single, consistent coordinate system.
    # Without this, the mouth_bbox returned by the gate would be in the
    # transposed coordinate space and image.crop() would slice the wrong
    # region from a phone photo with an orientation tag.
    original_format = img.format
    img = ImageOps.exif_transpose(img)
    img.format = original_format

    return img, multi_page_warning


def pil_to_model_input(pil_image, size):
    img = ImageOps.exif_transpose(pil_image).convert("RGB")
    img = img.resize(size, Image.NEAREST)
    return np.expand_dims(np.asarray(img, dtype=np.float32), axis=0)


def make_gradcam_heatmap(img_array, model, gap_layer_name):
    grad_model = keras.models.Model(
        model.inputs,
        [model.get_layer(gap_layer_name).input, model.output],
    )
    with tf.GradientTape() as tape:
        conv_output, preds = grad_model(img_array)
        class_channel = preds[:, 0]
    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()


def upsample_heatmap_to_image(heatmap_small, target_pil_image):
    target_w, target_h = ImageOps.exif_transpose(target_pil_image).size
    h_img = Image.fromarray(
        np.clip(heatmap_small * 255, 0, 255).astype(np.uint8), mode="L"
    )
    h_img = h_img.resize((target_w, target_h), Image.BILINEAR)
    return np.asarray(h_img, dtype=np.float32) / 255.0


def heatmap_to_boxes(
    heatmap_full,
    global_p,
    percentile=BOX_PERCENTILE,
    relative_floor=BOX_RELATIVE_FLOOR,
    min_area_frac=BOX_MIN_AREA_FRAC,
    max_boxes=BOX_MAX,
):
    H, W = heatmap_full.shape
    sigma = max(1.0, 0.02 * H)
    smoothed = scipy.ndimage.gaussian_filter(heatmap_full, sigma=sigma)

    peak = float(smoothed.max())
    if peak <= 1e-6:
        return []

    tau = max(np.percentile(smoothed, percentile), relative_floor * peak)
    binmask = smoothed >= tau
    binmask = scipy.ndimage.binary_closing(binmask, iterations=2)

    labels, n_components = scipy.ndimage.label(binmask)
    if n_components == 0:
        return []

    slices = scipy.ndimage.find_objects(labels)
    min_area = min_area_frac * H * W
    boxes = []
    for i, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        comp_mask = labels[sl] == i
        area = int(comp_mask.sum())
        if area < min_area:
            continue
        mean_act = float(smoothed[sl][comp_mask].mean())
        confidence = float(global_p) * (mean_act / peak)
        y0, y1 = sl[0].start, sl[0].stop
        x0, x1 = sl[1].start, sl[1].stop
        boxes.append((int(x0), int(y0), int(x1 - x0), int(y1 - y0), confidence))

    boxes.sort(key=lambda b: -b[4])
    return boxes[:max_boxes]


def draw_boxes_on_image(pil_image, boxes, color=BOX_COLOR):
    img = pil_image.copy()
    if not boxes:
        return img
    draw = ImageDraw.Draw(img)
    stroke = max(2, int(0.012 * min(img.size)))
    for x, y, w, h, _ in boxes:
        draw.rectangle([x, y, x + w, y + h], outline=color, width=stroke)
    return img


def overlay_gradcam(pil_image, heatmap, alpha=0.4):
    img = np.asarray(ImageOps.exif_transpose(pil_image).convert("RGB"), dtype=np.float32)
    heatmap_u8 = np.uint8(255 * heatmap)
    jet = mpl.colormaps["jet"]
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_u8]
    jet_heatmap = keras.utils.array_to_img(jet_heatmap).resize(
        (img.shape[1], img.shape[0]), Image.BILINEAR
    )
    overlay = np.clip(np.asarray(jet_heatmap, dtype=np.float32) * alpha + img, 0, 255)
    return Image.fromarray(overlay.astype(np.uint8))


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    model.predict(np.zeros((1, *IMG_SIZE, 3), dtype=np.float32), verbose=0)
    return model


@st.cache_resource
def find_gap_layer_name(_model):
    for layer in _model.layers:
        if isinstance(layer, keras.layers.GlobalAveragePooling2D):
            return layer.name
    raise ValueError("No GlobalAveragePooling2D layer found in model")


@st.cache_resource
def load_face_mesh():
    try:
        return build_face_mesh()
    except Exception:
        return None


@st.cache_resource
def load_clip_bundle():
    try:
        return build_clip_bundle()
    except Exception:
        return None


@st.cache_data(show_spinner=False, max_entries=10)
def cached_decode_and_gate(file_bytes, _face_mesh, _clip_bundle):
    image, multi_page_warning = decode_uploaded_image(file_bytes)
    if _face_mesh is None or _clip_bundle is None:
        return image, None, multi_page_warning
    gate_result = evaluate_gate(
        image,
        face_detector_fn=lambda im: detect_face_and_mouth(im, _face_mesh),
        clip_score_fn=lambda im: clip_oral_probability(im, _clip_bundle),
    )
    return image, gate_result, multi_page_warning


def main():
    st.set_page_config(
        page_title="DysPOLNet",
        layout="centered",
        page_icon=":microscope:",
    )

    st.title("DysPOLNet")
    st.caption(
        "Dysplasia risk estimation for oral leukoplakia photographs. "
        "Research use only — not a diagnostic device."
    )

    file = st.file_uploader(
        "Upload a close-up photograph of the lesion",
        type=SUPPORTED_UPLOAD_TYPES,
        help=(
            "Accepted: JPEG, PNG, BMP, TIFF, WebP, HEIC/HEIF, AVIF. Max 50 MB. "
            "Close-up without cheek retractors, teeth, or mouth mirrors if possible."
        ),
    )

    if file is None:
        st.info("Upload an image to get started.")
        return

    # Defer heavy model loads until a file is actually uploaded so the first
    # page load doesn't block on the ~150 MB CLIP weight download.
    model = load_model()
    gap_layer_name = find_gap_layer_name(model)
    face_mesh = load_face_mesh()
    clip_bundle = load_clip_bundle()

    try:
        with st.spinner("Checking that this is a close-up of the oral cavity..."):
            image, gate_result, multi_page_warning = cached_decode_and_gate(
                file.getvalue(), face_mesh, clip_bundle
            )
    except UnsupportedImageError as exc:
        st.error(str(exc))
        return

    if multi_page_warning:
        st.warning(multi_page_warning)

    if gate_result is not None:
        if gate_result.crop_box is not None:
            image = image.crop(gate_result.crop_box)
            if min(image.size) < 100:
                st.warning(
                    f"Auto-cropped region is small ({image.size[0]}×{image.size[1]} px); "
                    "the prediction may be unreliable. Consider uploading a closer photograph."
                )
            else:
                st.info(
                    "Image auto-cropped to mouth region. For best results, upload "
                    "close-up photographs of the lesion."
                )

        if gate_result.decision == "block":
            st.error(
                "This image does not appear to show the inside of an oral cavity "
                f"(content match: {gate_result.p_oral:.0%}). DysPOLNet is trained "
                "on intraoral photographs of leukoplakia; results on other content "
                "will not be meaningful."
            )
            override = st.checkbox(
                "Analyze anyway — I understand the result may be unreliable",
                key="gate_override",
            )
            if not override:
                return
        elif gate_result.decision == "warn":
            st.warning(
                f"Image content match to an oral cavity is moderate "
                f"({gate_result.p_oral:.0%}). The prediction may be less reliable "
                "than on a clear close-up."
            )

    try:
        with st.spinner("Analyzing image..."):
            img_array = pil_to_model_input(image, IMG_SIZE)
            raw_score = float(np.asarray(model.predict(img_array, verbose=0)).squeeze())
            calibrated = float(platt_calibrate(raw_score))
    except Exception:
        st.error("The model could not process this image. Please try a different photo.")
        return

    above_threshold = calibrated >= OPERATING_THRESHOLD

    with st.container(border=True):
        col_img, col_result = st.columns([1, 1], gap="large")
        with col_img:
            st.image(image, use_container_width=True)
            st.caption("Uploaded image")
        with col_result:
            st.metric(
                label="Dysplasia probability",
                value=format(calibrated, ".1%"),
                help=(
                    f"Operating threshold: {OPERATING_THRESHOLD:.0%} "
                    "(sensitivity >95% during model development)"
                ),
            )
            st.progress(min(max(calibrated, 0.0), 1.0))
            if above_threshold:
                st.markdown("**:red[HIGH RISK]** — above operating point")
            else:
                st.markdown("**:green[LOW RISK]** — below operating point")
            st.caption("The probability is more informative than the binary status.")

    tab_explain, tab_details = st.tabs(["Explainability", "Model details"])

    with tab_explain:
        try:
            with st.spinner("Generating explainability overlay..."):
                heatmap_small = make_gradcam_heatmap(img_array, model, gap_layer_name)
                heatmap_full = upsample_heatmap_to_image(heatmap_small, image)
                boxes = heatmap_to_boxes(heatmap_full, global_p=calibrated)
                overlay = overlay_gradcam(
                    image, heatmap_small, alpha=HEATMAP_ALPHA_WITH_BOXES
                )
                overlay = draw_boxes_on_image(overlay, boxes)
            st.image(overlay, use_container_width=True)
            box_count = len(boxes)
            if box_count == 0:
                st.caption(
                    "Grad-CAM heatmap shown; no individual region exceeded the "
                    "detection threshold. Intensity reflects relative model attention."
                )
            else:
                st.caption(
                    f"Grad-CAM heatmap with {box_count} highlighted "
                    f"region{'s' if box_count != 1 else ''} (orange). "
                    "Boxes mark areas of highest model attention; they are not "
                    "lesion boundaries."
                )
        except Exception:
            st.warning("Explainability overlay could not be generated for this image.")

    with tab_details:
        st.write("**Architecture:** EfficientNetB2, 300×300 input, single sigmoid output")
        st.write("**Calibration:** Platt scaling on validation predictions")
        st.write(f"**Raw model score:** `{raw_score:.4f}`")
        st.write(f"**Calibrated probability:** `{calibrated:.4f}`")
        st.write(f"**Operating threshold:** `{OPERATING_THRESHOLD:.0%}`")
        st.write("---")
        st.write("**Oral-cavity gate**")
        if gate_result is None:
            st.write("- Status: `disabled` (mediapipe / open_clip not installed)")
        else:
            st.write(f"- Method: `{gate_result.method}`")
            st.write(f"- Decision: `{gate_result.decision}`")
            st.write(f"- Content match (p_oral): `{gate_result.p_oral:.3f}`")
            st.write(
                f"- Pass / block thresholds: "
                f"`{GATE_PASS_THRESHOLD:.2f}` / `{GATE_BLOCK_THRESHOLD:.2f}`"
            )
            if gate_result.crop_box is not None:
                st.write(f"- Auto-crop box: `{gate_result.crop_box}`")

    st.divider()
    st.caption(
        "DysPOLNet • "
        "[Oral Cancer Research Theme, HKU](https://facdent.hku.hk/research/oral-cancer.html) • "
        "Research use only"
    )


if __name__ == "__main__":
    main()
