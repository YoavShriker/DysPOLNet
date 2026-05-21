import io
import os

import keras
import matplotlib as mpl
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("DYSPOLNET_MODEL", os.path.join(HERE, "..", "DysPOLNet.hdf5"))
IMG_SIZE = (300, 300)
OPERATING_THRESHOLD = 0.10

# Platt scaling coefficients from the original `lr` pickle
# (LogisticRegression: coef_=4.14699105, intercept_=-2.96500325).
PLATT_COEF = 4.14699105
PLATT_INTERCEPT = -2.96500325


def platt_calibrate(raw_score):
    return 1.0 / (1.0 + np.exp(-(PLATT_COEF * raw_score + PLATT_INTERCEPT)))


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


def main():
    st.set_page_config(
        page_title="DysPOLNet",
        layout="centered",
        page_icon=":microscope:",
    )

    model = load_model()
    gap_layer_name = find_gap_layer_name(model)

    st.title("DysPOLNet")
    st.caption(
        "Dysplasia risk estimation for oral leukoplakia photographs. "
        "Research use only — not a diagnostic device."
    )

    file = st.file_uploader(
        "Upload a close-up photograph of the lesion",
        type=["jpg", "jpeg", "png"],
        help="Without cheek retractors, teeth, or mouth mirrors if possible",
    )

    if file is None:
        st.info("Upload a JPG or PNG to get started.")
        return

    try:
        image = Image.open(io.BytesIO(file.getvalue()))
        image.load()
    except Exception:
        st.error("Could not read this file as an image. Please upload a valid JPG or PNG.")
        return

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
            with st.spinner("Generating explainability heatmap..."):
                heatmap = make_gradcam_heatmap(img_array, model, gap_layer_name)
                overlay = overlay_gradcam(image, heatmap)
            st.image(overlay, use_container_width=True)
            st.caption(
                "Grad-CAM: regions influencing the prediction. "
                "Intensity reflects relative model attention, not lesion severity."
            )
        except Exception:
            st.warning("Heatmap could not be generated for this image.")

    with tab_details:
        st.write("**Architecture:** EfficientNetB2, 300×300 input, single sigmoid output")
        st.write("**Calibration:** Platt scaling on validation predictions")
        st.write(f"**Raw model score:** `{raw_score:.4f}`")
        st.write(f"**Calibrated probability:** `{calibrated:.4f}`")
        st.write(f"**Operating threshold:** `{OPERATING_THRESHOLD:.0%}`")

    st.divider()
    st.caption(
        "DysPOLNet • "
        "[Oral Cancer Research Theme, HKU](https://facdent.hku.hk/research/oral-cancer.html) • "
        "Research use only"
    )


if __name__ == "__main__":
    main()
