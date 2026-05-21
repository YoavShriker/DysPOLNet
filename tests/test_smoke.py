import os
import sys

import keras
import numpy as np
import pytest
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from app import (  # noqa: E402
    BOX_MAX,
    PLATT_COEF,
    PLATT_INTERCEPT,
    UnsupportedImageError,
    decode_uploaded_image,
    draw_boxes_on_image,
    find_gap_layer_name,
    heatmap_to_boxes,
    make_gradcam_heatmap,
    pil_to_model_input,
    platt_calibrate,
    upsample_heatmap_to_image,
)

MODEL_PATH = os.path.join(HERE, "..", "DysPOLNet.hdf5")

# Golden values measured against the committed DysPOLNet.hdf5.
# Any change to model weights, preprocessing, or dependency versions
# that affects inference will move these — that's the regression signal.
GOLDEN_RNG42_RAW = 0.8855422735
GOLDEN_RNG42_CAL = 0.6698115015
GOLDEN_GRAY128_RAW = 0.7391895652
GOLDEN_GRAY128_CAL = 0.5250812463

TOL = 1e-3


@pytest.fixture(scope="module")
def model():
    return tf.keras.models.load_model(MODEL_PATH)


def test_platt_inline_matches_known_anchor_points():
    # Anchors come from the original `lr` pickle: raw=0.715 maps to cal=0.5
    # and raw=0.185 maps to cal=0.10. Verify the inline formula reproduces them.
    assert abs(platt_calibrate(0.7149770065769265) - 0.5) < 1e-6
    assert abs(platt_calibrate(0.18514114442966206) - 0.10) < 1e-6
    assert PLATT_COEF == pytest.approx(4.14699105)
    assert PLATT_INTERCEPT == pytest.approx(-2.96500325)


def test_model_has_gap_layer(model):
    name = find_gap_layer_name(model)
    assert isinstance(model.get_layer(name), keras.layers.GlobalAveragePooling2D)


def test_prediction_deterministic_random_input(model):
    rng = np.random.default_rng(42)
    x = rng.integers(0, 256, (1, 300, 300, 3)).astype(np.float32)
    raw = float(np.asarray(model.predict(x, verbose=0)).squeeze())
    cal = float(platt_calibrate(raw))
    assert abs(raw - GOLDEN_RNG42_RAW) < TOL
    assert abs(cal - GOLDEN_RNG42_CAL) < TOL


def test_prediction_constant_gray_input(model):
    x = np.full((1, 300, 300, 3), 128.0, dtype=np.float32)
    raw = float(np.asarray(model.predict(x, verbose=0)).squeeze())
    cal = float(platt_calibrate(raw))
    assert abs(raw - GOLDEN_GRAY128_RAW) < TOL
    assert abs(cal - GOLDEN_GRAY128_CAL) < TOL


def test_gradcam_produces_finite_heatmap(model):
    x = np.full((1, 300, 300, 3), 200.0, dtype=np.float32)
    name = find_gap_layer_name(model)
    hm = make_gradcam_heatmap(x, model, name)
    assert hm.ndim == 2
    assert np.isfinite(hm).all()
    assert hm.min() >= 0.0
    assert hm.max() <= 1.0 + 1e-6


def test_pil_to_model_input_handles_rgba_and_exif():
    from PIL import Image

    rgba = Image.new("RGBA", (640, 480), (200, 100, 50, 255))
    arr = pil_to_model_input(rgba, (300, 300))
    assert arr.shape == (1, 300, 300, 3)
    assert arr.dtype == np.float32
    assert arr.min() >= 0.0 and arr.max() <= 255.0


def _png_bytes(size=(64, 64), color=(128, 64, 32)):
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def test_decode_accepts_png():
    img, warn = decode_uploaded_image(_png_bytes())
    assert img.format == "PNG"
    assert warn is None


def test_decode_rejects_pdf():
    pdf_bytes = b"%PDF-1.4\n%fake content"
    with pytest.raises(UnsupportedImageError, match="PDF"):
        decode_uploaded_image(pdf_bytes)


def test_decode_rejects_svg():
    svg_bytes = b'<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg"></svg>'
    with pytest.raises(UnsupportedImageError, match="SVG"):
        decode_uploaded_image(svg_bytes)


def test_decode_rejects_dicom_magic():
    fake_dicom = b"\x00" * 128 + b"DICM" + b"\x00" * 100
    with pytest.raises(UnsupportedImageError, match="DICOM"):
        decode_uploaded_image(fake_dicom)


def test_decode_rejects_garbage():
    with pytest.raises(UnsupportedImageError):
        decode_uploaded_image(b"this is definitely not an image")


def test_decode_accepts_bmp_and_tiff_and_webp():
    import io
    from PIL import Image
    for fmt in ("BMP", "TIFF", "WEBP"):
        buf = io.BytesIO()
        Image.new("RGB", (32, 32), (10, 20, 30)).save(buf, format=fmt)
        img, warn = decode_uploaded_image(buf.getvalue())
        assert img.format == fmt
        assert warn is None


def test_decode_bakes_exif_rotation_into_pixels():
    import io
    from PIL import Image
    buf = io.BytesIO()
    wide = Image.new("RGB", (200, 100), (180, 60, 60))
    exif = wide.getexif()
    exif[0x0112] = 6  # Orientation tag: rotate 90 CW
    wide.save(buf, format="JPEG", exif=exif)
    decoded, _ = decode_uploaded_image(buf.getvalue())
    # After baking the EXIF orientation, a 200x100 wide image with tag=6
    # becomes a 100x200 tall image.
    assert decoded.size == (100, 200)
    # And the format attribute survives the transpose.
    assert decoded.format == "JPEG"


def test_decode_multipage_tiff_uses_first_page_with_warning():
    import io
    from PIL import Image
    frames = [Image.new("RGB", (16, 16), (i, i, i)) for i in (50, 100, 150)]
    buf = io.BytesIO()
    frames[0].save(buf, format="TIFF", save_all=True, append_images=frames[1:])
    img, warn = decode_uploaded_image(buf.getvalue())
    assert img.format == "TIFF"
    assert warn is not None and "Multi-page" in warn


def test_boxes_empty_on_flat_heatmap():
    flat = np.zeros((300, 300), dtype=np.float32)
    assert heatmap_to_boxes(flat, global_p=0.7) == []


def test_boxes_detect_single_hotspot():
    hm = np.zeros((300, 300), dtype=np.float32)
    hm[100:180, 120:200] = 1.0
    boxes = heatmap_to_boxes(hm, global_p=0.8)
    assert len(boxes) >= 1
    x, y, w, h, conf = boxes[0]
    # box should overlap the hotspot region
    assert x < 200 and (x + w) > 100
    assert y < 180 and (y + h) > 100
    # confidence is heuristic but bounded by global_p
    assert 0.0 < conf <= 0.8 + 1e-6


def test_boxes_cap_at_max():
    hm = np.zeros((300, 300), dtype=np.float32)
    # five well-separated hotspots
    for cy, cx in [(40, 40), (40, 250), (150, 150), (260, 40), (260, 250)]:
        hm[cy - 15:cy + 15, cx - 15:cx + 15] = 1.0
    boxes = heatmap_to_boxes(hm, global_p=0.9)
    assert len(boxes) <= BOX_MAX


def test_boxes_sorted_by_confidence_desc():
    hm = np.zeros((300, 300), dtype=np.float32)
    hm[40:80, 40:80] = 0.4  # weaker
    hm[160:230, 160:230] = 1.0  # stronger
    boxes = heatmap_to_boxes(hm, global_p=0.9)
    assert len(boxes) >= 2
    confs = [b[4] for b in boxes]
    assert confs == sorted(confs, reverse=True)


def test_upsample_heatmap_matches_image_shape():
    from PIL import Image
    small = np.random.RandomState(0).rand(10, 10).astype(np.float32)
    img = Image.new("RGB", (640, 480), (10, 20, 30))
    full = upsample_heatmap_to_image(small, img)
    assert full.shape == (480, 640)
    assert full.dtype == np.float32
    assert 0.0 <= full.min() and full.max() <= 1.0


def test_draw_boxes_preserves_image_size_and_no_op_on_empty():
    from PIL import Image
    img = Image.new("RGB", (200, 200), (50, 60, 70))
    out_empty = draw_boxes_on_image(img, [])
    assert out_empty.size == img.size
    out = draw_boxes_on_image(img, [(20, 30, 50, 40, 0.7)])
    assert out.size == img.size


from gate import (  # noqa: E402
    FACE_CROP_MOUTH_AREA_THRESHOLD,
    GATE_BLOCK_THRESHOLD,
    GATE_PASS_THRESHOLD,
    crop_box_around_mouth,
    decide_from_p_oral,
    evaluate_gate,
)


def test_decide_from_p_oral_thresholds():
    assert decide_from_p_oral(GATE_PASS_THRESHOLD) == "pass"
    assert decide_from_p_oral(GATE_PASS_THRESHOLD + 0.01) == "pass"
    assert decide_from_p_oral((GATE_PASS_THRESHOLD + GATE_BLOCK_THRESHOLD) / 2) == "warn"
    assert decide_from_p_oral(GATE_BLOCK_THRESHOLD) == "warn"
    assert decide_from_p_oral(GATE_BLOCK_THRESHOLD - 0.01) == "block"
    assert decide_from_p_oral(0.0) == "block"
    assert decide_from_p_oral(1.0) == "pass"


def test_crop_box_around_mouth_stays_in_bounds():
    img_size = (640, 480)
    mouth_bbox = (300, 220, 360, 260)
    left, top, right, bot = crop_box_around_mouth(img_size, mouth_bbox)
    assert 0 <= left < right <= 640
    assert 0 <= top < bot <= 480


def test_crop_box_clamps_when_mouth_near_edge():
    img_size = (200, 200)
    mouth_bbox = (0, 0, 60, 50)
    left, top, right, bot = crop_box_around_mouth(img_size, mouth_bbox)
    assert left == 0 and top == 0
    assert right <= 200 and bot <= 200


def test_gate_passes_when_face_with_large_mouth():
    from PIL import Image
    fake_image = Image.new("RGB", (300, 300), (200, 100, 100))

    def fake_face_detector(_):
        return {
            "img_size": (300, 300),
            "mouth_bbox": (50, 50, 250, 250),
            "mouth_area_frac": FACE_CROP_MOUTH_AREA_THRESHOLD + 0.1,
        }

    def fake_clip(_):
        raise AssertionError("CLIP should not be invoked when face fills frame")

    result = evaluate_gate(fake_image, fake_face_detector, fake_clip)
    assert result.decision == "pass"
    assert result.method == "face_close_up"
    assert result.face_detected is True
    assert result.crop_box is None


def test_gate_warns_and_crops_when_small_mouth():
    from PIL import Image
    fake_image = Image.new("RGB", (400, 400), (180, 120, 110))

    def fake_face_detector(_):
        return {
            "img_size": (400, 400),
            "mouth_bbox": (180, 220, 220, 240),
            "mouth_area_frac": 0.01,
        }

    def fake_clip(_):
        raise AssertionError("CLIP should not be invoked when face was found")

    result = evaluate_gate(fake_image, fake_face_detector, fake_clip)
    assert result.decision == "warn"
    assert result.method == "face_crop"
    assert result.crop_box is not None
    left, top, right, bot = result.crop_box
    assert 0 <= left < right <= 400
    assert 0 <= top < bot <= 400


def test_gate_falls_back_to_clip_when_no_face():
    from PIL import Image
    fake_image = Image.new("RGB", (300, 300), (100, 100, 100))

    def fake_face_detector(_):
        return None

    for p_oral, expected in [(0.9, "pass"), (0.45, "warn"), (0.10, "block")]:
        result = evaluate_gate(fake_image, fake_face_detector, lambda _, p=p_oral: p)
        assert result.method == "clip"
        assert result.face_detected is False
        assert result.decision == expected
        assert abs(result.p_oral - p_oral) < 1e-6


@pytest.mark.skipif(
    os.getenv("SKIP_HEAVY_TESTS") == "1",
    reason="CLIP / MediaPipe integration; requires internet for first-run weight download",
)
def test_face_mesh_and_clip_load_and_run():
    from PIL import Image
    from gate import build_face_mesh, build_clip_bundle, clip_oral_probability, detect_face_and_mouth
    try:
        fm = build_face_mesh()
        clip_bundle = build_clip_bundle()
    except Exception as exc:
        pytest.skip(f"Could not load gate dependencies: {exc}")
    img = Image.new("RGB", (224, 224), (150, 90, 80))
    face = detect_face_and_mouth(img, fm)
    assert face is None
    p = clip_oral_probability(img, clip_bundle)
    assert 0.0 <= p <= 1.0
