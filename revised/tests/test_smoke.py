import os
import sys

import keras
import numpy as np
import pytest
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from app import (  # noqa: E402
    PLATT_COEF,
    PLATT_INTERCEPT,
    find_gap_layer_name,
    make_gradcam_heatmap,
    pil_to_model_input,
    platt_calibrate,
)

MODEL_PATH = os.path.join(HERE, "..", "..", "DysPOLNet.hdf5")

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
