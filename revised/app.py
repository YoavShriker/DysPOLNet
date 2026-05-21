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

# Platt scaling coefficients extracted from the original `lr` pickle
# (LogisticRegression: coef_=4.14699105, intercept_=-2.96500325).
# Inlined to remove the pickle dependency and the sklearn import.
PLATT_COEF = 4.14699105
PLATT_INTERCEPT = -2.96500325


def platt_calibrate(raw_score):
    return 1.0 / (1.0 + np.exp(-(PLATT_COEF * raw_score + PLATT_INTERCEPT)))


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    # Warm up so the first user request doesn't pay the graph-tracing cost.
    model.predict(np.zeros((1, *IMG_SIZE, 3), dtype=np.float32), verbose=0)
    return model


@st.cache_resource
def find_gap_layer_name(_model):
    for layer in _model.layers:
        if isinstance(layer, keras.layers.GlobalAveragePooling2D):
            return layer.name
    raise ValueError("No GlobalAveragePooling2D layer found in model")


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


model = load_model()
gap_layer_name = find_gap_layer_name(model)

st.write(
    """
         # Predict Probability of Dysplasia in Oral Leukoplakia
         """
)
st.write(
    "Simple deployment of the ***:blue[DysPOLNet]*** model to predict dysplasia using lesion photographs"
)
file = st.file_uploader(
    "Please upload a close-up image file of the lesion without cheek retractors, teeth, or mouth mirrors if possible",
    type=["jpg", "png"],
)

if file is None:
    st.text("Please upload an image file in jpg or png format")
else:
    try:
        image = Image.open(io.BytesIO(file.getvalue()))
        image.load()
    except Exception:
        st.error("Could not read this file as an image. Please upload a valid JPG or PNG.")
        st.stop()

    st.image(image, use_container_width=True)
    st.caption("_Image Uploaded by_ USER")

    try:
        with st.spinner("Analyzing image..."):
            img_array = pil_to_model_input(image, IMG_SIZE)
            raw_score = float(np.asarray(model.predict(img_array, verbose=0)).squeeze())
            calibrated = float(platt_calibrate(raw_score))
    except Exception:
        st.error("The model could not process this image. Please try a different photo.")
        st.stop()

    prediction = format(calibrated, ".1%")

    st.markdown("###")
    st.subheader("**MODEL OUTPUTS**")
    st.write("--")
    st.write("Predicted probability of Dysplasia:", prediction)
    st.caption(
        "(Predicted probability with sensitivity above 95% during model development is **:orange[10%]**)"
    )
    st.write("--")
    if raw_score < 0.5:
        st.write("Suggested Binary Dysplasia Status:", "**:green[LOW RISK]**")
    else:
        st.write("Suggested Binary Dysplasia Status:", "**:red[HIGH RISK]**")
    st.caption(
        "Please note that the *Predicted Probability* is more informative than *Binary Status*"
    )
    st.write("--")
    st.write("Explainability Heatmap:")
    try:
        with st.spinner("Generating explainability heatmap..."):
            heatmap = make_gradcam_heatmap(img_array, model, gap_layer_name)
            overlay = overlay_gradcam(image, heatmap)
        st.image(overlay, use_container_width=True)
        st.caption(
            "_GradCAM heatmap showing region(s) influencing :blue[DysPOLNet’s] prediction_"
        )
    except Exception:
        st.warning("Heatmap could not be generated for this image.")
    st.markdown("####")
    st.markdown("####")
    st.markdown("####")
    st.markdown("####")
    st.markdown("####")
    st.write(
        "Group Website: [Oral Cancer Research Theme, HKU](https://facdent.hku.hk/research/oral-cancer.html)  |  2024"
    )
