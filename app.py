import tensorflow as tf
import keras
import tempfile
import matplotlib as mpl
import pandas as pd
from sklearn.linear_model import LogisticRegression as LR
import pickle
import streamlit as st
import os

preprocess_input = tf.keras.applications.efficientnet.preprocess_input

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('DysPOLNet.hdf5')  # replace with your model loading code
    return model

model = load_model()

st.write("""
         # Predict Probability of Dysplasia in Oral Leukoplakia
         """
         )
st.write('Simple deployment of the ***:blue[DysPOLNet]*** model to predict dysplasia using lesion photographs')
file = st.file_uploader("Please upload a close-up image file of the lesion without cheek retractors, teeth, or mouth mirrors if possible", type=["jpg", "png"])
if file:
        temp_dir = tempfile.mkdtemp()
        path = os.path.join(temp_dir, file.name)
        with open(path, "wb") as f:
                f.write(file.getvalue())

from PIL import Image, ImageOps
import numpy as np

img_size = (300,300)
def get_img_array(img_path, size):
    img = keras.utils.load_img(img_path, target_size=size)
    array = keras.utils.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array

last_conv_layer_name = "global_average_pooling2d"
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).input, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def display_gradcam(img_path, heatmap, alpha=0.9):
    # Load the original image
    img = keras.utils.load_img(img_path)
    img = keras.utils.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = mpl.colormaps["jet"]

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
    return superimposed_img
    
if file is None:
    st.text("Please upload an image file in jpg or png format")
else:
    image = Image.open(file)
    st.image(image, use_container_width=True)
    st.caption('_Image Uploaded by_ USER')
    img_array = preprocess_input(get_img_array(path, size=img_size))
    prediction0 = model.predict(img_array)
    prediction1 = np.array(prediction0)
    prediction2 = pd.DataFrame(prediction1)
    platt = pickle.load(open('lr', 'rb'))
    prediction3 = platt.predict_proba(prediction2.values.reshape(-1,1))[:,1]
    predictionx = float(prediction3)
    predictiony = float(prediction1)
    prediction = str(format(predictionx, ".1%"))
    st.markdown('###')
    st.subheader('**MODEL OUTPUTS**')
    st.write('--')
    st.write('Predicted probability of Dysplasia:', prediction)
    st.caption('(Predicted probability with sensitivity above 95% during model development is **:orange[10%]**)')
    st.write('--')
    if predictiony < 0.5:
        st.write('Suggested Binary Dysplasia Status:', "**:green[LOW RISK]**")
    elif predictiony > 0.5:
        st.write('Suggested Binary Dysplasia Status:', "**:red[HIGH RISK]**")
    st.caption('Please note that the *Predicted Probability* is more informative than *Binary Status*')
    st.write('--')
    st.write('Explainability Heatmap:')
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    image2=display_gradcam(path, heatmap)
    st.image(image2, use_container_width=True)
    st.caption('_GradCAM heatmap showing region(s) influencing :blue[DysPOLNet’s] prediction_')
    st.markdown('####')
    st.markdown('####')
    st.markdown('####')
    st.markdown('####')
    st.markdown('####')
    st.write("Group Website: [Oral Cancer Research Theme, HKU](https://facdent.hku.hk/research/oral-cancer.html)  |  2024")
    


