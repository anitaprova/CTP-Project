import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from keras.models import Model
from keras.applications import VGG19
from keras.models import load_model
from PIL import Image
import os

def build_unet_model(input_shape=(256, 256, 3)):
    inputs = Input(shape=input_shape)

    # Encoder (downsampling path)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    # Bottleneck with more filters
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(c4)

    # Decoder (upsampling path)
    u5 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c4)
    u5 = concatenate([u5, c3])  
    c5 = Conv2D(256, (3, 3), activation='relu', padding='same')(u5)
    c5 = Conv2D(256, (3, 3), activation='relu', padding='same')(c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c2])  
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same')(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c1])  
    c7 = Conv2D(64, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(64, (3, 3), activation='relu', padding='same')(c7)

    outputs = Conv2D(3, (1, 1), activation='tanh')(c7)

    model = Model(inputs, outputs)
    return model

vgg = VGG19(weights="imagenet", include_top=False)
vgg_model = Model(inputs=vgg.input, outputs=vgg.get_layer("block3_conv3").output)
vgg_model.trainable = False

def compute_perceptual(y_true, y_pred):
    y_true_vgg = tf.keras.applications.vgg19.preprocess_input(y_true)
    y_pred_vgg = tf.keras.applications.vgg19.preprocess_input(y_pred)

    y_true_vgg = vgg_model(y_true_vgg)
    y_pred_vgg = vgg_model(y_pred_vgg)

    perceptual_loss = tf.reduce_mean(tf.square(y_true_vgg - y_pred_vgg))
    return perceptual_loss


def compute_gradients(y_true, y_pred):
    dy_true, dx_true = tf.image.image_gradients(y_true * 255)
    dy_pred, dx_pred = tf.image.image_gradients(y_pred * 255)

    gradient_loss = tf.reduce_mean(
        tf.abs(dy_pred - dy_true) + tf.abs(dx_pred - dx_true), axis=-1
    )
    return gradient_loss

def combined_loss(y_true, y_pred):
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    perceptual_loss = compute_perceptual(y_true, y_pred)
    gradient_loss = compute_gradients(y_true, y_pred)

    total_loss = mse_loss + 0.1 * perceptual_loss + 0.1 * gradient_loss
    return total_loss

def preprocess_image(image, target_size=(256, 256)):
    original_size = image.size

    image = image.convert("RGB")
    image = image.resize(target_size)
    image = np.array(image) / 255.0
    return image, original_size

unet_model = load_model(
    "model/unet_model.keras",
    custom_objects={"combined_loss": combined_loss, "Model": build_unet_model},
)

# Styling 
st.set_page_config(page_icon="🎨")
st.title("🎨 Art Restoration")
with open('app.css') as f:
    css = f.read()
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

# Sidebar for sample images
st.sidebar.title("Options")
sample_dir = "sample_images"
sample_images = [f for f in os.listdir(sample_dir) if f.endswith(("jpeg", "jpg", "png"))]
selected_sample = st.sidebar.selectbox("Choose a sample image:", ["None"] + sample_images)

# Selecting Uploaded or Sample Image
uploaded_file = st.file_uploader("", type=["jpeg", "jpg", "png"])
if selected_sample != "None":
    image_path = os.path.join(sample_dir, selected_sample)
    image = Image.open(image_path)
    input_image, original_size = preprocess_image(image)
elif uploaded_file:
    image = Image.open(uploaded_file)
    input_image, original_size = preprocess_image(image)
else:
    image = None

# Running the model on image
if image:
    input_image = tf.expand_dims(input_image, axis=0)

    predicted_image = unet_model.predict(input_image)
    predicted_image = tf.squeeze(predicted_image, axis=0)
    predicted_image = predicted_image.numpy()

    # Normalize predicted image to [0.0, 1.0] for display
    predicted_image = np.clip(predicted_image, 0.0, 1.0)

    predicted_image = Image.fromarray((predicted_image * 255).astype(np.uint8))
    predicted_image = predicted_image.resize(original_size)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Input Image", use_container_width=True)
    with col2:
        st.image(predicted_image, caption="Restored Image", use_container_width=True)
else:
    st.write("Please upload an image or select a sample from the sidebar.")
