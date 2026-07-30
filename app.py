import streamlit as st
import numpy as np 
import tensorflow as tf
from PIL import Image

# Configure the page
st.set_page_config(page_title="Concrete Bridge Deck Crack Detector", page_icon="", layout="centered")

st.title("🌉 Bridge Deck Crack Detector")
st.write("Upload a concrete bridge deck surface image to check for cracks")

# Load the saved model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("models/crack_classifier.keras")
    return model

# Prediction function
# Confirmed class order: index 0 = Cracked, index 1 = Non_cracked
def predict(model, pil_image):
    img = pil_image.convert("RGB").resize((128, 128))
    arr = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
    prob_non_cracked = float(model.predict(arr, verbose=0)[0][0])
    prob_cracked = 1.0 - prob_non_cracked
    label = "Non-Cracked" if prob_non_cracked >= 0.5 else "Cracked"
    return label, prob_cracked * 100, prob_non_cracked * 100

# Build the UI
model = load_model()

import gc

uploaded_file = st.file_uploader("Upload a concrete surface image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Show image first
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Classifying..."):
        label, cracked_pct, non_cracked_pct = predict(model, img)
        # <-- Free RAM after each image 
        gc.collect()

    st.write(f"**Prediction:** {label}")
    st.progress(min(int(cracked_pct), 100), text=f"Cracked: {cracked_pct:.1f}%")
    st.progress(min(int(non_cracked_pct), 100), text=f"Non-Cracked: {non_cracked_pct:.1f}%")
