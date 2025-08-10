import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Class labels
CLASS_NAMES = [
    "animal_fish", "bass", "black_sea_sprat", "glit_head_bream", "horse_mackerel",
    "red_mullet", "red_sea_bream", "sea_bass", "shrimp", "striped_red_mullet", "trout"
]

# Load model
model = load_model("fish_classifier_model.h5")  # change to your model path

st.title("Fish Classifier 🐟")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = load_img(uploaded_file, target_size=(150, 150))  # match training size
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # IMPORTANT: match training rescale
    img_array = np.expand_dims(img_array, axis=0)  # shape (1, 150, 150, 3)

    # Predict
    predictions = model.predict(img_array)
    pred_index = np.argmax(predictions)
    pred_label = CLASS_NAMES[pred_index]
    confidence = predictions[0][pred_index] * 100

    st.markdown(f"**Prediction:** {pred_label} ({confidence:.2f}% confidence)")
