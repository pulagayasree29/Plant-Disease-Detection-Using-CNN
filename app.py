from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

import numpy as np
from PIL import Image
import pickle as pk
import streamlit as st

a = load_model(r"C:\Users\GAYASREE\Desktop\cnn\cotton_model.h5")

st.title("🌿 Leaf Disease Detection System")
st.markdown("""
This application helps identify leaf diseases using a deep learning model trained on six classes:
- Aphids
- Army worm
- Bacterial blight
- Healthy
- Powdery mildew
- Target spot

Upload a leaf image below, and the system will predict the disease or confirm if the leaf is healthy.
""")
classes = ['Aphids', 'Army worm', 'Bacterial blight', 'Healthy', 'Powdery mildew', 'Target spot']
st.sidebar.title("Class Information")
st.sidebar.markdown("Select a class to learn more about it and view an example image.")

# Sidebar content: Class information and example images
selected_class = st.sidebar.selectbox("Select a Class", classes)
st.sidebar.markdown(f"### About {selected_class}")
b=st.file_uploader("Choose an file",accept_multiple_files=False)
def preprocess_image(image):
    # Resize the image to match the target size used during training
    image = image.resize((180, 180))  # Ensure dimensions match target_size in your data generator
    # Convert the image to an array
    image_array = img_to_array(image)
    # Normalize pixel values (1./255 rescale)
    image_array = image_array / 255.0
    # Expand dimensions to simulate a batch (batch_size=1)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array
if b is not None:
    # Display uploaded image
    image = Image.open(b)
    st.image(image, width="stretch")

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make predictions
    prediction = a.predict(processed_image)
    predicted_class_index = np.argmax(prediction)  # Get the index of the highest probability
    predicted_class_name = classes[predicted_class_index]  # Map index to class name

    # Display the class name
    st.write(f"Predicted Class: {predicted_class_name}")

    # Optional: Show prediction probabilities
    st.write("Prediction Probabilities:")
    for i, class_name in enumerate(classes):
        st.write(f"{class_name}: {prediction[0][i]:.2f}")