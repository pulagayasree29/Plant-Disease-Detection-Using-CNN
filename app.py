from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

import numpy as np
from PIL import Image
import streamlit as st

# ✅ LOAD MODEL (IMPORTANT CHANGE)
model = load_model("cotton_model.h5")

# ================= UI DESIGN =================
st.markdown("""
<style>
body {
    background-color: #0B3D2E;
}
h1, h2, h3, p {
    color: #ECF0F1;
}
</style>
""", unsafe_allow_html=True)

st.title("🌿 Leaf Disease Detection System")

st.markdown("""
This application helps identify leaf diseases using Deep Learning.

### Classes:
- Healthy
- Army worm
- Bacterial blight
- Aphids
- Powdery mildew
- Target spot
""")

# ================= CLASS LIST =================
classes = ['Healthy', 'Army worm', 'Bacterial blight', 'Aphids', 'Powdery mildew', 'Target spot']

# ================= SIDEBAR =================
st.sidebar.title("Class Information")
selected_class = st.sidebar.selectbox("Select a Class", classes)

if selected_class == "Healthy":
    st.sidebar.markdown("A healthy cotton plant has vibrant green leaves.")

elif selected_class == "Army worm":
    st.sidebar.markdown("Armyworms are caterpillars that damage crops.")

elif selected_class == "Bacterial blight":
    st.sidebar.markdown("Caused by bacteria, affects leaf health.")

elif selected_class == "Aphids":
    st.sidebar.markdown("Small insects that suck nutrients.")

elif selected_class == "Powdery mildew":
    st.sidebar.markdown("Fungal disease causing white powder.")

elif selected_class == "Target spot":
    st.sidebar.markdown("Causes circular lesions on leaves.")

# ================= IMAGE PREPROCESS =================
def preprocess_image(image):
    image = image.resize((180, 180))
    image_array = img_to_array(image)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# ================= FILE UPLOAD =================
uploaded_file = st.file_uploader("📤 Upload a Leaf Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)

    # Preprocess
    processed_image = preprocess_image(image)

    # Prediction
    prediction = model.predict(processed_image)
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = classes[predicted_class_index]

    # ================= RESULT =================
    st.success(f"🌱 Predicted Class: {predicted_class_name}")

    # ================= DETAILS =================
    if predicted_class_index == 0:
        st.write("Type: Not a disease")
        st.write("Description: Healthy cotton plant")
        st.write("🌱 Control: No action needed. Maintain good care practices.")

    elif predicted_class_index == 1:
        st.write("Type: Pest")
        st.write("Description: Armyworms damage crops")
        st.write("🌿 Natural Control:")
        st.write("- Neem oil spray (weekly)")
        st.write("- Garlic + chili spray")
        st.write("- Handpicking larvae early morning")

    elif predicted_class_index == 2:
        st.write("Type: Disease")
        st.write("Description: Bacterial infection affecting leaves")
        st.write("🌿 Natural Control:")
        st.write("- Use turmeric water spray")
        st.write("- Neem oil helps reduce spread")
        st.write("- Remove infected leaves")

    elif predicted_class_index == 3:
        st.write("Type: Pest")
        st.write("Description: Aphids suck plant nutrients")
        st.write("🌿 Natural Control:")
        st.write("- Neem oil spray")
        st.write("- Soap water spray (mild)")
        st.write("- Introduce ladybugs (natural predators)")

    elif predicted_class_index == 4:
        st.write("Type: Disease")
        st.write("Description: Powdery mildew fungal infection")
        st.write("🌿 Natural Control:")
        st.write("- Baking soda spray (1 tsp + water)")
        st.write("- Milk spray (1:10 ratio)")
        st.write("- Improve air circulation")

    elif predicted_class_index == 5:
        st.write("Type: Disease")
        st.write("Description: Target spot causes circular lesions")
        st.write("🌿 Natural Control:")
        st.write("- Neem oil spray")
        st.write("- Remove infected leaves")
        st.write("- Use compost tea spray")

    # ================= PROBABILITIES =================
    st.write("### 📊 Prediction Probabilities")
    for i, class_name in enumerate(classes):
        st.write(f"{class_name}: {prediction[0][i]:.2f}")