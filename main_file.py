import streamlit as st
import tensorflow as tf
import numpy as np
import google.generativeai as genai
import os
from config import API_KEY

# Set up the API key
api_key = os.getenv("GEMINI_API_KEY")  # Use the correct API key
genai.configure(api_key=api_key)
generation_config = {"temperature": 0.9, "top_p": 1, "top_k": 1, "max_output_tokens": 2048}

# TensorFlow Model Prediction
def model_prediction(test_image):
    try:
        model = tf.keras.models.load_model("trained_plant_disease_model.keras")
        img = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
        input_arr = tf.keras.preprocessing.image.img_to_array(img)
        input_arr = np.array([input_arr])  # Convert single image to batch
        predictions = model.predict(input_arr)
        predicted_class = np.argmax(predictions, axis=1)[0]  # Get index of max in first dimension, then first element
        return predicted_class
    except Exception as e:
        st.error(f"Error during model prediction: {e}")
        return None  # Or set a default value for empty predictions


# Get disease information from Gemini API
def get_disease_info(disease_name):
    try:
        model = genai.GenerativeModel('gemini-1.0-pro-latest')
        prompt = f"Provide detailed information about the plant disease {disease_name}."
        response = model.generate_content(prompt)

        # Access the text content from the first part
        disease_info = response.parts[0].text

        return disease_info
    except Exception as e:
        st.error(f"Error fetching disease information: {e}")
        return None


# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Main Page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "/Users/balrajmalusare/Desktop/Plant_disease_detection/demo.jpeg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
    Welcome to AgroShield - A Plant Disease Recognition System! 🌿🔍

    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

# About Project
elif app_mode == "About":
    st.header("About")
    st.markdown("""
    #### About Dataset
    This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this GitHub repo.
    This dataset consists of about 87K RGB images of healthy and diseased crop leaves, categorized into 38 different classes. The total dataset is divided into an 80/20 ratio of training and validation sets, preserving the directory structure.
    A new directory containing 33 test images is created later for prediction purposes.
    #### Content
    1. train (70295 images)
    2. test (33 images)
    3. validation (17572 images)
    """)

# Prediction Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if test_image is not None:
        st.image(test_image, use_column_width=True)
    
    # Predict button
    if st.button("Predict"):
        with st.spinner("Processing image..."):
            result_index = model_prediction(test_image)
            if result_index is not None:
        
                # Reading Labels
                class_name = [
                'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
                'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
                'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                'Tomato___healthy'
                ]
                disease_name = class_name[result_index]
            st.success(f"Model is predicting it's {disease_name}")
        
            # Fetch detailed information using Gemini API
            disease_info = get_disease_info(disease_name)
            st.markdown(f"### Detailed Information about {disease_name}")
            st.write(disease_info)
else:
    st.error("Model prediction failed. Please try again.")