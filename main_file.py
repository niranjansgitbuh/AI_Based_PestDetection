# ============================================================
# AgroShield — Plant Disease Recognition System
# ============================================================
# This application uses:
#   - TensorFlow/Keras CNN model  → to classify plant diseases from leaf images
#   - Google Gemini API           → to fetch structured disease information
#   - Streamlit                   → for the interactive web UI
#
# Flow:
#   User uploads image → Model predicts disease (Top-3 with confidence)
#   → Gemini returns Symptoms / Causes / Treatment / Severity
#   → Results displayed in a clean, structured layout
# ============================================================

import streamlit as st
import tensorflow as tf
import numpy as np
import google.generativeai as genai
import os


# ──────────────────────────────────────────────────────────────
# SECTION 1: API CONFIGURATION
# ──────────────────────────────────────────────────────────────
# We read the Gemini API key from an environment variable (GEMINI_API_KEY).
# This is safer than hardcoding the key directly in the source code.
# To set it, run in terminal: export GEMINI_API_KEY="your_key_here"
# ──────────────────────────────────────────────────────────────

api_key = os.getenv("GEMINI_API_KEY")   # Fetch API key from environment
genai.configure(api_key=api_key)         # Initialize Gemini with the API key


# ──────────────────────────────────────────────────────────────
# SECTION 2: CLASS LABELS
# ──────────────────────────────────────────────────────────────
# These are the 38 output classes that the CNN model can predict.
# Each label follows the format: CropName___ConditionName
# The model's output neuron index maps directly to this list.
# Example: index 20 → 'Potato___Early_blight'
# ──────────────────────────────────────────────────────────────

CLASS_NAMES = [
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


# ──────────────────────────────────────────────────────────────
# SECTION 3: HELPER FUNCTION — Format Disease Name
# ──────────────────────────────────────────────────────────────
# The raw class names from the dataset use triple underscores (___) as
# separators and single underscores (_) in place of spaces.
# This function converts them into a human-readable format.
#
# Example:
#   Input:  'Tomato___Early_blight'
#   Output: 'Tomato: Early Blight'
# ──────────────────────────────────────────────────────────────

def format_disease_name(raw_name: str) -> str:
    parts = raw_name.split("___")                                                 # Split into [crop, condition]
    crop = parts[0].replace("_", " ").replace("(", "").replace(")", "").strip()  # Clean crop name
    condition = parts[1].replace("_", " ").strip() if len(parts) > 1 else ""     # Clean condition name
    return f"{crop}: {condition}" if condition else crop                          # Combine into readable label


# ──────────────────────────────────────────────────────────────
# SECTION 4: MODEL PREDICTION — Top-3 with Confidence Scores
# ──────────────────────────────────────────────────────────────
# Loads the pre-trained Keras model and runs inference on the uploaded image.
#
# Steps:
#   1. Load the saved .keras model from disk
#   2. Resize the image to 128x128 (the size the model was trained on)
#   3. Convert image to a NumPy array and add a batch dimension
#   4. Run model.predict() → get probability scores for all 38 classes
#   5. Pick the top-3 highest probability classes
#
# Returns:
#   top_index (int)  — index of the best prediction
#   top3 (list)      — list of (class_index, confidence_%) for top 3
# ──────────────────────────────────────────────────────────────

def model_prediction(test_image):
    try:
        # Load the trained CNN model (only loads once per prediction call)
        model = tf.keras.models.load_model("trained_plant_disease_model.keras")

        # Preprocess the image: resize to match model's expected input size
        img = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))

        # Convert PIL image → NumPy array of shape (128, 128, 3)
        input_arr = tf.keras.preprocessing.image.img_to_array(img)

        # Add batch dimension → shape becomes (1, 128, 128, 3)
        # The model expects a batch of images, even if it's just one
        input_arr = np.array([input_arr])

        # Run inference — output shape: (1, 38) → one probability per class
        predictions = model.predict(input_arr)[0]   # [0] removes the batch dimension → shape (38,)

        # Sort class indices by probability (descending) and pick top 3
        top3_indices = np.argsort(predictions)[::-1][:3]

        # Build list of (class_index, confidence_percentage) tuples
        top3 = [(int(i), float(predictions[i]) * 100) for i in top3_indices]

        return top3[0][0], top3   # Return best index + full top-3 list

    except Exception as e:
        st.error(f"Error during model prediction: {e}")
        return None, []           # Return safe defaults on failure


# ──────────────────────────────────────────────────────────────
# SECTION 5: GEMINI API — Structured Disease Info + Severity
# ──────────────────────────────────────────────────────────────
# Sends a carefully crafted prompt to the Gemini LLM asking it to
# act as a plant pathologist and return disease info in a fixed format.
#
# Why a structured prompt?
#   → So we can parse each section (Symptoms, Treatment, etc.) separately
#   → Allows us to display them in different UI cards instead of one blob of text
#
# Returns a dict with these keys:
#   severity, symptoms, causes, organic_treatment, chemical_treatment, prevention
# ──────────────────────────────────────────────────────────────

def get_disease_info(disease_name: str) -> dict:
    try:
        # Initialize the Gemini generative model
        model = genai.GenerativeModel('gemini-1.0-pro-latest')

        # Structured prompt — forces Gemini to respond in a parseable format
        # Using exact section headers so our parser can split them reliably
        prompt = f"""
You are an expert plant pathologist. Provide information about the plant disease: "{disease_name}".

Respond ONLY in the following format with these exact section headers:

SEVERITY: [one of: Mild | Moderate | Severe]

SYMPTOMS:
[2-3 sentences describing visible symptoms]

CAUSES:
[1-2 sentences about the causative agent]

ORGANIC TREATMENT:
[2-3 actionable organic/natural treatment steps]

CHEMICAL TREATMENT:
[2-3 specific chemical treatments with product names if possible]

PREVENTION:
[2-3 practical prevention tips]

Do not add any text outside these sections.
"""
        # Call the Gemini API and extract the raw text from the response
        response = model.generate_content(prompt)
        raw = response.parts[0].text.strip()   # .parts[0].text gets the first text block

        # ── Initialize empty section buckets ──────────────────
        # These will be filled as we parse through the response line by line
        sections = {
            "severity": "",
            "symptoms": "",
            "causes": "",
            "organic_treatment": "",
            "chemical_treatment": "",
            "prevention": "",
        }

        # ── Map header strings → dict keys ────────────────────
        # Used to identify which section a line belongs to
        section_map = {
            "SEVERITY":           "severity",
            "SYMPTOMS":           "symptoms",
            "CAUSES":             "causes",
            "ORGANIC TREATMENT":  "organic_treatment",
            "CHEMICAL TREATMENT": "chemical_treatment",
            "PREVENTION":         "prevention",
        }

        # ── Line-by-line parser ────────────────────────────────
        # We track the "current section" and accumulate lines into a buffer.
        # When we hit a new section header, we save the buffer to the previous section.
        current_key = None   # Which section we're currently inside
        buffer = []          # Lines accumulated for the current section

        for line in raw.splitlines():
            stripped = line.strip()
            matched = False

            # Check if this line starts a new section header
            for header, key in section_map.items():
                if stripped.upper().startswith(header + ":"):
                    # Save previous section's content before switching
                    if current_key:
                        sections[current_key] = "\n".join(buffer).strip()

                    # Switch to the new section
                    current_key = key

                    # Some headers have inline content (e.g., "SEVERITY: Mild")
                    inline = stripped[len(header) + 1:].strip()
                    buffer = [inline] if inline else []
                    matched = True
                    break

            # If not a header line, it's content — add to current section's buffer
            if not matched and current_key:
                buffer.append(stripped)

        # Don't forget to save the last section after the loop ends
        if current_key:
            sections[current_key] = "\n".join(buffer).strip()

        return sections

    except Exception as e:
        st.error(f"Error fetching disease information: {e}")
        return None   # Return None so the caller can handle the failure gracefully


# ──────────────────────────────────────────────────────────────
# SECTION 6: SEVERITY BADGE HELPER
# ──────────────────────────────────────────────────────────────
# Converts a severity string into a color-coded emoji badge for display.
# This makes severity immediately visible at a glance.
#
# Mild     → 🟢  (low urgency)
# Moderate → 🟡  (action needed)
# Severe   → 🔴  (urgent treatment required)
# ──────────────────────────────────────────────────────────────

def severity_badge(severity: str) -> str:
    s = severity.strip().lower()
    if s == "mild":
        return "🟢 **Mild**"
    elif s == "moderate":
        return "🟡 **Moderate**"
    elif s == "severe":
        return "🔴 **Severe**"
    return f"⚪ **{severity}**"   # Fallback for unexpected values


# ──────────────────────────────────────────────────────────────
# SECTION 7: CONFIDENCE THRESHOLD
# ──────────────────────────────────────────────────────────────
# If the model's top prediction confidence is below this value (%),
# we show a warning to the user instead of presenting the result
# as definitive. This prevents false confidence in blurry/unclear images.
# Adjust this value based on your model's performance characteristics.
# ──────────────────────────────────────────────────────────────

CONFIDENCE_THRESHOLD = 60.0   # Percentage (0–100)


# ──────────────────────────────────────────────────────────────
# SECTION 8: STREAMLIT UI — Sidebar Navigation
# ──────────────────────────────────────────────────────────────
# Streamlit re-runs the entire script on every user interaction.
# The sidebar selectbox controls which "page" is rendered below.
# ──────────────────────────────────────────────────────────────

st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])


# ──────────────────────────────────────────────────────────────
# PAGE 1: HOME
# ──────────────────────────────────────────────────────────────
# Landing page with project overview, instructions, and feature highlights.
# ──────────────────────────────────────────────────────────────

if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")

    # Display a demo/banner image from local disk
    image_path = "/Users/balrajmalusare/Desktop/Plant_disease_detection/demo.jpeg"
    st.image(image_path, use_column_width=True)

    st.markdown("""
    Welcome to **AgroShield** - A Plant Disease Recognition System! 🌿🔍

    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant,
    and our system will analyze it to detect any signs of diseases. Together, let's protect our
    crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results with confidence scores and detailed treatment recommendations.

    ### What's New ✨
    - 📊 **Top-3 Predictions** with confidence percentages
    - 🔴 **Severity Estimation** — Mild / Moderate / Severe
    - 💊 **Structured Treatment Info** — Organic & Chemical options

    ### Why Choose Us?
    - **Accuracy:** State-of-the-art machine learning for precise disease detection.
    - **User-Friendly:** Simple and intuitive interface.
    - **Fast and Efficient:** Results in seconds for quick decision-making.
    """)


# ──────────────────────────────────────────────────────────────
# PAGE 2: ABOUT
# ──────────────────────────────────────────────────────────────
# Information about the dataset used to train the model.
# ──────────────────────────────────────────────────────────────

elif app_mode == "About":
    st.header("About")
    st.markdown("""
    #### About Dataset
    This dataset is recreated using offline augmentation from the original dataset.
    It consists of ~87K RGB images of healthy and diseased crop leaves across **38 classes**.
    Split: 80% training / 20% validation.

    #### Content
    1. train (70,295 images)
    2. test (33 images)
    3. validation (17,572 images)
    """)


# ──────────────────────────────────────────────────────────────
# PAGE 3: DISEASE RECOGNITION (Main Feature)
# ──────────────────────────────────────────────────────────────
# This is the core page where:
#   1. User uploads a leaf image
#   2. Model predicts the disease with top-3 confidence scores
#   3. Gemini API returns structured disease info
#   4. Everything is displayed in a clean 2-column layout
# ──────────────────────────────────────────────────────────────

elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")

    # File uploader — restricts to image formats only
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])

    # Show a preview of the uploaded image immediately after upload
    if test_image is not None:
        st.image(test_image, use_column_width=True)

    # ── Predict Button ─────────────────────────────────────────
    # Everything below runs only when the user clicks "Predict"
    if st.button("Predict"):

        # Guard: ensure an image is uploaded before predicting
        if test_image is None:
            st.warning("Please upload an image before clicking Predict.")
        else:
            # ── Step 1: Run the CNN model ──────────────────────
            with st.spinner("Analyzing image..."):
                top_index, top3 = model_prediction(test_image)

            # If prediction failed (e.g. model file missing), stop here
            if top_index is None:
                st.error("Model prediction failed. Please try again with a clearer image.")
            else:
                # Extract the top prediction's details
                top_confidence = top3[0][1]                        # Confidence % of best prediction
                disease_raw    = CLASS_NAMES[top_index]            # Raw label e.g. 'Tomato___Early_blight'
                disease_label  = format_disease_name(disease_raw)  # Human-readable label

                # ── Step 2: Confidence Warning ─────────────────
                # If model isn't confident enough, warn the user
                # rather than presenting a potentially wrong result as fact
                if top_confidence < CONFIDENCE_THRESHOLD:
                    st.warning(
                        f"⚠️ Low confidence ({top_confidence:.1f}%). "
                        "The image may be unclear or the disease may not be in our database. "
                        "Please upload a clearer, closer image of the affected leaf."
                    )

                # ── Step 3: Display Primary Result ────────────
                st.success(f"✅ Predicted Disease: **{disease_label}**")

                # ── Step 4: Top-3 Confidence Bars ─────────────
                # Shows the model's top 3 guesses with visual progress bars
                # so users understand how certain the model is
                st.markdown("### 📊 Prediction Confidence")
                for rank, (idx, conf) in enumerate(top3):
                    label = format_disease_name(CLASS_NAMES[idx])
                    icon  = "🥇" if rank == 0 else ("🥈" if rank == 1 else "🥉")
                    st.markdown(f"{icon} **{label}** — `{conf:.1f}%`")
                    st.progress(min(conf / 100, 1.0))   # progress() expects a value between 0.0–1.0

                # ── Step 5: Fetch & Display Structured Info ────
                # Separate spinner here since this is a second async operation (API call)
                st.markdown("---")
                st.markdown(f"### 🌿 Detailed Report: {disease_label}")

                with st.spinner("Fetching disease information..."):
                    info = get_disease_info(disease_raw)   # Pass raw name for accurate Gemini results

                if info:
                    # ── Severity Badge ─────────────────────────
                    # Displayed prominently at the top of the report
                    if info.get("severity"):
                        st.markdown(f"**Severity:** {severity_badge(info['severity'])}")
                        st.markdown("")   # Small spacer

                    # ── Two-column layout ──────────────────────
                    # Left column  → Diagnostic info (Symptoms, Causes, Prevention)
                    # Right column → Action info (Organic & Chemical Treatment)
                    col1, col2 = st.columns(2)

                    with col1:
                        # st.info() renders a blue info box — used for descriptive/neutral content
                        if info.get("symptoms"):
                            st.markdown("#### 🔍 Symptoms")
                            st.info(info["symptoms"])

                        if info.get("causes"):
                            st.markdown("#### 🦠 Causes")
                            st.info(info["causes"])

                        if info.get("prevention"):
                            st.markdown("#### 🛡️ Prevention")
                            st.info(info["prevention"])

                    with col2:
                        # st.success() renders a green box — used for positive/natural treatments
                        if info.get("organic_treatment"):
                            st.markdown("#### 🌱 Organic Treatment")
                            st.success(info["organic_treatment"])

                        # st.warning() renders an orange box — used for chemical treatments (caution)
                        if info.get("chemical_treatment"):
                            st.markdown("#### 💊 Chemical Treatment")
                            st.warning(info["chemical_treatment"])

                else:
                    # Gemini API call returned None — likely an auth or network issue
                    st.error("Could not fetch disease information. Please check your Gemini API key.")
