import streamlit as st
import tensorflow as tf
import numpy as np
import google.generativeai as genai
import os

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

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

# ──────────────────────────────────────────────
# Helper: format class name for display
# ──────────────────────────────────────────────
def format_disease_name(raw_name: str) -> str:
    """Convert 'Tomato___Early_blight' → 'Tomato: Early Blight'"""
    parts = raw_name.split("___")
    crop = parts[0].replace("_", " ").replace("(", "").replace(")", "").strip()
    condition = parts[1].replace("_", " ").strip() if len(parts) > 1 else ""
    return f"{crop}: {condition}" if condition else crop


# ──────────────────────────────────────────────
# Enhancement #1 — Top-3 predictions with confidence
# ──────────────────────────────────────────────
def model_prediction(test_image):
    """
    Returns:
        top_index   (int)  — index of best prediction
        top3        (list) — [(index, confidence_pct), ...] top 3
    """
    try:
        model = tf.keras.models.load_model("trained_plant_disease_model.keras")
        img = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
        input_arr = tf.keras.preprocessing.image.img_to_array(img)
        input_arr = np.array([input_arr])
        predictions = model.predict(input_arr)[0]          # shape: (38,)

        top3_indices = np.argsort(predictions)[::-1][:3]   # top-3 descending
        top3 = [(int(i), float(predictions[i]) * 100) for i in top3_indices]

        return top3[0][0], top3
    except Exception as e:
        st.error(f"Error during model prediction: {e}")
        return None, []


# ──────────────────────────────────────────────
# Enhancement #2 & #3 — Structured info + severity
# ──────────────────────────────────────────────
def get_disease_info(disease_name: str) -> dict:
    """
    Returns a dict with keys:
        symptoms, causes, organic_treatment, chemical_treatment,
        prevention, severity
    Falls back to plain text if parsing fails.
    """
    try:
        model = genai.GenerativeModel('gemini-1.0-pro-latest')

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
        response = model.generate_content(prompt)
        raw = response.parts[0].text.strip()

        # Parse the structured response
        sections = {
            "severity": "",
            "symptoms": "",
            "causes": "",
            "organic_treatment": "",
            "chemical_treatment": "",
            "prevention": "",
        }

        section_map = {
            "SEVERITY":           "severity",
            "SYMPTOMS":           "symptoms",
            "CAUSES":             "causes",
            "ORGANIC TREATMENT":  "organic_treatment",
            "CHEMICAL TREATMENT": "chemical_treatment",
            "PREVENTION":         "prevention",
        }

        current_key = None
        buffer = []

        for line in raw.splitlines():
            stripped = line.strip()
            matched = False
            for header, key in section_map.items():
                if stripped.upper().startswith(header + ":"):
                    if current_key:
                        sections[current_key] = "\n".join(buffer).strip()
                    current_key = key
                    inline = stripped[len(header) + 1:].strip()
                    buffer = [inline] if inline else []
                    matched = True
                    break
            if not matched and current_key:
                buffer.append(stripped)

        if current_key:
            sections[current_key] = "\n".join(buffer).strip()

        return sections

    except Exception as e:
        st.error(f"Error fetching disease information: {e}")
        return None


# ──────────────────────────────────────────────
# Severity badge helper
# ──────────────────────────────────────────────
def severity_badge(severity: str) -> str:
    s = severity.strip().lower()
    if s == "mild":
        return "🟢 **Mild**"
    elif s == "moderate":
        return "🟡 **Moderate**"
    elif s == "severe":
        return "🔴 **Severe**"
    return f"⚪ **{severity}**"


# ──────────────────────────────────────────────
# Confidence threshold warning
# ──────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 60.0


# ──────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# ── Home ──────────────────────────────────────
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
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

# ── About ─────────────────────────────────────
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

# ── Disease Recognition ───────────────────────
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])

    if test_image is not None:
        st.image(test_image, use_column_width=True)

    if st.button("Predict"):
        if test_image is None:
            st.warning("Please upload an image before clicking Predict.")
        else:
            with st.spinner("Analyzing image..."):
                top_index, top3 = model_prediction(test_image)

            if top_index is None:
                st.error("Model prediction failed. Please try again with a clearer image.")
            else:
                top_confidence = top3[0][1]
                disease_raw    = CLASS_NAMES[top_index]
                disease_label  = format_disease_name(disease_raw)

                # ── Confidence warning ────────────────
                if top_confidence < CONFIDENCE_THRESHOLD:
                    st.warning(
                        f"⚠️ Low confidence ({top_confidence:.1f}%). "
                        "The image may be unclear or the disease may not be in our database. "
                        "Please upload a clearer, closer image of the affected leaf."
                    )

                # ── Primary result ────────────────────
                st.success(f"✅ Predicted Disease: **{disease_label}**")

                # ── Enhancement #1: Top-3 confidence bars ──
                st.markdown("### 📊 Prediction Confidence")
                for rank, (idx, conf) in enumerate(top3):
                    label = format_disease_name(CLASS_NAMES[idx])
                    icon  = "🥇" if rank == 0 else ("🥈" if rank == 1 else "🥉")
                    st.markdown(f"{icon} **{label}** — `{conf:.1f}%`")
                    st.progress(min(conf / 100, 1.0))

                # ── Enhancements #2 & #3: Structured info ──
                st.markdown("---")
                st.markdown(f"### 🌿 Detailed Report: {disease_label}")

                with st.spinner("Fetching disease information..."):
                    info = get_disease_info(disease_raw)

                if info:
                    # Severity badge (Enhancement #3)
                    if info.get("severity"):
                        st.markdown(f"**Severity:** {severity_badge(info['severity'])}")
                        st.markdown("")

                    col1, col2 = st.columns(2)

                    with col1:
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
                        if info.get("organic_treatment"):
                            st.markdown("#### 🌱 Organic Treatment")
                            st.success(info["organic_treatment"])

                        if info.get("chemical_treatment"):
                            st.markdown("#### 💊 Chemical Treatment")
                            st.warning(info["chemical_treatment"])
                else:
                    st.error("Could not fetch disease information. Please check your Gemini API key.")
