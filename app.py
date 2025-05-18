import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_v2_preprocess_input
import joblib
from pathlib import Path

st.set_page_config(page_title="Jackfruit Fruit Damage Classifier",
                   layout="wide", initial_sidebar_state="collapsed")

css = Path("styles.css").read_text()          # adjust path if needed
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# ------------------------------------------------------------------ CONFIG
CLASSES_FROM_YOUR_COLAB = ["Fruit_borer", "Fruit_fly", "Healthy", "Rhizopus_rot"]
CLASS_LABELS = {i: lbl for i, lbl in enumerate(CLASSES_FROM_YOUR_COLAB)}
IMAGE_SIZE = (224, 224)
MODEL_PATH_CNN = "jackfruit_cnn_feature_extractor.h5"
MODEL_PATH_SCALER = "jackfruit_feature_scaler.pkl"
MODEL_PATH_SVM = "jackfruit_svm_classifier.pkl"

# ------------------------------------------------------------------ LOAD MODELS
@st.cache_resource
def load_all_models():
    try:
        cnn = tf.keras.models.load_model(MODEL_PATH_CNN, compile=False)
        scaler = joblib.load(MODEL_PATH_SCALER)
        svm = joblib.load(MODEL_PATH_SVM)
        return cnn, scaler, svm
    except Exception as e:
        st.error(f"Error loading models: {e}. Please ensure model files are present and paths are correct.")
        st.stop()
cnn_model, scaler, svm_model = load_all_models()

# ------------------------------------------------------------------ HELPERS
def preprocess_image_for_prediction(img_pil):
    img = img_pil.resize(IMAGE_SIZE)
    arr = np.array(img)
    if arr.ndim == 2:
        arr = np.stack((arr,)*3, axis=-1)
    elif arr.shape[2] == 4:
        arr = arr[:, :, :3]
    arr = np.expand_dims(arr, 0)
    return mobilenet_v2_preprocess_input(arr.astype("float32"))

def predict_disease(img_pil):
    feats = cnn_model.predict(preprocess_image_for_prediction(img_pil))
    feats_scaled = scaler.transform(feats)
    idx = int(svm_model.predict(feats_scaled)[0])
    label = CLASS_LABELS.get(idx, f"Unknown Class ({idx})")

    conf = None
    probs = None
    if hasattr(svm_model, "predict_proba") and getattr(svm_model, 'probability', False):
        probs = svm_model.predict_proba(feats_scaled)
        conf = np.max(probs[0]) * 100
    elif hasattr(svm_model, "decision_function"):
        pass

    return label, conf, probs

# ------------------------------------------------------------------ SESSION STATE
for k in ("selected_image_pil", "analysis_results", "image_source"):
    if k not in st.session_state:
        st.session_state[k] = None


# ============================= PAGE HEADER =============================
st.markdown('<h1 class="main-header">Jackfruit Fruit Damage Classifier</h1>', unsafe_allow_html=True)


# ------------------------------------------------------------------ LAYOUT
col1, col2 = st.columns([6, 4])

# ============================= LEFT COLUMN =============================
with col1:
    st.markdown('<p class="column-title">Upload Jackfruit Image</p>', unsafe_allow_html=True)

    # Standard Streamlit File Uploader - its appearance will be modified by CSS
    uploaded = st.file_uploader(
        label="Drag and drop file here or click to browse", # Standard label
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
        key="jackfruit_uploader_styled_directly", # New key
        help="Upload a JPG, JPEG, or PNG image (max 200MB).",
        label_visibility="collapsed" 
    )

    if uploaded:
        unique_id = f"upload_{uploaded.name}_{uploaded.size}"
        if st.session_state.image_source != unique_id or st.session_state.selected_image_pil is None:
            try:
                st.session_state.selected_image_pil = Image.open(uploaded)
                st.session_state.analysis_results = None
                st.session_state.image_source = unique_id
                st.rerun()
            except Exception as e:
                st.error(f"Error opening uploaded image: {e}")
                st.session_state.selected_image_pil = None
                st.session_state.analysis_results = None
                st.session_state.image_source = None

# ============================= RIGHT COLUMN ============================
with col2:
    st.markdown('<p class="column-title">Analysis Results</p>',
                unsafe_allow_html=True)

    if st.session_state.selected_image_pil:
        st.image(st.session_state.selected_image_pil,
                 caption="Selected for Analysis", use_container_width=True)

        if st.button("Predict", key="diagnose_button", type="primary", use_container_width=True):
            with st.spinner("Analyzing…"):
                try:
                    pred, conf, probs = predict_disease(st.session_state.selected_image_pil)
                    st.session_state.analysis_results = {
                        "prediction": pred, "confidence": conf, "all_probs": probs
                    }
                except Exception as e:
                    st.session_state.analysis_results = {
                        "prediction": f"Error during analysis: {str(e)[:100]}...",
                        "confidence": None, "all_probs": None
                    }
            st.rerun()

    if st.session_state.analysis_results:
        res = st.session_state.analysis_results
        if st.session_state.selected_image_pil:
             st.markdown("---")
        st.subheader("Diagnosis:")

        prediction_text = str(res.get("prediction", ""))
        if "Error" in prediction_text or not prediction_text :
            st.error(f"⚠️ {prediction_text if prediction_text else 'Analysis did not return a result.'}")
        else:
            st.success(f"**Condition:** {res['prediction']}")
            if res.get("confidence") is not None:
                st.metric("Confidence", f"{res['confidence']:.2f}%")

            if res.get("all_probs") is not None and isinstance(res["all_probs"], np.ndarray):
                with st.expander("View Detailed Probabilities"):
                    prob_data = {CLASS_LABELS.get(i, f"Class {i}"): p
                                 for i, p in enumerate(res["all_probs"][0])}
                    st.bar_chart(prob_data)
            elif res.get("confidence") is None and (not hasattr(svm_model, "predict_proba") or not getattr(svm_model, 'probability', False)):
                 st.caption("Confidence scores are not available for this SVM model or probability was not enabled during training.")


    if not st.session_state.selected_image_pil and not st.session_state.analysis_results:
        st.markdown(
            '<p class="placeholder-text" style="text-align:center; padding-top:50px;">'
            'No image selected yet.</p>', unsafe_allow_html=True)
    elif st.session_state.selected_image_pil and not st.session_state.analysis_results:
        st.info("Click the **Predict** button above to analyze the uploaded image.")