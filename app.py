import streamlit as st
import numpy as np
import pickle
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image

# Page configuration
st.set_page_config(
    page_title="Poultry Disease Detection",
    page_icon="🐔",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Modern Mobile-App Style CSS
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main Container */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0 !important;
    }
    
    .block-container {
        padding: 1rem 1rem 2rem 1rem !important;
        max-width: 480px !important;
    }
    
    /* App Card */
    .app-card {
        background: white;
        border-radius: 30px;
        padding: 2rem 1.5rem;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        margin-bottom: 1rem;
    }
    
    /* Header */
    .app-header {
        text-align: center;
        margin-bottom: 1.5rem;
    }
    
    .app-icon {
        font-size: 4rem;
        margin-bottom: 0.5rem;
        animation: bounce 2s infinite;
    }
    
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    
    .app-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2d3748;
        margin: 0.5rem 0;
        line-height: 1.2;
    }
    
    .app-subtitle {
        font-size: 0.95rem;
        color: #718096;
        font-weight: 400;
    }
    
    /* Status Badge */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1.2rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Upload Section */
    .upload-section {
        background: linear-gradient(135deg, #f6f8fb 0%, #e9ecef 100%);
        border: 2px dashed #cbd5e0;
        border-radius: 20px;
        padding: 2rem 1rem;
        text-align: center;
        margin: 1.5rem 0;
        transition: all 0.3s ease;
    }
    
    .upload-section:hover {
        border-color: #667eea;
        background: linear-gradient(135deg, #f0f4ff 0%, #e9ecef 100%);
    }
    
    .upload-icon {
        font-size: 3rem;
        margin-bottom: 0.5rem;
        opacity: 0.6;
    }
    
    /* File Uploader Custom */
    .stFileUploader {
        background: transparent !important;
    }
    
    .stFileUploader > div {
        background: transparent !important;
        border: none !important;
    }
    
    /* Button Styles */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.9rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border: none;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        margin-top: 1rem;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 35px rgba(102, 126, 234, 0.5);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Result Card */
    .result-card {
        background: white;
        border-radius: 20px;
        padding: 2rem 1.5rem;
        margin: 1.5rem 0;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        animation: slideUp 0.5s ease;
    }
    
    @keyframes slideUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Result Header */
    .result-header {
        text-align: center;
        margin-bottom: 1.5rem;
    }
    
    .result-icon {
        font-size: 4rem;
        margin-bottom: 0.5rem;
    }
    
    .result-title {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .result-confidence {
        font-size: 1.2rem;
        font-weight: 500;
        opacity: 0.8;
    }
    
    /* Healthy Result */
    .healthy {
        background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%);
        color: #22543d;
    }
    
    .healthy .result-title {
        color: #22543d;
    }
    
    /* Disease Result */
    .disease {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: #742a2a;
    }
    
    .disease .result-title {
        color: #742a2a;
    }
    
    /* Probability Bars */
    .prob-container {
        margin: 1.5rem 0;
    }
    
    .prob-item {
        margin-bottom: 1rem;
    }
    
    .prob-label {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.3rem;
        font-size: 0.9rem;
        font-weight: 500;
        color: #4a5568;
    }
    
    .prob-bar {
        height: 10px;
        background: #e2e8f0;
        border-radius: 10px;
        overflow: hidden;
    }
    
    .prob-fill {
        height: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        transition: width 0.8s ease;
    }
    
    /* Recommendations Card */
    .recommend-card {
        background: linear-gradient(135deg, #ffeaa7 0%, #fdcb6e 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin-top: 1.5rem;
    }
    
    .recommend-card h4 {
        color: #744210;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .recommend-card ul {
        list-style: none;
        padding: 0;
        margin: 0;
    }
    
    .recommend-card li {
        color: #744210;
        padding: 0.4rem 0;
        padding-left: 1.5rem;
        position: relative;
        font-size: 0.9rem;
    }
    
    .recommend-card li:before {
        content: "✓";
        position: absolute;
        left: 0;
        font-weight: bold;
        color: #00b894;
    }
    
    /* Footer */
    .app-footer {
        text-align: center;
        color: white;
        padding: 1.5rem 0;
        font-size: 0.85rem;
        opacity: 0.9;
    }
    
    .footer-badge {
        display: inline-block;
        background: rgba(255,255,255,0.2);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.5rem 0;
        backdrop-filter: blur(10px);
    }
    
    /* Spinner Override */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    
    /* Image Display */
    .uploaded-image {
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        margin: 1rem 0;
    }
    
    /* Progress Bar Override */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    try:
        with open('classifier.pkl', 'rb') as f:
            svm_model = pickle.load(f)
        
        feature_extractor = MobileNetV2(
            weights='imagenet',
            include_top=False,
            pooling='avg',
            input_shape=(224, 224, 3)
        )
        
        return svm_model, feature_extractor
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return None, None

# Feature extraction
def extract_features(img, feature_extractor):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = feature_extractor.predict(img_array, verbose=0)
    return features

# Prediction
def predict_disease(img, svm_model, feature_extractor):
    features = extract_features(img, feature_extractor)
    prediction = svm_model.predict(features)
    
    try:
        probabilities = svm_model.predict_proba(features)[0]
    except:
        decision_scores = svm_model.decision_function(features)[0]
        exp_scores = np.exp(decision_scores - np.max(decision_scores))
        probabilities = exp_scores / exp_scores.sum()
    
    class_names = ['Coccidiosis', 'Healthy', 'Salmonella']
    predicted_class = class_names[prediction[0]]
    confidence = probabilities[prediction[0]] * 100
    
    return predicted_class, confidence, probabilities

# Main app
def main():
    # App Card Container
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    
    # Header
    st.markdown("""
        <div class="app-header">
            <div class="app-icon">🐔</div>
            <h1 class="app-title">Poultry Disease Detection</h1>
            <p class="app-subtitle">AI-Powered Health Diagnosis</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Load models
    svm_model, feature_extractor = load_models()
    
    if svm_model is None or feature_extractor is None:
        st.error("❌ Failed to load AI models")
        return
    
    # Status badge
    st.markdown("""
        <div style="text-align: center;">
            <span class="status-badge">
                <span>✓</span>
                <span>AI Models Ready</span>
            </span>
        </div>
    """, unsafe_allow_html=True)
    
    # Upload Section
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown('<div class="upload-icon">📸</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Tap to upload poultry image",
        type=['jpg', 'jpeg', 'png'],
        label_visibility="collapsed"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert('RGB')
        
        # Display uploaded image
        st.image(img, use_column_width=True, output_format='JPEG')
        
        # Analyze button
        if st.button("🔍 Analyze Image"):
            with st.spinner("🔬 Analyzing..."):
                try:
                    predicted_class, confidence, probabilities = predict_disease(
                        img, svm_model, feature_extractor
                    )
                    
                    # Close app card
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Result card
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    
                    # Result header
                    result_class = "healthy" if predicted_class == "Healthy" else "disease"
                    emoji = "✅" if predicted_class == "Healthy" else "⚠️"
                    
                    st.markdown(f"""
                        <div class="result-header {result_class}">
                            <div class="result-icon">{emoji}</div>
                            <h2 class="result-title">{predicted_class}</h2>
                            <p class="result-confidence">{confidence:.1f}% Confidence</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Probability breakdown
                    st.markdown("#### 📊 Detailed Analysis")
                    
                    class_names = ['Coccidiosis', 'Healthy', 'Salmonella']
                    class_icons = ['🦠', '✅', '🔬']
                    
                    for i, (class_name, icon) in enumerate(zip(class_names, class_icons)):
                        prob = probabilities[i] * 100
                        st.markdown(f"""
                            <div class="prob-item">
                                <div class="prob-label">
                                    <span>{icon} {class_name}</span>
                                    <span><strong>{prob:.1f}%</strong></span>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                        st.progress(float(probabilities[i]))
                    
                    # Recommendations
                    st.markdown('<div class="recommend-card">', unsafe_allow_html=True)
                    st.markdown("#### 💡 Recommendations")
                    
                    if predicted_class == "Coccidiosis":
                        st.markdown("""
                            <ul>
                                <li>Isolate affected birds immediately</li>
                                <li>Consult veterinarian for medication</li>
                                <li>Improve sanitation and hygiene</li>
                                <li>Use anticoccidial treatment</li>
                            </ul>
                        """, unsafe_allow_html=True)
                    elif predicted_class == "Salmonella":
                        st.markdown("""
                            <ul>
                                <li>Quarantine affected birds now</li>
                                <li>Seek immediate veterinary care</li>
                                <li>Enhance biosecurity measures</li>
                                <li>Disinfect environment thoroughly</li>
                            </ul>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                            <ul>
                                <li>Continue regular monitoring</li>
                                <li>Maintain good hygiene practices</li>
                                <li>Ensure proper nutrition</li>
                                <li>Keep health records updated</li>
                            </ul>
                        """, unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    else:
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
        <div class="app-footer">
            <div class="footer-badge">
                🔬 MobileNetV2 + SVM | 95.34% Accuracy
            </div>
            <p style="margin-top: 0.5rem; font-size: 0.75rem;">
                ⚠️ AI-based tool. Consult veterinarian for diagnosis.
            </p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
