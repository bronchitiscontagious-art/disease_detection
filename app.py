import streamlit as st
import numpy as np
import pickle
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image

# Page configuration
st.set_page_config(
    page_title="Poultry Disease Classification",
    page_icon="🐔",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 0.75rem;
        font-size: 1.1rem;
        border-radius: 8px;
        border: none;
        margin-top: 1rem;
    }
    .result-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .healthy {
        background-color: #d4edda;
        border: 2px solid #28a745;
    }
    .disease {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and feature extractor
@st.cache_resource
def load_models():
    try:
        # Load SVM model
        with open('MobileNetV2_SVM_model.pkl', 'rb') as f:
            svm_model = pickle.load(f)
        
        # Load MobileNetV2 for feature extraction
        feature_extractor = MobileNetV2(
            weights='imagenet',
            include_top=False,
            pooling='avg',
            input_shape=(224, 224, 3)
        )
        
        return svm_model, feature_extractor
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

# Feature extraction function
def extract_features(img, feature_extractor):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = feature_extractor.predict(img_array, verbose=0)
    return features

# Prediction function
def predict_disease(img, svm_model, feature_extractor):
    features = extract_features(img, feature_extractor)
    prediction = svm_model.predict(features)
    
    # Handle probability prediction
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
    st.title("🐔 Poultry Disease Classification")
    st.markdown("### AI-Powered Disease Detection System")
    st.markdown("Upload a poultry image to detect **Coccidiosis**, **Salmonella**, or check if it's **Healthy**")
    st.markdown("---")
    
    with st.spinner("Loading AI models..."):
        svm_model, feature_extractor = load_models()
    
    if svm_model is None or feature_extractor is None:
        st.error("❌ Failed to load models. Please check if 'MobileNetV2_SVM_model.pkl' exists.")
        return
    
    st.success("✅ Models loaded successfully!")
    
    uploaded_file = st.file_uploader(
        "Choose a poultry image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of poultry for disease detection"
    )
    
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert('RGB')
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(img, caption="Uploaded Image", use_column_width=True)
        
        if st.button("🔍 Analyze Image"):
            with st.spinner("Analyzing..."):
                try:
                    predicted_class, confidence, probabilities = predict_disease(
                        img, svm_model, feature_extractor
                    )
                    
                    st.markdown("---")
                    st.markdown("## 📊 Analysis Results")
                    
                    if predicted_class == "Healthy":
                        result_class = "healthy"
                        emoji = "✅"
                    else:
                        result_class = "disease"
                        emoji = "⚠️"
                    
                    st.markdown(f"""
                        <div class="result-box {result_class}">
                            <h2>{emoji} {predicted_class}</h2>
                            <h3>Confidence: {confidence:.2f}%</h3>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("### Detailed Probabilities:")
                    class_names = ['Coccidiosis', 'Healthy', 'Salmonella']
                    
                    for i, class_name in enumerate(class_names):
                        prob = probabilities[i] * 100
                        st.progress(float(probabilities[i]))
                        st.markdown(f"**{class_name}**: {prob:.2f}%")
                    
                    st.markdown("---")
                    st.markdown("### 💡 Recommendations:")
                    
                    if predicted_class == "Coccidiosis":
                        st.info("""
                        **Coccidiosis Detected**
                        - Isolate affected birds immediately
                        - Consult a veterinarian for medication
                        - Improve sanitation and hygiene
                        - Consider anticoccidial treatment
                        """)
                    elif predicted_class == "Salmonella":
                        st.warning("""
                        **Salmonella Detected**
                        - Quarantine affected birds
                        - Seek immediate veterinary care
                        - Enhance biosecurity measures
                        - Disinfect the environment thoroughly
                        """)
                    else:
                        st.success("""
                        **Healthy Birds**
                        - Continue regular health monitoring
                        - Maintain good hygiene practices
                        - Ensure proper nutrition and vaccination
                        - Keep records of health status
                        """)
                
                except Exception as e:
                    st.error(f"❌ Error during prediction: {str(e)}")
    
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: gray;'>
            <p>🔬 Powered by MobileNetV2 + SVM | Accuracy: 95.34%</p>
            <p>⚠️ This is an AI tool. Always consult a veterinarian for final diagnosis.</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
