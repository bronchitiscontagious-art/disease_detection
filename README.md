# 🐔 Poultry Disease Classification App

AI-powered web application for detecting poultry diseases using MobileNetV2 + SVM classifier.

## 🎯 Features

- **Real-time Disease Detection**: Upload poultry images and get instant predictions
- **3 Disease Classes**: Coccidiosis, Salmonella, and Healthy
- **High Accuracy**: 95.34% test accuracy
- **User-Friendly Interface**: Built with Streamlit
- **Confidence Scores**: Shows probability for each class

## 🚀 Model Performance

- **Training Accuracy**: 100%
- **Validation Accuracy**: 97.10%
- **Test Accuracy**: 95.34%
- **Architecture**: MobileNetV2 (feature extraction) + SVM (classification)

## 📦 Installation

### Local Setup

```bash
# Clone repository
git clone <your-repo-url>
cd poultry-disease-classifier

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
```

### Required Files

Make sure you have `MobileNetV2_SVM_model.pkl` in the root directory.

## 🌐 Deployment

### Railway Deployment

1. Push code to GitHub
2. Connect Railway to your GitHub repository
3. Railway will automatically detect and deploy the Streamlit app
4. Set environment variables if needed

## 🔧 Tech Stack

- **Framework**: Streamlit
- **ML Model**: MobileNetV2 + SVM
- **Deep Learning**: TensorFlow 2.10.0
- **Image Processing**: Pillow
- **ML Library**: scikit-learn

## 📊 Dataset

- **Total Images**: 6,436
- **Classes**: 
  - Coccidiosis: 2,103 images
  - Healthy: 2,057 images
  - Salmonella: 2,276 images
- **Balance Ratio**: 0.904 (Well Balanced)

## ⚠️ Disclaimer

This is an AI-based diagnostic tool. Always consult a qualified veterinarian for final diagnosis and treatment decisions.

## 📝 License

[Your License Here]

## 👨‍💻 Author

[Your Name]
