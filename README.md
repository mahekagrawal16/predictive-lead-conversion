# Predictive Lead Conversion

A machine learning-based system that predicts the likelihood of a sales lead converting, using behavioral and demographic data. Designed to support targeted marketing strategies by identifying high-potential leads.
## 🖼️ App Preview

### 🎨 User Interface
![UI Screenshot](assets/Screenshot1.png)
![UI Screenshot](assets/Screenshot2.png)
---

## 🔍 Overview

This project explores lead conversion prediction using supervised learning models. It involves preprocessing metadata, engineering meaningful features, and training models to classify whether a lead is likely to convert or not.

---
## 🚀 Features

- ✅ Real-time lead conversion prediction
- 📈 Confidence score visualization
- 🔍 SHAP-based feature contribution analysis
- 📝 Downloadable PDF prediction report
- 🎨 Stylish and interactive Streamlit UI
- ⚡ Lightweight and fast – ready for real-world use

---

## 📊 Model Details

- **Algorithm**: Random Forest Classifier
- **Accuracy**: 84%
- **ROC-AUC Score**: 0.89
- **Interpretability**: SHAP (SHapley Additive exPlanations)

---

## 🧠 Tech Stack

- Python 3.9+
- Streamlit
- pandas, matplotlib
- SHAP
- FPDF
- joblib, pickle

---

## 🛠️ Installation

1. **Clone the Repository**:

```bash    
git clone https://github.com/yourusername/predictive-lead-conversion.git
cd predictive-lead-conversion

2. **Install Dependencies**:
pip install -r requirements.txt    
Run the App Locally:

3. **Run the App Locally**:   
streamlit run app.py    
ℹ️ Make sure rf_model.pkl, label_encoders.pkl, x_columns.pkl, and X_train.pkl are present in the root folder.

🧾 Sample Output

After prediction, the app generates:

📊 Prediction result (Converted / Not Converted)
📌 SHAP-based top 5 influencing features
📄 Downloadable PDF report with lead inputs, model summary, and business ROI

📥 Files Included
.
├── app.py                    # Main Streamlit app
├── rf_model.pkl              # Trained Random Forest model
├── label_encoders.pkl        # Encoders for categorical variables
├── x_columns.pkl             # Feature column order
├── X_train.pkl               # Training data for SHAP
├── requirements.txt          # Python dependencies
└── README.md                 # This file

📚 How to Use

Fill in lead information manually or load a sample.
Click Predict Conversion.
Analyze prediction, top features, and explanation.
Download the PDF report for sharing or records.

---

## 📬 Contact

For any questions or collaboration requests, feel free to reach out:  
**Mahek Agrawal**  
📧 mahek.suresh.aug2004@gmail.com  
[LinkedIn](https://linkedin.com/in/mahek-agrawal-503819255)
