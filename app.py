import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
from fpdf import FPDF
import joblib

# Load model and metadata
rf = pickle.load(open('rf_model.pkl', 'rb'))
label_encoders = pickle.load(open('label_encoders.pkl', 'rb'))
X_columns = pickle.load(open('x_columns.pkl', 'rb'))
X_train = joblib.load("X_train.pkl")

st.set_page_config(page_title="Lead Conversion Predictor", layout="centered")
st.title("🎯 AI Lead Conversion Predictor")
st.markdown("Predict if a lead is likely to convert using trained machine learning model.")

# Sample load
if "sample" not in st.session_state:
    st.session_state.sample = {}

if st.button("🎲 Load Sample Lead"):
    sample = {
        'Lead Origin': 'Landing Page Submission',
        'Lead Source': 'Google',
        'City': 'Mumbai',
        'Total Time Spent on Website': 90,
        'Page Views Per Visit': 5,
        'Specialization': 'Management Studies',
        'How did you hear about us?': 'Online Search',
        'What is your current occupation': 'Unemployed',
        'Tags': 'Interested in Data Science',
    }
    for key, value in sample.items():
        st.session_state[key] = value

# User input
st.header("📥 Enter Lead Details")
user_input = {}
for col in X_columns:
    if col in label_encoders:
        options = label_encoders[col].classes_
        default = st.session_state.get(col, options[0])
        if default not in options:
            default = options[0]
        user_input[col] = st.selectbox(col, options, index=options.tolist().index(default))
    else:
        default_val = st.session_state.get(col, 0.0)
        user_input[col] = st.number_input(col, value=float(default_val))

# Optional insights
if 'Total Time Spent on Website' in user_input and user_input['Total Time Spent on Website'] < 30:
    st.warning("⚠️ This lead has very low engagement on the website.")

# Predict button
if st.button("🚀 Predict"):
    input_df = pd.DataFrame([user_input])
    for col, le in label_encoders.items():
        input_df[col] = le.transform(input_df[col])
    input_df = input_df[X_columns]

    prediction = rf.predict(input_df)[0]
    prob = rf.predict_proba(input_df)[0][1]

    pred_label = "Converted" if prediction == 1 else "Not Converted"
    st.success(f"Prediction: **{pred_label}** ({prob * 100:.2f}%)")

    # Confidence Meter
    def confidence_bar(prob):
        fig, ax = plt.subplots(figsize=(5, 0.5))
        ax.barh(0, prob, color='green' if prob > 0.5 else 'red')
        ax.set_xlim([0, 1])
        ax.set_yticks([])
        ax.set_title("Confidence Score", fontsize=10)
        st.pyplot(fig)

    confidence_bar(prob)

    # SHAP explanation
    # SHAP explanation
    st.subheader("🔍 SHAP Feature Contribution (Waterfall)")
    explainer = shap.Explainer(rf, X_train)
    shap_values = explainer(input_df)

    # Build single explanation for class 1
    single_explainer = shap.Explanation(
        values=shap_values.values[0][1],
        base_values=shap_values.base_values[0][1],
        data=shap_values.data[0],
        feature_names=shap_values.feature_names
    )

    # Waterfall plot
    shap.plots.waterfall(single_explainer, max_display=10)

    # Top 5 features
    vals = shap_values.values[0][1]
    feats = shap_values.feature_names
    top5 = sorted(zip(feats, vals), key=lambda x: abs(x[1]), reverse=True)[:5]

    st.subheader("🔑 Top 5 Influencing Features")
    for feat, val in top5:
        st.markdown(f"- **{feat}**: {'+' if val > 0 else '-'}{abs(val):.2f}")


    # PDF Report
    def generate_pdf(pred, prob, inputs):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, "Lead Conversion Prediction Report", ln=True, align='C')
        pdf.ln(10)
        pdf.cell(200, 10, f"Prediction: {pred}", ln=True)
        pdf.cell(200, 10, f"Confidence: {prob * 100:.2f}%", ln=True)
        pdf.ln(10)
        pdf.cell(200, 10, "Input Features:", ln=True)
        for k, v in inputs.items():
            pdf.cell(200, 10, f"{k}: {v}", ln=True)
        pdf.output("report.pdf")

    generate_pdf(pred_label, prob, user_input)
    with open("report.pdf", "rb") as file:
        st.download_button("📥 Download Report PDF", data=file, file_name="lead_report.pdf")
