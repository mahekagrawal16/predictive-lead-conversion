import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
from fpdf import FPDF
import joblib

# Load models and encoders
rf = pickle.load(open('rf_model.pkl', 'rb'))
label_encoders = pickle.load(open('label_encoders.pkl', 'rb'))
X_columns = pickle.load(open('x_columns.pkl', 'rb'))
X_train = joblib.load("X_train.pkl")

# Page config
st.set_page_config(page_title="Lead Conversion Predictor", layout="wide")

# Custom CSS
st.markdown("""
    <style>
        html, body, [data-testid="stAppViewContainer"] {
    background-color:rgba(15, 10, 15, 0.85);
    background-size: 600% 600%;
    animation: darkGradient 15s ease infinite;
    color: white !important;
    }

    @keyframes darkGradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .block-container {
        background-color: rgba(15, 10, 15, 0.85);
        padding: 4rem;
        color: white;
        border-radius: 20px;
        box-shadow: 0 0 20px rgba(0, 0, 0, 0.3);
    }
    

        h1, h2, h3 {
            color: #6b5b95;
            font-weight: bold;
        }

        .stButton>button {
            background-color: #f67280;
            color: white !important;
            padding: 0.6em 1.5em;
            border-radius: 10px;
            font-weight: 600;
            transition: 0.3s ease;
            box-shadow: 0 4px 12px rgba(255,182,185,0.4);
            border: none;
        }

        .stButton>button:hover {
            background-color: #ffb6b9;
            color: red !important;
            transform: scale(1.05);
        }

        .stDownloadButton>button {
            background-color: #808080;
            color: white;
            border-radius: 10px;
            font-weight: 600;
            transition: 0.3s ease;
            box-shadow: 0 4px 12px rgba(162,213,242,0.4);
            border: none;
        }

        .stDownloadButton>button:hover {
            background-color: #D3D3D3;
            color:black;
            transform: scale(1.05);
        }

        .stSidebar {
            background-color: #f8f4ff;
            color: #333;
        }

        .css-1d391kg, .css-1v0mbdj {
            color: #333;
        }
            /* Force labels and form headers to show in light color */
    label, .css-1p4vugk, .css-1cpxqw2, .st-emotion-cache-1cpxqw2, .st-emotion-cache-1p4vugk {
        color: #ffffff !important;
        font-weight: 500;
    }

    </style>
""", unsafe_allow_html=True)


# Title
st.title("📊AI Lead Conversion Predictor")
st.markdown("Use this tool to predict whether a lead will convert using a trained machine learning model. Get predictions, SHAP-based explanations, and a downloadable PDF report.")

# ---------------- Sidebar Section ----------------
st.sidebar.title("⚙️ Actions")

st.sidebar.info("""
Try a Sample Prediction!
""")
if st.sidebar.button("Load Sample Lead"):
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

st.sidebar.title("📖 Quick Guide")
st.sidebar.info("""
1. Enter lead details or use the demo.
2. Click Predict Conversion.
3. Analyze confidence and contributing features.
4. Download the prediction report as PDF.

✅ Tip: Focus on leads with high SHAP scores and website engagement!
""")
# Initialize toggle state
if "show_sidebar_info" not in st.session_state:
    st.session_state["show_sidebar_info"] = False

# Sidebar toggle button
with st.sidebar:
    if st.button("ⓘ Full User Guide"):
        st.session_state["show_sidebar_info"] = not st.session_state["show_sidebar_info"]

    # Conditional instruction content
    if st.session_state["show_sidebar_info"]:
        st.markdown("---")
        st.markdown("### 🧭 How to Use This App")
        st.markdown("""
        1. **Enter lead details** or use the sample.
        2. Click **Predict Conversion**.
        3. Analyze:
           - Confidence score
           - Top contributing features
           - SHAP explanation chart
        4. **Download the PDF report**.

        ---
        ### 🛠️ Technologies Used
        - **Python**
        - **Random Forest** for classification
        - **SHAP** for feature impact
        - **FPDF** for report generation
        - **Streamlit** for UI

        ---
        ### 📌 Notes
        - Higher website engagement ⬆️ = Better conversion chances.
        - Use this for smarter lead targeting & better ROI.
        """)

        st.markdown("---")

# Compact metrics using HTML
st.sidebar.markdown("""
<div style='display: flex; justify-content: space-around;margin-top:20px;font-size: 13px; text-align: center;'>
    <div style='padding: 15px 15px; background: #e0f7fa; border-radius: 8px;'>
        <div style='font-weight: bold;font-size: 14px;'>🧠 84%</div>
        <div style='font-size: 14px;'>Accuracy</div>
    </div>
    <div style='padding: 15px 15px; background: #e8f5e9; border-radius: 8px;'>
        <div style='font-weight: bold;font-size: 14px;'>📉 0.89</div>
        <div style='font-size: 14px;'>ROC-AUC</div>
    </div>
    <div style='padding:15px 15px; background: #fff3e0; border-radius: 8px;'>
        <div style='font-weight: bold;font-size: 14px;'>🔥 50%</div>
        <div style='font-size: 14px;'>Threshold</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown(
    """
    <div style='position: fixed; bottom: 30px; width: 18rem; text-align: center; font-size: 15px;'>
        🚀 <b>AI Lead Predictor</b><br>
        <strong>By Mahek Agrawal</strong> |
        <a href='https://linkedin.com/in/mahek-agrawal-503819255' target='_blank'>🔗 LinkedIn</a>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")
# User Input Form
st.header("Enter Lead Details")
input_form = {}
col1, col2, col3 = st.columns(3)
col_map = [col1, col2, col3]

for i, col in enumerate(X_columns):
    with col_map[i % 3]:
        if col in label_encoders:
            options = label_encoders[col].classes_.tolist()
            default = st.session_state.get(col, options[0])
            if default not in options:
                default = options[0]
            input_form[col] = st.selectbox(col, options, index=options.index(default))

        else:
            default_val = st.session_state.get(col, 0.0)
            input_form[col] = st.number_input(col, value=float(default_val))

# Predict
if st.button("🚀 Predict Conversion"):
    input_df = pd.DataFrame([input_form])
    for col, le in label_encoders.items():
        input_df[col] = le.transform(input_df[col])
    input_df = input_df[X_columns]

    prediction = rf.predict(input_df)[0]
    prob = rf.predict_proba(input_df)[0][1]
    pred_label = "Converted" if prediction == 1 else "Not Converted"

    st.markdown(f"""
    <div style="background-color: #e6f0ff; padding: 1rem 1.5rem;color: #084298; font-weight: 600;">
        🎉Prediction: <b>{pred_label}</b> ({prob * 100:.2f}% Confidence)
    </div>
    """, unsafe_allow_html=True)

    # Confidence Score
    st.subheader("Confidence Score")
    st.markdown("This shows the model's estimated probability of the lead converting:")

    fig, ax = plt.subplots(figsize=(9,0.2))
    ax.barh([0], [prob], color='green' if prob > 0.5 else 'red')
    ax.set_xlim([0, 1])
    ax.set_yticks([])
    ax.set_xlabel("Probability of Conversion")
    ax.set_title("Model Confidence")
    st.pyplot(fig)

    # SHAP-based explanation
    st.subheader("Feature Contribution Analysis")
    st.markdown("The chart below explains which features influenced the prediction most significantly.")

    explainer = shap.Explainer(rf, X_train)
    shap_values = explainer(input_df)

    # Create single explanation object
    single_explainer = shap.Explanation(
        values=shap_values.values[0][1],  # For class 'Converted'
        base_values=shap_values.base_values[0][1],
        data=shap_values.data[0],
        feature_names=shap_values.feature_names
    )

    # Waterfall plot for visual explanation
    shap.plots.waterfall(single_explainer, max_display=10)

    # Top 5 contributing features
    st.subheader("Top Influential Features")
    st.markdown("These are the top 5 features that influenced the prediction:")

    vals = shap_values.values[0][1]
    feats = shap_values.feature_names
    top5 = sorted(zip(feats, vals), key=lambda x: abs(x[1]), reverse=True)[:5]

    top5_df = pd.DataFrame(top5, columns=["Feature", "Impact"])
    top5_df["Direction"] = top5_df["Impact"].apply(lambda x: "Positive" if x > 0 else "Negative")
    top5_df["Impact"] = top5_df["Impact"].round(3)

    st.dataframe(top5_df[["Feature", "Direction", "Impact"]])

        
    class StyledPDFWithTable(FPDF):
        def header(self):
            self.set_font("Arial", 'B', 16)
            self.set_text_color(30, 60, 90)
            self.cell(0, 10, "Lead Conversion Prediction Report", ln=True, align='C')
            self.set_draw_color(160, 160, 160)
            self.set_line_width(0.5)
            self.line(10, 20, 200, 20)
            self.ln(5)

        def footer(self):
            self.set_y(-15)
            self.set_font("Arial", 'I', 10)
            self.set_text_color(120, 120, 120)
            self.cell(0, 10, "Generated by AI Lead Predictor | Developed by Mahek Agrawal", 0, 0, 'C')

    # PDF Generator
    def generate_pdf(pred, prob, inputs):
        pdf = StyledPDFWithTable()
        pdf.add_page()

        # Summary
        pdf.set_font("Arial", 'B', 14)
        pdf.set_text_color(0, 102, 102)
        pdf.cell(0, 10, "Prediction Summary", ln=True)
        pdf.set_font("Arial", '', 12)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 10, f"Prediction Result: {pred}", ln=True)
        pdf.cell(0, 10, f"Confidence Score: {prob * 100:.2f}%", ln=True)
        pdf.ln(5)
        pdf.set_draw_color(200, 200, 200)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(10)

        # Input Table
        pdf.set_font("Arial", 'B', 14)
        pdf.set_text_color(0, 102, 102)
        pdf.cell(0, 10, "Lead Input Details", ln=True)
        pdf.set_font("Arial", 'B', 12)
        pdf.set_fill_color(220, 230, 240)
        pdf.set_text_color(0)
        pdf.cell(70, 8, "Feature", border=1, align='C', fill=True)
        pdf.cell(120, 8, "Value", border=1, align='C', fill=True)
        pdf.ln()

        pdf.set_font("Arial", '', 12)
        for k, v in inputs.items():
            pdf.cell(70, 8, str(k), border=1)
            pdf.cell(120, 8, str(v), border=1)
            pdf.ln()
        pdf.output("report.pdf")


    generate_pdf(pred_label, prob, input_form)
    with open("report.pdf", "rb") as file:
        st.download_button("Download PDF Report", data=file.read(), file_name="lead_report.pdf", mime="application/pdf")
