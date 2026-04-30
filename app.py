# Import libraries
import streamlit as st
import numpy as np
import pickle

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

# Page configuration
st.set_page_config(page_title="Loan Approval System", layout="centered")

# Title
st.title("🏦 Loan Approval Prediction System")
 
st.info(
    "Prediction based on income, loan amount, credit history and education. "
    "Higher income and good credit history increase chances of approval."
)

st.markdown("### Enter Applicant Details")

# Layout (2 columns)
col1, col2 = st.columns(2)

with col1:
    income = st.number_input("Applicant Income", min_value=0)
    loan_amount = st.number_input("Loan Amount", min_value=0)

with col2:
    credit_history = st.selectbox("Credit History", ["Good", "Bad"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])

# Convert categorical inputs
credit = 1 if credit_history == "Good" else 0
edu = 0 if education == "Graduate" else 1

st.divider()

# Prediction
if st.button("🔍 Predict Loan Status"):

    if income == 0 or loan_amount == 0:
        st.warning("Please enter valid values")

    else:
        data = np.array([[income, loan_amount, credit, edu]])

        # Prediction
        result = model.predict(data)

        # Probability (IMPORTANT)
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(data)[0][1]  # probability of approval
            st.write(f"Approval Probability: {prob*100:.2f}%")

        # Result
        if result[0] == 1:
            st.success("✅ Loan Approved")
        else:
            st.error("❌ Loan Rejected")

        # Explanation
        if credit == 0:
            st.warning("Reason: Poor credit history reduces approval chances")

        elif income < 3000:
            st.warning("Reason: Low income may affect loan approval")

        else:
            st.info("Reason: Good financial profile")

# Footer
st.markdown("---")
st.caption("Machine Learning Project | Loan Approval Prediction")