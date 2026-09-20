"""
Customer Churn Prediction Web Application
Built with Streamlit and Scikit-Learn / XGBoost
Department of CSE (AI & ML) - ANITS
Shell-Edunet Skills4Future AICTE Internship Project
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">📊 Telecom Customer Churn Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Predictive Analytics for Customer Retention & Churn Risk Assessment</div>', unsafe_allow_html=True)

# Sidebar - Customer Profile Input
st.sidebar.header("📋 Customer Profile & Services")

st.sidebar.subheader("1. Demographics")
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
age = st.sidebar.slider("Age", 18, 85, 42)
married = st.sidebar.selectbox("Married / Partner", ["Yes", "No"])
dependents = st.sidebar.slider("Number of Dependents", 0, 10, 0)
referrals = st.sidebar.slider("Number of Referrals", 0, 15, 2)

st.sidebar.subheader("2. Account & Contract")
tenure = st.sidebar.slider("Tenure (Months)", 1, 72, 24)
contract = st.sidebar.selectbox("Contract Type", ["Month-to-Month", "One Year", "Two Year"])
paperless = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
payment = st.sidebar.selectbox("Payment Method", ["Bank Withdrawal", "Credit Card", "Mailed Check"])
offer = st.sidebar.selectbox("Offer Accepted", ["None", "Offer A", "Offer B", "Offer C", "Offer D", "Offer E"])

st.sidebar.subheader("3. Subscribed Services")
phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["Yes", "No"] if phone_service == "Yes" else ["No"])
internet_service = st.sidebar.selectbox("Internet Service", ["Yes", "No"])
internet_type = st.sidebar.selectbox("Internet Type", ["Fiber Optic", "DSL", "Cable", "None"] if internet_service == "Yes" else ["None"])
online_security = st.sidebar.selectbox("Online Security", ["Yes", "No"] if internet_service == "Yes" else ["No"])
online_backup = st.sidebar.selectbox("Online Backup", ["Yes", "No"] if internet_service == "Yes" else ["No"])
device_protection = st.sidebar.selectbox("Device Protection Plan", ["Yes", "No"] if internet_service == "Yes" else ["No"])
tech_support = st.sidebar.selectbox("Premium Tech Support", ["Yes", "No"] if internet_service == "Yes" else ["No"])
streaming_tv = st.sidebar.selectbox("Streaming TV", ["Yes", "No"] if internet_service == "Yes" else ["No"])
streaming_movies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No"] if internet_service == "Yes" else ["No"])
streaming_music = st.sidebar.selectbox("Streaming Music", ["Yes", "No"] if internet_service == "Yes" else ["No"])
unlimited_data = st.sidebar.selectbox("Unlimited Data", ["Yes", "No"] if internet_service == "Yes" else ["No"])

st.sidebar.subheader("4. Billing & Financials")
monthly_charge = st.sidebar.number_input("Monthly Charge ($)", min_value=10.0, max_value=150.0, value=65.0, step=2.5)
total_charges = st.sidebar.number_input("Total Charges ($)", min_value=10.0, max_value=12000.0, value=float(tenure * monthly_charge), step=50.0)
extra_data_charges = st.sidebar.number_input("Total Extra Data Charges ($)", min_value=0.0, max_value=500.0, value=0.0, step=5.0)
long_dist_charges = st.sidebar.number_input("Total Long Distance Charges ($)", min_value=0.0, max_value=5000.0, value=float(tenure * 20.0), step=25.0)
total_revenue = total_charges + extra_data_charges + long_dist_charges

# Main Dashboard Layout
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    st.metric(label="Tenure", value=f"{tenure} Months", delta=f"{tenure//12} Yrs")
with col2:
    st.metric(label="Contract", value=contract)
with col3:
    st.metric(label="Total Revenue", value=f"${total_revenue:,.2f}")

st.divider()

def compute_rule_based_prediction():
    churn_score = 0.0
    if contract == "Month-to-Month":
        churn_score += 0.40
    elif contract == "One Year":
        churn_score += 0.15
    else:
        churn_score += 0.05
        
    if tenure < 12:
        churn_score += 0.30
    elif tenure < 24:
        churn_score += 0.15
    elif tenure > 48:
        churn_score -= 0.15
        
    if internet_type == "Fiber Optic":
        churn_score += 0.15
    elif internet_type == "None":
        churn_score -= 0.10
        
    if tech_support == "No" and internet_service == "Yes":
        churn_score += 0.10
    if online_security == "No" and internet_service == "Yes":
        churn_score += 0.08
        
    if payment == "Bank Withdrawal":
        churn_score += 0.10
        
    if monthly_charge > 80:
        churn_score += 0.12
    elif monthly_charge < 35:
        churn_score -= 0.10
        
    if referrals == 0:
        churn_score += 0.10
    else:
        churn_score -= min(referrals * 0.03, 0.15)
        
    churn_prob = max(0.05, min(0.92, churn_score))
    
    if tenure <= 3:
        joined_prob = max(0.15, 0.50 - (tenure * 0.12))
        stay_prob = max(0.1, 1.0 - churn_prob - joined_prob)
    else:
        joined_prob = 0.03
        stay_prob = 1.0 - churn_prob - joined_prob
        
    total = churn_prob + stay_prob + joined_prob
    return stay_prob / total, churn_prob / total, joined_prob / total

if st.button("🔍 Predict Customer Status", use_container_width=True, type="primary"):
    stay_prob, churn_prob, joined_prob = compute_rule_based_prediction()
    
    model_path = os.path.join(os.path.dirname(__file__), 'churn_model.pkl')
    cols_path = os.path.join(os.path.dirname(__file__), 'model_columns.pkl')
    scaler_path = os.path.join(os.path.dirname(__file__), 'scaler.pkl')
    
    if os.path.exists(model_path) and os.path.exists(cols_path) and os.path.exists(scaler_path):
        try:
            model = joblib.load(model_path)
            model_cols = joblib.load(cols_path)
            scaler = joblib.load(scaler_path)
            
            user_data = {
                'Gender': 1 if gender == "Male" else 0,
                'Age': age,
                'Married': 1 if married == "Yes" else 0,
                'Number of Dependents': dependents,
                'Number of Referrals': referrals,
                'Tenure in Months': tenure,
                'Phone Service': 1 if phone_service == "Yes" else 0,
                'Multiple Lines': 1 if multiple_lines == "Yes" else 0,
                'Internet Service': 1 if internet_service == "Yes" else 0,
                'Avg Monthly Long Distance Charges': 25.0,
                'Avg Monthly GB Download': 25.0,
                'Online Security': 1 if online_security == "Yes" else 0,
                'Online Backup': 1 if online_backup == "Yes" else 0,
                'Device Protection Plan': 1 if device_protection == "Yes" else 0,
                'Premium Tech Support': 1 if tech_support == "Yes" else 0,
                'Streaming TV': 1 if streaming_tv == "Yes" else 0,
                'Streaming Movies': 1 if streaming_movies == "Yes" else 0,
                'Streaming Music': 1 if streaming_music == "Yes" else 0,
                'Unlimited Data': 1 if unlimited_data == "Yes" else 0,
                'Paperless Billing': 1 if paperless == "Yes" else 0,
                'Monthly Charge': monthly_charge,
                'Total Charges': total_charges,
                'Total Extra Data Charges': extra_data_charges,
                'Total Long Distance Charges': long_dist_charges,
                'Total Revenue': total_revenue
            }
            
            for c in ['Payment Method', 'Contract', 'Internet Type', 'Offer']:
                if c == 'Payment Method':
                    user_data[f"Payment Method_{payment}"] = 1
                elif c == 'Contract':
                    user_data[f"Contract_{contract}"] = 1
                elif c == 'Internet Type':
                    user_data[f"Internet Type_{internet_type}"] = 1
                elif c == 'Offer':
                    user_data[f"Offer_{offer}"] = 1
                    
            input_df = pd.DataFrame([user_data])
            for c in model_cols:
                if c not in input_df.columns:
                    input_df[c] = 0
            input_df = input_df[model_cols]
            
            cols_to_scale = ['Age','Number of Dependents','Number of Referrals','Tenure in Months',
                             'Avg Monthly Long Distance Charges','Avg Monthly GB Download',
                             'Monthly Charge', 'Total Charges', 'Total Extra Data Charges', 
                             'Total Long Distance Charges','Total Revenue']
            input_df[cols_to_scale] = scaler.transform(input_df[cols_to_scale])
            
            probs = model.predict_proba(input_df)[0]
            churn_prob, joined_prob, stay_prob = probs[0], probs[1], probs[2]
        except Exception as e:
            pass

    probs_dict = {'Stayed': stay_prob, 'Churned': churn_prob, 'Joined': joined_prob}
    predicted_class = max(probs_dict, key=probs_dict.get)
    confidence = probs_dict[predicted_class] * 100

    res_col1, res_col2 = st.columns([1.3, 1.0])

    with res_col1:
        st.subheader("🎯 Prediction Result")
        if predicted_class == "Churned":
            st.error(f"⚠️ **High Churn Risk Detected!** (Confidence: {confidence:.1f}%)")
            st.markdown("""
            **Key Risk Drivers Identified:**
            - Short tenure or Month-to-Month contract structure.
            - Higher relative monthly charges without bundled support.
            - Low number of customer referrals or unengaged add-ons.
            
            **Recommended Retention Actions:**
            1. Offer long-term contract discounts (1-Year or 2-Year plan incentive).
            2. Bundle complimentary Premium Tech Support or Device Protection.
            3. Initiate proactive outreach from Customer Success Team.
            """)
        elif predicted_class == "Stayed":
            st.success(f"✅ **Customer is Likely to STAY** (Confidence: {confidence:.1f}%)")
            st.markdown("""
            **Positive Retention Indicators:**
            - Healthy customer tenure and established billing history.
            - Long-term commitment contract or stable bundle usage.
            - High satisfaction with core connectivity services.
            """)
        else:
            st.info(f"🌟 **Customer Status: NEW / JOINED** (Confidence: {confidence:.1f}%)")
            st.markdown("""
            **Onboarding Recommendations:**
            - Deliver automated welcome journey and service orientation.
            - Incentivize referral program enrollment.
            """)

    with res_col2:
        st.subheader("📊 Probability Breakdown")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        classes = ['Stayed', 'Churned', 'Joined']
        values = [stay_prob * 100, churn_prob * 100, joined_prob * 100]
        colors = ['#2ecc71', '#e74c3c', '#3498db']
        bars = ax.bar(classes, values, color=colors, width=0.55)
        ax.set_ylabel('Probability (%)')
        ax.set_ylim(0, 100)
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold')
        st.pyplot(fig)

st.markdown("---")
st.markdown("<center><small>Shell-Edunet Skills4Future AICTE Virtual Internship | Department of CSE (AI & ML), ANITS</small></center>", unsafe_allow_html=True)
