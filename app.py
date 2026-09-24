import streamlit as st
import pickle


# Load model and vectorizer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('dv.pkl', 'rb') as f:
    dv = pickle.load(f)


# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)


# Title
st.title("📊 Customer Churn Prediction")
st.write(
    "Enter customer information below to predict the probability "
    "that the customer will churn."
)


# Customer information
gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

senior_citizen = st.selectbox(
    "Senior Citizen",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

partner = st.selectbox(
    "Partner",
    ["Yes", "No"]
)

dependents = st.selectbox(
    "Dependents",
    ["Yes", "No"]
)

tenure = st.number_input(
    "Tenure (months)",
    min_value=0,
    max_value=100,
    value=12
)

phoneservice = st.selectbox(
    "Phone Service",
    ["Yes", "No"]
)

multiplelines = st.selectbox(
    "Multiple Lines",
    ["Yes", "No", "No phone service"]
)

internetservice = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

onlinesecurity = st.selectbox(
    "Online Security",
    ["Yes", "No", "No internet service"]
)

onlinebackup = st.selectbox(
    "Online Backup",
    ["Yes", "No", "No internet service"]
)

deviceprotection = st.selectbox(
    "Device Protection",
    ["Yes", "No", "No internet service"]
)

techsupport = st.selectbox(
    "Tech Support",
    ["Yes", "No", "No internet service"]
)

streamingtv = st.selectbox(
    "Streaming TV",
    ["Yes", "No", "No internet service"]
)

streamingmovies = st.selectbox(
    "Streaming Movies",
    ["Yes", "No", "No internet service"]
)

contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

paperlessbilling = st.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

paymentmethod = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

monthlycharges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=50.0
)

totalcharges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=500.0
)


# Prediction
if st.button("Predict Churn"):

    customer = {
        "gender": gender,
        "senior_citizen": senior_citizen,
        "partner": partner,
        "dependents": dependents,
        "tenure": tenure,
        "phoneservice": phoneservice,
        "multiplelines": multiplelines,
        "internetservice": internetservice,
        "onlinesecurity": onlinesecurity,
        "onlinebackup": onlinebackup,
        "deviceprotection": deviceprotection,
        "techsupport": techsupport,
        "streamingtv": streamingtv,
        "streamingmovies": streamingmovies,
        "contract": contract,
        "paperlessbilling": paperlessbilling,
        "paymentmethod": paymentmethod,
        "monthlycharges": monthlycharges,
        "totalcharges": totalcharges
    }

    X = dv.transform([customer])

    probability = model.predict_proba(X)[0, 1]

    st.subheader("Prediction")

    st.metric(
        "Churn Probability",
        f"{probability * 100:.2f}%"
    )

    if probability >= 0.5:
        st.error("⚠️ Customer is likely to churn.")
    else:
        st.success("✅ Customer is unlikely to churn.")