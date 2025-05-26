import streamlit as st
import pandas as pd
import pickle

# Load the trained pipeline
with open("best_model.pkl", "rb") as f:
    loaded_model = pickle.load(f)

st.title("Insurance Premium Predictor")
st.write("Enter customer details to predict premium amount.")

def user_input():
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    gender = st.selectbox("Gender", options=["Male", "Female"]) 
    Annual_Income = st.number_input("Annual Income", min_value=0, value=50000)
    marital_status = st.selectbox("Marital Status", options=["Single", "Married", "Divorced"])
    Number_of_Dependents = st.number_input("Number of Dependents", min_value=0, max_value=5, value=1)
    education = st.selectbox("Education Level", options=["High School", "Bachelor's", "Master's", "PhD"])
    occupation = st.selectbox("Occupation", ["Salaried", "Self-Employed", "Retired"])
    Health_Score = st.slider("Health Score (0-100)", min_value=0.0, max_value=100.0, value=50.0)
    Location = st.selectbox("Location", options=["Urban", "Suburban", "Rural"])
    Previous_Claims = st.number_input("Previous Claims", min_value=0, max_value=10, value=0)
    Vehicle_Age = st.number_input("Vehicle Age (years)", min_value=0, max_value=20, value=5)
    Credit_Score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
    Insurance_Duration = st.number_input("Insurance Duration (years)", min_value=0, max_value=10, value=3)
    Policy_Start_Date = st.date_input("Policy Start Date", key="policy_start_date")
    Customer_Feedback = st.selectbox("Customer Feedback", options=["Poor", "Average", "Good", "Excellent"])
    Smoking_Status = st.selectbox("Smoking Status", options=["Yes", "No"])
    Exercise_Frequency = st.selectbox("Exercise Frequency", options=["None", "Rarely", "Occasional", "Monthly", "Weekly", "Daily"])
    Property_Type = st.selectbox("Property Type", options=["House", "Apartment", "Condo"])
    policy_type = st.selectbox("Policy Type", options=["Basic", "Comprehensive", "Premium"])
    policy_start_date = pd.to_datetime(Policy_Start_Date)
    policy_age_days = (pd.Timestamp.today() - policy_start_date).days
    policy_start_year = policy_start_date.year
    policy_start_month = policy_start_date.month
    policy_start_day = policy_start_date.day

    data = {
        "Age": age,
        "Gender": gender,
        "Annual Income": Annual_Income,
        "Marital Status": marital_status,
        "Number of Dependents": Number_of_Dependents,
        "Education Level": education,
        "Occupation": occupation,
        "Health Score": Health_Score,
        "Location": Location,
        "Previous Claims": Previous_Claims,
        "Vehicle Age": Vehicle_Age,
        "Credit Score": Credit_Score,
        "Insurance Duration": Insurance_Duration,
        "Policy Start Date": pd.to_datetime(Policy_Start_Date),
        "Customer Feedback": Customer_Feedback,
        "Smoking Status": Smoking_Status,
        "Exercise Frequency": Exercise_Frequency,
        "Property Type": Property_Type,
        "Policy Type": policy_type,
        "Policy Age (Days)": policy_age_days,
        "Policy Start Year": policy_start_year,
        "Policy Start Month": policy_start_month,
        "Policy Start Day": policy_start_day
    }

    return pd.DataFrame([data])

def preprocess_input(df):
    import pandas as pd
    import numpy as np

    # Handle missing values (numerical)
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col] = df[col].fillna(df[col].median())

    # Handle missing values (categorical)
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Date conversion
    df['Policy Start Date'] = pd.to_datetime(df['Policy Start Date'], errors='coerce')
    df['Policy Start Year'] = df['Policy Start Date'].dt.year
    df['Policy Start Month'] = df['Policy Start Date'].dt.month
    df['Policy Start Day'] = df['Policy Start Date'].dt.day
    df['Policy Age (Days)'] = (pd.to_datetime("today") - df['Policy Start Date']).dt.days

    # Age group
    age_bins = [18, 30, 45, 60, 100]
    age_labels = ['18–30', '31–45', '46–60', '60+']
    df['Age Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, include_lowest=True)

    # Income Bracket
    income_bins = [0, 30000, 60000, 100000, np.inf]
    income_labels = ['Low', 'Median', 'High', 'Very High']
    df['Income_Bracket'] = pd.cut(df['Annual Income'], bins=income_bins, labels=income_labels)

    # Credit Category
    credit_bins = [0, 400, 600, 800, np.inf]
    df['Credit_Category'] = pd.cut(df['Credit Score'], bins=credit_bins, labels=False)

    # Dependents Group
    def dependents_group(num):
        if num == 0:
            return 'None'
        elif num <= 2:
            return 'Few'
        else:
            return 'Many'
    df['dependents group'] = df['Number of Dependents'].apply(dependents_group)

    # Days since policy start
    today = pd.Timestamp.now()
    df['Days_Since_Policy_Start'] = (today - df['Policy Start Date']).dt.days

    # Customer Feedback encoding
    feedback_map = {'Poor': 0, 'Average': 1, 'Good': 2}
    df['Customer_Feedback_Score'] = df['Customer Feedback'].map(feedback_map)

    # Interaction Features
    df['Age_x_Health'] = df['Age'] * df['Health Score']
    df['CreditScore_x_PrevClaims'] = df['Credit Score'].fillna(0) * df['Previous Claims'].fillna(0)

    # Risk flags
    df['Is_Smoker'] = df['Smoking Status'].apply(lambda x: 1 if str(x).lower() == 'yes' else 0)
    df['Low_Credit_Score'] = df['Credit Score'].apply(lambda x: 1 if x < 600 else 0)
    df['Multiple_Claims'] = df['Previous Claims'].apply(lambda x: 1 if x > 2 else 0)

    # Exercise Frequency Encoding
    exercise_map = {'Daily': 4, 'Weekly': 3, 'Monthly': 2, 'Rarely': 1, 'Never': 0}
    df['Exercise_Freq_Score'] = df['Exercise Frequency'].map(exercise_map)

    # Interaction Income x Credit
    df['Income_x_Credit'] = df['Annual Income'] * df['Credit Score']

    # Convert categorical columns to string
    for col in df.select_dtypes(include=['object', 'category']).columns:
        df[col] = df[col].astype(str)

    return df


input_df = user_input()
if st.button("Estimate Premium"):
    processed_df = preprocess_input(input_df)
    prediction = loaded_model.predict(processed_df)
    st.success(f"Estimated Premium: ₹{prediction[0]:,.2f}")

