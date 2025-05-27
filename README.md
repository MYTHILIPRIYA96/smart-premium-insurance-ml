# smartpremium-insurance-ml
End-to-end machine learning solution to predict insurance premiums based on customer demographics and policy details. Includes data preprocessing, regression modeling, ML pipelines, experiment tracking with MLflow, and real-time prediction app deployment using Streamlit.

# SmartPremium: Predicting Insurance Costs with Machine Learning

**Objective**
To build and deploy a machine learning model that predicts insurance premium amounts using customer demographics, lifestyle data, and policy details. This helps insurers set data-driven, risk-adjusted pricing.

**Problem Statement**
Insurance companies need to accurately price policies based on risk. This project aims to automate premium estimation using a data-driven approach through regression models.

**Business Use Cases**
Insurance Companies: Optimize pricing models based on customer risk profiles.

Financial Institutions: Estimate risk for products bundled with insurance.

Healthcare Providers: Forecast long-term healthcare costs.

Customer Experience: Deliver real-time, personalized insurance quotes.

# Project Structure
├── best_model.pkl # Final saved model (Pickle)
├── train.csv # Training dataset
├── test.csv # Test dataset
├── SmartPremium.ipynb # Main script with full pipeline
├── README.md # Project documentation
├── app.py

## Features Used

- **Demographics:** Age, Gender, Marital Status
- **Health & Lifestyle:** Smoking Status, Exercise Frequency, Pre-existing Conditions
- **Financial:** Annual Income, Credit Score
- **Insurance History:** Number of Claims, Policy Type, Claim Amounts
- **Custom Engineered Features:**
  - Age Groups (bins)
  - Credit Score Categories
  - Interaction Features (e.g., Age × Income)
  - Feedback Encodings
  - Frequency Encodings for Categorical Columns

##  Pipeline Overview

1. **Data Ingestion**
2. **Exploratory Data Analysis (EDA)**
3. **Data Cleaning & Missing Value Imputation**
4. **Feature Engineering**
5. **Data Splitting (Train/Test)**
6. **Model Training**
   - Linear Regression
   - Decision Tree Regressor
   - Random Forest Regressor
   - XGBoost Regressor
7. **Hyperparameter Tuning**
   - `GridSearchCV` on best-performing models
8. **Model Evaluation**
   - RMSE, MAE, R² Score, RMSLE
9. **MLflow Integration**
   - Auto-logging models, parameters, and metrics
10. **Model Export**
   - Save the best model using 'pickle'

## Sample Evaluation Metrics

| Model              | RMSE   | MAE    | R²     | RMSLE  |
|-------------------|--------|--------|--------|--------|
| XGBoost Regressor | 150.23 | 102.56 | 0.8932 | 0.2123 |
| Random Forest      | 158.90 | 110.45 | 0.8765 | 0.2198 |

 Best model is logged in MLflow and saved as 'best_model.pkl'.


##  Getting Started

### 1. Clone the Repository

'''bash
git clone https://github.com/MYTHILIPRIYA96/smart-premium-insurance-ml
'''
# Install Dependencies
pip install -r requirements.txt

# Run the Pipeline

python  SmartPremium.ipynb

## Demo

Try it live on [Streamlit Cloud](https://smart-premium-insurance-ml-ihbhnfu8adlfm2g8e2kc4g.streamlit.app/)


