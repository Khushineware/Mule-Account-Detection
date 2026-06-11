# 🕵️ Mule Detection System (Fraud Detection using Machine Learning)

An end-to-end **machine learning system to detect mule/fraudulent bank accounts** using transaction behavior, feature engineering, and predictive modeling.

The system is designed to be modular, scalable, and production-ready with explainability support.

---

## 🚀 Project Objective

To detect **fraudulent or mule accounts** from banking transaction data using machine learning techniques like **XGBoost / Isolation-based models**, combined with feature engineering and explainability (SHAP).

---

## 🧠 Key Features

- Robust data preprocessing pipeline
- Handling missing values and sentinel values (`-1`)
- Automatic date-based feature engineering
- Label encoding for categorical features
- ML-based fraud detection model
- SHAP-based explainability support
- Reusable training and inference pipeline
- Modular Python architecture

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the repository


git clone https://github.com/your-username/mule-detection-system.git
cd mule-detection-system

### 2️⃣ Create a virtual environment
python -m venv .venv

Activate it:

Windows:
.venv\Scripts\activate

Mac/Linux:
source .venv/bin/activate

### 3️⃣ Install dependencies
pip install -r requirements.txt


### 4️⃣ Add dataset

Place your dataset inside:
data/transactions.csv

### ▶️ How to Run the Project
🔹 Train the Model
python src/train_model.py

## This will:
- Load and preprocess data
- Perform feature engineering
- Train ML model
- Save model artifacts inside /model
- Generate reports in /reports

🔹 Run Predictions
python src/predict.py

🔹 (Optional) Run App
If you are using Streamlit UI:
streamlit run app.py


### 🔄 Workflow Overview
Raw Data
   ↓
Preprocessing (cleaning + missing handling)
   ↓
Feature Engineering (behavioral + time-based features)
   ↓
Encoding (categorical → numeric)
   ↓
Model Training (XGBoost / Isolation Forest)
   ↓
Prediction + Explainability (SHAP)