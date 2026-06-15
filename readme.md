# 🕵️ FlowShield AI – Intelligent Mule Account Detection Platform

An end-to-end **AI-powered fraud detection system** designed to identify suspicious and mule accounts by analyzing transactional and behavioral banking data.

FlowShield AI combines **Machine Learning, Anomaly Detection, Feature Engineering, and Explainable AI** to proactively detect fraudulent account activity, reduce financial losses, and support faster investigation workflows.

---

## 📌 Dataset Access

Due to GitHub file size limitations, the original banking transaction dataset is not included in this repository.

### Dataset Download

Download the dataset and place it in the following directory:

```text
data/transactions.csv
```

After adding the dataset, the project can be trained and executed normally.

---

## 🚀 Project Objective

Financial institutions process millions of transactions daily, making manual fraud detection increasingly difficult. Traditional rule-based monitoring systems often struggle to identify evolving fraud patterns and suspicious account behavior.

FlowShield AI addresses this challenge by leveraging Machine Learning and behavioral analytics to:

* Detect suspicious and mule accounts
* Identify hidden fraud patterns
* Generate intelligent risk scores
* Support proactive fraud prevention
* Improve transparency through Explainable AI

---

## 🧠 Key Features

### 🔍 Intelligent Mule Account Detection

Identifies suspicious and mule accounts using transactional and behavioral patterns.

### 📊 Advanced Feature Engineering

Extracts meaningful behavioral, transactional, and temporal fraud indicators from raw banking data.

### 🧠 AI-Powered Risk Scoring

Assigns risk levels to accounts based on learned fraud patterns.

### ⚠️ Anomaly Detection Engine

Detects unusual account behavior and hidden fraud signals using Isolation Forest.

### 🔎 Explainable AI (SHAP)

Provides transparent explanations for model predictions and highlights influential features.

### 🚨 Intelligent Alert Generation

Flags high-risk accounts for proactive investigation and monitoring.

### 🛡️ Fraud Prevention Framework

Supports early intervention and reduces operational and financial risk.

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/mule-detection-system.git
cd mule-detection-system
```

### 2️⃣ Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment:

**Windows**

```bash
.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Add Dataset

Place the dataset file inside:

```text
data/transactions.csv
```

---

## ▶️ Running the Project

### Train the Model

```bash
python src/train_model.py
```

This process will:

* Load and preprocess transaction data
* Perform feature engineering
* Train fraud detection models
* Save model artifacts inside `/model`
* Generate analytical reports inside `/reports`

---

### Run Predictions

```bash
python src/predict.py
```

---

### Launch the Streamlit Dashboard

```bash
streamlit run app.py
```

---

## 🔄 Workflow Overview

```text
Banking Transaction Data
            ↓
      Data Preprocessing
            ↓
      Feature Engineering
            ↓
    Feature Transformation
            ↓
      Model Training
   (XGBoost + Isolation Forest)
            ↓
      Risk Classification
            ↓
     Explainability (SHAP)
            ↓
 Intelligent Alert Generation
            ↓
 Suspicious Account Detection
```

---

## 🛠️ Technology Stack

| Category             | Technologies          |
| -------------------- | --------------------- |
| Programming Language | Python                |
| Data Processing      | Pandas, NumPy         |
| Machine Learning     | XGBoost, Scikit-Learn |
| Anomaly Detection    | Isolation Forest      |
| Explainable AI       | SHAP                  |
| Visualization        | Matplotlib            |
| User Interface       | Streamlit             |
| Model Persistence    | Joblib                |
| Development Tools    | Git, GitHub, VS Code  |

---

## 📈 Expected Outcomes

* Early detection of mule and suspicious accounts
* Reduced financial fraud exposure
* Improved investigation efficiency
* Lower false-positive rates
* Explainable and transparent fraud predictions
* Scalable fraud monitoring framework

---

