An end-to-end Machine Learning project designed to help microfinance institutions predict the likelihood of loan default based on customer financial and demographic data. Built using Python, Streamlit, and Plotly, this project demonstrates ML, explainability, and deployment in a real-world context.


📌 Problem Statement
Loan default can severely affect microfinance institutions' sustainability. This system helps identify high-risk applicants before issuing loans, reducing non-performing loans (NPLs) and supporting better financial decisions.

📊 Features
Interactive Streamlit Web App

Predict default risk from user input

Model explainability with SHAP

Visual analytics dashboard (Plotly)

Clean and production-ready code

🧠 Machine Learning Workflow
Data Preprocessing

Handling missing values

Outlier detection

Feature engineering (e.g., debt-to-income ratio)

Model Training

Algorithms: XGBoost, Random Forest, LightGBM

Evaluation: Confusion Matrix, ROC-AUC, Precision-Recall

Model Explainability

SHAP values to interpret predictions

Transparent decision support for credit officers

Deployment

Streamlit interface

Real-time user input

Data visualization with Plotly

📁 Folder Structure
graphql
Copy
Edit
ML-loan-default/
│
├── data/                      # Raw and processed datasets
├── notebooks/                 # EDA and training notebooks
├── models/                    # Trained models
├── streamlit_app.py           # Streamlit app
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
🛠️ Tech Stack
Python (Pandas, Scikit-Learn, XGBoost, SHAP)

Streamlit

Plotly

Git/GitHub

(Optional: MLflow, Docker, SQL/NoSQL)
