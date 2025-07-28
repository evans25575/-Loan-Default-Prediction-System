# 💸 Loan Default Prediction System

An end-to-end Machine Learning project designed to help microfinance institutions predict the likelihood of loan default based on customer financial and demographic data. Built using Python, Streamlit, and Plotly, this project showcases ML modeling, explainability, and deployment in a real-world co

## 📌 Problem Statement

Loan defaults can severely affect the financial sustainability of microfinance institutions. This system helps identify high-risk applicants **before** issuing loans, aiming to reduce non-performing loans (NPLs) and improve credit decision-making.

---

## 📊 Key Features

- 🧾 **Interactive Web App**: Predict default risk from real-time user input via Streamlit.
- 🧠 **Model Explainability**: Uses SHAP to explain individual predictions.
- 📈 **Visual Analytics Dashboard**: Interactive insights using Plotly.
- ⚙️ **Clean, Modular Code**: Production-ready structure for easy extension.

---

## 🧠 Machine Learning Workflow

### 📍 Data Preprocessing
- Handle missing values
- Detect and treat outliers
- Engineer features (e.g., debt-to-income ratio, loan-to-asset)

### 🤖 Model Training
- Algorithms used: **XGBoost**, **Random Forest**, **LightGBM**
- Evaluation metrics: **Confusion Matrix**, **ROC-AUC**, **Precision-Recall Curve**

### 📌 Model Explainability
- SHAP values for global and local interpretation
- Transparency in credit risk decisioning

### 🚀 Deployment
- **Streamlit** interface
- Real-time predictions
- Data visualization using **Plotly**

---

## 📂 Project Structure

```bash
Loan-Default-Prediction-System/
├── data/               # Raw and processed datasets
├── notebooks/          # EDA and model training notebooks
├── models/             # Trained ML models (pickle files)
├── streamlit_app.py    # Streamlit dashboard script
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation


---

🛠️ Tech Stack

Category	Tools/Frameworks

Programming Language	Python
ML Libraries	Pandas, Scikit-learn, XGBoost, LightGBM
Explainability	SHAP
Frontend	Streamlit
Visualization	Plotly
Version Control	Git + GitHub
(Optional)	Docker, MLflow, SQL/NoSQL



---

🚀 How to Run

1. Clone the repo



git clone https://github.com/evans25575/-Loan-Default-Prediction-System.git
cd Loan-Default-Prediction-System

2. Install dependencies



pip install -r requirements.txt

3. Run the Streamlit app



streamlit run streamlit_app.py

Then open http://localhost:8501 in your browser.

## 💼 Business Value & Use Case

### 📊 Business Impact

This machine learning system is designed to support credit decision-making in Microfinance Institutions (MFIs), SACCOs, digital lenders, and commercial banks. If deployed effectively:

✅ It can reduce Non-Performing Loans (NPLs) by 15–30% by flagging high-risk applicants before loan issuance.

✅ It can save 2–4 hours per day in manual risk assessment by credit officers or analysts.

✅ It enables faster and more objective credit approvals, helping financial institutions disburse loans at scale without compromising on portfolio quality.

✅ Over time, it can be extended to include dynamic retraining and customer segmentation, unlocking long-term value through smarter credit analytics.


## 🏦 Target Users

The system is primarily built for the following user personas:

Credit Officers – To enter borrower info and get instant risk feedback before approving loans.

Loan Committee Members – To use model insights during loan vetting processes.

Risk Analysts / Data Teams – To analyze prediction patterns and update the model with new data.

Branch / Portfolio Managers – To view trends in loan default risk across borrower segments.

Product Teams – To integrate predictions into digital lending platforms or mobile apps.


## Model Assumptions & Limitations

While the system is robust, users should be aware of these constraints:

Input Data Bias: The model depends heavily on financial and demographic variables (e.g., income, age, loan size). Borrowers with informal income sources or missing documentation may be misclassified.

No Real-Time External Data: It does not integrate with external credit bureaus (e.g., CRBs), which can be critical in full-spectrum risk profiling.

Static Model: The deployed version is based on static training data. Performance may degrade if borrower behavior or macroeconomic conditions change. Regular retraining is recommended.

No Ethics Layer: This system doesn’t implement fairness checks (e.g., race, gender bias) or regulatory compliance — these should be added before real-world deployment.


## > 🧠 Recommendation: For production use, integrate this model into your loan management system, monitor performance, and schedule periodic retraining with updated borrower data.


👨‍💻 Author

Evans Kiplangat
🌐 https://evans-kiplangat-portfolio27.netlify.app/
🐙 


---

📜 License

MIT License
