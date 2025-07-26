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




👨‍💻 Author

Evans Kiplangat
🌐 https://evans-kiplangat-portfolio27.netlify.app/
🐙 


---

📜 License

MIT License
