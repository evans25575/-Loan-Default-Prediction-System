# ðŸ›¡ï¸ Loan Guard AI
### Intelligent Credit Risk Scoring for Digital Lenders & Microfinance Institutions

[![Live App](https://img.shields.io/badge/ðŸš€%20LIVE%20APP-Launch%20Loan%20Guard%20AI-006BA6?style=for-the-badge)](https://3v5yyiiskavaufqjdjckgn.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://3v5yyiiskavaufqjdjckgn.streamlit.app/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Model-FF6600?style=flat-square)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## ðŸ”— Try It Live

> **ðŸ‘‰ [https://3v5yyiiskavaufqjdjckgn.streamlit.app/](https://3v5yyiiskavaufqjdjckgn.streamlit.app/)**

No setup required. Enter customer financial data and get an instant credit risk prediction with full explainability.

---

## ðŸ“Œ Overview

**Loan Guard AI** is an end-to-end machine learning system that predicts the likelihood of loan default based on customer financial and demographic data. Designed for microfinance institutions, SACCOs, and digital lenders operating in emerging markets where traditional credit bureau data is limited or unavailable.

Built with Python, XGBoost, LightGBM, SHAP, and Streamlit â€” from raw data to deployed, explainable predictions.

---

## ðŸ’¼ Business Impact

| Metric | Result |
|---|---|
| ðŸ“‰ Reduction in Non-Performing Loans (NPLs) | **15â€“30% estimated** |
| â±ï¸ Time saved per day on manual risk review | **2â€“4 hours** |
| âš¡ Inference speed | **Real-time (sub-second)** |
| ðŸ¦ Target institutions | Microfinance, SACCOs, Digital Lenders |

---

## ðŸ§  Machine Learning Workflow

### 1. Data Preprocessing
- Handle missing values and detect/treat outliers
- Feature engineering: **debt-to-income ratio**, **loan-to-asset ratio**, repayment history signals

### 2. Model Training
| Algorithm | Type | Use |
|---|---|---|
| XGBoost | Gradient Boosting | Primary classifier |
| LightGBM | Gradient Boosting | Speed-optimised variant |
| Random Forest | Ensemble | Baseline comparison |

### 3. Evaluation
- Confusion Matrix
- ROC-AUC Curve
- Precision-Recall Curve

### 4. Explainability â€” SHAP
- **Global**: Which features most influence default risk across all customers?
- **Local**: Why did *this specific customer* get a high-risk score?
- Critical for regulatory transparency and loan officer trust

### 5. Deployment
- **Frontend**: Streamlit interactive web app
- **Inference**: Real-time predictions from user inputs
- **Visualisation**: Plotly-powered analytics dashboard

---

## ðŸ“Š Key Features

- ðŸ§¾ **Interactive Web App** â€” Enter applicant data, get instant risk prediction
- ðŸ§  **Model Explainability** â€” SHAP values for transparent credit decisions
- ðŸ“ˆ **Visual Analytics Dashboard** â€” Risk distribution, NPL trends, ROC curves
- âš™ï¸ **Modular, Production-Ready Code** â€” Structured for extension and scaling

---

## ðŸ“‚ Project Structure

```
Loan-Guard-AI/
â”œâ”€â”€ models/                  # Trained ML models (.pkl files)
â”œâ”€â”€ streamlit_app.py         # Main Streamlit dashboard
â”œâ”€â”€ train_model.py           # Model training pipeline
â”œâ”€â”€ loan_default_model.pkl   # Serialised production model
â”œâ”€â”€ requirements.txt         # Python dependencies
â””â”€â”€ README.md
```

---

## ðŸ› ï¸ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| ML Libraries | scikit-learn, XGBoost, LightGBM, pandas, numpy |
| Explainability | SHAP |
| Visualisation | Plotly |
| Interface | Streamlit |
| Version Control | Git + GitHub |
| Optional | Docker, MLflow |

---

## âš ï¸ Model Assumptions & Limitations

- **Input Bias** â€” Relies on structured financial data; may underrepresent informal earners
- **No External Credit Data** â€” CRB scores not yet integrated (planned)
- **Static Model** â€” Recommend quarterly retraining with fresh repayment data
- **No Fairness Layer** â€” Regulatory/ethical constraints (gender, age bias) not yet enforced

> ðŸ§  **For production use:** Integrate into a loan management system, add performance monitoring, and retrain quarterly.

---

## ðŸš€ Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/evans25575/-Loan-Default-Prediction-System.git
cd Loan-Default-Prediction-System

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run streamlit_app.py
```

Then open **http://localhost:8501** in your browser.

---

## ðŸ¦ Who Is This For?

| Stakeholder | How They Use It |
|---|---|
| Credit Officers | Instant risk feedback per applicant |
| Loan Committees | Data-driven insights during vetting |
| Risk Analysts | Monitor model performance, trigger retraining |
| Portfolio Managers | Track portfolio-level risk exposure |
| Product/Dev Teams | Integrate predictions into digital platforms |

---

## ðŸ‘¨â€ðŸ’» Author

**Evans Kiplangat** â€” Data Scientist | Credit & Risk Modelling

ðŸŒ [Portfolio](https://evans25575.github.io/Evans---portfolio-/) &nbsp;|&nbsp; ðŸ™ [GitHub](https://github.com/evans25575) &nbsp;|&nbsp; ðŸ“§ kiplaevans2018@gmail.com

---

## ðŸ“œ License

MIT License â€” free to use, modify, and distribute with attribution.
