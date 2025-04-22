import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
# Define the model path relative to this file
MODEL_PATH = Path(__file__).resolve().parent / "notebooks" / "loan_default_model.pkl"

@st.cache_resource
def load_model(path: Path = MODEL_PATH):
    """Load and cache the trained model pipeline."""
    if not path.exists():
        st.error(f"Model file not found at: {path}")
        st.stop()
    return joblib.load(path)

# Load the model
model = load_model()

st.title("Loan Default Prediction App")
st.subheader("Enter loan application details to predict default risk")

# Numeric inputs
loan_amnt = st.number_input("Loan Amount", min_value=500.0, max_value=40000.0, value=10000.0, step=100.0)
int_rate = st.number_input("Interest Rate (%)", min_value=5.0, max_value=30.0, value=12.0, step=0.1)
installment = st.number_input("Installment", min_value=50.0, max_value=2000.0, value=200.0, step=1.0)
annual_inc = st.number_input("Annual Income", min_value=1000.0, max_value=500000.0, value=50000.0, step=1000.0)
dti = st.number_input("Debt-to-Income Ratio", min_value=0.0, max_value=50.0, value=15.0, step=0.1)
delinq_2yrs = st.number_input("Delinquencies in last 2 years", min_value=0, max_value=10, value=0, step=1)
revol_util = st.number_input("Revolving Utilization (%)", min_value=0.0, max_value=150.0, value=50.0, step=0.1)
total_acc = st.number_input("Total Credit Accounts", min_value=1, max_value=100, value=10, step=1)

# Categorical inputs
term = st.selectbox("Term", ['36 months', '60 months'])
grade = st.selectbox("Grade", ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
sub_grade = st.selectbox("Sub Grade", [f"{g}{i}" for g in ['A','B','C','D','E','F','G'] for i in range(1,6)])
emp_length = st.selectbox("Employment Length", ['< 1 year','1 year','2 years','3 years','4 years','5 years','6 years','7 years','8 years','9 years','10+ years','n/a'])
home_ownership = st.selectbox("Home Ownership", ['OWN','MORTGAGE','RENT','OTHER','NONE','ANY'])
verification_status = st.selectbox("Verification Status", ['Verified','Source Verified','Not Verified'])
purpose = st.selectbox("Purpose", ['debt_consolidation','credit_card','home_improvement','major_purchase','small_business','car','wedding','medical','moving','vacation','house','renewable_energy','educational','other'])

# Collect input into DataFrame
input_dict = {
    'loan_amnt': loan_amnt,
    'int_rate': int_rate,
    'installment': installment,
    'annual_inc': annual_inc,
    'dti': dti,
    'delinq_2yrs': delinq_2yrs,
    'revol_util': revol_util,
    'total_acc': total_acc,
    'term': term,
    'grade': grade,
    'sub_grade': sub_grade,
    'emp_length': emp_length,
    'home_ownership': home_ownership,
    'verification_status': verification_status,
    'purpose': purpose
}
input_df = pd.DataFrame([input_dict])

# Predict
if st.button("Predict Default Risk"):
    proba = model.predict_proba(input_df)[0][1]
    pred = model.predict(input_df)[0]
    st.write(f"**Default Probability:** {proba:.2f}")
    st.write(f"**Prediction:** {'Default' if pred==1 else 'No Default'}")
