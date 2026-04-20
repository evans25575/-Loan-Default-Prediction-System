"""
train_model.py — Run this once to generate models/loan_default_model.pkl
Usage: python train_model.py
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier
import joblib, os

np.random.seed(42)
n = 5000

loan_amnt   = np.random.uniform(500, 40000, n)
int_rate    = np.random.uniform(5, 30, n)
installment = loan_amnt / np.random.uniform(12, 60, n)
annual_inc  = np.random.lognormal(10.8, 0.5, n)
dti         = np.random.uniform(0, 40, n)
delinq_2yrs = np.random.poisson(0.3, n)
revol_util  = np.random.uniform(0, 100, n)
total_acc   = np.random.randint(2, 50, n)

terms       = np.random.choice(['36 months','60 months'], n, p=[0.7,0.3])
grades      = np.random.choice(['A','B','C','D','E','F','G'], n, p=[0.2,0.25,0.2,0.15,0.1,0.07,0.03])
emp_lengths = np.random.choice(['< 1 year','1 year','2 years','3 years','4 years','5 years',
                                 '6 years','7 years','8 years','9 years','10+ years'], n)
home_owns   = np.random.choice(['OWN','MORTGAGE','RENT','OTHER'], n, p=[0.1,0.4,0.45,0.05])
ver_status  = np.random.choice(['Verified','Source Verified','Not Verified'], n)
purposes    = np.random.choice(['debt_consolidation','credit_card','home_improvement',
                                 'small_business','car','medical','other'], n,
                                p=[0.35,0.22,0.13,0.1,0.08,0.07,0.05])
sub_grades  = np.random.choice([f'{g}{i}' for g in ['A','B','C','D','E','F','G'] for i in range(1,6)], n)

grade_risk = {'A':0.04,'B':0.08,'C':0.14,'D':0.21,'E':0.28,'F':0.36,'G':0.44}
base_p = np.array([grade_risk[g] for g in grades])
base_p += (dti/200) + (delinq_2yrs*0.05) + (revol_util/400) - (annual_inc/2000000)
base_p = np.clip(base_p, 0.02, 0.85)
target = np.random.binomial(1, base_p)

df = pd.DataFrame({
    'loan_amnt': loan_amnt, 'int_rate': int_rate, 'installment': installment,
    'annual_inc': annual_inc, 'dti': dti, 'delinq_2yrs': delinq_2yrs,
    'revol_util': revol_util, 'total_acc': total_acc,
    'term': terms, 'grade': grades, 'sub_grade': sub_grades,
    'emp_length': emp_lengths, 'home_ownership': home_owns,
    'verification_status': ver_status, 'purpose': purposes,
    'loan_status': target
})

NUM_COLS = ['loan_amnt','int_rate','installment','annual_inc','dti','delinq_2yrs','revol_util','total_acc']
CAT_COLS = ['term','grade','sub_grade','emp_length','home_ownership','verification_status','purpose']

X = df.drop('loan_status', axis=1)
y = df['loan_status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), NUM_COLS),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CAT_COLS)
])

model = Pipeline([
    ('pre', preprocessor),
    ('clf', XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='logloss', random_state=42,
        scale_pos_weight=(y_train==0).sum()/(y_train==1).sum()
    ))
])

model.fit(X_train, y_train)
y_proba = model.predict_proba(X_test)[:,1]
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
print(classification_report(y_test, model.predict(X_test)))

os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/loan_default_model.pkl')
print("✅ Model saved to models/loan_default_model.pkl")
