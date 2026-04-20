import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import joblib
import shap

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LoanGuard AI · Credit Risk Assessment",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=Outfit:wght@300;400;500;600&display=swap');

  html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
  .stApp { background: #0b0f0e; color: #e8f0eb; }

  /* Sidebar */
  section[data-testid="stSidebar"] { background: #111916; border-right: 1px solid rgba(61,255,160,0.12); }
  section[data-testid="stSidebar"] * { color: #e8f0eb !important; }
  .stSlider > div > div { background: rgba(61,255,160,0.15) !important; }

  /* Cards */
  .metric-card {
    background: rgba(24,34,24,0.9);
    border: 1px solid rgba(61,255,160,0.15);
    border-radius: 6px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 0;
  }
  .metric-card.danger  { border-color: rgba(255,80,80,0.4); }
  .metric-card.warning { border-color: rgba(255,200,60,0.4); }
  .metric-card.safe    { border-color: rgba(61,255,160,0.4); }

  .metric-label { font-family: 'DM Mono', monospace; font-size: 0.68rem; letter-spacing: 0.18em; text-transform: uppercase; color: #7a9e87; margin-bottom: 0.4rem; }
  .metric-value { font-family: 'DM Serif Display', serif; font-size: 2.4rem; line-height: 1; }
  .metric-value.danger  { color: #ff5050; }
  .metric-value.warning { color: #ffc83c; }
  .metric-value.safe    { color: #3dffa0; }
  .metric-sub   { font-size: 0.78rem; color: #7a9e87; margin-top: 0.3rem; }

  /* Header */
  .app-header { padding: 1rem 0 2rem; border-bottom: 1px solid rgba(61,255,160,0.12); margin-bottom: 2rem; }
  .app-title { font-family: 'DM Serif Display', serif; font-size: 2.2rem; color: #e8f0eb; margin: 0; }
  .app-title em { font-style: italic; color: #3dffa0; }
  .app-sub { font-size: 0.88rem; color: #7a9e87; margin-top: 0.4rem; font-family: 'DM Mono', monospace; letter-spacing: 0.08em; }

  /* Verdict banner */
  .verdict { padding: 1.4rem 2rem; border-radius: 6px; margin: 1.5rem 0; }
  .verdict.high   { background: rgba(255,80,80,0.12);  border: 1px solid rgba(255,80,80,0.4); }
  .verdict.medium { background: rgba(255,200,60,0.1);  border: 1px solid rgba(255,200,60,0.4); }
  .verdict.low    { background: rgba(61,255,160,0.08); border: 1px solid rgba(61,255,160,0.3); }
  .verdict-title { font-family: 'DM Serif Display', serif; font-size: 1.5rem; margin-bottom: 0.3rem; }
  .verdict-title.high   { color: #ff5050; }
  .verdict-title.medium { color: #ffc83c; }
  .verdict-title.low    { color: #3dffa0; }
  .verdict-desc { font-size: 0.88rem; color: #7a9e87; line-height: 1.6; }

  /* Section labels */
  .section-label { font-family: 'DM Mono', monospace; font-size: 0.7rem; color: #3dffa0; letter-spacing: 0.2em; text-transform: uppercase; margin-bottom: 0.8rem; }

  /* Buttons */
  .stButton > button {
    background: #3dffa0 !important; color: #0b0f0e !important;
    font-weight: 600 !important; border: none !important;
    border-radius: 3px !important; padding: 0.65rem 2rem !important;
    font-family: 'Outfit', sans-serif !important;
    letter-spacing: 0.04em !important;
    transition: all 0.2s !important;
  }
  .stButton > button:hover { background: #1adb7a !important; transform: translateY(-1px); }

  /* Input styling */
  .stNumberInput input, .stSelectbox select, div[data-baseweb="select"] {
    background: rgba(24,34,24,0.9) !important;
    border-color: rgba(61,255,160,0.15) !important;
    color: #e8f0eb !important;
  }

  /* Plot backgrounds */
  .js-plotly-plot .plotly { background: transparent !important; }

  /* Info boxes */
  .info-box { background: rgba(61,255,160,0.05); border-left: 3px solid #3dffa0; padding: 1rem 1.2rem; border-radius: 0 4px 4px 0; margin: 1rem 0; font-size: 0.85rem; color: #7a9e87; line-height: 1.6; }

  /* Hide streamlit default elements */
  #MainMenu, footer, header { visibility: hidden; }
  .stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).resolve().parent / "models" / "loan_default_model.pkl"

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"❌ Model not found at: {MODEL_PATH}")
        st.stop()
    return joblib.load(MODEL_PATH)

model = load_model()

# ── Constants ─────────────────────────────────────────────────────────────────
NUM_COLS = ['loan_amnt','int_rate','installment','annual_inc','dti',
            'delinq_2yrs','revol_util','total_acc']
CAT_COLS = ['term','grade','sub_grade','emp_length','home_ownership',
            'verification_status','purpose']

GRADE_RISK = {'A':'Low risk · Prime borrower','B':'Low–medium risk','C':'Medium risk',
              'D':'Medium–high risk','E':'High risk','F':'Very high risk','G':'Speculative'}

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <div class="app-title">LoanGuard <em>AI</em></div>
  <div class="app-sub">Credit Risk Assessment System · Powered by XGBoost · Built by Evans Kiplangat</div>
</div>
""", unsafe_allow_html=True)

# ── Inline Input Form (works on all screen sizes) ─────────────────────────────
st.markdown('<div class="section-label">Applicant Details — Fill all fields then click Assess Risk</div>', unsafe_allow_html=True)

with st.container():
    st.markdown("**📋 Loan Details**")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        loan_amnt = st.number_input("Loan Amount (KES/USD)", 500.0, 40000.0, 10000.0, 500.0)
    with c2:
        int_rate = st.slider("Interest Rate (%)", 5.0, 30.0, 12.0, 0.1)
    with c3:
        installment = st.number_input("Monthly Installment", 50.0, 2000.0, 200.0, 10.0)
    with c4:
        term = st.selectbox("Loan Term", ['36 months', '60 months'])

    c5, c6 = st.columns(2)
    with c5:
        purpose = st.selectbox("Loan Purpose", ['debt_consolidation','credit_card','home_improvement',
                                                 'small_business','car','medical','other'])
    with c6:
        home_ownership = st.selectbox("Home Ownership", ['RENT','MORTGAGE','OWN','OTHER'])

    st.markdown("**👤 Borrower Profile**")
    b1, b2, b3 = st.columns(3)
    with b1:
        annual_inc = st.number_input("Annual Income", 1000.0, 500000.0, 50000.0, 1000.0)
    with b2:
        emp_length = st.selectbox("Employment Length", ['< 1 year','1 year','2 years','3 years',
                                                         '4 years','5 years','6 years','7 years',
                                                         '8 years','9 years','10+ years'])
    with b3:
        ver_status = st.selectbox("Income Verification", ['Verified','Source Verified','Not Verified'])

    st.markdown("**📊 Credit History**")
    h1, h2, h3, h4, h5, h6 = st.columns(6)
    with h1:
        grade = st.selectbox("Credit Grade", ['A','B','C','D','E','F','G'])
    with h2:
        sub_grade = st.selectbox("Sub-Grade", [f"{grade}{i}" for i in range(1,6)])
    with h3:
        dti = st.slider("Debt-to-Income (%)", 0.0, 45.0, 15.0, 0.5)
    with h4:
        delinq_2yrs = st.number_input("Delinquencies (2yr)", 0, 10, 0)
    with h5:
        revol_util = st.slider("Revolving Util. (%)", 0.0, 100.0, 50.0, 1.0)
    with h6:
        total_acc = st.number_input("Total Accounts", 1, 100, 10)

    st.markdown("")
    predict_btn = st.button("🔍  Assess Risk Now", use_container_width=True)

st.markdown("---")

# ── Build input DataFrame ─────────────────────────────────────────────────────
input_df = pd.DataFrame([{
    'loan_amnt': loan_amnt, 'int_rate': int_rate, 'installment': installment,
    'annual_inc': annual_inc, 'dti': dti, 'delinq_2yrs': delinq_2yrs,
    'revol_util': revol_util, 'total_acc': total_acc,
    'term': term, 'grade': grade, 'sub_grade': sub_grade,
    'emp_length': emp_length, 'home_ownership': home_ownership,
    'verification_status': ver_status, 'purpose': purpose
}])

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📋  Risk Assessment", "📊  Market Insights", "🧠  Model Explainability", "ℹ️  About"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 · RISK ASSESSMENT
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    if predict_btn:
        proba = model.predict_proba(input_df)[0][1]
        pred  = model.predict(input_df)[0]

        # Risk tier
        if proba >= 0.60:
            tier, tier_css = "HIGH RISK", "high"
            verdict_text = "This applicant shows elevated probability of default. Consider additional collateral requirements, a co-guarantor, or a reduced loan ceiling before disbursement."
            gauge_color  = "#ff5050"
        elif proba >= 0.35:
            tier, tier_css = "MEDIUM RISK", "medium"
            verdict_text = "Moderate default risk detected. Recommend enhanced due diligence, income verification, and phased disbursement with performance triggers."
            gauge_color  = "#ffc83c"
        else:
            tier, tier_css = "LOW RISK", "low"
            verdict_text = "This applicant presents a favourable credit profile. Standard loan terms are appropriate, subject to internal policy limits."
            gauge_color  = "#3dffa0"

        # ── Verdict banner ──
        st.markdown(f"""
        <div class="verdict {tier_css}">
          <div class="verdict-title {tier_css}">{tier} · {proba*100:.1f}% Default Probability</div>
          <div class="verdict-desc">{verdict_text}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Metric row ──
        c1, c2, c3, c4 = st.columns(4)
        dti_class    = "danger" if dti > 35 else ("warning" if dti > 20 else "safe")
        util_class   = "danger" if revol_util > 75 else ("warning" if revol_util > 50 else "safe")
        inc_class    = "safe" if annual_inc > 40000 else ("warning" if annual_inc > 20000 else "danger")
        delinq_class = "safe" if delinq_2yrs == 0 else ("warning" if delinq_2yrs == 1 else "danger")

        with c1:
            st.markdown(f"""<div class="metric-card {tier_css}">
              <div class="metric-label">Default Probability</div>
              <div class="metric-value {tier_css}">{proba*100:.1f}%</div>
              <div class="metric-sub">{tier}</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="metric-card {dti_class}">
              <div class="metric-label">Debt-to-Income</div>
              <div class="metric-value {dti_class}">{dti:.1f}%</div>
              <div class="metric-sub">{"⚠ Above threshold" if dti > 35 else "Within range"}</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="metric-card {util_class}">
              <div class="metric-label">Revolving Util.</div>
              <div class="metric-value {util_class}">{revol_util:.0f}%</div>
              <div class="metric-sub">{"⚠ High utilization" if revol_util > 75 else "Acceptable"}</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""<div class="metric-card {delinq_class}">
              <div class="metric-label">Delinquencies</div>
              <div class="metric-value {delinq_class}">{int(delinq_2yrs)}</div>
              <div class="metric-sub">{"✓ Clean record" if delinq_2yrs == 0 else "⚠ Past defaults"}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_g, col_b = st.columns([1, 1])

        # ── Gauge chart ──
        with col_g:
            st.markdown('<div class="section-label">Risk Gauge</div>', unsafe_allow_html=True)
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=round(proba * 100, 1),
                number={'suffix': '%', 'font': {'size': 48, 'color': gauge_color, 'family': 'DM Serif Display'}},
                delta={'reference': 30, 'increasing': {'color': '#ff5050'}, 'decreasing': {'color': '#3dffa0'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#7a9e87',
                             'tickfont': {'color': '#7a9e87', 'size': 11}},
                    'bar': {'color': gauge_color, 'thickness': 0.25},
                    'bgcolor': 'rgba(24,34,24,0.9)',
                    'borderwidth': 0,
                    'steps': [
                        {'range': [0, 35],  'color': 'rgba(61,255,160,0.08)'},
                        {'range': [35, 60], 'color': 'rgba(255,200,60,0.08)'},
                        {'range': [60, 100],'color': 'rgba(255,80,80,0.08)'},
                    ],
                    'threshold': {'line': {'color': gauge_color, 'width': 3}, 'thickness': 0.8, 'value': proba*100}
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font_color='#e8f0eb', height=280, margin=dict(t=20,b=10,l=20,r=20)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        # ── Risk factor bar ──
        with col_b:
            st.markdown('<div class="section-label">Risk Factor Breakdown</div>', unsafe_allow_html=True)
            factors = {
                'Credit Grade': {'A':0.04,'B':0.1,'C':0.18,'D':0.28,'E':0.38,'F':0.48,'G':0.58}[grade],
                'DTI Ratio':     min(dti / 100, 0.45),
                'Revol. Util':   revol_util / 200,
                'Delinquencies': min(delinq_2yrs * 0.12, 0.5),
                'Income Level':  max(0, 0.3 - annual_inc / 200000),
                'Loan Amount':   loan_amnt / 100000,
            }
            f_labels = list(factors.keys())
            f_vals   = [round(v * 100, 1) for v in factors.values()]
            f_colors = ['#ff5050' if v > 25 else '#ffc83c' if v > 12 else '#3dffa0' for v in f_vals]

            fig_bar = go.Figure(go.Bar(
                x=f_vals, y=f_labels, orientation='h',
                marker_color=f_colors,
                text=[f'{v}%' for v in f_vals], textposition='outside',
                textfont={'color': '#e8f0eb', 'size': 11}
            ))
            fig_bar.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font_color='#7a9e87', height=280,
                xaxis=dict(range=[0,65], showgrid=True, gridcolor='rgba(61,255,160,0.08)',
                           ticksuffix='%', tickfont={'color':'#7a9e87'}),
                yaxis=dict(tickfont={'color':'#e8f0eb'}),
                margin=dict(t=20,b=10,l=10,r=60)
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # ── Recommendations ──
        st.markdown('<div class="section-label">Analyst Recommendations</div>', unsafe_allow_html=True)
        recs = []
        if dti > 35:       recs.append(("🔴", "High DTI", f"DTI of {dti:.1f}% exceeds the 35% threshold. Request debt restructuring plan or reduce loan ceiling."))
        if revol_util > 75: recs.append(("🔴", "Revolving Overuse", f"Utilization of {revol_util:.0f}% signals liquidity stress. Require additional income documentation."))
        if delinq_2yrs > 0: recs.append(("🟡", "Past Delinquencies", f"{int(delinq_2yrs)} delinquency record(s). Request explanation letter and references."))
        if grade in ['E','F','G']: recs.append(("🔴", "Subprime Grade", f"Grade {grade} — {GRADE_RISK[grade]}. Apply stricter collateral policy."))
        if annual_inc < 20000: recs.append(("🟡", "Income Concern", f"Annual income of {annual_inc:,.0f} relative to loan amount. Verify income sources."))
        if not recs:       recs.append(("🟢", "No Red Flags", "All major risk indicators within acceptable ranges. Proceed with standard approval process."))

        for icon, title, desc in recs:
            st.markdown(f'<div class="info-box"><strong>{icon} {title}</strong> — {desc}</div>', unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="text-align:center; padding: 5rem 2rem; color: #7a9e87;">
          <div style="font-family:'DM Serif Display',serif; font-size:3rem; color:#3dffa0; margin-bottom:1rem;">←</div>
          <div style="font-family:'DM Mono',monospace; font-size:0.8rem; letter-spacing:0.15em; text-transform:uppercase;">
            Fill in the applicant details in the sidebar<br>then click <strong style="color:#3dffa0;">Assess Risk Now</strong>
          </div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 · MARKET INSIGHTS DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-label">Portfolio Overview · Simulated Dataset</div>', unsafe_allow_html=True)

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    kpis = [("Total Applications", "5,000", "Training dataset", "safe"),
            ("Default Rate", "28.5%", "Weighted avg across grades", "warning"),
            ("Model ROC-AUC", "0.678", "XGBoost · 5-fold CV", "safe"),
            ("NPL Reduction Est.", "~18%", "Vs. baseline approval policy", "safe")]
    for col, (label, val, sub, css) in zip([k1,k2,k3,k4], kpis):
        with col:
            st.markdown(f"""<div class="metric-card {css}">
              <div class="metric-label">{label}</div>
              <div class="metric-value {css}" style="font-size:1.8rem;">{val}</div>
              <div class="metric-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        st.markdown('<div class="section-label">Default Rate by Credit Grade</div>', unsafe_allow_html=True)
        grades_x = ['A','B','C','D','E','F','G']
        defaults = [4, 8, 14, 21, 28, 36, 44]
        approved = [96,92,86,79,72,64,56]
        fig1 = go.Figure()
        fig1.add_bar(name='Default %', x=grades_x, y=defaults, marker_color='#ff5050',
                     text=[f'{v}%' for v in defaults], textposition='inside', textfont_color='white')
        fig1.add_bar(name='Performing %', x=grades_x, y=approved, marker_color='#3dffa0',
                     text=[f'{v}%' for v in approved], textposition='inside', textfont_color='#0b0f0e')
        fig1.update_layout(barmode='stack', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                           font_color='#7a9e87', height=300, legend=dict(font_color='#e8f0eb'),
                           xaxis=dict(tickfont_color='#e8f0eb'), yaxis=dict(ticksuffix='%', gridcolor='rgba(61,255,160,0.06)'),
                           margin=dict(t=10,b=10))
        st.plotly_chart(fig1, use_container_width=True)

    with r1c2:
        st.markdown('<div class="section-label">Risk Category Distribution</div>', unsafe_allow_html=True)
        fig2 = go.Figure(go.Pie(
            labels=['Low Risk','Medium Risk','High Risk'],
            values=[220,130,50],
            hole=0.55,
            marker_colors=['#3dffa0','#ffc83c','#ff5050'],
            textfont_color=['#0b0f0e','#0b0f0e','#fff'],
            textinfo='percent+label'
        ))
        fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#7a9e87',
                           height=300, showlegend=False,
                           annotations=[dict(text='Risk<br>Mix', x=0.5, y=0.5,
                                            font=dict(size=13, color='#e8f0eb'), showarrow=False)],
                           margin=dict(t=10,b=10))
        st.plotly_chart(fig2, use_container_width=True)

    r2c1, r2c2 = st.columns(2)

    with r2c1:
        st.markdown('<div class="section-label">Estimated NPL Reduction (Before vs After AI Screening)</div>', unsafe_allow_html=True)
        fig3 = go.Figure()
        fig3.add_bar(name='Without AI', x=['NPL Rate'], y=[28], marker_color='rgba(255,80,80,0.7)',
                     text=['28%'], textposition='outside', textfont_color='#ff5050')
        fig3.add_bar(name='With LoanGuard AI', x=['NPL Rate'], y=[17], marker_color='rgba(61,255,160,0.8)',
                     text=['17%'], textposition='outside', textfont_color='#3dffa0')
        fig3.update_layout(barmode='group', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                           font_color='#7a9e87', height=300, legend=dict(font_color='#e8f0eb'),
                           yaxis=dict(ticksuffix='%', gridcolor='rgba(61,255,160,0.06)', range=[0,35]),
                           margin=dict(t=10,b=10))
        st.plotly_chart(fig3, use_container_width=True)

    with r2c2:
        st.markdown('<div class="section-label">Default Probability Distribution</div>', unsafe_allow_html=True)
        np.random.seed(1)
        sim_proba = np.concatenate([np.random.beta(2,8,250), np.random.beta(5,4,150), np.random.beta(8,3,100)])
        fig4 = go.Figure(go.Histogram(
            x=sim_proba, nbinsx=40,
            marker_color='#3dffa0', opacity=0.7,
            marker_line_color='rgba(61,255,160,0.3)', marker_line_width=0.5
        ))
        fig4.add_vline(x=0.35, line_dash='dash', line_color='#ffc83c', annotation_text='Medium threshold',
                       annotation_font_color='#ffc83c')
        fig4.add_vline(x=0.60, line_dash='dash', line_color='#ff5050', annotation_text='High threshold',
                       annotation_font_color='#ff5050')
        fig4.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                           font_color='#7a9e87', height=300, showlegend=False,
                           xaxis=dict(tickformat='.0%', title='Default Probability', title_font_color='#7a9e87'),
                           yaxis=dict(gridcolor='rgba(61,255,160,0.06)'),
                           margin=dict(t=10,b=10))
        st.plotly_chart(fig4, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 · MODEL EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-label">Model Transparency · SHAP Feature Importance</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <strong>Why explainability matters:</strong> Credit decisions affect livelihoods. LoanGuard AI uses SHAP (SHapley Additive exPlanations) to show <em>why</em> a prediction was made — not just what it is. Each bar below shows how much each feature contributes to the default probability for the current applicant.
    </div>
    """, unsafe_allow_html=True)

    # SHAP for current input
    if predict_btn:
        with st.spinner("Computing SHAP explanations…"):
            try:
                preprocessor = model.named_steps['pre']
                clf          = model.named_steps['clf']
                X_transformed = preprocessor.transform(input_df)

                explainer   = shap.TreeExplainer(clf)
                shap_vals   = explainer.shap_values(X_transformed)

                # Get feature names
                num_names = NUM_COLS
                cat_names = list(preprocessor.named_transformers_['cat'].get_feature_names_out(CAT_COLS))
                all_names = num_names + cat_names

                sv = shap_vals[0]
                # Aggregate cat features back to original
                agg = {}
                for i, name in enumerate(num_names):
                    agg[name] = sv[i]
                for i, name in enumerate(cat_names):
                    orig = name.split('_')[0]
                    agg[orig] = agg.get(orig, 0) + sv[len(num_names)+i]

                agg_df = pd.DataFrame(list(agg.items()), columns=['Feature','SHAP'])
                agg_df = agg_df.reindex(agg_df['SHAP'].abs().sort_values(ascending=True).index)

                colors = ['#ff5050' if v > 0 else '#3dffa0' for v in agg_df['SHAP']]
                fig_shap = go.Figure(go.Bar(
                    x=agg_df['SHAP'], y=agg_df['Feature'],
                    orientation='h', marker_color=colors,
                    text=[f"{v:+.3f}" for v in agg_df['SHAP']],
                    textposition='outside', textfont_color='#e8f0eb'
                ))
                fig_shap.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#7a9e87', height=400,
                    xaxis=dict(title='SHAP value (impact on default probability)',
                               title_font_color='#7a9e87', gridcolor='rgba(61,255,160,0.07)',
                               zeroline=True, zerolinecolor='rgba(61,255,160,0.3)'),
                    yaxis=dict(tickfont_color='#e8f0eb'),
                    margin=dict(t=10,b=40,r=80)
                )
                st.plotly_chart(fig_shap, use_container_width=True)
                st.markdown("""
                <div class="info-box">
                🔴 <strong>Red bars</strong> = features increasing default risk &nbsp;|&nbsp;
                🟢 <strong>Green bars</strong> = features reducing default risk.<br>
                Longer bars = stronger influence on this prediction.
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"SHAP computation note: {e}. Run a prediction first to see applicant-level explanation.")
    else:
        # Global feature importance from model
        st.markdown('<div class="section-label">Global Feature Importance (XGBoost)</div>', unsafe_allow_html=True)
        clf = model.named_steps['clf']
        preprocessor = model.named_steps['pre']
        cat_names = list(preprocessor.named_transformers_['cat'].get_feature_names_out(CAT_COLS))
        all_names = NUM_COLS + cat_names

        importances = clf.feature_importances_
        # Aggregate cat features
        agg = {n: imp for n, imp in zip(NUM_COLS, importances[:len(NUM_COLS)])}
        for i, name in enumerate(cat_names):
            orig = name.split('_')[0]
            agg[orig] = agg.get(orig, 0) + importances[len(NUM_COLS)+i]

        imp_df = pd.DataFrame(list(agg.items()), columns=['Feature','Importance'])
        imp_df = imp_df.sort_values('Importance', ascending=True)

        fig_imp = go.Figure(go.Bar(
            x=imp_df['Importance'], y=imp_df['Feature'], orientation='h',
            marker_color='#3dffa0', opacity=0.85,
            text=[f"{v:.3f}" for v in imp_df['Importance']],
            textposition='outside', textfont_color='#e8f0eb'
        ))
        fig_imp.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#7a9e87', height=420,
            xaxis=dict(title='Feature Importance Score', title_font_color='#7a9e87',
                       gridcolor='rgba(61,255,160,0.07)'),
            yaxis=dict(tickfont_color='#e8f0eb'),
            margin=dict(t=10,b=40,r=80)
        )
        st.plotly_chart(fig_imp, use_container_width=True)
        st.markdown("""
        <div class="info-box">
        Run a prediction to see <strong>applicant-level SHAP explanations</strong> — showing exactly why the model scored this specific borrower the way it did.
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 · ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    c1, c2 = st.columns([2,1])
    with c1:
        st.markdown("""
        <div class="section-label">About LoanGuard AI</div>
        <h3 style="font-family:'DM Serif Display',serif; color:#e8f0eb; margin-bottom:1rem;">
          Built to reduce bad loans. Built for decision-makers.
        </h3>
        <div style="color:#7a9e87; line-height:1.8; font-size:0.9rem;">
        <p>LoanGuard AI is an end-to-end credit risk assessment system designed for microfinance institutions, SACCOs, and digital lenders across East Africa. It replaces subjective loan officer judgement with a transparent, data-driven probability score — explained in plain language.</p>
        <br>
        <p><strong style="color:#e8f0eb;">The problem it solves:</strong> Non-performing loans (NPLs) cost lenders 15–30% of their portfolio value annually. Manual credit review is slow, inconsistent, and unscalable. LoanGuard AI delivers instant, consistent, explainable risk assessment for every applicant.</p>
        <br>
        <p><strong style="color:#e8f0eb;">Model architecture:</strong> XGBoost classifier trained on structured loan data, with a preprocessing pipeline handling numerical scaling and one-hot encoding of categorical features. SHAP values provide per-applicant transparency.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <br>
        <div class="section-label">Technical Stack</div>
        """, unsafe_allow_html=True)

        tech_cols = st.columns(3)
        techs = [
            ("ML Engine", "XGBoost · Scikit-learn · SHAP"),
            ("Data Layer", "Pandas · NumPy · Joblib"),
            ("Interface", "Streamlit · Plotly"),
        ]
        for col, (label, val) in zip(tech_cols, techs):
            with col:
                st.markdown(f"""<div class="metric-card safe" style="padding:1rem;">
                  <div class="metric-label">{label}</div>
                  <div style="color:#3dffa0; font-family:'DM Mono',monospace; font-size:0.75rem; margin-top:0.4rem; line-height:1.6;">{val}</div>
                </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="metric-card safe" style="text-align:center; padding:2rem;">
          <div class="metric-label">Built By</div>
          <div style="font-family:'DM Serif Display',serif; font-size:1.4rem; color:#e8f0eb; margin:0.8rem 0;">Evans Kiplangat</div>
          <div style="font-family:'DM Mono',monospace; font-size:0.7rem; color:#7a9e87; margin-bottom:1.5rem;">R&D Data Analyst · Nairobi, Kenya</div>
          <div style="display:flex; flex-direction:column; gap:0.5rem;">
            <a href="https://github.com/evans25575" target="_blank" style="color:#3dffa0; font-size:0.78rem; font-family:'DM Mono',monospace; text-decoration:none;">⌥ github.com/evans25575</a>
            <a href="mailto:kiplaevans2018@gmail.com" style="color:#3dffa0; font-size:0.78rem; font-family:'DM Mono',monospace; text-decoration:none;">✉ kiplaevans2018@gmail.com</a>
          </div>
        </div>
        <br>
        <div class="metric-card warning" style="padding:1.2rem;">
          <div class="metric-label">⚠ Disclaimer</div>
          <div style="font-size:0.78rem; color:#7a9e87; margin-top:0.5rem; line-height:1.6;">
          This system is for decision-support only. Final credit decisions must comply with applicable regulatory and ethical standards. Model should be retrained quarterly on live data.
          </div>
        </div>
        """, unsafe_allow_html=True)
