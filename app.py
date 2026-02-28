"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          AML FRAUD DETECTION  —  Streamlit Web Application                  ║
║          Author : Khalid Dharif  |  Graduation Project                       ║
║          Dataset: IBM Transactions for Anti-Money Laundering (AML)           ║
╚══════════════════════════════════════════════════════════════════════════════╝

Run locally:
    streamlit run app.py

Deploy:
    Push to GitHub → connect at streamlit.io/cloud
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION  (must be first Streamlit call)
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AML Fraud Detector",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #0f1117; }

    /* Fraud alert box */
    .fraud-box {
        background: linear-gradient(135deg, #ff4444, #cc0000);
        border-radius: 12px; padding: 20px; text-align: center;
        color: white; font-size: 1.6rem; font-weight: bold;
        box-shadow: 0 4px 20px rgba(255,68,68,0.4);
        animation: pulse 1.5s ease-in-out infinite;
    }
    /* Legitimate box */
    .legit-box {
        background: linear-gradient(135deg, #00c853, #007b33);
        border-radius: 12px; padding: 20px; text-align: center;
        color: white; font-size: 1.6rem; font-weight: bold;
        box-shadow: 0 4px 20px rgba(0,200,83,0.4);
    }
    /* Metric card */
    .metric-card {
        background: #1e2130; border-radius: 10px; padding: 16px;
        text-align: center; border: 1px solid #2d3250;
    }
    .metric-value { font-size: 2rem; font-weight: bold; color: #4fc3f7; }
    .metric-label { font-size: 0.85rem; color: #9e9e9e; margin-top: 4px; }

    /* Section header */
    .section-header {
        font-size: 1.1rem; font-weight: 600; color: #81d4fa;
        border-bottom: 2px solid #1976d2; padding-bottom: 6px;
        margin-bottom: 14px;
    }
    @keyframes pulse {
        0%, 100% { box-shadow: 0 4px 20px rgba(255,68,68,0.4); }
        50%       { box-shadow: 0 4px 30px rgba(255,68,68,0.8); }
    }
    /* Hide Streamlit footer */
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ARTIFACT LOADER  (cached — only runs once per session)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading model artifacts…")
def load_artifacts():
    """Load all saved joblib artifacts from the models/ directory."""
    base = os.path.join(os.path.dirname(__file__), "models")
    try:
        return {
            "model":        joblib.load(os.path.join(base, "aml_best_model.joblib")),
            "scaler":       joblib.load(os.path.join(base, "aml_scaler.joblib")),
            "le_from":      joblib.load(os.path.join(base, "aml_le_from_bank.joblib")),
            "le_to":        joblib.load(os.path.join(base, "aml_le_to_bank.joblib")),
            "le_pair":      joblib.load(os.path.join(base, "aml_le_bank_pair.joblib")),
            "le_currency":  joblib.load(os.path.join(base, "aml_le_currency.joblib")),
            "le_format":    joblib.load(os.path.join(base, "aml_le_format.joblib")),
            "feat_cols":    joblib.load(os.path.join(base, "aml_feature_columns.joblib")),
            "model_name":   joblib.load(os.path.join(base, "aml_model_name.joblib")),
        }
    except FileNotFoundError as e:
        st.error(f"❌ Model artifact not found: {e}\n\nRun the notebook first to generate the `.joblib` files, then place them in a `models/` folder.")
        st.stop()


def safe_int(value, default=0):
    """Convert potentially dirty numeric values to int."""
    if pd.isna(value):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def safe_float(value, default=0.0):
    """Convert potentially dirty numeric values to float."""
    if pd.isna(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_alert_reasons(inputs: dict, features: dict, proba: float, threshold: float) -> list:
    """Generate analyst-friendly reason codes to support investigations."""
    reasons = []
    amount_received = max(features.get("Amount Received", 0.0), 1.0)
    amount_diff = features.get("Amount_Diff", 0.0)
    diff_ratio = amount_diff / amount_received

    if proba >= threshold:
        reasons.append("Model probability exceeds active fraud threshold")
    if features.get("Is_High_Risk_Hour", 0):
        reasons.append("Transaction occurs in high-risk hour window (00:00-05:59)")
    if features.get("Is_Weekend", 0):
        reasons.append("Transaction scheduled on weekend")
    if inputs.get("recv_currency") != inputs.get("pay_currency"):
        reasons.append("Currency mismatch between debit and credit legs")
    if diff_ratio >= 0.15:
        reasons.append("Large payment/receipt mismatch suggests layering risk")
    if features.get("From_Bank_Fraud_History", 0) >= 5:
        reasons.append("Sender bank has elevated historical fraud count")
    if features.get("Bank_Pair_Fraud_History", 0) >= 3:
        reasons.append("Bank corridor has elevated historical fraud count")
    if features.get("Amount Received", 0) >= 100000:
        reasons.append("High-value transaction above enhanced due diligence trigger")

    return reasons or ["No dominant risk driver detected by current features"]


def compute_case_priority(proba: float, amount_diff: float, amount_received: float, from_hist: int, pair_hist: int) -> int:
    """Return a 0-100 composite priority score for fraud operations triage."""
    ratio = amount_diff / max(amount_received, 1.0)
    score = (
        proba * 65
        + min(ratio, 1.0) * 15
        + min(from_hist / 10, 1.0) * 10
        + min(pair_hist / 10, 1.0) * 10
    )
    return int(round(min(max(score, 0), 100)))


def recommend_case_action(priority_score: int) -> tuple:
    """Map priority to next action and target SLA."""
    if priority_score >= 85:
        return "Escalate immediately, hold transaction pending analyst decision", "Immediate"
    if priority_score >= 65:
        return "Level-2 AML review and customer verification", "< 4 hours"
    if priority_score >= 45:
        return "Queue for analyst screening with enhanced monitoring", "< 24 hours"
    return "Monitor only, no immediate intervention", "Next cycle"


def queue_bucket(priority_score: int) -> str:
    """Queue assignment used by operations teams."""
    if priority_score >= 85:
        return "P1-Critical"
    if priority_score >= 65:
        return "P2-High"
    if priority_score >= 45:
        return "P3-Medium"
    return "P4-Low"


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def predict(artifacts: dict, inputs: dict, threshold: float = 0.40) -> dict:
    """
    Core inference function. Takes raw input dict, applies full preprocessing
    pipeline, and returns structured prediction result.
    """
    le_from    = artifacts["le_from"]
    le_to      = artifacts["le_to"]
    le_pair    = artifacts["le_pair"]
    le_cur     = artifacts["le_currency"]
    le_fmt     = artifacts["le_format"]
    scaler     = artifacts["scaler"]
    clf        = artifacts["model"]
    feat_cols  = artifacts["feat_cols"]

    # ── Input validation ──────────────────────────────────────────────────────
    if inputs["from_bank"] not in le_from.classes_:
        return {"error": f"Bank ID {inputs['from_bank']} not in training data."}
    if inputs["to_bank"] not in le_to.classes_:
        return {"error": f"Bank ID {inputs['to_bank']} not in training data."}

    from_code = int(le_from.transform([inputs["from_bank"]])[0])
    to_code   = int(le_to.transform([inputs["to_bank"]])[0])
    pair_str  = f"{from_code}-{to_code}"

    if pair_str not in le_pair.classes_:
        return {"error": f"Bank pair {inputs['from_bank']}→{inputs['to_bank']} not in training data."}
    pair_code = int(le_pair.transform([pair_str])[0])

    for val, le, field in [
        (inputs["recv_currency"], le_cur, "receiving_currency"),
        (inputs["pay_currency"],  le_cur, "payment_currency"),
        (inputs["pay_format"],    le_fmt, "payment_format"),
    ]:
        if val not in le.classes_:
            return {"error": f"'{val}' is not valid for {field}."}

    # ── Feature construction ───────────────────────────────────────────────────
    hour    = inputs["hour"]
    day     = inputs["day"]
    weekday = inputs["weekday"]
    amount_r = inputs["amount_received"]
    amount_p = inputs["amount_paid"]

    # Cyclical encodings expected by the trained model.
    hour_sin = float(np.sin(2 * np.pi * hour / 24))
    hour_cos = float(np.cos(2 * np.pi * hour / 24))
    month_sin = float(np.sin(2 * np.pi * inputs["month"] / 12))
    month_cos = float(np.cos(2 * np.pi * inputs["month"] / 12))
    day_sin = float(np.sin(2 * np.pi * day / 31))
    day_cos = float(np.cos(2 * np.pi * day / 31))

    # Fallback operational features for real-time scoring when aggregate history
    # stats are not supplied by upstream transaction profiling systems.
    sender_tx_count = int(max(inputs["from_fraud_hist"] * 12 + 1, 1))
    sender_avg_amount = float(amount_r)
    sender_std_amount = float(max(amount_r * 0.15, 1.0))
    amount_zscore = float((amount_r - sender_avg_amount) / sender_std_amount)
    unique_bank_connections = int(max(inputs["pair_fraud_hist"] + 1, 1))

    features = {
        "Amount Received":         amount_r,
        "Receiving Currency":      int(le_cur.transform([inputs["recv_currency"]])[0]),
        "Amount Paid":             amount_p,
        "Payment Currency":        int(le_cur.transform([inputs["pay_currency"]])[0]),
        "Payment Format":          int(le_fmt.transform([inputs["pay_format"]])[0]),
        "hour_sin":                hour_sin,
        "hour_cos":                hour_cos,
        "month_sin":               month_sin,
        "month_cos":               month_cos,
        "day_sin":                 day_sin,
        "day_cos":                 day_cos,
        "Hour":                    hour,
        "Day":                     day,
        "Weekday":                 weekday,
        "Month":                   inputs["month"],
        "Is_Weekend":              int(weekday >= 5),
        "Is_Mid_Month":            int(10 <= day <= 17),
        "Is_High_Risk_Hour":       int(hour in [0, 1, 2, 3, 4, 5]),
        "Amount_Diff":             abs(amount_p - amount_r),
        "From_Bank_Fraud_History": inputs["from_fraud_hist"],
        "Bank_Pair_Fraud_History": inputs["pair_fraud_hist"],
        "Sender_Tx_Count":         sender_tx_count,
        "Sender_Avg_Amount":       sender_avg_amount,
        "Sender_Std_Amount":       sender_std_amount,
        "Amount_ZScore":           amount_zscore,
        "Unique_Bank_Connections": unique_bank_connections,
        "From_Bank_Code":          from_code,
        "To_Bank_Code":            to_code,
        "Bank_Pair_Code":          pair_code,
    }

    # Tolerant alignment: keeps scoring stable if model artifacts evolve.
    row = pd.DataFrame([features]).reindex(columns=feat_cols, fill_value=0)
    row_scaled = scaler.transform(row)
    proba      = float(clf.predict_proba(row_scaled)[0][1])
    pred       = int(proba >= threshold)

    # Risk bands
    if proba < 0.20:   risk, risk_color = "🟢 LOW",      "#4caf50"
    elif proba < 0.40: risk, risk_color = "🟡 MODERATE", "#ffb300"
    elif proba < 0.65: risk, risk_color = "🟠 HIGH",     "#ff7043"
    else:              risk, risk_color = "🔴 CRITICAL",  "#f44336"

    reasons = build_alert_reasons(inputs, features, proba, threshold)
    priority_score = compute_case_priority(
        proba=proba,
        amount_diff=features["Amount_Diff"],
        amount_received=features["Amount Received"],
        from_hist=features["From_Bank_Fraud_History"],
        pair_hist=features["Bank_Pair_Fraud_History"],
    )
    next_action, sla = recommend_case_action(priority_score)

    return {
        "pred":       pred,
        "proba":      proba,
        "risk":       risk,
        "risk_color": risk_color,
        "features":   features,
        "reasons":    reasons,
        "priority_score": priority_score,
        "queue":      queue_bucket(priority_score),
        "next_action": next_action,
        "sla":        sla,
    }


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: GAUGE CHART
# ══════════════════════════════════════════════════════════════════════════════
def make_gauge(probability: float) -> go.Figure:
    color = "#4caf50" if probability < 0.25 else \
            "#ffb300" if probability < 0.40 else \
            "#ff7043" if probability < 0.65 else "#f44336"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(probability * 100, 2),
        title={"text": "Fraud Probability (%)", "font": {"size": 16, "color": "#e0e0e0"}},
        number={"suffix": "%", "font": {"size": 36, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#9e9e9e",
                     "tickfont": {"color": "#9e9e9e"}},
            "bar":  {"color": color},
            "bgcolor": "#1e2130",
            "steps": [
                {"range": [0,  25], "color": "#1b5e20"},
                {"range": [25, 40], "color": "#827717"},
                {"range": [40, 65], "color": "#bf360c"},
                {"range": [65, 100],"color": "#b71c1c"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 3},
                "thickness": 0.8,
                "value": probability * 100
            }
        }
    ))
    fig.update_layout(
        height=280, margin=dict(t=40, b=10, l=20, r=20),
        paper_bgcolor="#0f1117", font_color="#e0e0e0"
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: FEATURE BREAKDOWN BAR CHART
# ══════════════════════════════════════════════════════════════════════════════
def make_feature_chart(features: dict) -> go.Figure:
    # Show selected features in a horizontal bar
    display = {
        "Amount Received":          features.get("Amount Received", 0),
        "Amount Paid":              features.get("Amount Paid", 0),
        "Amount Diff":              features.get("Amount_Diff", 0),
        "Fraud History (Bank)":     features.get("From_Bank_Fraud_History", 0),
        "Fraud History (Pair)":     features.get("Bank_Pair_Fraud_History", 0),
        "Is Weekend":               features.get("Is_Weekend", 0),
        "Is High-Risk Hour":        features.get("Is_High_Risk_Hour", 0),
    }
    labels = list(display.keys())
    values = list(display.values())
    colors = ["#ef5350" if v > 0 else "#42a5f5" for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation='h',
        marker_color=colors, opacity=0.85,
    ))
    fig.update_layout(
        title="Transaction Feature Values",
        height=300, margin=dict(t=40, b=10, l=10, r=10),
        paper_bgcolor="#1e2130", plot_bgcolor="#1e2130",
        font_color="#e0e0e0",
        xaxis={"gridcolor": "#2d3250"},
        yaxis={"gridcolor": "#2d3250"},
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE  (for history log)
# ══════════════════════════════════════════════════════════════════════════════
if "history" not in st.session_state:
    st.session_state.history = []
if "alert_log" not in st.session_state:
    st.session_state.alert_log = []
if "last_batch_results" not in st.session_state:
    st.session_state.last_batch_results = pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://img.icons8.com/color/96/bank-building.png", width=72)
    st.title("AML Fraud Detector")
    st.caption("Graduation Project · Khalid Dharif")
    st.markdown("---")

    # Navigation
    page = st.radio(
        "Navigate",
        ["🔍 Single Transaction", "📋 Batch Scoring", "🚨 Operations Dashboard", "📊 Model Info", "📖 About"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("**Detection Policy**")
    policy = st.selectbox(
        "Policy Profile",
        options=["Balanced", "Aggressive Detection", "Precision Focus"],
        index=0,
        help="Aggressive lowers threshold for recall; Precision raises threshold to reduce false alerts."
    )
    policy_thresholds = {
        "Aggressive Detection": 0.30,
        "Balanced": 0.40,
        "Precision Focus": 0.55,
    }
    threshold = st.slider(
        "Fraud Threshold",
        min_value=0.10, max_value=0.90, value=policy_thresholds[policy], step=0.05,
        help="Probability above which a transaction is flagged as fraud. "
             "Lower = more sensitive (more alerts). Higher = more precise (fewer alerts)."
    )
    st.caption(f"Profile default: {policy_thresholds[policy]:.0%}")
    st.markdown(f"*Current threshold: `{threshold:.0%}`*")
    st.markdown("---")
    st.markdown("**Risk Operations Context**")
    st.caption("Priority queues: P1 Critical, P2 High, P3 Medium, P4 Low")
    st.caption("Analyst SLA is based on composite case priority score")
    st.markdown("---")
    st.markdown("**Model Source**")
    st.caption("IBM Synthetic AML Transactions")
    st.caption("~500K transactions | ~0.2% fraud rate")


# ══════════════════════════════════════════════════════════════════════════════
# LOAD ARTIFACTS
# ══════════════════════════════════════════════════════════════════════════════
artifacts = load_artifacts()
known_from_banks = sorted(artifacts["le_from"].classes_.tolist())
known_to_banks   = sorted(artifacts["le_to"].classes_.tolist())
known_currencies = sorted(artifacts["le_currency"].classes_.tolist())
known_formats    = sorted(artifacts["le_format"].classes_.tolist())


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — SINGLE TRANSACTION SCORING
# ══════════════════════════════════════════════════════════════════════════════
if page == "🔍 Single Transaction":
    st.title("🏦 AML Transaction Risk Scorer")
    st.markdown(f"*Active model: **{artifacts['model_name']}**  |  Fraud threshold: **{threshold:.0%}***")
    st.markdown("---")

    # ── Input form ────────────────────────────────────────────────────────────
    with st.form("transaction_form"):
        c1, c2, c3 = st.columns(3)

        # Column 1 — Bank & Amount
        with c1:
            st.markdown('<div class="section-header">🏛️ Banks & Amounts</div>', unsafe_allow_html=True)
            from_bank = st.selectbox("Sending Bank ID", options=known_from_banks, index=0)
            to_bank   = st.selectbox("Receiving Bank ID", options=known_to_banks, index=min(5, len(known_to_banks) - 1))
            amount_r  = st.number_input("Amount Received ($)", min_value=0.01, value=1_858.96, step=100.0, format="%.2f")
            amount_p  = st.number_input("Amount Paid ($)",     min_value=0.01, value=1_858.96, step=100.0, format="%.2f")

        # Column 2 — Currency & Format
        with c2:
            st.markdown('<div class="section-header">💱 Currency & Format</div>', unsafe_allow_html=True)
            recv_currency = st.selectbox("Receiving Currency", options=known_currencies,
                                         index=known_currencies.index("US Dollar") if "US Dollar" in known_currencies else 0)
            pay_currency  = st.selectbox("Payment Currency",   options=known_currencies,
                                         index=known_currencies.index("US Dollar") if "US Dollar" in known_currencies else 0)
            pay_format    = st.selectbox("Payment Format",     options=known_formats,
                                         index=known_formats.index("Wire Transfer") if "Wire Transfer" in known_formats else 0)

        # Column 3 — Time & History
        with c3:
            st.markdown('<div class="section-header">⏰ Time & Risk Context</div>', unsafe_allow_html=True)
            now     = datetime.now()
            hour    = st.slider("Hour of Day",    0, 23, now.hour)
            day     = st.slider("Day of Month",   1, 31, now.day)
            weekday = st.slider("Weekday (0=Mon, 6=Sun)", 0, 6, now.weekday())
            month   = st.slider("Month",          1, 12, now.month)
            st.markdown("**Historical Risk Context**")
            from_hist = st.number_input("Sender Bank Prior Frauds",  0, 10000, 0)
            pair_hist = st.number_input("Bank Pair Prior Frauds",     0, 10000, 0)

        submitted = st.form_submit_button("🔎 Score Transaction", use_container_width=True, type="primary")

    # ── Run prediction ─────────────────────────────────────────────────────────
    if submitted:
        inputs = dict(
            from_bank=from_bank, to_bank=to_bank,
            amount_received=amount_r, amount_paid=amount_p,
            recv_currency=recv_currency, pay_currency=pay_currency,
            pay_format=pay_format,
            hour=hour, day=day, weekday=weekday, month=month,
            from_fraud_hist=from_hist, pair_fraud_hist=pair_hist,
        )

        with st.spinner("Scoring transaction…"):
            time.sleep(0.4)   # small delay for UX realism
            result = predict(artifacts, inputs, threshold)

        if "error" in result:
            st.error(f"⚠️ {result['error']}")
        else:
            st.markdown("---")
            # ── Verdict banner ────────────────────────────────────────────────
            if result["pred"] == 1:
                st.markdown(f'<div class="fraud-box">🚨 FRAUD DETECTED  ·  {result["risk"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="legit-box">✅ LEGITIMATE TRANSACTION  ·  {result["risk"]}</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Metrics row ───────────────────────────────────────────────────
            m1, m2, m3, m4 = st.columns(4)
            prob_pct = result["proba"] * 100
            with m1:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{prob_pct:.1f}%</div><div class="metric-label">Fraud Probability</div></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="metric-card"><div class="metric-value">${amount_r:,.0f}</div><div class="metric-label">Amount Received</div></div>', unsafe_allow_html=True)
            with m3:
                st.markdown(f'<div class="metric-card"><div class="metric-value">${abs(amount_p-amount_r):,.0f}</div><div class="metric-label">Amount Difference</div></div>', unsafe_allow_html=True)
            with m4:
                is_risky = from_hist + pair_hist
                st.markdown(f'<div class="metric-card"><div class="metric-value">{is_risky}</div><div class="metric-label">Total Prior Frauds</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            q1, q2, q3 = st.columns(3)
            with q1:
                st.metric("Queue", result["queue"])
            with q2:
                st.metric("Priority Score", f'{result["priority_score"]}/100')
            with q3:
                st.metric("Analyst SLA", result["sla"])

            st.info(f'**Next Action:** {result["next_action"]}')
            with st.expander("Why this alert was scored this way"):
                for reason in result["reasons"]:
                    st.write(f"- {reason}")

            # ── Charts ────────────────────────────────────────────────────────
            ch1, ch2 = st.columns(2)
            with ch1:
                st.plotly_chart(make_gauge(result["proba"]), use_container_width=True)
            with ch2:
                st.plotly_chart(make_feature_chart(result["features"]), use_container_width=True)

            # ── Log to history ────────────────────────────────────────────────
            st.session_state.history.append({
                "Time":        datetime.now().strftime("%H:%M:%S"),
                "From→To":     f"{from_bank} → {to_bank}",
                "Amount":      f"${amount_r:,.2f}",
                "Format":      pay_format,
                "Fraud %":     f"{prob_pct:.1f}%",
                "Risk":        result["risk"],
                "Queue":       result["queue"],
                "Verdict":     "FRAUD" if result["pred"] else "OK",
            })
            st.session_state.alert_log.append({
                "scored_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "from_bank": from_bank,
                "to_bank": to_bank,
                "amount_received": amount_r,
                "amount_paid": amount_p,
                "fraud_probability": round(prob_pct, 2),
                "risk": result["risk"],
                "queue": result["queue"],
                "priority_score": result["priority_score"],
                "next_action": result["next_action"],
                "sla": result["sla"],
                "reasons": " | ".join(result["reasons"]),
            })

            # ── Latest 5 scored ───────────────────────────────────────────────
            if len(st.session_state.history) > 1:
                st.markdown("---")
                st.markdown("**📋 Recent Transactions (this session)**")
                hist_df = pd.DataFrame(st.session_state.history[-5:][::-1])
                st.dataframe(hist_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — BATCH SCORING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Batch Scoring":
    st.title("📋 Batch Transaction Scoring")
    st.markdown("Upload a CSV file with transaction data to score multiple transactions at once.")

    st.markdown("""
    **Required columns:**
    `from_bank`, `to_bank`, `amount_received`, `amount_paid`, `recv_currency`,
    `pay_currency`, `pay_format`, `hour`, `day`, `weekday`, `month`,
    `from_fraud_hist` *(optional)*, `pair_fraud_hist` *(optional)*
    """)

    # Demo CSV download
    demo_data = pd.DataFrame([
        {"from_bank": 3402, "to_bank": 1120, "amount_received": 1858.96, "amount_paid": 1858.96,
         "recv_currency": "US Dollar", "pay_currency": "US Dollar", "pay_format": "Wire Transfer",
         "hour": 14, "day": 8, "weekday": 1, "month": 9, "from_fraud_hist": 0, "pair_fraud_hist": 0},
        {"from_bank": 3402, "to_bank": 1120, "amount_received": 47500, "amount_paid": 47500,
         "recv_currency": "Bitcoin", "pay_currency": "Bitcoin", "pay_format": "Bitcoin",
         "hour": 2, "day": 15, "weekday": 6, "month": 9, "from_fraud_hist": 8, "pair_fraud_hist": 4},
    ])
    st.download_button("⬇️ Download Sample CSV", demo_data.to_csv(index=False),
                       "aml_sample.csv", "text/csv")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df_upload = pd.read_csv(uploaded)
        if df_upload.empty:
            st.warning("Uploaded CSV is empty. Please upload at least one transaction.")
        else:
            st.write(f"Loaded **{len(df_upload):,}** transactions. Scoring…")
            prog = st.progress(0)
            results = []

            for i, row in df_upload.iterrows():
                inputs = {
                    "from_bank":       safe_int(row.get("from_bank", known_from_banks[0]), known_from_banks[0]),
                    "to_bank":         safe_int(row.get("to_bank", known_to_banks[0]), known_to_banks[0]),
                    "amount_received": max(safe_float(row.get("amount_received", 0.01), 0.01), 0.01),
                    "amount_paid":     max(safe_float(row.get("amount_paid", 0.01), 0.01), 0.01),
                    "recv_currency":   str(row.get("recv_currency", "US Dollar")).strip(),
                    "pay_currency":    str(row.get("pay_currency", "US Dollar")).strip(),
                    "pay_format":      str(row.get("pay_format", "Wire Transfer")).strip(),
                    "hour":            min(max(safe_int(row.get("hour", 12), 12), 0), 23),
                    "day":             min(max(safe_int(row.get("day", 1), 1), 1), 31),
                    "weekday":         min(max(safe_int(row.get("weekday", 0), 0), 0), 6),
                    "month":           min(max(safe_int(row.get("month", 1), 1), 1), 12),
                    "from_fraud_hist": max(safe_int(row.get("from_fraud_hist", 0), 0), 0),
                    "pair_fraud_hist": max(safe_int(row.get("pair_fraud_hist", 0), 0), 0),
                }
                r = predict(artifacts, inputs, threshold)

                if "error" in r:
                    out = {
                        "Verdict": "ERROR",
                        "Fraud_%": np.nan,
                        "Risk_Level": "N/A",
                        "Queue": "Validation",
                        "Priority_Score": np.nan,
                        "SLA": "N/A",
                        "Next_Action": "Correct input data and rescore",
                        "Reason_Codes": r["error"],
                    }
                else:
                    out = {
                        "Verdict": "FRAUD" if r["pred"] else "LEGITIMATE",
                        "Fraud_%": round(r["proba"] * 100, 2),
                        "Risk_Level": r["risk"],
                        "Queue": r["queue"],
                        "Priority_Score": r["priority_score"],
                        "SLA": r["sla"],
                        "Next_Action": r["next_action"],
                        "Reason_Codes": " | ".join(r["reasons"]),
                    }
                results.append(out)
                prog.progress((i + 1) / len(df_upload))

            results_df = pd.DataFrame(results)
            scored_df = pd.concat([df_upload.reset_index(drop=True), results_df], axis=1)
            st.session_state.last_batch_results = scored_df.copy()

            fraud_mask = scored_df["Verdict"] == "FRAUD"
            error_mask = scored_df["Verdict"] == "ERROR"
            n_fraud = int(fraud_mask.sum())
            n_error = int(error_mask.sum())
            fraud_rate = n_fraud / len(scored_df) * 100

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Transactions", f"{len(scored_df):,}")
            c2.metric("Fraud Alerts", f"{n_fraud:,}", f"{fraud_rate:.1f}%")
            c3.metric("Validation Errors", f"{n_error:,}")
            c4.metric("Policy", policy)
            st.success(f"Completed scoring. {n_fraud:,} transactions queued as fraud alerts.")

            if n_fraud > 0:
                queue_counts = scored_df.loc[fraud_mask, "Queue"].value_counts().reset_index()
                queue_counts.columns = ["Queue", "Count"]
                fig_q = px.bar(
                    queue_counts,
                    x="Queue",
                    y="Count",
                    color="Queue",
                    title="Fraud Alert Queue Distribution",
                    color_discrete_map={
                        "P1-Critical": "#ef5350",
                        "P2-High": "#ff7043",
                        "P3-Medium": "#ffb300",
                        "P4-Low": "#42a5f5",
                        "Validation": "#9e9e9e",
                    }
                )
                fig_q.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#1e2130", font_color="#e0e0e0")
                st.plotly_chart(fig_q, use_container_width=True)

            st.dataframe(scored_df, use_container_width=True, hide_index=True)
            st.download_button(
                "⬇️ Download Case Queue CSV",
                scored_df.to_csv(index=False),
                "aml_case_queue.csv",
                "text/csv"
            )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — OPERATIONS DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🚨 Operations Dashboard":
    st.title("🚨 Fraud Operations Dashboard")
    st.markdown("Session-level investigator view of alerts and queue load.")

    if not st.session_state.alert_log and st.session_state.last_batch_results.empty:
        st.info("No scored transactions yet. Run single or batch scoring first.")
    else:
        alert_df = pd.DataFrame(st.session_state.alert_log)
        if not alert_df.empty:
            st.markdown("### Live Alert Log (Single Scoring)")
            m1, m2, m3 = st.columns(3)
            m1.metric("Alerts Logged", len(alert_df))
            m2.metric("Avg Priority", f"{alert_df['priority_score'].mean():.1f}")
            m3.metric("P1/P2 Share", f"{(alert_df['queue'].isin(['P1-Critical', 'P2-High']).mean() * 100):.1f}%")

            fig_live = px.histogram(
                alert_df, x="priority_score", color="queue", nbins=20,
                title="Priority Score Distribution (Single Scoring)"
            )
            fig_live.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#1e2130", font_color="#e0e0e0")
            st.plotly_chart(fig_live, use_container_width=True)
            st.dataframe(alert_df.iloc[::-1], use_container_width=True, hide_index=True)

        if not st.session_state.last_batch_results.empty:
            st.markdown("### Last Batch Queue Snapshot")
            batch_df = st.session_state.last_batch_results.copy()
            fraud_batch = batch_df[batch_df["Verdict"] == "FRAUD"]
            if not fraud_batch.empty:
                b1, b2, b3 = st.columns(3)
                b1.metric("Fraud Alerts", int((batch_df["Verdict"] == "FRAUD").sum()))
                b2.metric("Avg Fraud %", f"{fraud_batch['Fraud_%'].mean():.1f}%")
                b3.metric("Top Queue", fraud_batch["Queue"].mode().iloc[0])
                st.dataframe(
                    fraud_batch.sort_values(by="Priority_Score", ascending=False).head(20),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.caption("Last batch has no fraud alerts.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL INFO
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Info":
    st.title("📊 Model & Performance Overview")

    model_name = artifacts["model_name"]
    st.markdown(f"### Active Model: **{model_name}**")

    model_descriptions = {
        "Logistic Regression": "Linear classifier. Fast and interpretable but cannot capture non-linear patterns.",
        "Decision Tree":       "Single decision tree. Highly interpretable rules but prone to overfitting.",
        "Random Forest":       "Bagging ensemble of 200 decision trees. Robust and stable — excellent general-purpose model.",
        "XGBoost":             "Level-wise gradient boosting. Strong regularisation; typically low false-positive rate.",
        "LightGBM":            "Leaf-wise gradient boosting (Microsoft). 10–100× faster than XGBoost on large datasets; competitive AUC.",
        "CatBoost":            "Ordered boosting (Yandex). Minimal overfitting; often achieves the highest AUC on financial tabular data.",
    }

    # Model comparison table
    st.markdown("### 6-Model Comparison")
    comparison = pd.DataFrame({
        "Model": list(model_descriptions.keys()),
        "Family": ["Linear", "Tree", "Bagging", "Boosting", "Boosting", "Boosting"],
        "Speed": ["⚡⚡⚡", "⚡⚡⚡", "⚡⚡", "⚡⚡", "⚡⚡⚡", "⚡"],
        "AUC (typical)": ["~0.877", "~0.903", "~0.935", "~0.926", "~0.938", "~0.940"],
        "Key Strength": [
            "Interpretability", "Rule extraction", "Stability",
            "Low false positives", "Speed + accuracy", "Robustness"
        ],
        "Description": list(model_descriptions.values()),
    })
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### Feature Dictionary")
    feat_info = pd.DataFrame({
        "Feature": [
            "Amount Received", "Amount Paid", "Amount_Diff",
            "Receiving Currency", "Payment Currency", "Payment Format",
            "Hour", "Day", "Weekday", "Month",
            "Is_Weekend", "Is_Mid_Month", "Is_High_Risk_Hour",
            "From_Bank_Fraud_History", "Bank_Pair_Fraud_History",
            "From_Bank_Code", "To_Bank_Code", "Bank_Pair_Code"
        ],
        "Type": ["numeric"]*3 + ["encoded"]*3 + ["numeric"]*4 + ["binary"]*3 + ["numeric"]*2 + ["encoded"]*3,
        "Description": [
            "Amount credited to receiver (receiving currency)",
            "Amount debited from sender (payment currency)",
            "Absolute difference — non-zero hints at FX conversion layering",
            "Label-encoded receiving currency",
            "Label-encoded payment currency",
            "Label-encoded payment method",
            "Hour of transaction (0–23)",
            "Day of month (1–31)",
            "Day of week (0=Monday)",
            "Month (1–12)",
            "1 if Saturday or Sunday",
            "1 if day 10–17",
            "1 if hour 0–5 (midnight to 6 AM)",
            "Cumulative prior frauds from this sending bank",
            "Cumulative prior frauds on this bank corridor",
            "Label-encoded From Bank ID",
            "Label-encoded To Bank ID",
            "Label-encoded bank pair string",
        ]
    })
    st.dataframe(feat_info, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📖 About":
    st.title("📖 About This Project")
    st.markdown("""
    ## Anti-Money Laundering Detection using Machine Learning
    **Graduation Project** · Khalid Dharif

    ---

    ### What is Money Laundering?
    Money laundering is the process of disguising the proceeds of crime as legitimate income.
    It costs the global economy an estimated **$800 billion – $2 trillion per year** (UNODC).
    Traditional rule-based AML systems struggle to keep pace with sophisticated, adaptive schemes.

    ### What This System Does
    This application scores financial transactions in real time using a machine learning model
    trained on IBM's synthetic AML dataset (~500,000 transactions). It outputs:
    - A **fraud probability** (0–100%)
    - A **risk level** (Low / Moderate / High / Critical)
    - A **verdict** (Legitimate / Fraud Detected)

    ### The ML Pipeline
    | Stage | Detail |
    |-------|--------|
    | Dataset | IBM Synthetic AML — `LI-Small_Trans.csv` |
    | Class balance | SMOTE (applied to training set only) |
    | Models compared | Logistic Regression, Decision Tree, Random Forest, XGBoost, **LightGBM**, **CatBoost** |
    | Best model | Selected by ROC-AUC on held-out 30% test set |
    | Key features | Temporal signals, amount difference, cumulative bank fraud history |

    ### Technical Stack
    - **ML:** scikit-learn, XGBoost, LightGBM, CatBoost, imbalanced-learn
    - **App:** Streamlit + Plotly
    - **Deployment:** Streamlit Community Cloud (via GitHub)

    ---
    *Built with ❤️ for the IBM AML Graduation Project.*
    """)
