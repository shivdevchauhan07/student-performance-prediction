from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "student_performance_model.pkl"

FEATURES = [
    "study_hours_per_day",
    "attendance_percent",
    "previous_score",
    "assignment_score",
    "sleep_hours_per_day",
    "class_participation_1_to_5",
]

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom styling: responsive, clean, presentation-friendly.
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #0b1220 0%, #111827 55%, #0b1220 100%);
        color: #f8fafc;
    }
    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 2.5rem;
    }
    .hero {
        padding: 1.5rem 1.6rem;
        border: 1px solid rgba(148,163,184,.25);
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(30,41,59,.95), rgba(15,23,42,.95));
        margin-bottom: 1.2rem;
    }
    .eyebrow {
        color: #5eead4;
        font-size: .82rem;
        font-weight: 700;
        letter-spacing: .12em;
        text-transform: uppercase;
    }
    .hero h1 {
        margin: .35rem 0 .5rem 0;
        font-size: clamp(1.8rem, 4vw, 2.7rem);
        line-height: 1.15;
        color: #f8fafc;
    }
    .hero p {
        color: #cbd5e1;
        margin-bottom: 0;
        font-size: 1rem;
    }
    div[data-testid="stForm"] {
        border: 1px solid rgba(148,163,184,.25);
        border-radius: 18px;
        padding: 1.15rem;
        background: rgba(15,23,42,.78);
    }
    div[data-testid="stMetric"] {
        background: rgba(15,23,42,.9);
        border: 1px solid rgba(45,212,191,.35);
        padding: 1rem 1.1rem;
        border-radius: 16px;
    }
    .result-note {
        color: #cbd5e1;
        font-size: .9rem;
    }
    div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
        width: 100%;
        min-height: 2.8rem;
        border-radius: 12px;
        font-weight: 700;
        border: 1px solid #2dd4bf;
    }
    [data-testid="stCaptionContainer"] {
        color: #94a3b8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Machine Learning • BCA Project</div>
      <h1>🎓 Student Performance Predictor</h1>
      <p>Explore how learning-related indicators may relate to an estimated final score.
      This is an educational demonstration, not an official grading tool.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

top1, top2, top3 = st.columns(3)
with top1:
    st.markdown("**📚 Study habits**")
    st.caption("Daily study time and sleep")
with top2:
    st.markdown("**📝 Academic indicators**")
    st.caption("Previous and assignment scores")
with top3:
    st.markdown("**🤝 Engagement**")
    st.caption("Attendance and participation")

if not MODEL_PATH.exists():
    st.error(
        "The trained model file is missing. Ensure "
        "`student_performance_model.pkl` is committed in the same GitHub folder as `app.py`."
    )
    st.info("The model file is required to make predictions. Do not upload an empty placeholder file.")
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
except Exception as exc:
    st.error("The model file could not be loaded. Check that the correct trained .pkl file is in the repository.")
    st.caption(f"Technical detail: {type(exc).__name__}")
    st.stop()

st.subheader("Enter student indicators")
st.caption("Adjust the values below, then tap **Predict score**.")

with st.form("prediction_form"):
    left, right = st.columns(2)
    with left:
        study = st.slider("Study hours per day", 0.5, 8.0, 3.0, 0.5)
        attendance = st.slider("Attendance (%)", 45, 100, 80)
        previous = st.slider("Previous score", 0, 100, 65)
    with right:
        assignment = st.slider("Assignment score", 0, 100, 70)
        sleep = st.slider("Sleep hours per day", 4.0, 9.0, 7.0, 0.5)
        participation = st.slider("Class participation (1–5)", 1, 5, 3)
    submitted = st.form_submit_button("✨ Predict score")

if submitted:
    row = pd.DataFrame(
        [{
            "study_hours_per_day": study,
            "attendance_percent": attendance,
            "previous_score": previous,
            "assignment_score": assignment,
            "sleep_hours_per_day": sleep,
            "class_participation_1_to_5": participation,
        }],
        columns=FEATURES,
    )
    try:
        estimate = float(model.predict(row)[0])
        estimate = max(0.0, min(100.0, estimate))
    except Exception as exc:
        st.error("Prediction failed. Check the model and feature columns.")
        st.caption(f"Technical detail: {type(exc).__name__}")
        st.stop()

    st.divider()
    st.subheader("Prediction result")
    metric_col, band_col = st.columns([1, 1])
    with metric_col:
        st.metric("Estimated final score", f"{estimate:.1f} / 100")
        st.progress(int(round(estimate)))
    with band_col:
        if estimate >= 75:
            st.success("**Indicative band: Strong performance**")
            st.write("The model estimates a relatively high score for these inputs.")
        elif estimate >= 50:
            st.info("**Indicative band: Moderate performance**")
            st.write("The model estimates a mid-range score for these inputs.")
        else:
            st.warning("**Indicative band: May need additional support**")
            st.write("Consider reviewing learning support needs with a teacher.")

    with st.expander("View the values used for this prediction"):
        st.dataframe(
            pd.DataFrame(
                {
                    "Indicator": [
                        "Study hours per day",
                        "Attendance",
                        "Previous score",
                        "Assignment score",
                        "Sleep hours per day",
                        "Class participation",
                    ],
                    "Value": [
                        f"{study:g} hours",
                        f"{attendance}%",
                        f"{previous}/100",
                        f"{assignment}/100",
                        f"{sleep:g} hours",
                        f"{participation}/5",
                    ],
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

st.divider()
st.markdown("### About this project")
about1, about2 = st.columns(2)
with about1:
    st.write("**Model:** trained machine-learning regression model")
    st.write("**Inputs:** six learning-related indicators")
with about2:
    st.write("**Purpose:** educational project demonstration")
    st.write("**Data note:** the demo uses synthetic data; predictions are not validated for real students.")

st.caption(
    "For learning and demonstration only. Predictions are approximate and should support—not replace—teacher judgement."
)
