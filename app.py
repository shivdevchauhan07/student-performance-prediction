"""
Student Performance Prediction web app.
Run after installing requirements and training the model:
    python train_model.py
    streamlit run app.py
"""
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

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="centered")
st.title("🎓 Student Performance Prediction")
st.write(
    "Enter learning-related indicators to estimate a student's final score. "
    "This is an educational demonstration, not an official grading tool."
)

if not MODEL_PATH.exists():
    st.error("Model file not found. First run: python train_model.py")
    st.stop()

model = joblib.load(MODEL_PATH)

with st.form("prediction_form"):
    c1, c2 = st.columns(2)
    with c1:
        study = st.slider("Study hours per day", 0.5, 8.0, 3.0, 0.5)
        attendance = st.slider("Attendance (%)", 45, 100, 80)
        previous = st.slider("Previous score", 0, 100, 65)
    with c2:
        assignment = st.slider("Assignment score", 0, 100, 70)
        sleep = st.slider("Sleep hours per day", 4.0, 9.0, 7.0, 0.5)
        participation = st.slider("Class participation (1–5)", 1, 5, 3)
    submitted = st.form_submit_button("Predict score")

if submitted:
    row = pd.DataFrame([{
        "study_hours_per_day": study,
        "attendance_percent": attendance,
        "previous_score": previous,
        "assignment_score": assignment,
        "sleep_hours_per_day": sleep,
        "class_participation_1_to_5": participation,
    }], columns=FEATURES)
    estimate = float(model.predict(row)[0])
    estimate = max(0.0, min(100.0, estimate))
    st.metric("Estimated final score", f"{estimate:.1f} / 100")
    if estimate >= 75:
        st.success("Indicative band: Strong performance")
    elif estimate >= 50:
        st.info("Indicative band: Moderate performance")
    else:
        st.warning("Indicative band: May need additional support")
    st.caption("Prediction is approximate. It should support—not replace—teacher judgement.")
