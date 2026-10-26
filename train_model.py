"""
Train the Student Performance Prediction model.
The included CSV is synthetic demonstration data, not real student records.
Run: python train_model.py
"""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import math

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "student_performance.csv"
MODEL_PATH = ROOT / "student_performance_model.pkl"
FEATURES = [
    "study_hours_per_day",
    "attendance_percent",
    "previous_score",
    "assignment_score",
    "sleep_hours_per_day",
    "class_participation_1_to_5",
]

def main():
    data = pd.read_csv(DATA_PATH)
    X = data[FEATURES]
    y = data["final_score"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    model = RandomForestRegressor(
        n_estimators=160, random_state=42, min_samples_leaf=2
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"Test rows: {len(X_test)}")
    print(f"MAE: {mean_absolute_error(y_test, predictions):.2f}")
    print(f"RMSE: {math.sqrt(mean_squared_error(y_test, predictions)):.2f}")
    print(f"R2: {r2_score(y_test, predictions):.3f}")
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to: {MODEL_PATH}")

if __name__ == "__main__":
    main()
