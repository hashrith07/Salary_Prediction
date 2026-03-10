from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
import traceback

app = FastAPI(
    title="Salary Prediction API",
    description="Predicts salary in INR based on experience, education, job title and gender",
    version="1.0.0"
)

# Load model and columns at startup
try:
    model = joblib.load("salary_prediction_model.pkl")
    model_columns = joblib.load("model_columns.pkl")
except Exception as e:
    print(f"Failed to load model or columns: {e}")
    raise


class SalaryInput(BaseModel):
    years_of_experience: float
    education_level: str
    job_title: str
    gender: str


@app.get("/")
def root():
    return {"message": "Salary Prediction API is running"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "model_type": type(model).__name__
    }


@app.post("/predict")
def predict_salary(input_data: SalaryInput):
    try:
        # Convert to DataFrame (single row)
        df_input = pd.DataFrame([input_data.dict()])

        # One-hot encode
        df_encoded = pd.get_dummies(df_input)

        # Align with training columns
        df_encoded = df_encoded.reindex(columns=model_columns, fill_value=0)

        # Predict
        prediction_array = model.predict(df_encoded)

        # Safely extract single float value
        if isinstance(prediction_array, np.ndarray):
            if prediction_array.size == 1:
                predicted_salary = float(prediction_array.item())
            elif prediction_array.ndim == 1 and len(prediction_array) == 1:
                predicted_salary = float(prediction_array[0])
            elif prediction_array.ndim == 2 and prediction_array.shape[0] == 1:
                predicted_salary = float(prediction_array[0, 0])
            else:
                raise ValueError(f"Unexpected prediction shape: {prediction_array.shape}")
        else:
            predicted_salary = float(prediction_array)

        # Calculate range
        min_salary = predicted_salary * 0.9
        max_salary = predicted_salary * 1.1

        # Indian rupee formatting helper
        def format_inr(value):
            return f"₹{int(round(value)):,}"

        return {
            "predicted_salary": round(predicted_salary, 2),
            "salary_range": f"{format_inr(min_salary)} - {format_inr(max_salary)}",
            "currency": "INR"
        }

    except Exception as e:
        return {
            "error": str(e),
            "detail": traceback.format_exc()   # optional — remove in production if you want
        }, 500