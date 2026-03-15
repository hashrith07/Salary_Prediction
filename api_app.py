from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import os

app = FastAPI(
    title="Salary Prediction API",
    description="Predicts salary using ML model with real-world constraints",
    version="5.0.0"
)

# -----------------------------------------------------
# Load Model + Encoders
# -----------------------------------------------------

MODEL_DIR = os.path.dirname(__file__)

try:
    model = joblib.load(os.path.join(MODEL_DIR, "salary_model.pkl"))
    encoders = joblib.load(os.path.join(MODEL_DIR, "encoders.pkl"))

    JOB_ENCODER = encoders.get("Job_Title")
    GENDER_ENCODER = encoders.get("Gender")
    EDU_ENCODER = encoders.get("Education_Level")

    if not all([model, JOB_ENCODER, GENDER_ENCODER, EDU_ENCODER]):
        raise ValueError("Model or encoders missing")

except Exception as e:
    raise RuntimeError(f"Failed to load model/encoders: {str(e)}")


# -----------------------------------------------------
# Input Schema
# -----------------------------------------------------

class SalaryInput(BaseModel):

    age: int = Field(..., ge=18, le=65)

    gender: str

    education_level: str

    job_title: str

    years_of_experience: float = Field(..., ge=0, le=45)


# -----------------------------------------------------
# Real World Validation Rules
# -----------------------------------------------------

def validate_constraints(data: SalaryInput):

    errors = []

    # Age vs Experience
    if data.years_of_experience > (data.age - 18):
        errors.append("Years of experience cannot exceed age - 18.")

    # Education vs age
    if data.education_level == "Bachelor's" and data.age < 20:
        errors.append("Bachelor's degree holders are usually at least 20.")

    if data.education_level == "Master's" and data.age < 22:
        errors.append("Master's degree holders are usually at least 22.")

    if data.education_level == "PhD" and data.age < 25:
        errors.append("PhD holders are usually at least 25.")

    # Role rules
    if "Manager" in data.job_title and data.years_of_experience < 5:
        errors.append("Manager roles usually require ≥5 years experience.")

    if "Senior" in data.job_title and data.years_of_experience < 4:
        errors.append("Senior roles usually require ≥4 years experience.")

    if "Intern" in data.job_title and data.years_of_experience > 1:
        errors.append("Intern roles cannot have more than 1 year experience.")

    return errors


# -----------------------------------------------------
# Prediction Endpoint
# -----------------------------------------------------

@app.post("/predict")
async def predict(data: SalaryInput):

    try:

        errors = []

        # Category validation

        if data.gender not in GENDER_ENCODER.classes_:
            errors.append(
                f"Invalid gender '{data.gender}'. Allowed: {list(GENDER_ENCODER.classes_)}"
            )

        if data.education_level not in EDU_ENCODER.classes_:
            errors.append(
                f"Invalid education level '{data.education_level}'. Allowed: {list(EDU_ENCODER.classes_)}"
            )

        if data.job_title not in JOB_ENCODER.classes_:
            errors.append(
                f"Invalid job title '{data.job_title}'. Example allowed: {list(JOB_ENCODER.classes_)[:10]}"
            )

        # Real-world rules
        errors.extend(validate_constraints(data))

        if errors:
            raise HTTPException(status_code=422, detail=errors)

        # Encode inputs

        gender_encoded = int(GENDER_ENCODER.transform([data.gender])[0])
        edu_encoded = int(EDU_ENCODER.transform([data.education_level])[0])
        job_encoded = int(JOB_ENCODER.transform([data.job_title])[0])

        row = {
            "Age": float(data.age),
            "Gender": gender_encoded,
            "Education_Level": edu_encoded,
            "Job_Title": job_encoded,
            "Years_of_Experience": float(data.years_of_experience)
        }

        df = pd.DataFrame([row])

        # -----------------------------------------------------
        # Model Prediction (original dataset is US-based)
        # -----------------------------------------------------

        usd_salary = float(model.predict(df)[0])

        # Safety bounds for US salary
        usd_salary = max(30000, min(usd_salary, 200000))

        # -----------------------------------------------------
        # Adjust to Indian market
        # -----------------------------------------------------

        # Approx scaling (India salaries ~20% of US tech salaries)
        indian_salary_usd = usd_salary * 0.2

        usd_to_inr = 83

        indian_salary_inr = indian_salary_usd * usd_to_inr
        monthly_inr = indian_salary_inr / 12

        # Prediction uncertainty
        margin = indian_salary_inr * 0.15

        return {

            "predicted_salary_india_annual_inr":
                round(indian_salary_inr, 2),

            "predicted_salary_india_monthly_inr":
                round(monthly_inr, 2),

            "salary_range_inr":
                [
                    round(indian_salary_inr - margin, 2),
                    round(indian_salary_inr + margin, 2)
                ],

            "note":
                "Prediction adjusted for Indian tech market using scaling from US dataset"

        }

    except HTTPException as e:
        raise e

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------------------
# Get Valid Categories
# -----------------------------------------------------

@app.get("/valid-values")
async def valid_values():

    return {

        "genders": GENDER_ENCODER.classes_.tolist(),

        "education_levels": EDU_ENCODER.classes_.tolist(),

        "job_titles": JOB_ENCODER.classes_.tolist()
    }


# -----------------------------------------------------
# For Streamlit Compatibility
# -----------------------------------------------------

@app.get("/categories")
async def categories():

    return {

        "Gender": GENDER_ENCODER.classes_.tolist(),

        "Education_Level": EDU_ENCODER.classes_.tolist(),

        "Job_Title": JOB_ENCODER.classes_.tolist()
    }


# -----------------------------------------------------
# Health Check
# -----------------------------------------------------

@app.get("/health")
async def health():

    return {
        "status": "healthy",
        "model_loaded": model is not None
    }