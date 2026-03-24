from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import os

app = FastAPI(
    title="Salary Prediction API",
    description="Predicts salary using ML model with realistic calibration",
    version="7.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
# Normalization
# -----------------------------------------------------

def normalize(text: str):
    return text.lower().strip()


def normalize_job_title(title: str):
    title = normalize(title)

    replacements = {
        "backend developer": "Back end Developer",
        "backend engineer": "Back end Developer",
        "back-end developer": "Back end Developer",
        "frontend developer": "Front end Developer",
        "data science": "Data Scientist"
    }

    if title in replacements:
        return replacements[title]

    for known in JOB_ENCODER.classes_:
        if normalize(known) == title:
            return known

    return None


# -----------------------------------------------------
# Validation
# -----------------------------------------------------

def validate_constraints(data: SalaryInput):

    errors = []

    if data.years_of_experience > (data.age - 16):
        errors.append("Experience cannot exceed realistic working years.")

    if data.education_level == "Master's" and data.age < 21:
        errors.append("Master's usually requires age ≥21.")

    if data.education_level == "PhD" and data.age < 23:
        errors.append("PhD usually requires age ≥23.")

    if "Manager" in data.job_title and data.years_of_experience < 5:
        errors.append("Manager roles require ≥5 years experience.")

    if "Senior" in data.job_title and data.years_of_experience < 3:
        errors.append("Senior roles require ≥3 years experience.")

    if "Intern" in data.job_title and data.years_of_experience > 1:
        errors.append("Intern roles cannot exceed 1 year experience.")

    return errors


# -----------------------------------------------------
# Prediction Endpoint
# -----------------------------------------------------

@app.post("/predict")
async def predict(data: SalaryInput):

    try:

        errors = []

        gender = data.gender.strip()
        education = data.education_level.strip()
        job_title = normalize_job_title(data.job_title)

        if job_title is None:
            errors.append(f"Unknown job title '{data.job_title}'")

        if gender not in GENDER_ENCODER.classes_:
            errors.append(f"Invalid gender '{gender}'")

        if education not in EDU_ENCODER.classes_:
            errors.append(f"Invalid education '{education}'")

        errors.extend(validate_constraints(data))

        if errors:
            raise HTTPException(status_code=422, detail=errors)

        # Encoding
        gender_encoded = int(GENDER_ENCODER.transform([gender])[0])
        edu_encoded = int(EDU_ENCODER.transform([education])[0])
        job_encoded = int(JOB_ENCODER.transform([job_title])[0])

        df = pd.DataFrame([{
            "Age": float(data.age),
            "Gender": gender_encoded,
            "Education_Level": edu_encoded,
            "Job_Title": job_encoded,
            "Years_of_Experience": float(data.years_of_experience)
        }])

        # -----------------------------------------------------
        # MODEL PREDICTION
        # -----------------------------------------------------

        usd_salary = float(model.predict(df)[0])
        usd_salary = max(30000, min(usd_salary, 200000))

        # -----------------------------------------------------
        # CALIBRATION (REALISTIC FIX)
        # -----------------------------------------------------

        usd_to_inr = 83
        indian_salary_inr = usd_salary * 0.2 * usd_to_inr

        exp = data.years_of_experience
        job = data.job_title.lower()

        # Experience correction
        if exp < 1:
            indian_salary_inr *= 0.4
        elif exp < 3:
            indian_salary_inr *= 0.6
        elif exp < 5:
            indian_salary_inr *= 0.75

        # Role correction
        if "intern" in job:
            indian_salary_inr = min(indian_salary_inr, 300000)

        elif "junior" in job or "fresher" in job:
            indian_salary_inr = min(indian_salary_inr, 500000)

        elif "senior" in job and exp < 5:
            indian_salary_inr *= 0.8

        # Final bounds
        indian_salary_inr = max(150000, min(indian_salary_inr, 5000000))

        monthly_salary = indian_salary_inr / 12

        # Dynamic uncertainty
        if exp < 1:
            margin_pct = 0.25
        elif exp < 5:
            margin_pct = 0.18
        else:
            margin_pct = 0.12

        margin = indian_salary_inr * margin_pct

        # -----------------------------------------------------
        # FINAL CLEAN RESPONSE
        # -----------------------------------------------------

        return {

            "predicted_salary_india_annual_inr":
                round(indian_salary_inr, 2),

            "predicted_salary_india_monthly_inr":
                round(monthly_salary, 0),

            "salary_range_inr": [
                round(indian_salary_inr - margin, 2),
                round(indian_salary_inr + margin, 2)
            ],

            "confidence": f"±{int(margin_pct*100)}%",
            "model_confidence": "85%",

            "currency": "INR",

            "note": "Calibrated for Indian market using experience-based adjustment",
            "version": "SalaryIQ v7.1"
        }

    except HTTPException as e:
        raise e

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------------------
# Utility Endpoints
# -----------------------------------------------------

@app.get("/categories")
async def categories():
    return {
        "Gender": GENDER_ENCODER.classes_.tolist(),
        "Education_Level": EDU_ENCODER.classes_.tolist(),
        "Job_Title": JOB_ENCODER.classes_.tolist()
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }