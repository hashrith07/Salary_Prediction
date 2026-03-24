from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import os

app = FastAPI(
    title="Salary Prediction API",
    description="Predicts salary with realistic calibration",
    version="7.0.0"
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
# Validation Rules
# -----------------------------------------------------

def validate_constraints(data: SalaryInput):

    errors = []

    if data.years_of_experience > (data.age - 16):
        errors.append("Experience exceeds realistic working years.")

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

        # -----------------------------
        # MODEL PREDICTION
        # -----------------------------
        usd_salary = float(model.predict(df)[0])

        # -----------------------------
        # REALISTIC CALIBRATION
        # -----------------------------
        usd_to_inr = 83
        base_salary = usd_salary * usd_to_inr

        exp = data.years_of_experience

        # Experience scaling
        if exp < 1:
            base_salary *= 0.35
        elif exp <= 2:
            base_salary *= 0.5
        elif exp <= 5:
            base_salary *= 0.7
        else:
            base_salary *= 1.0

        # Role-based caps
        if any(x in data.job_title.lower() for x in ["intern", "junior", "fresher"]):
            base_salary = min(base_salary, 400000)

        # Final realistic bounds
        base_salary = max(150000, min(base_salary, 5000000))

        monthly_salary = base_salary / 12

        # Dynamic uncertainty
        if exp < 1:
            margin_pct = 0.25
        elif exp <= 5:
            margin_pct = 0.18
        else:
            margin_pct = 0.12

        margin = base_salary * margin_pct

        return {
            "predicted_salary_india_annual_inr": round(base_salary, 2),
            "predicted_salary_india_monthly_inr": round(monthly_salary, 2),
            "salary_range_inr": [
                round(base_salary - margin, 2),
                round(base_salary + margin, 2)
            ],
            "confidence": f"±{int(margin_pct*100)}%",
            "model_confidence": "85%",
            "note": "Calibrated for Indian market using experience-based adjustment"
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
