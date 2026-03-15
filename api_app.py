from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import os

app = FastAPI(
    title="Salary Prediction API",
    description="Predicts salary using ML model with realistic validation",
    version="6.0.0"
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
# Normalization Functions
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

    # fallback to original format if already correct
    for known in JOB_ENCODER.classes_:
        if normalize(known) == title:
            return known

    return None


# -----------------------------------------------------
# Real World Validation Rules
# -----------------------------------------------------

def validate_constraints(data: SalaryInput):

    errors = []

    # Age vs Experience
    if data.years_of_experience > (data.age - 16):
        errors.append("Experience cannot exceed realistic working years.")

    # Education vs age (more flexible)

    if data.education_level == "Master's" and data.age < 21:
        errors.append("Master's students are usually ≥21.")

    if data.education_level == "PhD" and data.age < 23:
        errors.append("PhD candidates are usually ≥23.")

    # Role constraints

    if "Manager" in data.job_title and data.years_of_experience < 5:
        errors.append("Manager roles usually require ≥5 years experience.")

    if "Senior" in data.job_title and data.years_of_experience < 3:
        errors.append("Senior roles usually require ≥3 years experience.")

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

        # Normalize inputs
        gender = data.gender.strip()
        education = data.education_level.strip()

        job_title = normalize_job_title(data.job_title)

        if job_title is None:
            errors.append(
                f"Unknown job title '{data.job_title}'. Example: {list(JOB_ENCODER.classes_)[:8]}"
            )

        if gender not in GENDER_ENCODER.classes_:
            errors.append(
                f"Invalid gender '{gender}'. Allowed: {list(GENDER_ENCODER.classes_)}"
            )

        if education not in EDU_ENCODER.classes_:
            errors.append(
                f"Invalid education level '{education}'. Allowed: {list(EDU_ENCODER.classes_)}"
            )

        errors.extend(validate_constraints(data))

        if errors:
            raise HTTPException(status_code=422, detail=errors)

        # Encode inputs

        gender_encoded = int(GENDER_ENCODER.transform([gender])[0])
        edu_encoded = int(EDU_ENCODER.transform([education])[0])
        job_encoded = int(JOB_ENCODER.transform([job_title])[0])

        row = {
            "Age": float(data.age),
            "Gender": gender_encoded,
            "Education_Level": edu_encoded,
            "Job_Title": job_encoded,
            "Years_of_Experience": float(data.years_of_experience)
        }

        df = pd.DataFrame([row])

        # Model prediction

        usd_salary = float(model.predict(df)[0])

        # Bound unrealistic values
        usd_salary = max(30000, min(usd_salary, 200000))

        # Convert to Indian market estimate
        indian_salary_usd = usd_salary * 0.2

        usd_to_inr = 83

        indian_salary_inr = indian_salary_usd * usd_to_inr

        monthly_salary = indian_salary_inr / 12

        # Uncertainty range
        margin = indian_salary_inr * 0.15

        return {

            "predicted_salary_india_annual_inr":
                round(indian_salary_inr, 2),

            "predicted_salary_india_monthly_inr":
                round(monthly_salary, 2),

            "salary_range_inr":
                [
                    round(max(0, indian_salary_inr - margin), 2),
                    round(indian_salary_inr + margin, 2)
                ],

            "note":
                "Prediction estimated for Indian market using scaled US dataset"

        }

    except HTTPException as e:
        raise e

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------------------
# Category Endpoints
# -----------------------------------------------------

@app.get("/valid-values")
async def valid_values():

    return {

        "genders": GENDER_ENCODER.classes_.tolist(),

        "education_levels": EDU_ENCODER.classes_.tolist(),

        "job_titles": JOB_ENCODER.classes_.tolist()
    }


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
