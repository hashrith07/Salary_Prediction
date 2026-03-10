import streamlit as st
import requests

# ────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────

# Change this URL when you deploy the API
API_URL = "http://127.0.0.1:8000/predict"          # local development
# API_URL = "https://your-api-name.onrender.com/predict"   # when deployed

st.set_page_config(
    page_title="Salary Predictor",
    page_icon="💰",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ────────────────────────────────────────────────
# TITLE & DESCRIPTION
# ────────────────────────────────────────────────

st.title("💰 Salary Prediction")
st.markdown("Estimate your salary based on experience, education, role and gender")

# ────────────────────────────────────────────────
# INPUT FORM
# ────────────────────────────────────────────────

with st.form("salary_form"):
    col1, col2 = st.columns(2)

    with col1:
        experience = st.number_input(
            "Years of Experience",
            min_value=0.0,
            max_value=40.0,
            value=3.0,
            step=0.5,
            format="%.1f",
            help="Total professional work experience in years"
        )

    with col2:
        gender = st.selectbox(
            "Gender",
            options=["Male", "Female", "Other"],
            index=0
        )

    education = st.selectbox(
        "Education Level",
        options=[
            "High School",
            "Diploma",
            "Bachelor's",
            "Master's",
            "PhD"
        ],
        index=2
    )

    job_title = st.text_input(
        "Job Title / Role",
        value="Software Engineer",
        max_chars=80,
        help="Be specific (e.g. Data Analyst, Senior Developer, Marketing Manager)"
    )

    submitted = st.form_submit_button("Predict Salary", type="primary", use_container_width=True)

# ────────────────────────────────────────────────
# PREDICTION
# ────────────────────────────────────────────────

if submitted:
    with st.spinner("Getting prediction..."):
        payload = {
            "years_of_experience": experience,
            "education_level": education,
            "job_title": job_title.strip(),
            "gender": gender
        }

        try:
            response = requests.post(API_URL, json=payload, timeout=12)

            if response.status_code == 200:
                result = response.json()

                st.success("Prediction successful!")

                col_a, col_b = st.columns([3, 2])

                with col_a:
                    st.metric(
                        "Estimated Salary",
                        f"₹{result['predicted_salary']:,.2f}",
                    )

                with col_b:
                    st.markdown("**Salary Range**")
                    st.markdown(f"**{result['salary_range']}**")

                if "currency" in result:
                    st.caption(f"Currency: {result['currency']}")

            else:
                try:
                    error_msg = response.json().get("error", "Unknown error")
                    st.error(f"API returned error ({response.status_code}): {error_msg}")
                except:
                    st.error(f"API error ({response.status_code}): {response.text[:200]}...")

        except requests.exceptions.RequestException as e:
            st.error("Could not connect to the prediction API")
            st.info(str(e))
            st.caption("Is your API server running? (uvicorn ... --reload)")

# ────────────────────────────────────────────────
# FOOTER / INFO
# ────────────────────────────────────────────────

st.markdown("---")
st.caption("Model trained on historical salary data • Predictions are estimates only")
st.caption("API endpoint: " + API_URL)