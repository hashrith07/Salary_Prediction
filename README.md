# 💼 Salary Prediction System

A Machine Learning powered web application that predicts employee salaries based on attributes such as age, education level, job title, gender, and years of experience.

This project demonstrates the integration of **Machine Learning, FastAPI, and Streamlit** to build a complete ML-powered web application.

---

## 🚀 Features

- Machine Learning model trained for salary prediction
- FastAPI backend for real-time predictions
- Interactive Streamlit frontend
- Data validation and realistic prediction constraints
- REST API integration between frontend and backend

---

## 🧠 Machine Learning Pipeline

1. Data preprocessing and cleaning
2. Feature encoding for categorical variables
3. Model training using Scikit-Learn
4. Model serialization using Joblib
5. API integration for predictions

---

## 🛠 Tech Stack

- Python
- Scikit-Learn
- FastAPI
- Streamlit
- Pandas
- NumPy
- Joblib

---

## 📂 Project Structure

```
Salary_Prediction
│
├── model/
│   ├── salary_model.pkl
│   └── encoders.pkl
│
├── api/
│   └── main.py
│
├── frontend/
│   └── streamlit_app.py
│
└── notebooks/
    └── salary_prediction.ipynb
```

---

## ▶️ Running the Project

### 1️⃣ Start FastAPI backend

```bash
uvicorn main:app --reload
```

API will run at:

```
http://127.0.0.1:8000
```

---

### 2️⃣ Start Streamlit frontend

```bash
streamlit run streamlit_app.py
```

App will open at:

```
http://localhost:8501
```

---

## 📊 Example Prediction

Input

- Age: 25  
- Education: Master's  
- Job Title: Software Engineer  
- Experience: 3 years  

Output

```
Predicted Salary Range:
₹7,00,000 – ₹9,00,000 per year
```

---

## 📌 Future Improvements

- Deploy application using Docker
- Add model comparison (RandomForest vs XGBoost)
- Add salary analytics dashboard
- Integrate real-world salary datasets

---

## 👨‍💻 Author

Hashrith Naga

GitHub: https://github.com/hashrith07
