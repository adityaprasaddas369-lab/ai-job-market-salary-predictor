
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="AI Salary Predictor",
    page_icon="📊",
    layout="wide",
)

@st.cache_resource
def load_model():
    return joblib.load("models/salary_model.joblib")

st.title("AI Job Market & Salary Predictor")
st.write(
    "Explore estimated salaries for technology roles. "
    "This project currently uses synthetic demonstration "
    "data, not verified UK salary statistics."
)

try:
    model = load_model()
except FileNotFoundError:
    st.error(
        "Model file not found. Run train_model.py first "
        "to generate the model."
    )
    st.stop()

role = st.selectbox(
    "Job role",
    [
        "Data Scientist",
        "Machine Learning Engineer",
        "Data Analyst",
        "AI Engineer",
        "Software Engineer",
    ],
)

location = st.selectbox(
    "Location",
    ["London", "Bristol", "Manchester", "Leeds"],
)

education = st.selectbox(
    "Education",
    ["Bachelors", "Masters", "PhD"],
)

experience = st.slider(
    "Years of experience", 0, 15, 1
)

skills = st.slider(
    "Number of relevant skills", 2, 12, 5
)

if st.button("Predict salary"):
    features = pd.DataFrame([{
        "job_role": role,
        "location": location,
        "education": education,
        "experience_years": experience,
        "skills_count": skills,
    }])

    prediction = model.predict(features)[0]

    st.metric(
        "Estimated annual salary",
        f"£{prediction:,.0f}",
    )

    st.caption(
        "Illustrative estimate from a synthetic dataset. "
        "Not a real-world salary benchmark."
    )
  
