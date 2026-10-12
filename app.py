
import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="AI Academic Advisor",
    page_icon="🎓",
    layout="centered"
)

# Load trained AI model
@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "academic_advisor_model.pkl"
    with open(model_path, "rb") as file:
        return pickle.load(file)

try:
    model = load_model()
except (FileNotFoundError, OSError, pickle.UnpicklingError) as e:
    st.error(f"Unable to load the AI model: {e}")
    st.stop()

# Application header
st.title("🎓 AI Academic Advisor System")
st.write(
    "Get personalized academic guidance using "
    "Artificial Intelligence and Machine Learning."
)

st.divider()

# Student information
st.subheader("Student Information")

student_name = st.text_input(
    "Full Name",
    placeholder="Enter your name"
)

level = st.selectbox(
    "Academic Level",
    ["100 Level", "200 Level", "300 Level", "400 Level"]
)

department = st.text_input(
    "Department",
    value="Computer Science"
)

# Academic performance
st.subheader("Academic Performance")

attendance = st.slider(
    "Class Attendance (%)",
    min_value=0,
    max_value=100,
    value=75
)

study_hours = st.slider(
    "Daily Study Hours",
    min_value=0.0,
    max_value=12.0,
    value=3.0,
    step=0.5
)

average_score = st.slider(
    "Average Academic Score (%)",
    min_value=0,
    max_value=100,
    value=65
)

st.divider()

# AI prediction
if st.button("Generate Academic Advice", type="primary"):

    if not student_name.strip():
        st.warning("Please enter your name.")

    else:
        features = pd.DataFrame([{
            "attendance": attendance,
            "study_hours": study_hours,
            "average_score": average_score
        }])

        prediction = model.predict(features)[0]

        st.subheader("Academic Assessment")

        st.write(f"Student: {student_name}")
        st.write(f"Department: {department}")
        st.write(f"Level: {level}")

        if prediction == "Good Standing":
            st.success("Academic Status: Good Standing")
            st.write(
                "Your academic indicators suggest "
                "good performance."
            )
            st.info(
                "Advice: Maintain consistent study habits, "
                "attend lectures regularly, and challenge "
                "yourself with advanced academic projects."
            )

        elif prediction == "Needs Improvement":
            st.warning("Academic Status: Needs Improvement")
            st.write(
                "Some of your academic indicators "
                "could be improved."
            )
            st.info(
                "Advice: Create a weekly study timetable, "
                "revise difficult topics, join study groups, "
                "and seek feedback from lecturers."
            )

        else:
            st.error("Academic Status: At Risk")
            st.write(
                "Your academic indicators suggest "
                "you may need additional support."
            )
            st.info(
                "Advice: Meet with an academic advisor, "
                "prioritize challenging courses, improve "
                "lecture attendance, and develop a "
                "structured study plan."
            )

        # Additional personalized recommendations
        st.subheader("Additional Recommendations")

        if attendance < 75:
            st.write(
                "• Improve your lecture attendance."
            )

        if study_hours < 3:
            st.write(
                "• Consider increasing your daily "
                "study time gradually."
            )

        if average_score < 60:
            st.write(
                "• Focus on difficult subjects and "
                "practice past examination questions."
            )

        st.caption(
            "This is a demonstration system trained "
            "on synthetic data. Predictions are advisory "
            "and are not official academic assessments."
        )
