import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# Page config
st.set_page_config(
    page_title="Student Performance Prediction System",
    page_icon="🎓",
    layout="wide"
)

# Load model files
model = joblib.load("models/student_performance_model.pkl")
encoder = joblib.load("models/activity_encoder.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")


# App title
st.title("🎓 Student Performance Prediction System")

st.write(
    "This machine learning app predicts a student's performance index "
    "based on study hours, previous scores, sleep hours, extracurricular activity, "
    "and sample question paper practice."
)


# Sidebar
st.sidebar.header("Enter Student Details")

hours_studied = st.sidebar.slider(
    "Hours Studied",
    min_value=1,
    max_value=10,
    value=5
)

previous_scores = st.sidebar.slider(
    "Previous Scores",
    min_value=0,
    max_value=100,
    value=70
)

extracurricular = st.sidebar.selectbox(
    "Extracurricular Activities",
    ["Yes", "No"]
)

sleep_hours = st.sidebar.slider(
    "Sleep Hours",
    min_value=3,
    max_value=10,
    value=7
)

sample_papers = st.sidebar.slider(
    "Sample Question Papers Practiced",
    min_value=0,
    max_value=10,
    value=5
)


# Convert extracurricular activity
extra_encoded = encoder.transform([extracurricular])[0]


# Input dataframe
input_data = pd.DataFrame({
    "Hours Studied": [hours_studied],
    "Previous Scores": [previous_scores],
    "Extracurricular Activities": [extra_encoded],
    "Sleep Hours": [sleep_hours],
    "Sample Question Papers Practiced": [sample_papers]
})

input_data = input_data[feature_columns]


# Prediction
if st.sidebar.button("Predict Performance"):
    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Predicted Performance Index",
            value=f"{prediction:.2f}/100"
        )

    with col2:
        if prediction >= 60:
            st.success("Predicted Result: Pass")
        else:
            st.error("Predicted Result: Fail")

    # Recommendation section
    st.subheader("Personalized Recommendations")

    recommendations = []

    if hours_studied < 5:
        recommendations.append("Increase study hours to at least 5-6 hours daily.")

    if previous_scores < 60:
        recommendations.append("Focus on improving weak subjects from previous exams.")

    if sleep_hours < 6:
        recommendations.append("Improve sleep routine for better concentration.")

    if sample_papers < 5:
        recommendations.append("Practice more sample question papers before exams.")

    if extracurricular == "No":
        recommendations.append(
            "Try participating in extracurricular activities for balanced development."
        )

    if recommendations:
        for rec in recommendations:
            st.write("- " + rec)
    else:
        st.success("Good academic habits. Keep maintaining consistency.")


# Feature importance
st.subheader("Feature Importance Analysis")

importance = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

st.dataframe(importance_df, use_container_width=True)

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(importance_df["Feature"], importance_df["Importance"])
ax.set_xlabel("Importance Score")
ax.set_ylabel("Features")
ax.set_title("Factors Affecting Student Performance")
ax.invert_yaxis()

st.pyplot(fig)


# Dataset preview
st.subheader("Dataset Preview")

try:
    df = pd.read_csv("data/Student_Performance.csv")
    st.dataframe(df.head(20), use_container_width=True)
except FileNotFoundError:
    st.warning("Dataset file not found. Please add Student_Performance.csv inside data folder.")