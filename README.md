# 🎓 Student Performance Prediction System

A beginner-friendly Machine Learning web application that predicts a student's **Performance Index** using academic and lifestyle-related factors such as study hours, previous scores, sleep hours, extracurricular activities, and sample question paper practice.

---

## 📌 Project Overview

The **Student Performance Prediction System** is a simple yet useful Machine Learning project built with Python.

It uses a Kaggle dataset to predict how well a student may perform based on different input factors.

This project is designed to show:

- Data preprocessing
- Machine Learning model training
- Regression-based prediction
- Feature importance analysis
- Streamlit web app development
- Personalized recommendations

---

## 🎯 Objective

The main goal of this project is to predict a student's **Performance Index** and help understand which factors affect academic performance the most.

The system also provides basic improvement suggestions based on the entered student details.

---

## 📂 Dataset

Dataset used:

**Student Performance Dataset from Kaggle**

The dataset contains student-related information such as:

| Column Name | Description |
|---|---|
| Hours Studied | Number of hours studied by the student |
| Previous Scores | Marks scored in previous exams |
| Extracurricular Activities | Whether the student participates in activities |
| Sleep Hours | Average sleep hours per day |
| Sample Question Papers Practiced | Number of sample papers practiced |
| Performance Index | Final performance score of the student |

---

## 🛠️ Tech Stack

| Technology | Use |
|---|---|
| Python | Main programming language |
| Pandas | Data loading and preprocessing |
| Scikit-learn | Model training and evaluation |
| Matplotlib | Feature importance visualization |
| Streamlit | Web app interface |
| Joblib | Saving and loading ML model |

---

## 📁 Project Structure


student-performance-prediction/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
│
├── data/
│   └── Student_Performance.csv
│
└── models/
    ├── student_performance_model.pkl
    ├── activity_encoder.pkl
    └── feature_columns.pkl




⚙️ How the Project Works
1️⃣ Load Dataset

The dataset is loaded from:

data/Student_Performance.csv
2️⃣ Data Preprocessing

The categorical column:

Extracurricular Activities

is converted into numerical form using Label Encoding.

Example:

Yes → 1
No → 0
3️⃣ Model Training

A Random Forest Regressor model is trained to predict the student's Performance Index.

4️⃣ Model Evaluation

The model is evaluated using:

Mean Absolute Error
Mean Squared Error
R² Score
5️⃣ Web App Prediction

The trained model is loaded into a Streamlit app where users can enter student details and get predictions.

🚀 Features
Predicts student performance index
Shows pass/fail result based on predicted score
Provides personalized improvement recommendations
Displays feature importance analysis
Shows dataset preview
Simple and beginner-friendly interface
Easy to run locally
🧠 Input Features

The app takes the following inputs:

Hours Studied
Previous Scores
Extracurricular Activities
Sleep Hours
Sample Question Papers Practiced
📊 Output

The app gives:

Predicted Performance Index
Pass/Fail result
Personalized recommendations
Feature importance chart
📈 Feature Importance

The project includes feature importance analysis to show which factors have the highest impact on student performance.

This helps understand whether performance is more affected by:

Previous scores
Study hours
Sleep hours
Practice papers
Extracurricular activities
💡 Recommendation System

The app gives simple recommendations such as:

Increase study hours
Improve sleep routine
Practice more sample papers
Focus on weak subjects
Maintain consistency
