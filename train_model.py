import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Create models folder if not exists
os.makedirs("models", exist_ok=True)

# Load Kaggle dataset
df = pd.read_csv("data/Student_Performance.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Encode categorical column
encoder = LabelEncoder()
df["Extracurricular Activities"] = encoder.fit_transform(
    df["Extracurricular Activities"]
)

# Features and target
X = df.drop("Performance Index", axis=1)
y = df["Performance Index"]

# Save feature columns
joblib.dump(list(X.columns), "models/feature_columns.pkl")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Student Performance Prediction Model")
print("------------------------------------")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R2 Score: {r2:.2f}")

# Save model and encoder
joblib.dump(model, "models/student_performance_model.pkl")
joblib.dump(encoder, "models/activity_encoder.pkl")

print("\nModel saved successfully in models folder.")