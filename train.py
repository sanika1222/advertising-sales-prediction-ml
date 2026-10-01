# train.py
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# 1. Load Data
print("📦 Loading dataset...")
df = pd.read_csv("data/advertising_data.csv")

# 2. Data Preprocessing
print("🧹 Preprocessing data...")
df.drop_duplicates(inplace=True)
df.fillna(df.mean(), inplace=True)  # Handle any missing values

# Separate features and target
X = df[['TV', 'Radio', 'Newspaper', 'Digital', 'Social_Media']]
y = df['Sales']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling (Crucial for clean linear regression patterns)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Model Building
print("🚀 Training Linear Regression Model...")
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Generate predictions
y_pred = model.predict(X_test_scaled)

# 4. Model Evaluation
print("\n📊 Model Evaluation Metrics:")
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = metrics.r2_score(y_test, y_pred)

print(f"   • Mean Absolute Error (MAE): {mae:.4f}")
print(f"   • Mean Squared Error (MSE): {mse:.4f}")
print(f"   • Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"   • R² Score (Coefficient of Determination): {r2:.4f}")

# Display feature impact coefficients
print("\n🎯 Feature Coefficients (Impact Analysis):")
for feature, coef in zip(X.columns, model.coef_):
    print(f"   • {feature}: {coef:.4f}")

# 5. Save Model and Preprocessing Components
print("\n💾 Saving models to artifacts/ directory...")
os.makedirs("artifacts", exist_ok=True)

with open("artifacts/sales_model.pkl", "wb") as m_file:
    pickle.dump(model, m_file)

with open("artifacts/scaler.pkl", "wb") as s_file:
    pickle.dump(scaler, s_file)

print("✅ Training complete and models successfully stored!")
