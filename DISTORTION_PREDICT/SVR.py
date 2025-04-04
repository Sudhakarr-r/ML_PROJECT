import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data
data = pd.read_csv("max_disp_1.csv")
X = data.drop(columns=['model', 'strut', 'lattice_size']) 
y = data['max_disp']  

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SVR pipeline
svr_pipeline = Pipeline([
    ('scaler', StandardScaler()),  
    ('regressor', SVR(kernel='rbf', C=1.0, epsilon=0.1))  
])

# Fit and predict
svr_pipeline.fit(X_train, y_train)
y_pred = svr_pipeline.predict(X_test)

# Evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("SVR Model Performance:")
print(f"Mean Absolute Error (MAE): {mae:.6f}")
print(f"Mean Squared Error (MSE): {mse:.6f}")
print(f"R^2 Score: {r2:.6f}")

# ===================
# Plot 1: Actual vs Predicted
# ===================
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Ideal line
plt.xlabel("Actual Max Displacement")
plt.ylabel("Predicted Max Displacement")
plt.title("Actual vs Predicted Max Displacement")
plt.grid(True)
plt.tight_layout()
plt.show()

# ===================
# Plot 2: Residuals Plot
# ===================
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted Max Displacement")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted")
plt.grid(True)
plt.tight_layout()
plt.show()

# ===================
# Plot 3: Error Distribution
# ===================
plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True, bins=30)
plt.xlabel("Prediction Error (Residuals)")
plt.title("Distribution of Prediction Errors")
plt.grid(True)
plt.tight_layout()
plt.show()
