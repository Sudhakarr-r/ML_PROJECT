import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

# Load dataset
data = pd.read_csv("max_disp_1.csv")

# Feature and target selection
X = data.drop(columns=['model', 'strut', 'lattice_size'])  
y = data['max_disp']

# Split into train and test sets
X_train = X.iloc[:559].copy()
y_train = y.iloc[:559].copy()
X_test = X.iloc[559:].copy()
y_test = y.iloc[559:].copy()

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Set up 5-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# SVR Model
svr_pipeline = Pipeline([
    ('scaler', StandardScaler()),  
    ('regressor', SVR(kernel='rbf', C=1.0, epsilon=0.1))  
])
svr_pipeline.fit(X_train_scaled, y_train)
y_pred_svr = svr_pipeline.predict(X_test_scaled)

print("SVR Model Performance on Test Set:")
print(f"MAE: {mean_absolute_error(y_test, y_pred_svr):.6f}")
print(f"MSE: {mean_squared_error(y_test, y_pred_svr):.6f}")
print(f"R^2: {r2_score(y_test, y_pred_svr):.6f}")

cv_mae_svr = -cross_val_score(svr_pipeline, X_train_scaled, y_train, cv=kf, scoring='neg_mean_absolute_error')
cv_mse_svr = -cross_val_score(svr_pipeline, X_train_scaled, y_train, cv=kf, scoring='neg_mean_squared_error')
cv_r2_svr = cross_val_score(svr_pipeline, X_train_scaled, y_train, cv=kf, scoring='r2')

print("\nSVR Model 5-Fold Cross-Validation:")
print(f"Mean MAE: {cv_mae_svr.mean():.6f} ± {cv_mae_svr.std():.6f}")
print(f"Mean MSE: {cv_mse_svr.mean():.6f} ± {cv_mse_svr.std():.6f}")
print(f"Mean R²: {cv_r2_svr.mean():.6f} ± {cv_r2_svr.std():.6f}")

# ANN Model
ann_model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
ann_model.fit(X_train_scaled, y_train)
y_pred_ann = ann_model.predict(X_test_scaled)

print("\nANN Model Performance on Test Set:")
print(f"MAE: {mean_absolute_error(y_test, y_pred_ann):.6f}")
print(f"MSE: {mean_squared_error(y_test, y_pred_ann):.6f}")
print(f"R^2: {r2_score(y_test, y_pred_ann):.6f}")

cv_mae_ann = -cross_val_score(ann_model, X_train_scaled, y_train, cv=kf, scoring='neg_mean_absolute_error')
cv_mse_ann = -cross_val_score(ann_model, X_train_scaled, y_train, cv=kf, scoring='neg_mean_squared_error')
cv_r2_ann = cross_val_score(ann_model, X_train_scaled, y_train, cv=kf, scoring='r2')

print("\nANN Model 5-Fold Cross-Validation:")
print(f"Mean MAE: {cv_mae_ann.mean():.6f} ± {cv_mae_ann.std():.6f}")
print(f"Mean MSE: {cv_mse_ann.mean():.6f} ± {cv_mse_ann.std():.6f}")
print(f"Mean R²: {cv_r2_ann.mean():.6f} ± {cv_r2_ann.std():.6f}")

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("\nRandom Forest Model Performance on Test Set:")
print(f"MAE: {mean_absolute_error(y_test, y_pred_rf):.6f}")
print(f"MSE: {mean_squared_error(y_test, y_pred_rf):.6f}")
print(f"R^2: {r2_score(y_test, y_pred_rf):.6f}")

cv_mae_rf = -cross_val_score(rf_model, X_train, y_train, cv=kf, scoring='neg_mean_absolute_error')
cv_mse_rf = -cross_val_score(rf_model, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')
cv_r2_rf = cross_val_score(rf_model, X_train, y_train, cv=kf, scoring='r2')

print("\nRandom Forest Model 5-Fold Cross-Validation:")
print(f"Mean MAE: {cv_mae_rf.mean():.6f} ± {cv_mae_rf.std():.6f}")
print(f"Mean MSE: {cv_mse_rf.mean():.6f} ± {cv_mse_rf.std():.6f}")
print(f"Mean R²: {cv_r2_rf.mean():.6f} ± {cv_r2_rf.std():.6f}")

# XGBoost Model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

print("\nXGBoost Model Performance on Test Set:")
print(f"MAE: {mean_absolute_error(y_test, y_pred_xgb):.6f}")
print(f"MSE: {mean_squared_error(y_test, y_pred_xgb):.6f}")
print(f"R^2: {r2_score(y_test, y_pred_xgb):.6f}")

cv_mae_xgb = -cross_val_score(xgb_model, X_train, y_train, cv=kf, scoring='neg_mean_absolute_error')
cv_mse_xgb = -cross_val_score(xgb_model, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')
cv_r2_xgb = cross_val_score(xgb_model, X_train, y_train, cv=kf, scoring='r2')

print("\nXGBoost Model 5-Fold Cross-Validation:")
print(f"Mean MAE: {cv_mae_xgb.mean():.6f} ± {cv_mae_xgb.std():.6f}")
print(f"Mean MSE: {cv_mse_xgb.mean():.6f} ± {cv_mse_xgb.std():.6f}")
print(f"Mean R²: {cv_r2_xgb.mean():.6f} ± {cv_r2_xgb.std():.6f}")

# ---------- Visualization Part ----------

# Dictionary for predictions
model_preds = {
    'SVR': y_pred_svr,
    'ANN': y_pred_ann,
    'Random Forest': y_pred_rf,
    'XGBoost': y_pred_xgb
}

# Actual vs Predicted plots
plt.figure(figsize=(14, 10))
for i, (model_name, y_pred) in enumerate(model_preds.items()):
    plt.subplot(2, 2, i + 1)
    plt.scatter(y_test, y_pred, alpha=0.6, color='teal')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{model_name} - Actual vs Predicted")
    plt.grid(True)

plt.tight_layout()
plt.show()

# Collect performance metrics
mae_scores = [mean_absolute_error(y_test, y_pred_svr),
              mean_absolute_error(y_test, y_pred_ann),
              mean_absolute_error(y_test, y_pred_rf),
              mean_absolute_error(y_test, y_pred_xgb)]

mse_scores = [mean_squared_error(y_test, y_pred_svr),
              mean_squared_error(y_test, y_pred_ann),
              mean_squared_error(y_test, y_pred_rf),
              mean_squared_error(y_test, y_pred_xgb)]

r2_scores = [r2_score(y_test, y_pred_svr),
             r2_score(y_test, y_pred_ann),
             r2_score(y_test, y_pred_rf),
             r2_score(y_test, y_pred_xgb)]

model_names = ['SVR', 'ANN', 'Random Forest', 'XGBoost']

# Bar charts for MAE, MSE, R²
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

axs[0].bar(model_names, mae_scores, color='skyblue')
axs[0].set_title('Mean Absolute Error')
axs[0].set_ylabel('MAE')

axs[1].bar(model_names, mse_scores, color='salmon')
axs[1].set_title('Mean Squared Error')
axs[1].set_ylabel('MSE')

axs[2].bar(model_names, r2_scores, color='lightgreen')
axs[2].set_title('R² Score')
axs[2].set_ylabel('R²')

for ax in axs:
    ax.set_ylim(bottom=0)
    ax.grid(axis='y')

plt.suptitle("Model Performance Comparison", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
