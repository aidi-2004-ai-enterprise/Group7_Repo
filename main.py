import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_score, classification_report, confusion_matrix, precision_recall_curve
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt

# 1. Load and Prepare the Dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# 2. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. XGBoost Classifier with GridSearchCV to maximize precision
param_grid = {
    'n_estimators': [100],
    'max_depth': [3],
    'learning_rate': [0.1],
    'subsample': [1.0],
    'colsample_bytree': [1.0]
}

xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid,
                           scoring='precision', cv=3, verbose=0, n_jobs=-1)

grid_search.fit(X_train_scaled, y_train)
best_model = grid_search.best_estimator_

# 5. Get predicted probabilities and tune threshold
y_proba = best_model.predict_proba(X_test_scaled)[:, 1]

# Compute precision-recall curve
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

# Find the threshold that gives the highest precision
optimal_idx = np.argmax(precisions)
optimal_threshold = thresholds[optimal_idx]
print(f"\n🔍 Best Threshold for Max Precision: {optimal_threshold:.3f}")
print(f"Precision at this threshold: {precisions[optimal_idx]:.3f}")

# 6. Use custom threshold to make predictions
y_pred_custom = (y_proba >= optimal_threshold).astype(int)

# 7. Evaluation
print("\nClassification Report:\n", classification_report(y_test, y_pred_custom))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_custom))

# 8. Optional: Plot Precision-Recall Curve
plt.figure(figsize=(8, 6))
plt.plot(thresholds, precisions[:-1], label="Precision")
plt.plot(thresholds, recalls[:-1], label="Recall")
plt.axvline(optimal_threshold, color='red', linestyle='--', label=f'Optimal Threshold = {optimal_threshold:.2f}')
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision and Recall vs Threshold")
plt.legend()
plt.grid(True)
plt.show()
