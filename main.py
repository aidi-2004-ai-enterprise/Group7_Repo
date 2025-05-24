import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Load example dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# -----------------------------
# Person A: Split the dataset
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("✅ Dataset split complete.")

# -----------------------------
# Person B: Create XGBoost model
# -----------------------------
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
print("✅ XGBoost model created.")

# -----------------------------
# Person C: Fit the model
# -----------------------------
model.fit(X_train, y_train)
print("✅ Model training complete.")

# Optional: evaluate model (anyone can add)
accuracy = model.score(X_test, y_test)
print(f"🎯 Accuracy: {accuracy:.2%}")
