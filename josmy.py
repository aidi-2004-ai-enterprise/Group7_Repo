import xgboost as xgb
import shap
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
import warnings

warnings.filterwarnings("ignore")

def compute_shap_interactions(X, y, model):
    """Compute SHAP interaction values for feature pairs."""
    explainer = shap.TreeExplainer(model)
    shap_interactions = explainer.shap_interaction_values(X)
    
    # Average absolute interaction values across samples
    interaction_scores = np.abs(shap_interactions).mean(axis=0)
    
    # Zero out diagonal (self-interactions)
    np.fill_diagonal(interaction_scores, 0)
    
    return interaction_scores

def top_k_interactions(interaction_scores, top_k):
    """Extract top-k feature pairs based on SHAP interaction scores."""
    n = interaction_scores.shape[0]
    indices = [(i, j) for i in range(n) for j in range(i + 1, n)]
    scores = [(i, j, interaction_scores[i, j]) for i, j in indices]
    scores.sort(key=lambda x: -x[2])  # descending
    return [list(pair[:2]) for pair in scores[:top_k]]

# Load dataset
data = load_boston()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initial XGBoost training (for SHAP)
dtrain = xgb.DMatrix(X_train, label=y_train)
params = {
    "objective": "reg:squarederror",
    "max_depth": 4,
    "learning_rate": 0.1,
    "verbosity": 0
}
model = xgb.train(params, dtrain, num_boost_round=50)

# Compute SHAP interaction scores
interaction_scores = compute_shap_interactions(X_train, y_train, model)

# Select top-k interactions
top_interactions = top_k_interactions(interaction_scores, top_k=3)

# Convert to interaction_constraints format
interaction_constraints = str(top_interactions)

# Final model training with interaction constraints
dtrain_final = xgb.DMatrix(X_train, label=y_train, feature_names=data.feature_names)
dtest_final = xgb.DMatrix(X_test, label=y_test, feature_names=data.feature_names)

final_params = params.copy()
final_params["interaction_constraints"] = interaction_constraints

final_model = xgb.train(final_params, dtrain_final, num_boost_round=100)
preds = final_model.predict(dtest_final)

# Evaluation
rmse = np.sqrt(np.mean((preds - y_test) ** 2))
print(f"✅ RMSE with SHAP-guided interaction constraints: {rmse:.4f}")
print(f"✅ Top interactions used: {top_interactions}")
