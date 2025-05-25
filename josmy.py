import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression

def discover_interactions(X, y, top_k=5):
    """Discover top K feature interactions using mutual information."""
    interactions = []
    n_features = X.shape[1]
    for i in range(n_features):
        for j in range(i + 1, n_features):
            # Generate interaction feature
            interaction = X[:, i] * X[:, j]
            mi = mutual_info_regression(interaction.reshape(-1, 1), y)
            interactions.append(((i, j), mi[0]))
    
    # Sort and select top interactions
    interactions.sort(key=lambda x: -x[1])
    top_interactions = [pair for pair, _ in interactions[:top_k]]
    return top_interactions

def convert_to_constraint_format(interactions):
    """Convert pairwise interactions to XGBoost interaction constraint format."""
    # Each interaction set must be a list of feature indices
    return [list(pair) for pair in interactions]

# Load dataset
data = load_boston()
X = data.data
y = data.target
feature_names = data.feature_names

# Discover feature interactions
interactions = discover_interactions(X, y, top_k=3)
constraint_format = convert_to_constraint_format(interactions)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Convert to DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)

# Train with interaction constraints
params = {
    "objective": "reg:squarederror",
    "interaction_constraints": str(constraint_format),
    "max_depth": 4,
    "learning_rate": 0.1,
    "verbosity": 1
}

booster = xgb.train(params, dtrain, num_boost_round=100)

# Evaluate
preds = booster.predict(dtest)
rmse = np.sqrt(np.mean((preds - y_test) ** 2))
print(f"RMSE with adaptive interaction constraints: {rmse:.4f}")
