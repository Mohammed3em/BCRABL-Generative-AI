"""
===============================================================================
Project : BCRABL-AI
Script  : 26_train_best_xgboost.py

Purpose
-------
Train the final optimized XGBoost model using the best Optuna parameters.

===============================================================================
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBRegressor

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

from config import RANDOM_STATE
from data_loader import load_data

# =============================================================================
# Paths
# =============================================================================

PARAM_FILE = Path(
    "07_Results/XGBoost_Optuna/best_parameters.json"
)

RESULT_DIR = Path(
    "07_Results/XGBoost_Final"
)

MODEL_DIR = Path(
    "04_AI_Models/Models"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# =============================================================================
# Load Parameters
# =============================================================================

with open(PARAM_FILE, "r", encoding="utf-8") as f:

    params = json.load(f)

params["objective"] = "reg:squarederror"
params["random_state"] = RANDOM_STATE
params["verbosity"] = 0
params["n_jobs"] = -1

# =============================================================================
# Load Dataset
# =============================================================================

(
    X_train,
    y_train,
    X_valid,
    y_valid,
    X_test,
    y_test
) = load_data()

print("=" * 80)
print("FINAL XGBOOST MODEL")
print("=" * 80)

print("\nBest Hyperparameters\n")

for k, v in params.items():

    print(f"{k:<22}: {v}")

# =============================================================================
# Train Model
# =============================================================================

model = XGBRegressor(

    **params

)

print("\nTraining XGBoost...\n")

model.fit(

    X_train,

    y_train

)

print("Training Finished.\n")

# =============================================================================
# Prediction
# =============================================================================

train_pred = model.predict(X_train)
valid_pred = model.predict(X_valid)
test_pred = model.predict(X_test)

# =============================================================================
# Metrics
# =============================================================================

def calculate_metrics(y_true, y_pred):

    r2 = r2_score(y_true, y_pred)

    rmse = np.sqrt(

        mean_squared_error(

            y_true,

            y_pred

        )

    )

    mae = mean_absolute_error(

        y_true,

        y_pred

    )

    return r2, rmse, mae

train_r2, train_rmse, train_mae = calculate_metrics(
    y_train,
    train_pred
)

valid_r2, valid_rmse, valid_mae = calculate_metrics(
    y_valid,
    valid_pred
)

test_r2, test_rmse, test_mae = calculate_metrics(
    y_test,
    test_pred
)

performance = pd.DataFrame({

    "Dataset":[
        "Train",
        "Validation",
        "Test"
    ],

    "R2":[
        train_r2,
        valid_r2,
        test_r2
    ],

    "RMSE":[
        train_rmse,
        valid_rmse,
        test_rmse
    ],

    "MAE":[
        train_mae,
        valid_mae,
        test_mae
    ]

})

performance.to_csv(

    RESULT_DIR /

    "performance_summary.csv",

    index=False,

    encoding="utf-8-sig"

)

print(performance.round(4))
# =============================================================================
# Save Model
# =============================================================================

joblib.dump(
    model,
    MODEL_DIR / "best_xgboost.pkl"
)

# =============================================================================
# Save Predictions
# =============================================================================

datasets = {
    "train": (y_train, train_pred),
    "validation": (y_valid, valid_pred),
    "test": (y_test, test_pred)
}

for name, (y_true, y_pred) in datasets.items():

    pd.DataFrame({

        "Observed": y_true,
        "Predicted": y_pred,
        "Residual": y_true - y_pred

    }).to_csv(

        RESULT_DIR / f"{name}_predictions.csv",

        index=False,

        encoding="utf-8-sig"

    )

# =============================================================================
# Feature Importance
# =============================================================================

importance = pd.DataFrame({

    "Feature": X_train.columns,

    "Importance": model.feature_importances_

})

importance = importance.sort_values(

    by="Importance",

    ascending=False

)

importance.to_csv(

    RESULT_DIR / "feature_importance.csv",

    index=False,

    encoding="utf-8-sig"

)

top20 = importance.head(20)

plt.figure(figsize=(8,6))

plt.barh(

    top20["Feature"],

    top20["Importance"],

    color="royalblue"

)

plt.gca().invert_yaxis()

plt.xlabel("Importance", fontsize=12)

plt.ylabel("Fingerprint Bit", fontsize=12)

plt.title("Top 20 Feature Importance", fontsize=15)

plt.tight_layout()

plt.savefig(

    RESULT_DIR / "feature_importance_top20.png",

    dpi=600,

    bbox_inches="tight"

)

plt.close()

# =============================================================================
# Observed vs Predicted
# =============================================================================

plt.figure(figsize=(7,6))

plt.scatter(

    y_test,

    test_pred,

    color="royalblue",

    edgecolors="black",

    linewidth=0.4,

    s=55,

    alpha=0.70

)

mn = min(y_test.min(), test_pred.min())
mx = max(y_test.max(), test_pred.max())

plt.plot(

    [mn, mx],

    [mn, mx],

    "k--",

    linewidth=2

)

stats = (

    f"R² = {test_r2:.3f}\n"

    f"RMSE = {test_rmse:.3f}\n"

    f"MAE = {test_mae:.3f}"

)

plt.text(

    0.05,

    0.95,

    stats,

    transform=plt.gca().transAxes,

    fontsize=11,

    verticalalignment="top",

    bbox=dict(

        facecolor="white",

        edgecolor="black",

        boxstyle="round"

    )

)

plt.xlabel("Observed pIC50", fontsize=13, fontweight="bold")
plt.ylabel("Predicted pIC50", fontsize=13, fontweight="bold")
plt.title("Observed vs Predicted (Test Set)", fontsize=15, fontweight="bold")

plt.grid(alpha=0.30, linestyle=":")

plt.tight_layout()

plt.savefig(

    RESULT_DIR / "observed_vs_predicted_test.png",

    dpi=600,

    bbox_inches="tight"

)

plt.close()

# =============================================================================
# Residual Plot
# =============================================================================

plt.figure(figsize=(7,6))

residuals = y_test - test_pred

plt.scatter(

    test_pred,

    residuals,

    color="royalblue",

    edgecolors="black",

    linewidth=0.4,

    s=55,

    alpha=0.70

)

plt.axhline(

    0,

    color="black",

    linestyle="--",

    linewidth=2

)

plt.text(

    0.98,

    0.98,

    stats,

    transform=plt.gca().transAxes,

    fontsize=11,

    horizontalalignment="right",

    verticalalignment="top",

    bbox=dict(

        facecolor="white",

        edgecolor="black",

        boxstyle="round"

    )

)

plt.xlabel("Predicted pIC50", fontsize=13, fontweight="bold")
plt.ylabel("Residual", fontsize=13, fontweight="bold")
plt.title("Residual Plot (Test Set)", fontsize=15, fontweight="bold")

plt.grid(alpha=0.30, linestyle=":")

plt.tight_layout()

plt.savefig(

    RESULT_DIR / "residual_plot_test.png",

    dpi=600,

    bbox_inches="tight"

)

plt.close()

# =============================================================================
# Final Report
# =============================================================================

print()
print("=" * 80)
print("FINAL XGBOOST COMPLETED")
print("=" * 80)

print()

print(performance.round(4))

print()

print("Model")
print("-" * 80)
print(MODEL_DIR / "best_xgboost.pkl")

print()

print("Results Folder")
print("-" * 80)
print(RESULT_DIR)

print()
print("=" * 80)
