"""
===============================================================================
Project : BCRABL-AI
Script  : 23_train_best_random_forest.py

Purpose
-------
Train the final optimized Random Forest model using the best hyperparameters
identified by Optuna and evaluate its performance.

===============================================================================
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

from config import RANDOM_STATE
from data_loader import load_data

# =============================================================================
# Directories
# =============================================================================

PARAM_FILE = Path(
    "07_Results/Random_Forest_Optuna/best_parameters.json"
)

RESULT_DIR = Path(
    "07_Results/Random_Forest_Final"
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
# Load Best Parameters
# =============================================================================

with open(
    PARAM_FILE,
    "r",
    encoding="utf-8"
) as f:

    params = json.load(f)

params["random_state"] = RANDOM_STATE
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
print("FINAL RANDOM FOREST MODEL")
print("=" * 80)

print("\nBest Hyperparameters\n")

for k, v in params.items():
    print(f"{k:<22}: {v}")

# =============================================================================
# Build Model
# =============================================================================

model = RandomForestRegressor(
    **params
)

print("\nTraining Model...\n")

model.fit(
    X_train,
    y_train
)

print("Training Completed.\n")

# =============================================================================
# Prediction
# =============================================================================

train_pred = model.predict(X_train)
valid_pred = model.predict(X_valid)
test_pred = model.predict(X_test)

# =============================================================================
# Evaluation Function
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

# =============================================================================
# Performance Summary
# =============================================================================

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
    MODEL_DIR / "best_random_forest.pkl"
)

# =============================================================================
# Save Prediction Files
# =============================================================================

datasets = {
    "train": (y_train, train_pred),
    "validation": (y_valid, valid_pred),
    "test": (y_test, test_pred)
}

for name, (y_true, y_pred) in datasets.items():

    df = pd.DataFrame({

        "Observed": y_true,
        "Predicted": y_pred,
        "Residual": y_true - y_pred

    })

    df.to_csv(

        RESULT_DIR / f"{name}_predictions.csv",

        index=False,

        encoding="utf-8-sig"

    )

# =============================================================================
# Publication-Quality Observed vs Predicted
# =============================================================================

plt.figure(figsize=(7,6))

plt.scatter(

    y_test,

    test_pred,

    s=55,

    color="royalblue",

    edgecolors="black",

    linewidth=0.4,

    alpha=0.70

)

minimum = min(y_test.min(), test_pred.min())
maximum = max(y_test.max(), test_pred.max())

plt.plot(

    [minimum, maximum],

    [minimum, maximum],

    color="black",

    linestyle="--",

    linewidth=2

)

plt.xlabel(
    "Observed pIC50",
    fontsize=13,
    fontweight="bold"
)

plt.ylabel(
    "Predicted pIC50",
    fontsize=13,
    fontweight="bold"
)

plt.title(
    "Observed vs Predicted (Test Set)",
    fontsize=15,
    fontweight="bold"
)

plt.grid(
    linestyle=":",
    alpha=0.30
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

plt.tight_layout()

plt.savefig(

    RESULT_DIR / "observed_vs_predicted_test.png",

    dpi=600,

    bbox_inches="tight"

)

plt.close()

# =============================================================================
# Publication-Quality Residual Plot
# =============================================================================

plt.figure(figsize=(7,6))

residuals = y_test - test_pred

plt.scatter(

    test_pred,

    residuals,

    s=55,

    color="royalblue",

    edgecolors="black",

    linewidth=0.4,

    alpha=0.70

)

plt.axhline(

    y=0,

    color="black",

    linestyle="--",

    linewidth=2

)

plt.xlabel(

    "Predicted pIC50",

    fontsize=13,

    fontweight="bold"

)

plt.ylabel(

    "Residual",

    fontsize=13,

    fontweight="bold"

)

plt.title(

    "Residual Plot (Test Set)",

    fontsize=15,

    fontweight="bold"

)

plt.grid(

    linestyle=":",

    alpha=0.30

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
print("FINAL RANDOM FOREST COMPLETED")
print("=" * 80)

print()

print(performance.round(4))

print()

print("Saved Model")
print("-" * 80)
print(MODEL_DIR / "best_random_forest.pkl")

print()

print("Results Folder")
print("-" * 80)
print(RESULT_DIR)

print()
print("=" * 80)