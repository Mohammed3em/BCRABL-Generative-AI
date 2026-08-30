"""
===============================================================================
Project : BCRABL-AI
Script  : 22_optuna_random_forest.py

Purpose
-------
Bayesian hyperparameter optimization of Random Forest using Optuna with
5-Fold Cross-Validation for BCR-ABL pIC50 prediction.

Author : Mohammed Abdulaali
===============================================================================
"""

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import optuna
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score

from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from config import RANDOM_STATE
from data_loader import load_data

warnings.filterwarnings("ignore")

# =============================================================================
# Output Directories
# =============================================================================

RESULT_DIR = Path(
    "07_Results/Random_Forest_Optuna"
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
print("OPTUNA RANDOM FOREST OPTIMIZATION")
print("=" * 80)
print()

print(f"Training Samples   : {len(X_train):,}")
print(f"Validation Samples : {len(X_valid):,}")
print(f"Testing Samples    : {len(X_test):,}")

print()

# =============================================================================
# Cross Validation
# =============================================================================

cv = KFold(

    n_splits=5,

    shuffle=True,

    random_state=RANDOM_STATE

)
# =============================================================================
# Objective Function
# =============================================================================

def objective(trial):
    """
    Objective function for Optuna optimization.
    """

    params = {

        "n_estimators": trial.suggest_int(
            "n_estimators",
            200,
            1200,
            step=100
        ),

        "max_depth": trial.suggest_int(
            "max_depth",
            5,
            50
        ),

        "min_samples_split": trial.suggest_int(
            "min_samples_split",
            2,
            20
        ),

        "min_samples_leaf": trial.suggest_int(
            "min_samples_leaf",
            1,
            10
        ),

        "max_features": trial.suggest_categorical(
            "max_features",
            [
                "sqrt",
                "log2",
                None
            ]
        ),

        "bootstrap": trial.suggest_categorical(
            "bootstrap",
            [
                True,
                False
            ]
        ),

        "random_state": RANDOM_STATE,

        "n_jobs": -1

    }

    model = RandomForestRegressor(

        **params

    )

    scores = cross_val_score(

        model,

        X_train,

        y_train,

        cv=cv,

        scoring="r2",

        n_jobs=-1

    )

    mean_score = scores.mean()

    trial.set_user_attr(

        "cv_std",

        float(scores.std())

    )

    return mean_score


# =============================================================================
# Create Study
# =============================================================================

study = optuna.create_study(

    direction="maximize",

    sampler=TPESampler(

        seed=RANDOM_STATE

    ),

    pruner=MedianPruner(

        n_startup_trials=10,

        n_warmup_steps=0

    )

)

print("=" * 80)
print("OPTUNA OPTIMIZATION STARTED")
print("=" * 80)
print()

N_TRIALS = 100
# =============================================================================
# Run Optimization
# =============================================================================

study.optimize(

    objective,

    n_trials=N_TRIALS,

    show_progress_bar=True

)

# =============================================================================
# Best Trial
# =============================================================================

best_trial = study.best_trial

best_params = best_trial.params

best_score = best_trial.value

best_cv_std = best_trial.user_attrs["cv_std"]

print()
print("=" * 80)
print("OPTIMIZATION FINISHED")
print("=" * 80)

print()

print(f"Best Mean CV R² : {best_score:.4f}")

print(f"CV Std          : {best_cv_std:.4f}")

print()

print("Best Hyperparameters")

print("-" * 80)

for key, value in best_params.items():

    print(f"{key:<25} {value}")

# =============================================================================
# Save Best Parameters
# =============================================================================

with open(

    RESULT_DIR / "best_parameters.json",

    "w",

    encoding="utf-8"

) as f:

    json.dump(

        best_params,

        f,

        indent=4

    )

# =============================================================================
# Save Trial History
# =============================================================================

history = study.trials_dataframe()

history.to_csv(

    RESULT_DIR / "trial_history.csv",

    index=False,

    encoding="utf-8-sig"

)

# =============================================================================
# Build Final Model
# =============================================================================

final_parameters = best_params.copy()

final_parameters.update({

    "random_state": RANDOM_STATE,

    "n_jobs": -1

})

final_model = RandomForestRegressor(

    **final_parameters

)

print()

print("=" * 80)

print("TRAINING FINAL MODEL")

print("=" * 80)

print()

final_model.fit(

    X_train,

    y_train

)
# =============================================================================
# Predictions
# =============================================================================

train_pred = final_model.predict(

    X_train

)

valid_pred = final_model.predict(

    X_valid

)

test_pred = final_model.predict(

    X_test

)

# =============================================================================
# Evaluation Metrics
# =============================================================================

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

train_r2 = r2_score(

    y_train,

    train_pred

)

valid_r2 = r2_score(

    y_valid,

    valid_pred

)

test_r2 = r2_score(

    y_test,

    test_pred

)

train_rmse = np.sqrt(

    mean_squared_error(

        y_train,

        train_pred

    )

)

valid_rmse = np.sqrt(

    mean_squared_error(

        y_valid,

        valid_pred

    )

)

test_rmse = np.sqrt(

    mean_squared_error(

        y_test,

        test_pred

    )

)

train_mae = mean_absolute_error(

    y_train,

    train_pred

)

valid_mae = mean_absolute_error(

    y_valid,

    valid_pred

)

test_mae = mean_absolute_error(

    y_test,

    test_pred

)

# =============================================================================
# Performance Table
# =============================================================================

performance = pd.DataFrame({

    "Dataset": [

        "Train",

        "Validation",

        "Test"

    ],

    "R2": [

        train_r2,

        valid_r2,

        test_r2

    ],

    "RMSE": [

        train_rmse,

        valid_rmse,

        test_rmse

    ],

    "MAE": [

        train_mae,

        valid_mae,

        test_mae

    ]

})

performance.to_csv(

    RESULT_DIR / "performance_summary.csv",

    index=False,

    encoding="utf-8-sig"

)

# =============================================================================
# Save Best Model
# =============================================================================

joblib.dump(

    final_model,

    MODEL_DIR / "best_random_forest.pkl"

)

# =============================================================================
# Prediction Files
# =============================================================================

train_predictions = pd.DataFrame({

    "Observed": y_train,

    "Predicted": train_pred,

    "Residual": y_train - train_pred

})

validation_predictions = pd.DataFrame({

    "Observed": y_valid,

    "Predicted": valid_pred,

    "Residual": y_valid - valid_pred

})

test_predictions = pd.DataFrame({

    "Observed": y_test,

    "Predicted": test_pred,

    "Residual": y_test - test_pred

})

train_predictions.to_csv(

    RESULT_DIR / "train_predictions.csv",

    index=False,

    encoding="utf-8-sig"

)

validation_predictions.to_csv(

    RESULT_DIR / "validation_predictions.csv",

    index=False,

    encoding="utf-8-sig"

)

test_predictions.to_csv(

    RESULT_DIR / "test_predictions.csv",

    index=False,

    encoding="utf-8-sig"

)
# =============================================================================
# Visualization
# =============================================================================

import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Observed vs Predicted (Test)
# -----------------------------------------------------------------------------

plt.figure(figsize=(6, 6))

plt.scatter(
    y_test,
    test_pred,
    alpha=0.70
)

minimum = min(
    y_test.min(),
    test_pred.min()
)

maximum = max(
    y_test.max(),
    test_pred.max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    "r--",
    linewidth=2
)

plt.xlabel("Observed pActivity")

plt.ylabel("Predicted pActivity")

plt.title("Random Forest (Optimized)\nObserved vs Predicted")

plt.tight_layout()

plt.savefig(

    RESULT_DIR / "observed_vs_predicted_test.png",

    dpi=300,

    bbox_inches="tight"

)

plt.close()

# -----------------------------------------------------------------------------
# Residual Plot
# -----------------------------------------------------------------------------

residuals = y_test - test_pred

plt.figure(figsize=(6,6))

plt.scatter(

    test_pred,

    residuals,

    alpha=0.70

)

plt.axhline(

    y=0,

    color="red",

    linestyle="--",

    linewidth=2

)

plt.xlabel("Predicted pActivity")

plt.ylabel("Residual")

plt.title("Residual Plot")

plt.tight_layout()

plt.savefig(

    RESULT_DIR / "residual_plot.png",

    dpi=300,

    bbox_inches="tight"

)

plt.close()

# =============================================================================
# Official Optuna Figures
# =============================================================================

from optuna.visualization import (

    plot_optimization_history,

    plot_param_importances,

    plot_slice,

    plot_parallel_coordinate

)

# Optimization History

fig = plot_optimization_history(study)

fig.write_image(

    RESULT_DIR / "optimization_history.png"

)

# Parameter Importance

fig = plot_param_importances(study)

fig.write_image(

    RESULT_DIR / "parameter_importance.png"

)

# Slice Plot

fig = plot_slice(study)

fig.write_image(

    RESULT_DIR / "slice_plot.png"

)

# Parallel Coordinate Plot

fig = plot_parallel_coordinate(study)

fig.write_image(

    RESULT_DIR / "parallel_coordinate.png"

)

# =============================================================================
# Save Trial Summary
# =============================================================================

trial_summary = pd.DataFrame({

    "Metric": [

        "Trials",

        "Best Mean CV R2",

        "CV Standard Deviation",

        "Train R2",

        "Validation R2",

        "Test R2"

    ],

    "Value": [

        len(study.trials),

        best_score,

        best_cv_std,

        train_r2,

        valid_r2,

        test_r2

    ]

})

trial_summary.to_csv(

    RESULT_DIR / "optimization_summary.csv",

    index=False,

    encoding="utf-8-sig"

)
