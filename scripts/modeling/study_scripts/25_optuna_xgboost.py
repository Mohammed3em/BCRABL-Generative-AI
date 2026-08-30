"""
===============================================================================
Project : BCRABL-AI
Script  : 25_optuna_xgboost.py

Purpose
-------
Hyperparameter optimization of XGBoost using Optuna.

===============================================================================
"""

from pathlib import Path
import json
import warnings

import optuna
import pandas as pd

from xgboost import XGBRegressor

from sklearn.model_selection import KFold, cross_val_score

from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from config import RANDOM_STATE
from data_loader import load_data

warnings.filterwarnings("ignore")

# =============================================================================
# Directories
# =============================================================================

RESULT_DIR = Path(
    "07_Results/XGBoost_Optuna"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

(
    X_train,
    y_train,
    X_valid,
    y_valid,
    X_test,
    y_test
) = load_data()

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

# =============================================================================
# Objective Function
# =============================================================================

def objective(trial):

    params = {

        "objective": "reg:squarederror",

        "n_estimators":
        trial.suggest_int(
            "n_estimators",
            200,
            1200,
            step=100
        ),

        "learning_rate":
        trial.suggest_float(
            "learning_rate",
            0.01,
            0.30,
            log=True
        ),

        "max_depth":
        trial.suggest_int(
            "max_depth",
            3,
            12
        ),

        "min_child_weight":
        trial.suggest_int(
            "min_child_weight",
            1,
            10
        ),

        "gamma":
        trial.suggest_float(
            "gamma",
            0,
            5
        ),

        "subsample":
        trial.suggest_float(
            "subsample",
            0.6,
            1.0
        ),

        "colsample_bytree":
        trial.suggest_float(
            "colsample_bytree",
            0.6,
            1.0
        ),

        "reg_alpha":
        trial.suggest_float(
            "reg_alpha",
            0,
            5
        ),

        "reg_lambda":
        trial.suggest_float(
            "reg_lambda",
            0.1,
            10,
            log=True
        ),

        "random_state": RANDOM_STATE,

        "verbosity": 0,

        "n_jobs": -1

    }

    model = XGBRegressor(
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

    trial.set_user_attr(
        "cv_std",
        scores.std()
    )

    return scores.mean()

study = optuna.create_study(

    direction="maximize",

    sampler=TPESampler(
        seed=RANDOM_STATE
    ),

    pruner=MedianPruner()

)

study.optimize(

    objective,

    n_trials=100,

    show_progress_bar=True

)
# =============================================================================
# Best Trial
# =============================================================================

best_trial = study.best_trial

best_params = best_trial.params

best_score = best_trial.value

best_std = best_trial.user_attrs["cv_std"]

print()
print("=" * 80)
print("OPTIMIZATION FINISHED")
print("=" * 80)

print(f"\nBest Mean CV R² : {best_score:.4f}")
print(f"CV Std          : {best_std:.4f}")

print("\nBest Parameters")
print("-" * 80)

for key, value in best_params.items():
    print(f"{key:<25}: {value}")

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
# Save Optimization Summary
# =============================================================================

summary = pd.DataFrame({

    "Metric": [

        "Trials",
        "Best Mean CV R2",
        "CV Standard Deviation"

    ],

    "Value": [

        len(study.trials),
        best_score,
        best_std

    ]

})

summary.to_csv(

    RESULT_DIR / "optimization_summary.csv",

    index=False,

    encoding="utf-8-sig"

)

# =============================================================================
# Final Report
# =============================================================================

print()
print("=" * 80)
print("XGBOOST OPTUNA COMPLETED")
print("=" * 80)

print()

print("Generated Files")
print("-" * 80)

print(RESULT_DIR / "best_parameters.json")
print(RESULT_DIR / "trial_history.csv")
print(RESULT_DIR / "optimization_summary.csv")

print()

print("=" * 80)