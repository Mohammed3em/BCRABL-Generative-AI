"""
===============================================================================
Project : BCRABL-AI
Script  : 21_cross_validation.py

Purpose
-------
Evaluate the baseline Random Forest model using 5-fold Cross-Validation
on the training dataset.

Author : Mohammed Abdulaali
===============================================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

from config import (
    RF_PARAMS,
    RANDOM_STATE
)

from data_loader import load_data

# =============================================================================
# Output Directory
# =============================================================================

RESULT_DIR = Path(
    "07_Results/Cross_Validation"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# =============================================================================
# Load Training Data
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
print("5-FOLD CROSS VALIDATION")
print("=" * 80)

print(f"Training Samples : {len(X_train):,}")

# =============================================================================
# KFold
# =============================================================================

kf = KFold(

    n_splits=5,

    shuffle=True,

    random_state=RANDOM_STATE

)

results = []

fold = 1
# =============================================================================
# Cross Validation Loop
# =============================================================================

for train_index, validation_index in kf.split(X_train):

    print(f"Fold {fold}/5")

    X_tr = X_train.iloc[train_index]
    X_val = X_train.iloc[validation_index]

    y_tr = y_train.iloc[train_index]
    y_val = y_train.iloc[validation_index]

    model = RandomForestRegressor(

        **RF_PARAMS

    )

    model.fit(

        X_tr,

        y_tr

    )

    prediction = model.predict(

        X_val

    )

    r2 = r2_score(

        y_val,

        prediction

    )

    rmse = np.sqrt(

        mean_squared_error(

            y_val,

            prediction

        )

    )

    mae = mean_absolute_error(

        y_val,

        prediction

    )

    results.append({

        "Fold": fold,

        "Training_Size": len(X_tr),

        "Validation_Size": len(X_val),

        "R2": r2,

        "RMSE": rmse,

        "MAE": mae

    })

    print(f"   R²   : {r2:.4f}")
    print(f"   RMSE : {rmse:.4f}")
    print(f"   MAE  : {mae:.4f}")
    print()

    fold += 1

# =============================================================================
# Results DataFrame
# =============================================================================

results_df = pd.DataFrame(

    results

)

results_df.to_csv(

    RESULT_DIR / "cross_validation_results.csv",

    index=False,

    encoding="utf-8-sig"

)
# =============================================================================
# Summary Statistics
# =============================================================================

mean_r2 = results_df["R2"].mean()
std_r2 = results_df["R2"].std()

mean_rmse = results_df["RMSE"].mean()
std_rmse = results_df["RMSE"].std()

mean_mae = results_df["MAE"].mean()
std_mae = results_df["MAE"].std()

summary = pd.DataFrame({

    "Metric": [

        "Mean R2",
        "Std R2",

        "Mean RMSE",
        "Std RMSE",

        "Mean MAE",
        "Std MAE"

    ],

    "Value": [

        mean_r2,
        std_r2,

        mean_rmse,
        std_rmse,

        mean_mae,
        std_mae

    ]

})

summary.to_csv(

    RESULT_DIR / "cross_validation_summary.csv",

    index=False,

    encoding="utf-8-sig"

)

# =============================================================================
# Boxplot
# =============================================================================

plt.figure(figsize=(6, 5))

plt.boxplot(

    results_df["R2"],

    vert=True,

    patch_artist=True,

    labels=["R²"]

)

plt.ylabel("R² Score")

plt.title("5-Fold Cross-Validation Performance")

plt.grid(

    axis="y",

    alpha=0.30

)

plt.tight_layout()

plt.savefig(

    RESULT_DIR / "cross_validation_boxplot.png",

    dpi=300,

    bbox_inches="tight"

)

plt.close()

# =============================================================================
# Fold Performance Plot
# =============================================================================

plt.figure(figsize=(7,5))

plt.plot(

    results_df["Fold"],

    results_df["R2"],

    marker="o",

    linewidth=2

)

plt.xticks(results_df["Fold"])

plt.xlabel("Fold")

plt.ylabel("R²")

plt.title("R² Across 5 Cross-Validation Folds")

plt.grid(alpha=0.30)

plt.tight_layout()

plt.savefig(

    RESULT_DIR / "cross_validation_r2.png",

    dpi=300,

    bbox_inches="tight"

)

plt.close()
# =============================================================================
# Final Console Report
# =============================================================================

print()
print("=" * 80)
print("5-FOLD CROSS VALIDATION COMPLETED")
print("=" * 80)

print()

print("Average Performance")
print("-" * 80)

print(f"Mean R²        : {mean_r2:.4f} ± {std_r2:.4f}")
print(f"Mean RMSE      : {mean_rmse:.4f} ± {std_rmse:.4f}")
print(f"Mean MAE       : {mean_mae:.4f} ± {std_mae:.4f}")

print()

print("Individual Fold Results")
print("-" * 80)

print(results_df.round(4))

print()

print("Generated Files")
print("-" * 80)

print(RESULT_DIR / "cross_validation_results.csv")
print(RESULT_DIR / "cross_validation_summary.csv")
print(RESULT_DIR / "cross_validation_boxplot.png")
print(RESULT_DIR / "cross_validation_r2.png")

print()

print("=" * 80)
print("CROSS VALIDATION FINISHED SUCCESSFULLY")
print("=" * 80)