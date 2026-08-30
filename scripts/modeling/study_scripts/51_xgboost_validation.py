"""
===============================================================================
Project : BCRABL-AI

File:
    51_xgboost_validation.py

Purpose:
    5-Fold Cross Validation for Final XGBoost Model

===============================================================================
"""


import numpy as np
import pandas as pd

from pathlib import Path

import joblib

from sklearn.model_selection import KFold
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)


# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(
    r"D:\BCRABL-AI"
)


DATA_PATH = (

    BASE_DIR

    /

    "03_Feature_Engineering"

    /

    "morgan_features.csv"

)


MODEL_PATH = (
    BASE_DIR
    /
    "02_Data_Curation"
    /
    "Scripts"
    /
    "04_AI_Models"
    /
    "Models"
    /
    "xgboost.pkl"
)


OUTPUT_DIR = (

    BASE_DIR

    /

    "07_Results"

    /

    "XGBoost_CV"

)


OUTPUT_DIR.mkdir(

    exist_ok=True

)


# =============================================================================
# LOAD DATA
# =============================================================================


print("="*80)

print("XGBOOST 5-FOLD CROSS VALIDATION")

print("="*80)


df = pd.read_csv(

    DATA_PATH

)


print()

print("Dataset Shape:")

print(df.shape)


print()

print("Columns:")

print(df.columns.tolist())



# =============================================================================
# TARGET
# =============================================================================


TARGET = "pActivity"



if TARGET not in df.columns:

    raise ValueError(

        f"{TARGET} column not found"

    )



X = df.drop(

    columns=[TARGET]

)


y = df[TARGET]



# Remove SMILES if exists

if "SMILES" in X.columns:

    X = X.drop(

        columns=["SMILES"]

    )
    remove_columns = [
    "Compound_ID",
    "SMILES"
]

for col in remove_columns:
    if col in X.columns:
        X = X.drop(columns=[col])



print()

print("Features:")

print(X.shape[1])



# =============================================================================
# LOAD MODEL
# =============================================================================


model = joblib.load(

    MODEL_PATH

)


print()

print("XGBoost Model Loaded")



# =============================================================================
# 5 FOLD CV
# =============================================================================


kf = KFold(

    n_splits=5,

    shuffle=True,

    random_state=42

)


results = []



for fold, (train_idx, test_idx) in enumerate(

    kf.split(X),

    start=1

):


    print()

    print("-"*80)

    print(

        f"Fold {fold}"

    )


    X_train = X.iloc[train_idx]

    X_test = X.iloc[test_idx]


    y_train = y.iloc[train_idx]

    y_test = y.iloc[test_idx]



    model.fit(

        X_train,

        y_train

    )


    prediction = model.predict(

        X_test

    )



    r2 = r2_score(

        y_test,

        prediction

    )


    rmse = np.sqrt(

        mean_squared_error(

            y_test,

            prediction

        )

    )


    mae = mean_absolute_error(

        y_test,

        prediction

    )



    print(

        f"R2   : {r2:.4f}"

    )


    print(

        f"RMSE : {rmse:.4f}"

    )


    print(

        f"MAE  : {mae:.4f}"

    )



    results.append(

        {

            "Fold": fold,

            "R2": r2,

            "RMSE": rmse,

            "MAE": mae

        }

    )



# =============================================================================
# SAVE RESULTS
# =============================================================================


results_df = pd.DataFrame(

    results

)



summary = pd.DataFrame(

    {

        "Metric":

        [

            "R2",

            "RMSE",

            "MAE"

        ],


        "Mean":

        [

            results_df["R2"].mean(),

            results_df["RMSE"].mean(),

            results_df["MAE"].mean()

        ],


        "SD":

        [

            results_df["R2"].std(),

            results_df["RMSE"].std(),

            results_df["MAE"].std()

        ]

    }

)



print()

print("="*80)

print("CV SUMMARY")

print("="*80)


print(summary)



results_df.to_csv(

    OUTPUT_DIR /

    "fold_results.csv",

    index=False

)



summary.to_csv(

    OUTPUT_DIR /

    "cv_summary.csv",

    index=False

)



print()

print("Saved:")

print(OUTPUT_DIR)


print()

print("="*80)

print("XGBOOST CV COMPLETED")

print("="*80)