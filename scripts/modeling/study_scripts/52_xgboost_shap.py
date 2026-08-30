"""
===============================================================================
Project : BCRABL-AI

File:
    52_xgboost_shap.py

Purpose:
    XGBoost Feature Contribution Analysis
    (SHAP equivalent using native XGBoost)

===============================================================================
"""


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from pathlib import Path

import joblib

import xgboost as xgb



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
    "XGBoost_SHAP"

)


OUTPUT_DIR.mkdir(

    exist_ok=True

)



# =============================================================================
# LOAD DATA
# =============================================================================


print("="*80)

print("XGBOOST FEATURE CONTRIBUTION ANALYSIS")

print("="*80)



df = pd.read_csv(

    DATA_PATH

)



TARGET = "pActivity"



X = df.drop(

    columns=[TARGET]

)



for col in [

    "Compound_ID",

    "SMILES"

]:

    if col in X.columns:

        X = X.drop(

            columns=[col]

        )



print()

print("Features:")

print(X.shape)



# =============================================================================
# LOAD MODEL
# =============================================================================


model = joblib.load(

    MODEL_PATH

)


print()

print("XGBoost Model Loaded")



booster = model.get_booster()



# =============================================================================
# SAMPLE DATA
# =============================================================================


sample_size = min(

    1000,

    len(X)

)


X_sample = X.sample(

    sample_size,

    random_state=42

)



# =============================================================================
# NATIVE XGBOOST CONTRIBUTIONS
# =============================================================================


print()

print("Calculating feature contributions...")



dmatrix = xgb.DMatrix(

    X_sample

)



contributions = booster.predict(

    dmatrix,

    pred_contribs=True

)



# Last column is bias term

contributions = contributions[:, :-1]



print()

print("Calculation completed")



# =============================================================================
# FEATURE IMPORTANCE TABLE
# =============================================================================


importance = pd.DataFrame(

    {

        "Feature":

        X.columns,


        "Mean_ABS_SHAP":

        np.abs(contributions).mean(axis=0)

    }

)



importance = importance.sort_values(

    by="Mean_ABS_SHAP",

    ascending=False

)



importance.to_csv(

    OUTPUT_DIR /

    "All_SHAP_features.csv",

    index=False

)



importance.head(20).to_csv(

    OUTPUT_DIR /

    "Top20_SHAP_features.csv",

    index=False

)



# =============================================================================
# BAR PLOT TOP 20 FEATURES
# =============================================================================


top20 = importance.head(20)



plt.figure(

    figsize=(10,7)

)



plt.barh(

    top20["Feature"][::-1],

    top20["Mean_ABS_SHAP"][::-1]

)



plt.xlabel(

    "Mean |Contribution|"

)


plt.ylabel(

    "Morgan Fingerprint Bit"

)


plt.title(

    "Top 20 Features Contributing to XGBoost Prediction"

)


plt.tight_layout()



plt.savefig(

    OUTPUT_DIR /

    "Feature_Contribution_Top20.png",

    dpi=600,

    bbox_inches="tight"

)



plt.close()



# =============================================================================
# SUMMARY PLOT
# =============================================================================


plt.figure(

    figsize=(10,8)

)



top_features = top20["Feature"].values



feature_index = [

    list(X.columns).index(f)

    for f in top_features

]



summary_values = contributions[:, feature_index]



plt.boxplot(

    summary_values,

    labels=top_features,

    vert=False

)



plt.xlabel(

    "Contribution value"

)


plt.title(

    "Top 20 XGBoost Molecular Features"

)


plt.tight_layout()



plt.savefig(

    OUTPUT_DIR /

    "Contribution_Summary_Top20.png",

    dpi=600,

    bbox_inches="tight"

)



plt.close()



# =============================================================================
# FINISH
# =============================================================================


print()

print("="*80)

print("FEATURE CONTRIBUTION ANALYSIS COMPLETED")

print("="*80)


print()

print("Saved:")

print(OUTPUT_DIR)