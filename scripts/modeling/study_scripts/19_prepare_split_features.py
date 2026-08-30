"""
===============================================================================
Project : BCRABL-AI
Script  : 19_prepare_split_features.py

Purpose
-------
Generate feature matrices for Train / Validation / Test
using the predefined scaffold split.

Author : Mohammed Abdulaali
===============================================================================
"""

from pathlib import Path
import pandas as pd

# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_FILE = (
    PROJECT_ROOT
    / "03_Feature_Engineering"
    / "morgan_features.csv"
)

SPLIT_DIR = (
    PROJECT_ROOT
    / "01_Data"
    / "Modeling"
    / "Split"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "03_Feature_Engineering"
    / "Split"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "07_Results"
    / "Feature_Split"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Read files
# =============================================================================

features = pd.read_csv(FEATURE_FILE)

train = pd.read_csv(SPLIT_DIR / "train.csv")
validation = pd.read_csv(SPLIT_DIR / "validation.csv")
test = pd.read_csv(SPLIT_DIR / "test.csv")

# =============================================================================
# Merge by SMILES
# =============================================================================

train_features = train.merge(
    features,
    on=["SMILES", "pActivity"],
    how="left"
)

validation_features = validation.merge(
    features,
    on=["SMILES", "pActivity"],
    how="left"
)

test_features = test.merge(
    features,
    on=["SMILES", "pActivity"],
    how="left"
)

# =============================================================================
# Save
# =============================================================================

train_features.to_csv(
    OUTPUT_DIR / "train_features.csv",
    index=False,
    encoding="utf-8-sig"
)

validation_features.to_csv(
    OUTPUT_DIR / "validation_features.csv",
    index=False,
    encoding="utf-8-sig"
)

test_features.to_csv(
    OUTPUT_DIR / "test_features.csv",
    index=False,
    encoding="utf-8-sig"
)

# =============================================================================
# Summary
# =============================================================================

summary = pd.DataFrame({
    "Dataset": ["Train", "Validation", "Test"],
    "Records": [
        len(train_features),
        len(validation_features),
        len(test_features)
    ],
    "Missing Features": [
        train_features.isna().sum().sum(),
        validation_features.isna().sum().sum(),
        test_features.isna().sum().sum()
    ]
})

summary.to_csv(
    RESULT_DIR / "feature_split_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

# =============================================================================
# Console
# =============================================================================

print("=" * 80)
print("FEATURE SPLIT COMPLETED")
print("=" * 80)

print(summary)

print("\nOutput Folder")
print(OUTPUT_DIR)

print("\nResults Folder")
print(RESULT_DIR)