"""
===============================================================================
Project : BCRABL-AI
Script  : data_loader.py

Purpose
-------
Load Train / Validation / Test datasets and return feature matrices
and target vectors for machine learning models.

Author : Mohammed Abdulaali
===============================================================================
"""

import pandas as pd

from config import (
    TRAIN_FILE,
    VALIDATION_FILE,
    TEST_FILE,
)

# =============================================================================
# Load Data
# =============================================================================

def load_data():
    """
    Load Train / Validation / Test feature datasets.

    Returns
    -------
    X_train, y_train,
    X_valid, y_valid,
    X_test, y_test
    """

    # -------------------------------------------------------------------------
    # Read CSV files
    # -------------------------------------------------------------------------

    train = pd.read_csv(TRAIN_FILE)
    validation = pd.read_csv(VALIDATION_FILE)
    test = pd.read_csv(TEST_FILE)

    # -------------------------------------------------------------------------
    # Feature Columns
    # -------------------------------------------------------------------------

    feature_columns = [
        col for col in train.columns
        if col.startswith("FP_")
    ]

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    X_train = train[feature_columns]
    y_train = train["pActivity"]

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    X_valid = validation[feature_columns]
    y_valid = validation["pActivity"]

    # -------------------------------------------------------------------------
    # Test
    # -------------------------------------------------------------------------

    X_test = test[feature_columns]
    y_test = test["pActivity"]

    return (
        X_train,
        y_train,
        X_valid,
        y_valid,
        X_test,
        y_test
    )


# =============================================================================
# Dataset Summary
# =============================================================================

def dataset_summary():

    train = pd.read_csv(TRAIN_FILE)
    validation = pd.read_csv(VALIDATION_FILE)
    test = pd.read_csv(TEST_FILE)

    feature_columns = [
        c for c in train.columns
        if c.startswith("FP_")
    ]

    print("=" * 80)
    print("DATASET SUMMARY")
    print("=" * 80)

    print(f"Training Set    : {len(train):,}")
    print(f"Validation Set  : {len(validation):,}")
    print(f"Testing Set     : {len(test):,}")
    print(f"Fingerprint Bits: {len(feature_columns):,}")

    print()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":

    dataset_summary()