"""
===============================================================================
Project : BCRABL-AI
Script  : 18_prepare_features.py

Purpose
-------
Generate Morgan Fingerprints (ECFP4) for classical
machine learning models.

Input
-----
bcrabl_ic50_scaffold.csv

Output
------
morgan_features.csv

===============================================================================
"""

from pathlib import Path

import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem

# =============================================================================
# Settings
# =============================================================================

RADIUS = 2          # ECFP4
N_BITS = 2048

# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "01_Data"
    / "Modeling"
    / "bcrabl_ic50_scaffold.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "03_Feature_Engineering"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "07_Results"
    / "Feature_Engineering"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================

df = pd.read_csv(INPUT_FILE)

# =============================================================================
# Morgan Fingerprints
# =============================================================================

fingerprints = []

valid = 0
invalid = 0

for smi in df["SMILES"]:

    mol = Chem.MolFromSmiles(smi)

    if mol is None:

        invalid += 1

        fingerprints.append([0] * N_BITS)

        continue

    valid += 1

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=RADIUS,
        nBits=N_BITS
    )

    fingerprints.append(list(fp))

# =============================================================================
# Feature Matrix
# =============================================================================

feature_names = [

    f"FP_{i}"

    for i in range(N_BITS)

]

fp_df = pd.DataFrame(
    fingerprints,
    columns=feature_names
)

final_df = pd.concat(

    [

        df[["Compound_ID", "SMILES", "pActivity"]],

        fp_df

    ],

    axis=1

)

# =============================================================================
# Save
# =============================================================================

OUTPUT_FILE = OUTPUT_DIR / "morgan_features.csv"

final_df.to_csv(

    OUTPUT_FILE,

    index=False,

    encoding="utf-8-sig"

)

summary = pd.DataFrame({

    "Metric": [

        "Total Molecules",

        "Valid Molecules",

        "Invalid Molecules",

        "Fingerprint Size",

        "Radius"

    ],

    "Value": [

        len(df),

        valid,

        invalid,

        N_BITS,

        RADIUS

    ]

})

summary.to_csv(

    RESULT_DIR /

    "feature_summary.csv",

    index=False,

    encoding="utf-8-sig"

)

# =============================================================================
# Console
# =============================================================================

print("=" * 80)
print("FEATURE ENGINEERING COMPLETED")
print("=" * 80)

print(f"Total Molecules : {len(df):,}")
print(f"Valid Molecules : {valid:,}")
print(f"Invalid         : {invalid:,}")

print(f"Fingerprint     : Morgan (ECFP4)")
print(f"Radius          : {RADIUS}")
print(f"Bits            : {N_BITS}")

print("\nOutput File")
print(OUTPUT_FILE)