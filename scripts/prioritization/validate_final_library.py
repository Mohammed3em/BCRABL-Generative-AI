#!/usr/bin/env python3
"""Validate the released 50,000-member pre-docking library."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "prioritized" / "bcrabl_final_50000.csv"


def close(value: float, expected: float, tolerance: float = 5e-6) -> None:
    if not np.isclose(value, expected, atol=tolerance, rtol=0):
        raise AssertionError(f"Expected {expected}, found {value}")


def main() -> int:
    data = pd.read_csv(INPUT, low_memory=False)
    if len(data) != 50_000 or data["SMILES"].nunique() != 50_000:
        raise AssertionError("Expected 50,000 unique molecules")
    scaffold_class = data["Murcko_Scaffold"].fillna("[ACYCLIC]")
    if scaffold_class.nunique() != 50_000:
        raise AssertionError("Expected 50,000 scaffold classes including the acyclic class")
    if not (data["Consensus_pActivity"] >= 5.0).all():
        raise AssertionError("Activity threshold violation")
    if not (data["Max_Morgan_Tanimoto"] < 0.50).all():
        raise AssertionError("Morgan-similarity threshold violation")
    if not (data["Max_Murcko_Scaffold_Similarity"] < 0.50).all():
        raise AssertionError("Murcko-similarity threshold violation")
    if not (data["SA_Score"] <= 4.0).all():
        raise AssertionError("SA threshold violation")
    if not (data["Lipinski_Violations"] <= 1).all():
        raise AssertionError("Lipinski threshold violation")

    checks = {
        "Consensus_pActivity": (5.555298, 5.474303),
        "Max_Morgan_Tanimoto": (0.327748, 0.323000),
        "Max_Murcko_Scaffold_Similarity": (0.390477, 0.394000),
        "SA_Score": (2.889469, 2.820000),
    }
    for column, (mean, median) in checks.items():
        close(float(data[column].mean()), mean)
        close(float(data[column].median()), median)
    print("PASS: final 50,000 library counts, scaffold separation, thresholds, and summaries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
