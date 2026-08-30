#!/usr/bin/env python3
"""Validate the compact BCR::ABL1 publication data release."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
SPLITS = ROOT / "data" / "splits"
ACYCLIC = "[ACYCLIC]"


def fail(message: str) -> None:
    raise AssertionError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    modeling_path = PROCESSED / "bcrabl_ic50_modeling.csv"
    scaffold_path = PROCESSED / "bcrabl_ic50_scaffold.csv"
    modeling = pd.read_csv(modeling_path)
    scaffold = pd.read_csv(scaffold_path)

    expected_modeling = [
        "Compound_ID", "SMILES", "IC50_nM", "pActivity", "Measurements", "Sources"
    ]
    if modeling.columns.tolist() != expected_modeling:
        fail("Unexpected modeling columns")
    if scaffold.columns.tolist() != expected_modeling + ["Scaffold"]:
        fail("Unexpected scaffold columns")
    if len(modeling) != 7912 or modeling["SMILES"].nunique() != 7912:
        fail("The modeling dataset must contain 7,912 unique structures")
    if modeling["Compound_ID"].nunique() != 7912:
        fail("Compound_ID values are not unique")
    if not modeling[expected_modeling].equals(scaffold[expected_modeling]):
        fail("The scaffold table does not preserve the modeling records")

    scaffold_class = scaffold["Scaffold"].fillna(ACYCLIC)
    if scaffold_class.nunique() != 3341:
        fail("Expected 3,341 scaffold classes including the acyclic class")
    if int(scaffold["Scaffold"].isna().sum()) != 27:
        fail("Expected 27 acyclic compounds serialized with an empty scaffold")

    expected = {"train": 6373, "validation": 757, "test": 782}
    expected_scaffolds = {"train": 2672, "validation": 334, "test": 335}
    split_scaffolds: dict[str, set[str]] = {}
    split_smiles: list[str] = []
    lookup = scaffold[["SMILES", "Scaffold", "pActivity"]]

    for name, n_rows in expected.items():
        part = pd.read_csv(SPLITS / f"{name}.csv")
        if part.columns.tolist() != ["SMILES", "pActivity"]:
            fail(f"Unexpected columns in {name}.csv")
        if len(part) != n_rows or part["SMILES"].nunique() != n_rows:
            fail(f"Unexpected row or unique-SMILES count in {name}.csv")
        joined = part.merge(lookup, on=["SMILES", "pActivity"], how="left")
        if len(joined) != len(part):
            fail(f"Non-unique merge while validating {name}.csv")
        classes = set(joined["Scaffold"].fillna(ACYCLIC))
        if len(classes) != expected_scaffolds[name]:
            fail(f"Unexpected scaffold count in {name}.csv")
        split_scaffolds[name] = classes
        split_smiles.extend(part["SMILES"].tolist())

    if len(split_smiles) != len(set(split_smiles)):
        fail("A structure occurs in more than one split")
    if set(split_smiles) != set(modeling["SMILES"]):
        fail("The fixed partitions do not cover the full modeling dataset")
    for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        if split_scaffolds[left] & split_scaffolds[right]:
            fail(f"Scaffold leakage detected between {left} and {right}")

    implied = 9.0 - np.log10(modeling["IC50_nM"].astype(float))
    delta = (implied - modeling["pActivity"].astype(float)).abs()
    print("PASS: release structure, counts, coverage, and scaffold separation")
    print(f"Compounds: {len(modeling):,}; scaffold classes: {scaffold_class.nunique():,}")
    print("Splits: train=6,373; validation=757; test=782; leakage=0")
    print(
        "Aggregation note: "
        f"{int((delta > 1e-6).sum())} rows have separately aggregated median IC50 and "
        "median pIC50 values that are not exact inverses."
    )
    print("SHA-256:")
    for path in sorted((ROOT / "data").rglob("*")):
        if path.is_file():
            print(f"{sha256(path)}  {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, FileNotFoundError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
