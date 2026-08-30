#!/usr/bin/env python3
"""Apply the final multi-objective filters and scaffold-aware selection."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem.Scaffolds import MurckoScaffold


TARGET_SIZE = 50_000
EXPECTED_FILTER_COUNTS = {
    "Initial library": 494_897,
    "pActivity >=5": 367_801,
    "Morgan <0.5": 311_264,
    "Murcko <0.5": 209_422,
    "SA <=4": 188_232,
    "Lipinski <=1": 188_232,
    "After PAINS removal": 181_719,
}


def murcko_smiles(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES in characterized input: {smiles}")
    scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
    result = Chem.MolToSmiles(scaffold, canonical=True)
    return result if result else "[ACYCLIC]"


def pains_mask(smiles: pd.Series) -> list[bool]:
    parameters = FilterCatalogParams()
    parameters.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    catalog = FilterCatalog(parameters)
    keep = []
    for text in smiles:
        molecule = Chem.MolFromSmiles(text)
        keep.append(molecule is not None and not catalog.HasMatch(molecule))
    return keep


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Characterized 494,897-molecule CSV")
    parser.add_argument("output", type=Path, help="Output CSV for the final library")
    parser.add_argument("--statistics", type=Path, default=None)
    args = parser.parse_args()

    data = pd.read_csv(args.input, low_memory=False)
    required = {
        "SMILES", "Consensus_pActivity", "Max_Morgan_Tanimoto",
        "Max_Murcko_Scaffold_Similarity", "SA_Score", "Lipinski_Violations",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    rows: list[dict[str, int | str]] = []
    def record(stage: str, frame: pd.DataFrame) -> None:
        rows.append({"Stage": stage, "Count": len(frame)})

    record("Initial library", data)
    data = data.loc[data["Consensus_pActivity"] >= 5.0].copy()
    record("pActivity >=5", data)
    data = data.loc[data["Max_Morgan_Tanimoto"] < 0.50].copy()
    record("Morgan <0.5", data)
    data = data.loc[data["Max_Murcko_Scaffold_Similarity"] < 0.50].copy()
    record("Murcko <0.5", data)
    data = data.loc[data["SA_Score"] <= 4.0].copy()
    record("SA <=4", data)
    data = data.loc[data["Lipinski_Violations"] <= 1].copy()
    record("Lipinski <=1", data)
    data = data.loc[pains_mask(data["SMILES"])].copy()
    record("After PAINS removal", data)

    observed = {row["Stage"]: row["Count"] for row in rows}
    if observed != EXPECTED_FILTER_COUNTS:
        raise AssertionError(f"Filter counts differ from the final study run: {observed}")

    data["Murcko_Scaffold"] = data["SMILES"].map(murcko_smiles)
    activity = data["Consensus_pActivity"] / data["Consensus_pActivity"].max()
    molecular_novelty = 1.0 - data["Max_Morgan_Tanimoto"]
    scaffold_novelty = 1.0 - data["Max_Murcko_Scaffold_Similarity"]
    accessibility = 1.0 - data["SA_Score"] / data["SA_Score"].max()
    data["HTVS_Priority_Score"] = (
        0.45 * activity
        + 0.25 * molecular_novelty
        + 0.20 * scaffold_novelty
        + 0.10 * accessibility
    )

    ranked = data.sort_values("HTVS_Priority_Score", ascending=False)
    final = ranked.drop_duplicates("Murcko_Scaffold", keep="first").head(TARGET_SIZE).copy()
    if len(final) != TARGET_SIZE or final["Murcko_Scaffold"].nunique() != TARGET_SIZE:
        raise AssertionError("Expected 50,000 molecules from 50,000 scaffold classes")
    final.insert(0, "Compound_ID", [f"BCRL{index:05d}" for index in range(1, TARGET_SIZE + 1)])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(args.output, index=False)
    if args.statistics:
        args.statistics.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.statistics, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
