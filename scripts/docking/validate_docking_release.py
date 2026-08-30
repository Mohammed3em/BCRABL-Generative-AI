#!/usr/bin/env python3
"""Validate the released XP500 workbook and top-candidate table."""

from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
WORKBOOK = ROOT / "data" / "supplementary" / "Supplementary_Data_S1.xlsx"
TOP10 = ROOT / "results" / "docking" / "top10_ai_candidates.csv"


def main() -> int:
    workbook = openpyxl.load_workbook(WORKBOOK, read_only=True, data_only=True)
    sheet = workbook["XP_Unique_502"]
    rows = list(sheet.iter_rows(min_row=2, values_only=True))
    record_types = [row[0] for row in rows]
    if len(rows) != 502:
        raise AssertionError(f"Expected 502 XP records; found {len(rows)}")
    if record_types.count("AI-generated") != 500 or record_types.count("Reference") != 2:
        raise AssertionError("Expected 500 AI-generated records and two references")

    top10 = pd.read_csv(TOP10)
    if len(top10) != 10 or top10["Compound_ID"].nunique() != 10:
        raise AssertionError("Expected ten unique AI-generated candidates")
    expected = {
        "BCRL18077": -17.627,
        "BCRL46354": -16.947,
        "BCRL34603": -16.584,
        "BCRL37224": -16.042,
    }
    observed = dict(zip(top10["Compound_ID"], top10["glide gscore"]))
    for compound, score in expected.items():
        if compound not in observed or not np.isclose(float(observed[compound]), score):
            raise AssertionError(f"Unexpected XP result for {compound}")

    references = {row[2]: float(row[3]) for row in rows if row[0] == "Reference"}
    if not np.isclose(references.get("Imatinib", np.nan), -16.894):
        raise AssertionError("Unexpected imatinib XP score")
    if not np.isclose(references.get("Ponatinib", np.nan), -16.297):
        raise AssertionError("Unexpected ponatinib XP score")
    print("PASS: XP500 composition, Top10 candidates, and reference scores validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
