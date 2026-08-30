#!/usr/bin/env python3
"""Recalculate reported model metrics from released prediction tables."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "modeling"
MODELS = ["XGBoost_Final", "Random_Forest_Final", "GNN", "ChemBERTa", "MolFormer"]
SPLITS = ["Train", "Validation", "Test"]
EXPECTED_ROWS = {"Train": 6373, "Validation": 757, "Test": 782}
TOLERANCE = 5e-6


def metrics(observed: np.ndarray, predicted: np.ndarray) -> tuple[float, float, float]:
    residual = observed - predicted
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((observed - observed.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    mae = float(np.mean(np.abs(residual)))
    return r2, rmse, mae


def main() -> int:
    failures: list[str] = []
    rows: list[dict[str, object]] = []

    for model in MODELS:
        reported = pd.read_csv(RESULTS / model / "performance_summary.csv")
        for split in SPLITS:
            predictions = pd.read_csv(RESULTS / model / f"{split.lower()}_predictions.csv")
            if len(predictions) != EXPECTED_ROWS[split]:
                failures.append(
                    f"{model}/{split}: expected {EXPECTED_ROWS[split]} rows, found {len(predictions)}"
                )
            observed = predictions["Observed"].to_numpy(dtype=float)
            predicted = predictions["Predicted"].to_numpy(dtype=float)
            calculated = metrics(observed, predicted)
            selected = reported.loc[reported["Dataset"].str.lower() == split.lower()]
            if len(selected) != 1:
                failures.append(f"{model}/{split}: missing or duplicate reported row")
                continue
            expected = tuple(float(selected.iloc[0][key]) for key in ("R2", "RMSE", "MAE"))
            differences = tuple(abs(a - b) for a, b in zip(calculated, expected))
            if max(differences) > TOLERANCE:
                failures.append(f"{model}/{split}: metric mismatch {differences}")
            rows.append({
                "Model": model,
                "Dataset": split,
                "N": len(predictions),
                "R2": calculated[0],
                "RMSE": calculated[1],
                "MAE": calculated[2],
            })

    table = pd.DataFrame(rows)
    test = table.loc[table["Dataset"] == "Test"].sort_values("R2", ascending=False)
    print(test.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    if failures:
        print("FAIL:", *failures, sep="\n- ", file=sys.stderr)
        return 1
    print("PASS: all 15 model/split metric rows reproduce from the prediction files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
