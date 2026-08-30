#!/usr/bin/env python3
"""Reconstructed BCRABLScore reference implementation.

This file implements manuscript Equations 2–4. It was reconstructed after the
original final source file was unavailable; it must not be described as the
historical script used during the completed REINVENT4 run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "models" / "best_xgboost.pkl"
DEFAULT_TRAIN = ROOT / "data" / "splits" / "train.csv"
FP_RADIUS = 2
FP_SIZE = 2048
ACTIVITY_WEIGHT = 0.80
NOVELTY_WEIGHT = 0.20


class BCRABLReward:
    """Calculate the final activity–novelty reward described in the paper."""

    def __init__(self, model_path: Path = DEFAULT_MODEL, train_path: Path = DEFAULT_TRAIN):
        self.model = joblib.load(model_path)
        self.generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=FP_RADIUS, fpSize=FP_SIZE
        )
        train = pd.read_csv(train_path, usecols=["SMILES"])
        if len(train) != 6373:
            raise ValueError(f"Expected 6,373 QSAR training compounds; found {len(train):,}")
        self.training_fingerprints = []
        for smiles in train["SMILES"]:
            molecule = Chem.MolFromSmiles(str(smiles))
            if molecule is None:
                raise ValueError(f"Invalid training SMILES: {smiles}")
            self.training_fingerprints.append(self.generator.GetFingerprint(molecule))

    def score(self, smiles: str) -> dict[str, float | str | bool]:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return {
                "SMILES": smiles,
                "Valid": False,
                "Predicted_pIC50": 0.0,
                "Activity_score": 0.0,
                "Max_training_similarity": 0.0,
                "Novelty_score": 0.0,
                "Reward": 0.0,
            }

        fingerprint = self.generator.GetFingerprint(molecule)
        features = pd.DataFrame(
            [list(fingerprint)], columns=[f"FP_{index}" for index in range(FP_SIZE)]
        )
        predicted_pic50 = float(self.model.predict(features)[0])

        # Manuscript Equation 2: A_RL = clip((pIC50_XGB - 5) / 5, 0, 1)
        activity_score = float(np.clip((predicted_pic50 - 5.0) / 5.0, 0.0, 1.0))

        # Manuscript Equation 3: N_RL = 1 - max(T_Morgan)
        similarities = DataStructs.BulkTanimotoSimilarity(
            fingerprint, self.training_fingerprints
        )
        maximum_similarity = float(max(similarities))
        novelty_score = 1.0 - maximum_similarity

        # Manuscript Equation 4: R = 0.80 A_RL + 0.20 N_RL
        reward = ACTIVITY_WEIGHT * activity_score + NOVELTY_WEIGHT * novelty_score
        canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)
        return {
            "SMILES": canonical,
            "Valid": True,
            "Predicted_pIC50": predicted_pic50,
            "Activity_score": activity_score,
            "Max_training_similarity": maximum_similarity,
            "Novelty_score": novelty_score,
            "Reward": float(reward),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("smiles", help="One SMILES string to score")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    args = parser.parse_args()
    result = BCRABLReward(args.model, args.train).score(args.smiles)
    for key, value in result.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
