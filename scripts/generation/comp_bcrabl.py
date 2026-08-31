from __future__ import annotations

__all__ = ["BCRABLScore"]

from typing import List
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from .component_results import ComponentResults
from .add_tag import add_tag


# =============================================================================
# Load XGBoost model
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]

MODEL = joblib.load(ROOT / "models" / "best_xgboost.pkl")


# =============================================================================
# Morgan fingerprint generator
# =============================================================================

GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048
)


# =============================================================================
# Training fingerprints for novelty
# =============================================================================

TRAIN = pd.read_csv(ROOT / "data" / "splits" / "train.csv")


TRAIN_FPS = []

for smi in TRAIN["SMILES"]:

    mol = Chem.MolFromSmiles(smi)

    if mol:

        TRAIN_FPS.append(
            GENERATOR.GetFingerprint(mol)
        )


# =============================================================================
# REINVENT parameters
# =============================================================================

@add_tag("__parameters")
class Parameters:
    pass


# =============================================================================
# BCR-ABL Scoring Component
# =============================================================================

@add_tag("__component")
class BCRABLScore:


    def __init__(self, params=None):
        pass


    def __call__(self, smilies: List[str]):

        scores = []


        for smi in smilies:


            mol = Chem.MolFromSmiles(smi)


            # Invalid SMILES

            if mol is None:

                scores.append(0.0)

                continue



            # -------------------------------------------------------------
            # Morgan fingerprint
            # -------------------------------------------------------------

            fp = GENERATOR.GetFingerprint(mol)



            # -------------------------------------------------------------
            # XGBoost activity prediction
            # -------------------------------------------------------------

            X = pd.DataFrame(
                [list(fp)],
                columns=[f"FP_{i}" for i in range(2048)]
            )


            activity = MODEL.predict(X)[0]


            activity_score = np.clip(
                (activity - 5) / 5,
                0,
                1
            )



            # -------------------------------------------------------------
            # Novelty calculation
            # -------------------------------------------------------------

            sims = [

                DataStructs.TanimotoSimilarity(fp, train_fp)

                for train_fp in TRAIN_FPS

            ]


            novelty = 1 - max(sims)



            # -------------------------------------------------------------
            # Final reward
            # Activity focused
            # -------------------------------------------------------------

            reward = (

                0.80 * activity_score

                +

                0.20 * novelty

            )


            scores.append(
                float(reward)
            )



        # Debug

        print("INPUT:", len(smilies))

        print("SCORES:", len(scores))



        return ComponentResults(

            [

                np.array(
                    scores,
                    dtype=np.float32
                )

            ]

        )
