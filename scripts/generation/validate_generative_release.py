#!/usr/bin/env python3
"""Validate the compact REINVENT4 configuration and TL corpus."""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]


def require(pattern: str, text: str, label: str) -> None:
    if not re.search(pattern, text, flags=re.MULTILINE):
        raise AssertionError(f"Missing expected setting: {label}")


def main() -> int:
    corpus = ROOT / "corpora" / "bcrabl_transfer_learning_5624.smi"
    smiles = [line.strip() for line in corpus.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(smiles) != 5624:
        raise AssertionError("TL corpus must contain 5,624 non-empty training records")
    if len(set(smiles)) != 5617:
        raise AssertionError("Expected 5,617 distinct SMILES in the historical TL input")

    tl = (ROOT / "configs" / "transfer_learning.toml").read_text(encoding="utf-8")
    rl = (ROOT / "configs" / "reinforcement_learning.toml").read_text(encoding="utf-8")
    sampling = (ROOT / "configs" / "sampling_1m.toml").read_text(encoding="utf-8")
    require(r"num_epochs\s*=\s*20", tl, "20 TL epochs")
    require(r"batch_size\s*=\s*50", tl, "TL batch size 50")
    require(r'type\s*=\s*"dap"', rl, "DAP learning strategy")
    require(r"sigma\s*=\s*128", rl, "sigma 128")
    require(r"rate\s*=\s*0\.0001", rl, "learning rate 1e-4")
    require(r"batch_size\s*=\s*64", rl, "RL batch size 64")
    require(r'type\s*=\s*"IdenticalMurckoScaffold"', rl, "Murcko diversity filter")
    require(r"bucket_size\s*=\s*25", rl, "bucket size 25")
    require(r"minscore\s*=\s*0\.5", rl, "minimum diversity-filter score 0.5")
    require(r"max_score\s*=\s*0\.8", rl, "target score 0.80")
    require(r"min_steps\s*=\s*25", rl, "minimum 25 steps")
    require(r"max_steps\s*=\s*1000", rl, "maximum 1,000 steps")
    require(r"num_smiles\s*=\s*1000000", sampling, "one million samples")
    require(r"unique_molecules\s*=\s*true", sampling, "duplicate control")
    require(r"randomize_smiles\s*=\s*false", sampling, "production randomization disabled")
    print("PASS: 5,624 TL records (5,617 distinct SMILES) and final REINVENT4 settings validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
