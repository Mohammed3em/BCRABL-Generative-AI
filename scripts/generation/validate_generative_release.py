#!/usr/bin/env python3
"""Validate the compact REINVENT4 configuration and TL corpus."""

from pathlib import Path
import hashlib
import pickletools
import re
import zipfile


ROOT = Path(__file__).resolve().parents[2]


def require(pattern: str, text: str, label: str) -> None:
    if not re.search(pattern, text, flags=re.MULTILINE):
        raise AssertionError(f"Missing expected setting: {label}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archive_strings(path: Path) -> set[str]:
    with zipfile.ZipFile(path) as archive:
        payload_name = next(name for name in archive.namelist() if name.endswith("/data.pkl"))
        payload = archive.read(payload_name)
    return {
        argument
        for opcode, argument, _ in pickletools.genops(payload)
        if opcode.name in {"BINUNICODE", "SHORT_BINUNICODE", "UNICODE"}
        and isinstance(argument, str)
    }


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
    require(
        r'model_file\s*=\s*"models/BCRABL_stage1\.chkpt"',
        sampling,
        "production sampling from the final RL checkpoint",
    )

    prior = ROOT / "models" / "BCRABL_REINVENT_prior.model"
    rl_checkpoint = ROOT / "models" / "BCRABL_stage1.chkpt"
    reward_source = ROOT / "scripts" / "generation" / "comp_bcrabl.py"
    for artifact in (prior, rl_checkpoint, reward_source, ROOT / "models" / "best_xgboost.pkl"):
        if not artifact.is_file() or artifact.stat().st_size == 0:
            raise AssertionError(f"Missing or empty generative artifact: {artifact.relative_to(ROOT)}")
    if sha256(prior) == sha256(rl_checkpoint):
        raise AssertionError("The RL checkpoint must differ from the transfer-learning prior")
    if "RL" not in archive_strings(rl_checkpoint):
        raise AssertionError("The released checkpoint does not contain RL provenance metadata")

    reward = reward_source.read_text(encoding="utf-8")
    require(r"radius\s*=\s*2", reward, "Morgan radius 2")
    require(r"fpSize\s*=\s*2048", reward, "2,048-bit Morgan fingerprint")
    require(r"0\.80\s*\*\s*activity_score", reward, "activity reward weight 0.80")
    require(r"0\.20\s*\*\s*novelty", reward, "novelty reward weight 0.20")
    print("PASS: 5,624 TL records (5,617 distinct SMILES), final REINVENT4 settings, original reward source, TL prior, and RL checkpoint validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
