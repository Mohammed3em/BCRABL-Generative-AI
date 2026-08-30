# Generative-model reproducibility

## Final workflow represented here

- REINVENT4 v4.8.24
- Transfer-learning input: 5,624 SMILES records (5,617 distinct text representations)
- Transfer learning: 20 epochs, batch size 50, CUDA
- RL strategy: DAP, learning rate 1 × 10⁻⁴, sigma 128, batch size 64, randomized SMILES augmentation
- Reward: `R = 0.80 A_RL + 0.20 N_RL`
- Activity: `A_RL = clip((predicted pIC50_XGB - 5) / 5, 0, 1)`
- Novelty: `N_RL = 1 - maximum Morgan Tanimoto similarity` to the 6,373-compound QSAR training partition
- Morgan fingerprints: radius 2, 2,048 bits
- Diversity filter: `IdenticalMurckoScaffold`, bucket size 25, minimum score 0.5
- Termination: target score 0.80, minimum 25 steps, maximum 1,000 steps
- Production sampling: 1,000,000 sequences, duplicate control on, randomized SMILES off

## Reconstructed reward source

The original final custom-component source file was not retained. The available historical script implemented an earlier, incompatible reward containing QED and is deliberately excluded. `bcrabl_reward_reconstructed.py` was created from manuscript Equations 2–4 and the final study settings. It reproduces the documented mathematical score, but it is not evidence of the exact historical source code or software interface used for the completed RL run.

This distinction must remain visible in the public repository and Data Availability documentation. The reconstructed implementation should not be described as an original recovered file.

The released TOML files preserve the final scientific settings; only input/output paths were adapted to the repository directory structure.

### Historical TL-input note

The exact `FINAL_clean3` file referenced by the completed transfer-learning configuration contains 5,624 non-empty records and seven repeated SMILES strings (5,617 distinct lines). The file is preserved unchanged to represent the actual training input. This small duplication does not alter the reported input-row count, but the manuscript should avoid stating that all duplicate representations had been removed unless the authors can document a different final input file.

## Excluded derived artifacts

Intermediate TL checkpoints, duplicated model copies, exploratory scripts, figure-generation scripts, and large pairwise-similarity arrays are excluded. The compact release includes the final XGBoost surrogate required by the reconstructed reward implementation. The final REINVENT prior/agent weights are not included, so the TOML files preserve the final settings but are not an end-to-end executable reproduction from a fresh clone.
