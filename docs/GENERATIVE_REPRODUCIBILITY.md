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
- Production generation: 1,000,000 sampled sequences before validity and uniqueness processing

## Original reward source

The original REINVENT4 custom component used for the completed RL run was recovered and is released as `scripts/generation/comp_bcrabl.py`. It implements manuscript Equations 2–4 through the `BCRABLScore` endpoint. Only the historical absolute Windows paths for the XGBoost model and QSAR training partition were adapted to repository-relative paths; the fingerprint parameters, activity normalization, novelty calculation, reward weights, invalid-SMILES handling, and REINVENT4 interface are unchanged.

The released TOML files preserve the transfer-learning and reinforcement-learning settings; input/output paths were adapted to the repository directory structure. `models/BCRABL_REINVENT_prior.model` is the final 20-epoch target-focused transfer-learning prior. In staged learning, this prior initializes the agent, and `models/BCRABL_stage1.chkpt` is the resulting RL checkpoint.

### Historical TL-input note

The exact `FINAL_clean3` file referenced by the completed transfer-learning configuration contains 5,624 non-empty records and seven repeated SMILES strings (5,617 distinct lines). The file is preserved unchanged to represent the actual training input. This small duplication does not alter the reported input-row count, but the manuscript should avoid stating that all duplicate representations had been removed unless the authors can document a different final input file.

## Included and excluded model artifacts

The release includes the optimized XGBoost activity surrogate, final target-focused TL prior, final RL checkpoint, original reward component, historical TL corpus, and TL/RL configurations. Intermediate TL checkpoints, redundant initialization copies, exploratory scripts, figure-generation scripts, and large pairwise-similarity arrays are excluded. The generic pretrained `reinvent.prior` is part of the standard REINVENT4 distribution and is required only to repeat transfer learning from its original starting point.
