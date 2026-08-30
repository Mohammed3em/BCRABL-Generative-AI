# Predictive-model reproducibility

## Released evidence

The release includes the exact train, validation, and independent-test predictions for all five models, their reported performance summaries, available training histories, XGBoost five-fold cross-validation results, optimized XGBoost and Random Forest parameters, and the final benchmark table. `scripts/modeling/verify_reported_metrics.py` recalculates every reported R², RMSE, and MAE value directly from the prediction tables.

## Common evaluation design

All models used the same fixed Bemis–Murcko partitions: 6,373 training compounds, 757 validation compounds, and 782 independent-test compounds. XGBoost and Random Forest used 2,048-bit Morgan fingerprints with radius 2. Optuna optimization was restricted to the training partition. The independent-test partition was not used for fitting or hyperparameter selection.

## Model families

| Released label | Model | Representation |
|---|---|---|
| `XGBoost_Final` | XGBoost regressor | Morgan radius 2, 2,048 bits |
| `Random_Forest_Final` | Random Forest regressor | Morgan radius 2, 2,048 bits |
| `GNN` | Graph Isomorphism Network | Molecular atom-bond graph |
| `ChemBERTa` | ChemBERTa-77M-MLM regression head | Canonical SMILES |
| `MolFormer` | IBM MolFormer-XL regression head | Canonical SMILES |

## Full retraining boundary

The preserved development scripts are scientifically useful but are not yet a single environment-independent command-line workflow. They retain paths from the original Windows project layout, and the deep-learning workflows depend on pretrained checkpoints, GPU configuration, and library versions that were not fully frozen in the uploaded archive. For this reason:

1. The prediction tables and independent metric verifier are the authoritative reproducibility layer in this release.
2. Do not state that every model can be reproduced bit-for-bit from a fresh clone yet.
3. Before final public release, record the exact Python, RDKit, scikit-learn, XGBoost, PyTorch, PyTorch Geometric, Transformers, and model-checkpoint versions.
4. The compact optimized XGBoost surrogate required by the reconstructed reward is included. Other fitted-model binaries are excluded from GitHub; the two Random Forest files are approximately 258 MB and 383 MB.

## Interpretation

The metrics establish predictive performance on a scaffold-separated test set, not unrestricted out-of-distribution performance. Predicted pIC50 values are computational estimates and should not be presented as experimental activity measurements.
