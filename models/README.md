# Model artifacts

`best_xgboost.pkl` is the optimized XGBoost activity surrogate loaded by the original `BCRABLScore` custom component.

- `BCRABL_REINVENT_prior.model` is the final target-focused model after 20 transfer-learning epochs.
- `BCRABL_stage1.chkpt` is the final staged-learning checkpoint containing the RL-updated network and diversity-filter state.

Redundant initialization copies are omitted; `BCRABL_stage1.chkpt` is the authoritative released RL endpoint.

Other fitted QSAR model binaries are also excluded because the released prediction tables and independent metric verifier provide the compact reproducibility layer for the reported benchmark.
