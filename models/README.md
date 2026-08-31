# Model artifacts

`best_xgboost.pkl` is the optimized XGBoost activity surrogate loaded by the original `BCRABLScore` custom component.

- `BCRABL_REINVENT_prior.model` is the final target-focused model after 20 transfer-learning epochs.
- `BCRABL_stage1.chkpt` is the final staged-learning checkpoint containing the RL-updated network and diversity-filter state.

The historical files named `BCRABL_REINVENT_agent.model` and `BCRABL_REINVENT_agent_RL1000steps_backup.model.model` were byte-identical to the TL prior. They represented the agent initialization rather than the RL-updated endpoint and are intentionally omitted to avoid misleading duplication. Production sampling loads `BCRABL_stage1.chkpt`.

Other fitted QSAR model binaries are also excluded because the released prediction tables and independent metric verifier provide the compact reproducibility layer for the reported benchmark.
