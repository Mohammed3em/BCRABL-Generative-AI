# Final repository audit

Audit date: 2026-08-30

## Outcome

The released scientific tables and validation evidence are internally consistent with the manuscript values examined in this audit. All five repository validators pass after the corrections listed below.

## Independently validated

- Curated modeling data: 7,912 unique compounds.
- Scaffold classes: 3,341 when the acyclic class is counted explicitly.
- Fixed partitions: 6,373 train, 757 validation, and 782 independent test; no molecule or scaffold leakage.
- Predictive results: all 15 model-by-split R², RMSE, and MAE rows reproduce from the released prediction files.
- Independent test ranking: XGBoost, Random Forest, MolFormer, GNN, then ChemBERTa.
- Transfer-learning corpus: 5,624 records and 5,617 distinct SMILES lines.
- Final documented RL reward and REINVENT4 settings.
- Final pre-docking library: 50,000 unique molecules and 50,000 scaffold classes when the acyclic class is counted.
- Docking workbook: 500 AI-generated XP records plus imatinib and ponatinib; Top10 and reference scores match the manuscript.

## Corrections included in this package

1. In Supplementary Data S1, `BCRL32452` has HBA = 10. Its Lipinski violation count was corrected from 2 to 1; the existing `Within ≤1 Lipinski Violation = Yes` and `Project Filter Pass = Yes` values are therefore consistent.
2. `metadata/SHA256SUMS` was regenerated after the supplementary workbook correction and expanded to cover the principal released scientific artifacts.
3. `scripts/validate_release.py` now verifies the checksum manifest in addition to dataset integrity.
4. The placeholder repository URL in `CITATION.cff` was replaced with the actual GitHub URL, and the release version was set to 1.0.0.
5. The Data Availability Statement was revised for a public GitHub-only release and all placeholder DOI/release fields were removed.
6. README validation commands now include the generative, prioritization, and docking validators.
7. Repository documentation now states explicitly that final REINVENT prior/agent weights, most fitted model binaries, raw docking projects, and MD trajectories are not included. This prevents an unsupported claim of end-to-end rerunnability from a fresh clone.

## Manuscript wording still requiring correction

These are manuscript edits, not repository-data errors:

1. The Methods currently state that duplicate representations were removed before producing the 5,624-member transfer-learning corpus. The preserved input has 5,624 rows but 5,617 distinct SMILES. Recommended replacement: “The resulting transfer-learning input contained 5,624 molecular records after generator-specific preprocessing and vocabulary filtering.”
2. The Methods currently say that final down-selection did not enforce a strict one-compound-per-scaffold rule. The released final CSV and selection logic retain one highest-ranked representative per scaffold class. Recommended replacement: “Bemis–Murcko scaffold information was incorporated during final down-selection by retaining the highest-ranked representative from each scaffold class, thereby limiting enrichment of closely related chemical families.”

## Reproducibility boundary

The repository supports independent checking of the curated data, fixed partitions, reported predictive metrics, final generative settings, prioritization outputs, and final docking tables. It does not support bit-for-bit regeneration of the completed REINVENT run, raw Glide workflow, or MD trajectories because the corresponding historical weights and raw project files are not released. This limitation must remain explicit.
