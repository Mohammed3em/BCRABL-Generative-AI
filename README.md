# BCR::ABL1 generative AI study — data and modeling release

This repository contains the curated IC50 dataset, fixed Bemis–Murcko scaffold partitions, predictive-model scripts and evaluation outputs, and Supplementary Data S1 for the manuscript:

> *Generative AI for BCR::ABL1 Inhibitor Design: Predictive Modeling, Target-Focused Fine-Tuning, Reinforcement Learning, and Multi-Level Validation*

Author: Mohammed Abdulaali Sahib, Department of Pharmaceutical Chemistry, College of Pharmacy, University of Kerbala, Iraq. ORCID: [0009-0006-9992-5653](https://orcid.org/0009-0006-9992-5653).

## What is included

| Path | Purpose | Records |
|---|---|---:|
| `data/processed/bcrabl_ic50_modeling.csv` | Final compound-level IC50 modeling table | 7,912 |
| `data/processed/bcrabl_ic50_scaffold.csv` | Modeling table with Bemis–Murcko scaffold | 7,912 |
| `data/splits/train.csv` | Fixed training partition | 6,373 |
| `data/splits/validation.csv` | Fixed validation partition | 757 |
| `data/splits/test.csv` | Fixed independent scaffold-test partition | 782 |
| `data/supplementary/Supplementary_Data_S1.xlsx` | Ranked AI-generated candidates and associated descriptors | 502 data records in the main sheet |

The partitions contain 2,672, 334, and 335 scaffold classes, respectively, with no scaffold shared across subsets.

## Important scaffold convention

Twenty-seven valid acyclic molecules have an empty Bemis–Murcko scaffold string. They are treated as one explicit **acyclic scaffold class** during counting and leakage checks. Therefore, the full dataset contains 3,341 scaffold classes: 3,340 non-empty scaffold strings plus one acyclic class. All 27 acyclic molecules occur in the validation partition.

## Activity aggregation convention

For structures with repeated IC50 measurements, `pActivity` and `IC50_nM` were each aggregated by their median. Because the logarithm is nonlinear, the two medians are not always exact mathematical inverses when a compound has an even number of measurements. Predictive modeling used `pActivity` (median pIC50) as stated in the manuscript; `IC50_nM` is retained as a descriptive median.

## Validate the release

Install the lightweight validation dependencies and run:

```bash
python -m pip install -r requirements.txt
python scripts/validate_release.py
python scripts/modeling/verify_reported_metrics.py
```

The commands check dataset integrity and independently recalculate R², RMSE, and MAE from the released prediction files.

## Predictive modeling

Five regressors were evaluated on the same fixed scaffold partitions: XGBoost, Random Forest, GIN, ChemBERTa-77M-MLM, and IBM MolFormer-XL. XGBoost had the strongest independent-test performance (R² = 0.7516, RMSE = 0.8064, MAE = 0.5989), followed by Random Forest (R² = 0.7377, RMSE = 0.8286, MAE = 0.6028). Exact prediction tables, training histories where available, optimized tree-model parameters, and the benchmark summary are stored under `results/modeling/`.

The scripts in `scripts/modeling/study_scripts/` preserve the code used during model development. See `docs/QSAR_REPRODUCIBILITY.md` before attempting full retraining because the original scripts retain their working-directory assumptions and deep models require separate GPU-capable environments.

## Data provenance

Experimental records originated from ChEMBL target `CHEMBL2096618` and complementary BindingDB records. Exact quantitative values reported in nM were retained, censored observations were removed, structures were standardized with RDKit, and modeling was restricted to IC50. See [`docs/PROVENANCE.md`](docs/PROVENANCE.md) for the documented processing chain and current reproducibility limits.

## Repository scope

This repository is a publication-oriented release, not the complete 802 MB working directory. Duplicate copies, raw database exports, fingerprint matrices, fitted model binaries, temporary outputs, docking files, and MD trajectories are intentionally excluded. Large reusable artifacts should be deposited in a versioned research repository such as Zenodo rather than committed directly to GitHub.

## Citation

Please cite the associated article after publication. Machine-readable author and title metadata are provided in `CITATION.cff`; replace the placeholder article DOI and repository DOI when assigned.

## License and source attribution

See [`LICENSES.md`](LICENSES.md). The upstream databases retain their respective terms and must be attributed. No license for the author-created code has been selected in this draft release.
