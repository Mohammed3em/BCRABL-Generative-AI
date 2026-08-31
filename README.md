# BCR::ABL1 generative AI study — data and modeling release

This repository contains the curated IC50 dataset, fixed Bemis–Murcko scaffold partitions, predictive-model evidence, and target-focused REINVENT4 transfer-learning and reinforcement-learning artifacts for the manuscript:

> *Generative AI for BCR::ABL1 Inhibitor Design: Predictive Modeling, Target-Focused Fine-Tuning, Reinforcement Learning, and Multi-Level Validation*

**Authors:** Mohammed Abdulaali Sahib and Haydar Mohammad-Salim

## What is included

| Path | Purpose | Records |
|---|---|---:|
| `data/processed/bcrabl_ic50_modeling.csv` | Final compound-level IC50 modeling table | 7,912 |
| `data/processed/bcrabl_ic50_scaffold.csv` | Modeling table with Bemis–Murcko scaffold | 7,912 |
| `data/splits/train.csv` | Fixed training partition | 6,373 |
| `data/splits/validation.csv` | Fixed validation partition | 757 |
| `data/splits/test.csv` | Fixed independent scaffold-test partition | 782 |
| `data/supplementary/Supplementary_Data_S1.xlsx` | Ranked AI-generated candidates and associated descriptors | 502 data records in the main sheet |
| `models/BCRABL_REINVENT_prior.model` | Final target-focused model after transfer learning | 1 model |
| `models/BCRABL_stage1.chkpt` | Final reinforcement-learning checkpoint | 1 checkpoint |
| `scripts/generation/comp_bcrabl.py` | Original REINVENT4 custom scoring component | 1 script |

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
python scripts/generation/validate_generative_release.py
python scripts/prioritization/validate_final_library.py
python scripts/docking/validate_docking_release.py
```

The commands check dataset integrity, independently recalculate R², RMSE, and MAE from the released prediction files, verify the released generative model artifacts and settings, and validate the final 50,000-member library and docking tables.

## Predictive modeling

Five regressors were evaluated on the same fixed scaffold partitions: XGBoost, Random Forest, GIN, ChemBERTa-77M-MLM, and IBM MolFormer-XL. XGBoost had the strongest independent-test performance (R² = 0.7516, RMSE = 0.8064, MAE = 0.5989), followed by Random Forest (R² = 0.7377, RMSE = 0.8286, MAE = 0.6028). Exact prediction tables, training histories where available, optimized tree-model parameters, and the benchmark summary are stored under `results/modeling/`.

The scripts in `scripts/modeling/study_scripts/` preserve the code used during model development. See `docs/QSAR_REPRODUCIBILITY.md` before attempting full retraining because the original scripts retain their working-directory assumptions and deep models require separate GPU-capable environments.

## Generative modeling

The historical transfer-learning input contains 5,624 records (5,617 distinct SMILES). REINVENT4 transfer learning used 20 epochs, a batch size of 50, and CUDA acceleration. Reinforcement learning used the DAP strategy (learning rate 1 × 10⁻⁴, sigma 128, batch size 64) with randomized SMILES augmentation. The final reward was `0.80 × activity + 0.20 × novelty`, with invalid structures assigned zero. The `IdenticalMurckoScaffold` diversity filter used a bucket size of 25 and minimum score 0.5. The production generation campaign sampled 1,000,000 sequences before validity and uniqueness processing.

The original REINVENT4 custom scoring component is provided as `scripts/generation/comp_bcrabl.py`. Its model and training-data paths were adapted to the repository layout; the scoring equations and REINVENT4 component interface were preserved. The released target-focused transfer-learning prior and final RL checkpoint are stored under `models/`. See `docs/GENERATIVE_REPRODUCIBILITY.md`.

## Generation validation and prioritization

Production sampling yielded 723,314 valid unique molecules spanning 499,968 Bemis–Murcko scaffolds. Internal diversity was 0.877; 93.58% of generated molecules had maximum Morgan similarity below 0.60 to the full 7,912-compound experimental reference, and 99.83% were non-exact reproductions of the transfer-learning input.

Progressive multi-objective prioritization retained 494,897 molecules after physicochemical/Lipinski eligibility, 367,801 after consensus predicted pIC50 ≥ 5.0, 311,264 after Morgan similarity < 0.50, 209,422 after Murcko similarity < 0.50, 188,232 after SA ≤ 4.0, and 181,719 after PAINS removal. The released 50,000-member pre-docking library was selected by the final priority score while retaining the highest-ranked representative from each scaffold class. See `docs/PRIORITIZATION.md` and `data/prioritized/bcrabl_final_50000.csv`.

## Structure-based screening

The final 50,000-member library was screened against inactive DFG-out ABL1 (PDB 2HYY) using Glide HTVS, SP, and XP. Redocking of crystallographic imatinib gave a heavy-atom RMSD of 0.877 Å. The workflow advanced 10,280 compounds to SP, approximately 1,030 after an SP GlideScore threshold of −13.0 and complementary selection criteria, and 500 compounds to XP. Supplementary Data S1 contains the complete XP500 set plus imatinib and ponatinib. The ten highest-ranked AI-generated candidates are in `results/docking/top10_ai_candidates.csv`.

## Data provenance

Experimental records originated from ChEMBL target `CHEMBL2096618` and complementary BindingDB records. Exact quantitative values reported in nM were retained, censored observations were removed, structures were standardized with RDKit, and modeling was restricted to IC50. See [`docs/PROVENANCE.md`](docs/PROVENANCE.md) for the documented processing chain and current reproducibility limits.

## Repository scope

This repository is a publication-oriented release, not the complete working directory. Duplicate copies, raw database exports, large fingerprint matrices, most fitted-model binaries, temporary outputs, raw docking project files, and MD trajectories are intentionally excluded. The compact optimized XGBoost surrogate, target-focused transfer-learning prior, final RL checkpoint, original custom scoring component, transfer-learning corpus, and TL/RL configurations are included. Retraining transfer learning from the generic REINVENT prior requires the standard pretrained `reinvent.prior` distributed with REINVENT4 v4.8.24.

## Citation

Please cite the associated article after publication. Machine-readable author, title, version, and repository metadata are provided in `CITATION.cff`; add the article DOI when assigned.

## License and source attribution

See [`LICENSES.md`](LICENSES.md). Author-created code and documentation are released under the MIT License; upstream databases retain their respective terms and must be attributed.
