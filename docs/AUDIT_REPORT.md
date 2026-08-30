# Dataset and script audit

## Release decision

The final dataset and fixed splits are internally consistent with the manuscript counts and are suitable for a versioned data release after the missing provenance details listed below are completed. The original 802 MB archive should not be uploaded to GitHub as-is.

## Verified findings

| Check | Result |
|---|---|
| Final compounds | 7,912 rows and 7,912 unique SMILES |
| Compound identifiers | 7,912 unique IDs |
| Source labels | 7,354 BindingDB; 541 both sources; 17 ChEMBL |
| Scaffold classes | 3,340 non-empty + 1 acyclic class = 3,341 |
| Fixed partitions | 6,373 train; 757 validation; 782 test |
| Scaffold classes by split | 2,672 train; 334 validation; 335 test |
| Partition coverage | Every modeling SMILES appears exactly once |
| Scaffold leakage | None detected |
| Python files | 55 total; no syntax errors; 3 empty files |

## Duplicate and storage findings

- The same final modeling CSV occurs in five locations with identical SHA-256 content.
- The scaffold table and each split occur twice with identical content.
- The archive includes approximately 642 MB of Random Forest pickle files; the largest single model file is approximately 383 MB and exceeds GitHub's normal per-file limit.
- Feature matrices, model binaries, temporary outputs, and copied results should not be versioned alongside the compact data release.

## Scientific notes requiring explicit wording

- Empty scaffold fields represent 27 valid acyclic molecules, not failed RDKit parsing.
- The median `pActivity` is the predictive target. Separately aggregated median `IC50_nM` values are not always its exact inverse for repeated measurements.
- The released candidates are computationally generated and prioritized. Docking, MD, MM-GBSA, and predicted pIC50 are not experimental activity measurements.

## Items to resolve before submission

1. Record the exact ChEMBL release/version and access date.
2. Record the exact BindingDB download filename/release and access date.
3. Add the missing ChEMBL retrieval script and the authoritative BindingDB extraction/filtering code.
4. Select one master-building policy and remove the alternative implementation.
5. Freeze the software environment, especially Python, RDKit, pandas, NumPy, and scikit-learn versions.
6. Add the article DOI, repository DOI, and GitHub release URL when assigned.
7. Select an explicit license for author-created code.
