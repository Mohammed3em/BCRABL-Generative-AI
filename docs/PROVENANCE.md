# Provenance and reproducibility notes

## Documented processing chain

1. Retrieve records for human BCR::ABL1 (`CHEMBL2096618`) from ChEMBL and complementary records from BindingDB.
2. Retain valid structures and exact quantitative measurements in nM; remove censored values.
3. Keep IC50, EC50, Ki, and Kd in the integrated master table, but restrict predictive modeling to IC50.
4. Parse and sanitize structures with RDKit, remove CXSMILES annotations, and canonicalize SMILES.
5. Group retained IC50 records by canonical SMILES; use median pIC50 as the modeling response and retain source provenance.
6. Generate achiral Bemis–Murcko scaffolds and assign each scaffold class to one fixed partition using seed 42.

## Current reproducibility boundary

The uploaded working archive contains final and intermediate tables plus many analysis scripts, but it does not contain a complete, unambiguous acquisition-to-release workflow:

- the ChEMBL programmatic retrieval script was not present;
- `04_filter_bcrabl_targets.py` and `Methods_Source/01_Dataset/script/01_build_dataset_master.py` are empty;
- two different master-building scripts implement different duplicate policies;
- many scripts assume a directory depth that does not match their location in the uploaded archive;
- database release identifiers and exact access dates are not stored in a machine-readable metadata file.

Consequently, this release preserves the exact final modeling tables and fixed partitions and provides an independent validator. Before journal submission, add the ChEMBL retrieval code, the BindingDB source filename/release, database access dates, software versions, and the single authoritative curation pipeline. Do not claim that the raw-to-final dataset is fully reproducible until those items are supplied.

## Upstream attribution

- ChEMBL target: `CHEMBL2096618`
- BindingDB: complementary BCR::ABL1 bioactivity records

The final table retains record-level source labels. Database-specific assay identifiers and publication identifiers remain in the raw/intermediate working data and are not present at compound level in this compact release.
