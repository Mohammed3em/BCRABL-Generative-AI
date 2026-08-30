# Data dictionary

## Processed modeling data

| Column | Type | Meaning |
|---|---|---|
| `Compound_ID` | string | Study-local stable identifier assigned after activity sorting |
| `SMILES` | string | Standardized canonical molecular representation used in the study |
| `IC50_nM` | float | Median reported IC50 in nanomolar units for the standardized structure |
| `pActivity` | float | Median pIC50 used as the regression target |
| `Measurements` | integer | Number of retained IC50 records aggregated for the structure |
| `Sources` | string | Contributing database provenance: `ChEMBL`, `BindingDB`, or both |
| `Scaffold` | string | Achiral Bemis–Murcko scaffold SMILES; empty for valid acyclic molecules |

The split files contain `SMILES` and `pActivity`. They are fixed study partitions and should not be regenerated for benchmark comparisons.

## Supplementary Data S1

The workbook contains the complete cleaned XP-ranked table (`XP_Unique_502`), a top-candidate table (`Top10_Main`), and a README sheet. The workbook distinguishes AI-generated candidates from reference inhibitors and reports docking, consensus prediction, similarity, physicochemical, and chemical-quality fields. Docking scores and predicted pIC50 values are computational prioritization measures, not experimental validation.
