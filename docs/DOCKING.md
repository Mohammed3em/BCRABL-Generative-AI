# Hierarchical structure-based screening

The final AI-prioritized library was screened with Glide against inactive DFG-out ABL1 (PDB 2HYY). The receptor was prepared with the Schrödinger Protein Preparation Wizard, and the grid was centered on the crystallographic imatinib-binding region spanning the ATP site and adjacent hydrophobic pocket. Redocking was considered successful at heavy-atom RMSD <2.0 Å and produced 0.877 Å.

Screening followed HTVS → SP → XP. Of 50,000 candidates, 10,280 advanced to SP. An SP GlideScore threshold of ≤−13.0 was combined with consensus-predicted pIC50, XGBoost–Random Forest agreement, and chemical diversity to yield approximately 1,030 candidates. Diversity-aware down-selection retained 500 compounds for XP, with the best-scoring pose retained per compound. Imatinib and ponatinib were processed using the same XP protocol.

Docking scores are comparative within-protocol prioritization values, not experimental potency or binding free-energy measurements.

## Released evidence and boundary

- Supplementary Data S1: 500 unique AI-generated XP records plus two references
- Top ten AI-generated XP candidates and associated AI/ADME descriptors
- Reference XP scores and redocking summary
- Independent validator for the XP workbook and Top10 table

The original HTVS and SP Glide raw exports, receptor grid files, and Maestro project files were not present in the audited archive. Therefore, the intermediate counts are documented from the final manuscript but cannot be independently regenerated from this repository alone. These proprietary-software artifacts may be archived separately if required and redistribution is permitted.
