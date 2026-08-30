# Generation validation and multi-objective prioritization

Production sampling generated 723,314 valid unique molecules. Novelty and memorization were evaluated separately: reference-space novelty used all 7,912 curated experimental compounds, whereas exact memorization used the historical 5,624-record transfer-learning input. Internal diversity used two reproducible 5,000-molecule samples (seeds 42 and 123) and 2,048-bit radius-2 Morgan fingerprints.

The characterized population was filtered sequentially using consensus predicted pIC50, whole-molecule similarity, scaffold similarity, synthetic accessibility, Lipinski violations, and PAINS. The final score weighted normalized activity (0.45), molecular novelty (0.25), scaffold novelty (0.20), and synthetic accessibility (0.10). The highest-ranked representative from each scaffold class was retained for the final 50,000-member pre-docking library.

The full 723,314-molecule generated table is not included in GitHub because it is a large derived artifact. The final 50,000-member library, exact filter counts, summaries, and validation code are included. A versioned archival repository may be used for the full generated table if required by peer review.
