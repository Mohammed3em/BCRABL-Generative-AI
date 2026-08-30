# Manuscript alignment notes

These items were identified while auditing the exact released files. Apply them to the manuscript together during final repository alignment.

## 1. Transfer-learning input duplication

The configured historical TL file contains 5,624 records but 5,617 distinct SMILES lines. Seven representations are repeated. Revise wording that currently implies complete removal of all duplicates, unless another documented final training file is identified.

Suggested wording:

> The resulting transfer-learning input contained 5,624 molecular records after generator-specific preprocessing and vocabulary filtering.

## 2. Final scaffold selection

The final 50,000-member CSV and recovered selection script show one highest-ranked representative per scaffold class. This conflicts with the current phrase “without enforcing a strict one-compound-per-scaffold rule.”

Suggested Methods wording:

> Bemis–Murcko scaffold information was incorporated during final down-selection by retaining the highest-ranked representative from each scaffold class, thereby limiting enrichment of closely related chemical families.

Suggested Results sentence:

> The final 50,000-compound library represented 50,000 distinct scaffold classes when the acyclic class was counted explicitly.
