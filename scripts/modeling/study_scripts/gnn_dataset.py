"""
===============================================================================
Project : BCRABL-AI
Script  : 30_gnn_dataset.py

Purpose
-------
Load scaffold split datasets and convert SMILES into
PyTorch Geometric graph objects.

===============================================================================
"""

from pathlib import Path

import torch
import pandas as pd

from rdkit import Chem

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# =============================================================================
# Dataset Paths
# =============================================================================

ROOT = Path("01_Data/Modeling/Split")

TRAIN_FILE = ROOT / "train.csv"
VALID_FILE = ROOT / "validation.csv"
TEST_FILE  = ROOT / "test.csv"

# =============================================================================
# Read CSV Files
# =============================================================================

train_df = pd.read_csv(TRAIN_FILE)

valid_df = pd.read_csv(VALID_FILE)
test_df  = pd.read_csv(TEST_FILE)

print("=" * 80)
print("DATASET LOADED")
print("=" * 80)

print(f"Train      : {len(train_df):,}")
print(f"Validation : {len(valid_df):,}")
print(f"Test       : {len(test_df):,}")

print("=" * 80)
# =============================================================================
# Atom Features
# =============================================================================

def atom_features(atom):

    return [

        atom.GetAtomicNum(),

        atom.GetDegree(),

        atom.GetFormalCharge(),

        atom.GetTotalNumHs(),

        atom.GetImplicitValence(),

        int(atom.GetIsAromatic()),

        int(atom.IsInRing()),

        int(atom.GetHybridization())

    ]

# =============================================================================
# Bond Features
# =============================================================================

def bond_features(bond):

    return [

        bond.GetBondTypeAsDouble(),

        int(bond.GetIsConjugated()),

        int(bond.IsInRing())

    ]

# =============================================================================
# SMILES → Graph
# =============================================================================

def smiles_to_graph(smiles, target):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:

        return None

    # -------------------------
    # Node Features
    # -------------------------

    x = torch.tensor(

        [

            atom_features(atom)

            for atom in mol.GetAtoms()

        ],

        dtype=torch.float

    )

    # -------------------------
    # Edge Features
    # -------------------------

    edge_index = []

    edge_attr = []

    for bond in mol.GetBonds():

        i = bond.GetBeginAtomIdx()

        j = bond.GetEndAtomIdx()

        bf = bond_features(bond)

        edge_index.append([i, j])
        edge_index.append([j, i])

        edge_attr.append(bf)
        edge_attr.append(bf)

    edge_index = torch.tensor(

        edge_index,

        dtype=torch.long

    ).t().contiguous()

    edge_attr = torch.tensor(

        edge_attr,

        dtype=torch.float

    )

    y = torch.tensor(

        [target],

        dtype=torch.float

    )

    return Data(

        x=x,

        edge_index=edge_index,

        edge_attr=edge_attr,

        y=y

    )
# =============================================================================
# Convert DataFrame → Graph Dataset
# =============================================================================

def dataframe_to_dataset(df):

    dataset = []

    for _, row in df.iterrows():

        graph = smiles_to_graph(

            row["SMILES"],

            row["pActivity"]

        )

        if graph is not None:

            dataset.append(graph)

    return dataset

# =============================================================================
# Build Graph Datasets
# =============================================================================

print("\nConverting SMILES to Graphs ...\n")

train_dataset = dataframe_to_dataset(train_df)

valid_dataset = dataframe_to_dataset(valid_df)

test_dataset = dataframe_to_dataset(test_df)

print("Conversion Completed.\n")

print("=" * 80)
print("GRAPH DATASET SUMMARY")
print("=" * 80)

print(f"Training Graphs   : {len(train_dataset):,}")
print(f"Validation Graphs : {len(valid_dataset):,}")
print(f"Testing Graphs    : {len(test_dataset):,}")

sample = train_dataset[0]

print("\nSample Graph")
print("-" * 80)

print(sample)

print(f"\nNode Features : {sample.x.shape}")
print(f"Edge Index    : {sample.edge_index.shape}")
print(f"Edge Features : {sample.edge_attr.shape}")
print(f"Target Shape  : {sample.y.shape}")

# =============================================================================
# PyTorch Geometric DataLoaders
# =============================================================================

from torch_geometric.loader import DataLoader

BATCH_SIZE = 64

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# =============================================================================
# Export
# =============================================================================

__all__ = [
    "train_loader",
    "valid_loader",
    "test_loader",
    "train_dataset",
    "valid_dataset",
    "test_dataset"
]

print()
print("=" * 80)
print("GNN DATASET READY")
print("=" * 80)