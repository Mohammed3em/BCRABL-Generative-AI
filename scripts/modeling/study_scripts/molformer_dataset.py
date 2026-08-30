"""
===============================================================================
Project : BCRABL-AI

File:
    34_molformer_dataset.py

Purpose:
    MolFormer Dataset

===============================================================================
"""

from pathlib import Path

import pandas as pd

from datasets import Dataset

from transformers import (

    AutoTokenizer,

    DataCollatorWithPadding

)

# =============================================================================
# Configuration
# =============================================================================

MODEL_NAME = "ibm/MoLFormer-XL-both-10pct"

BATCH_SIZE = 16

MAX_LENGTH = 256

TRAIN_FILE = Path(

    "01_Data/Modeling/Split/train.csv"

)

VALID_FILE = Path(

    "01_Data/Modeling/Split/validation.csv"

)

TEST_FILE = Path(

    "01_Data/Modeling/Split/test.csv"

)

print("=" * 80)
print("MOLFORMER DATASET")
print("=" * 80)
print()
# =============================================================================
# Load Dataset
# =============================================================================

train_df = pd.read_csv(TRAIN_FILE)

valid_df = pd.read_csv(VALID_FILE)

test_df = pd.read_csv(TEST_FILE)

print(f"Training Samples   : {len(train_df):,}")

print(f"Validation Samples : {len(valid_df):,}")

print(f"Testing Samples    : {len(test_df):,}")

# =============================================================================
# Save SMILES
# =============================================================================

train_smiles = train_df["SMILES"].tolist()

valid_smiles = valid_df["SMILES"].tolist()

test_smiles = test_df["SMILES"].tolist()

# =============================================================================
# HuggingFace Dataset
# =============================================================================

train_dataset = Dataset.from_pandas(

    train_df

)

valid_dataset = Dataset.from_pandas(

    valid_df

)

test_dataset = Dataset.from_pandas(

    test_df

)

# =============================================================================
# Tokenizer
# =============================================================================

tokenizer = AutoTokenizer.from_pretrained(

    MODEL_NAME,

    trust_remote_code=True

)
# =============================================================================
# Tokenization Function
# =============================================================================

def tokenize(batch):

    return tokenizer(

        batch["SMILES"],

        truncation=True,

        max_length=MAX_LENGTH,

        padding=False

    )

# =============================================================================
# Tokenize Datasets
# =============================================================================

train_dataset = train_dataset.map(

    tokenize,

    batched=True,

    batch_size=1000,

    desc="Tokenizing Training"

)

valid_dataset = valid_dataset.map(

    tokenize,

    batched=True,

    batch_size=1000,

    desc="Tokenizing Validation"

)

test_dataset = test_dataset.map(

    tokenize,

    batched=True,

    batch_size=1000,

    desc="Tokenizing Testing"

)

print()

print("=" * 80)
print("TOKENIZATION FINISHED")
print("=" * 80)

# =============================================================================
# Rename Target
# =============================================================================

train_dataset = train_dataset.rename_column(

    "pActivity",

    "labels"

)

valid_dataset = valid_dataset.rename_column(

    "pActivity",

    "labels"

)

test_dataset = test_dataset.rename_column(

    "pActivity",

    "labels"

)
# =============================================================================
# Data Collator
# =============================================================================

data_collator = DataCollatorWithPadding(

    tokenizer=tokenizer,

    padding=True,

    return_tensors="pt"

)

# =============================================================================
# PyTorch Format
# =============================================================================

train_dataset.set_format(

    type="torch",

    columns=[

        "input_ids",

        "attention_mask",

        "labels"

    ]

)

valid_dataset.set_format(

    type="torch",

    columns=[

        "input_ids",

        "attention_mask",

        "labels"

    ]

)

test_dataset.set_format(

    type="torch",

    columns=[

        "input_ids",

        "attention_mask",

        "labels"

    ]

)

# =============================================================================
# Export
# =============================================================================

__all__ = [

    "MODEL_NAME",

    "MAX_LENGTH",

    "BATCH_SIZE",

    "tokenizer",

    "data_collator",

    "train_dataset",

    "valid_dataset",

    "test_dataset",

    "train_smiles",

    "valid_smiles",

    "test_smiles"

]
# =============================================================================
# Dataset Summary
# =============================================================================

if __name__ == "__main__":

    print()

    print("=" * 80)
    print("MOLFORMER DATASET READY")
    print("=" * 80)

    print()

    print(f"Training Samples   : {len(train_dataset):,}")
    print(f"Validation Samples : {len(valid_dataset):,}")
    print(f"Testing Samples    : {len(test_dataset):,}")

    print()

    sample = train_dataset[0]

    print("Dataset Keys")
    print("-" * 80)

    print(sample.keys())

    print()

    print("Input IDs Length :", len(sample["input_ids"]))
    print("Attention Length :", len(sample["attention_mask"]))
    print("Target :", float(sample["labels"]))

    print()

    print("=" * 80)
    print("MOLFORMER DATASET COMPLETED")
    print("=" * 80)