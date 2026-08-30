"""
===============================================================================
Project : BCRABL-AI

File:
    chemberta_dataset.py

Purpose:
    Prepare ChemBERTa datasets for pActivity prediction

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

MODEL_NAME = "DeepChem/ChemBERTa-77M-MLM"

MAX_LENGTH = 256

BATCH_SIZE = 16

# =============================================================================
# Project Paths
# =============================================================================

DATA_DIR = Path(

    "01_Data/Modeling/Split"

)

TRAIN_FILE = DATA_DIR / "train.csv"

VALID_FILE = DATA_DIR / "validation.csv"

TEST_FILE = DATA_DIR / "test.csv"

# =============================================================================
# Load CSV Files
# =============================================================================

train_df = pd.read_csv(TRAIN_FILE)

valid_df = pd.read_csv(VALID_FILE)

test_df = pd.read_csv(TEST_FILE)

# =============================================================================
# Keep Required Columns
# =============================================================================

REQUIRED_COLUMNS = [

    "SMILES",

    "pActivity"

]

train_df = train_df[REQUIRED_COLUMNS].copy()

valid_df = valid_df[REQUIRED_COLUMNS].copy()

test_df = test_df[REQUIRED_COLUMNS].copy()

# =============================================================================
# Rename Target
# =============================================================================

train_df.rename(

    columns={

        "pActivity": "labels"

    },

    inplace=True

)

valid_df.rename(

    columns={

        "pActivity": "labels"

    },

    inplace=True

)

test_df.rename(

    columns={

        "pActivity": "labels"

    },

    inplace=True

)

print("=" * 80)
print("CHEMBERTA DATASET")
print("=" * 80)

print()

print(f"Training Samples   : {len(train_df):,}")
print(f"Validation Samples : {len(valid_df):,}")
print(f"Testing Samples    : {len(test_df):,}")

# =============================================================================
# Tokenizer
# =============================================================================

tokenizer = AutoTokenizer.from_pretrained(

    MODEL_NAME,

    use_fast=True

)

data_collator = DataCollatorWithPadding(

    tokenizer=tokenizer,

    return_tensors="pt"

)
# =============================================================================
# Convert to Hugging Face Dataset
# =============================================================================

train_dataset = Dataset.from_pandas(

    train_df,

    preserve_index=False

)

valid_dataset = Dataset.from_pandas(

    valid_df,

    preserve_index=False

)

test_dataset = Dataset.from_pandas(

    test_df,

    preserve_index=False

)

# =============================================================================
# Tokenization
# =============================================================================

def tokenize_function(batch):

    return tokenizer(

        batch["SMILES"],

        truncation=True,

        max_length=MAX_LENGTH

    )

train_dataset = train_dataset.map(

    tokenize_function,

    batched=True,

    desc="Tokenizing Training"

)

valid_dataset = valid_dataset.map(

    tokenize_function,

    batched=True,

    desc="Tokenizing Validation"

)

test_dataset = test_dataset.map(

    tokenize_function,

    batched=True,

    desc="Tokenizing Testing"

)

# =============================================================================
# Preserve SMILES
# =============================================================================

train_smiles = train_df["SMILES"].tolist()

valid_smiles = valid_df["SMILES"].tolist()

test_smiles = test_df["SMILES"].tolist()

# =============================================================================
# Remove Original Text Column
# =============================================================================

train_dataset = train_dataset.remove_columns(

    "SMILES"

)

valid_dataset = valid_dataset.remove_columns(

    "SMILES"

)

test_dataset = test_dataset.remove_columns(

    "SMILES"

)

print()

print("=" * 80)

print("TOKENIZATION FINISHED")

print("=" * 80)
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
    print("CHEMBERTA DATASET READY")
    print("=" * 80)

    print()

    print("Training Samples   :", len(train_dataset))
    print("Validation Samples :", len(valid_dataset))
    print("Testing Samples    :", len(test_dataset))

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
    print("CHEMBERTA DATASET COMPLETED")
    print("=" * 80)
