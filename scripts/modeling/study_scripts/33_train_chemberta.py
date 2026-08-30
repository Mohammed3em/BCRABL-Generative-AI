"""
===============================================================================
Project : BCRABL-AI

File:
    33_train_chemberta.py

Purpose:
    Fine-tune ChemBERTa for pActivity Prediction

===============================================================================
"""

from pathlib import Path
import copy
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from torch.optim import AdamW

from transformers import get_linear_schedule_with_warmup

from tqdm import tqdm

from sklearn.metrics import (

    r2_score,

    mean_squared_error,

    mean_absolute_error

)

from chemberta_dataset import (

    train_dataset,

    valid_dataset,

    test_dataset,

    data_collator,

    train_smiles,

    valid_smiles,

    test_smiles,

    BATCH_SIZE

)

from chemberta_model import (

    model,

    DEVICE

)

# =============================================================================
# Reproducibility
# =============================================================================

SEED = 42

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True

torch.backends.cudnn.benchmark = False

# =============================================================================
# DataLoaders
# =============================================================================

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    collate_fn=data_collator

)

valid_loader = DataLoader(

    valid_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    collate_fn=data_collator

)

test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    collate_fn=data_collator

)

# =============================================================================
# Output Directories
# =============================================================================

MODEL_DIR = Path(

    "04_AI_Models/Models"

)

RESULT_DIR = Path(

    "07_Results/ChemBERTa"

)

MODEL_DIR.mkdir(

    parents=True,

    exist_ok=True

)

RESULT_DIR.mkdir(

    parents=True,

    exist_ok=True

)

print("=" * 80)
print("CHEMBERTA TRAINING")
print("=" * 80)

print()

print(f"Device : {DEVICE}")
print(f"Training Samples : {len(train_dataset):,}")
print(f"Validation Samples : {len(valid_dataset):,}")
print(f"Testing Samples : {len(test_dataset):,}")
# =============================================================================
# Hyperparameters
# =============================================================================

EPOCHS = 50

LEARNING_RATE = 1e-5

WEIGHT_DECAY = 0.01

WARMUP_RATIO = 0.10

PATIENCE = 7

# =============================================================================
# Loss Function
# =============================================================================

criterion = nn.MSELoss()

# =============================================================================
# Optimizer
# =============================================================================

optimizer = AdamW(

    model.parameters(),

    lr=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY

)

# =============================================================================
# Learning Rate Scheduler
# =============================================================================

total_training_steps = len(train_loader) * EPOCHS

warmup_steps = int(

    total_training_steps * WARMUP_RATIO

)

scheduler = get_linear_schedule_with_warmup(

    optimizer=optimizer,

    num_warmup_steps=warmup_steps,

    num_training_steps=total_training_steps

)

# =============================================================================
# Mixed Precision
# =============================================================================

scaler = torch.cuda.amp.GradScaler(

    enabled=torch.cuda.is_available()

)

# =============================================================================
# Early Stopping Variables
# =============================================================================

best_validation_loss = np.inf

best_epoch = 0

best_model = None

early_counter = 0

# =============================================================================
# Training History
# =============================================================================

history = {

    "epoch": [],

    "train_loss": [],

    "validation_loss": [],

    "train_r2": [],

    "validation_r2": [],

    "learning_rate": []

}

print()

print("=" * 80)

print("TRAINING CONFIGURATION")

print("=" * 80)

print()

print(f"Epochs          : {EPOCHS}")

print(f"Learning Rate   : {LEARNING_RATE}")

print(f"Weight Decay    : {WEIGHT_DECAY}")

print(f"Warmup Steps    : {warmup_steps}")

print(f"Batch Size      : {BATCH_SIZE}")

print(f"Mixed Precision : {torch.cuda.is_available()}")

print()
# =============================================================================
# Training Function
# =============================================================================

def train_one_epoch():

    model.train()

    running_loss = 0.0

    predictions = []

    targets = []

    progress = tqdm(

        train_loader,

        desc="Training",

        leave=False

    )

    for batch in progress:

        input_ids = batch["input_ids"].to(DEVICE)

        attention_mask = batch["attention_mask"].to(DEVICE)

        labels = batch["labels"].float().to(DEVICE)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(

            enabled=torch.cuda.is_available()

        ):

            outputs = model(

                input_ids=input_ids,

                attention_mask=attention_mask

            )

            loss = criterion(

                outputs,

                labels

            )

        scaler.scale(

            loss

        ).backward()

        scaler.step(

            optimizer

        )

        scaler.update()

        scheduler.step()

        running_loss += loss.item()

        predictions.extend(

            outputs.detach().cpu().numpy()

        )

        targets.extend(

            labels.detach().cpu().numpy()

        )

        progress.set_postfix(

            loss=f"{loss.item():.4f}"

        )

    epoch_loss = running_loss / len(train_loader)

    epoch_r2 = r2_score(

        targets,

        predictions

    )

    return (

        epoch_loss,

        epoch_r2,

        np.array(targets),

        np.array(predictions)

    )
# =============================================================================
# Validation Function
# =============================================================================

@torch.no_grad()

def validate_one_epoch(loader):

    model.eval()

    running_loss = 0.0

    predictions = []

    targets = []

    for batch in tqdm(

        loader,

        desc="Validation",

        leave=False

    ):

        input_ids = batch["input_ids"].to(DEVICE)

        attention_mask = batch["attention_mask"].to(DEVICE)

        labels = batch["labels"].float().to(DEVICE)

        with torch.cuda.amp.autocast(

            enabled=torch.cuda.is_available()

        ):

            outputs = model(

                input_ids=input_ids,

                attention_mask=attention_mask

            )

            loss = criterion(

                outputs,

                labels

            )

        running_loss += loss.item()

        predictions.extend(

            outputs.cpu().numpy()

        )

        targets.extend(

            labels.cpu().numpy()

        )

    epoch_loss = running_loss / len(loader)

    epoch_r2 = r2_score(

        targets,

        predictions

    )

    epoch_rmse = np.sqrt(

        mean_squared_error(

            targets,

            predictions

        )

    )

    epoch_mae = mean_absolute_error(

        targets,

        predictions

    )

    return (

        epoch_loss,

        epoch_r2,

        epoch_rmse,

        epoch_mae,

        np.array(targets),

        np.array(predictions)

    )
# =============================================================================
# Training Loop
# =============================================================================

print()

print("=" * 80)
print("START TRAINING")
print("=" * 80)

for epoch in range(EPOCHS):

    print()

    print(f"Epoch {epoch + 1}/{EPOCHS}")

    # -------------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------------

    train_loss, train_r2, _, _ = train_one_epoch()

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    (
        valid_loss,
        valid_r2,
        valid_rmse,
        valid_mae,
        _,
        _

    ) = validate_one_epoch(

        valid_loader

    )

    current_lr = optimizer.param_groups[0]["lr"]

    history["epoch"].append(

        epoch + 1

    )

    history["train_loss"].append(

        train_loss

    )

    history["validation_loss"].append(

        valid_loss

    )

    history["train_r2"].append(

        train_r2

    )

    history["validation_r2"].append(

        valid_r2

    )

    history["learning_rate"].append(

        current_lr

    )

    print(

        f"Train Loss : {train_loss:.4f} | "

        f"Train R² : {train_r2:.4f}"

    )

    print(

        f"Valid Loss : {valid_loss:.4f} | "

        f"Valid R² : {valid_r2:.4f} | "

        f"RMSE : {valid_rmse:.4f} | "

        f"MAE : {valid_mae:.4f}"

    )

    # -------------------------------------------------------------------------
    # Save Best Model
    # -------------------------------------------------------------------------

    if valid_loss < best_validation_loss:

        best_validation_loss = valid_loss

        best_epoch = epoch + 1

        early_counter = 0

        best_model = copy.deepcopy(

            model.state_dict()

        )

        torch.save(

            best_model,

            MODEL_DIR / "best_chemberta.pt"

        )

        print("✓ Best model saved.")

    else:

        early_counter += 1

        print(

            f"No improvement "

            f"({early_counter}/{PATIENCE})"

        )

    # -------------------------------------------------------------------------
    # Early Stopping
    # -------------------------------------------------------------------------

    if early_counter >= PATIENCE:

        print()

        print("Early stopping activated.")

        break
    # =============================================================================
# Load Best Model
# =============================================================================

print()

print("=" * 80)
print("LOADING BEST MODEL")
print("=" * 80)

model.load_state_dict(

    torch.load(

        MODEL_DIR / "best_chemberta.pt",

        map_location=DEVICE

    )

)

print(f"Best Epoch : {best_epoch}")

# =============================================================================
# Evaluation Function
# =============================================================================

@torch.no_grad()

def evaluate(loader):

    model.eval()

    predictions = []

    targets = []

    for batch in tqdm(

        loader,

        leave=False

    ):

        input_ids = batch["input_ids"].to(DEVICE)

        attention_mask = batch["attention_mask"].to(DEVICE)

        labels = batch["labels"].float().to(DEVICE)

        outputs = model(

            input_ids=input_ids,

            attention_mask=attention_mask

        )

        predictions.extend(

            outputs.cpu().numpy()

        )

        targets.extend(

            labels.cpu().numpy()

        )

    predictions = np.array(

        predictions

    )

    targets = np.array(

        targets

    )

    r2 = r2_score(

        targets,

        predictions

    )

    rmse = np.sqrt(

        mean_squared_error(

            targets,

            predictions

        )

    )

    mae = mean_absolute_error(

        targets,

        predictions

    )

    return (

        targets,

        predictions,

        r2,

        rmse,

        mae

    )

# =============================================================================
# Evaluate Train / Validation / Test
# =============================================================================

train_y, train_pred, train_r2, train_rmse, train_mae = evaluate(

    train_loader

)

valid_y, valid_pred, valid_r2, valid_rmse, valid_mae = evaluate(

    valid_loader

)

test_y, test_pred, test_r2, test_rmse, test_mae = evaluate(

    test_loader

)

# =============================================================================
# Performance Summary
# =============================================================================

summary = pd.DataFrame(

    {

        "Dataset": [

            "Train",

            "Validation",

            "Test"

        ],

        "R2": [

            train_r2,

            valid_r2,

            test_r2

        ],

        "RMSE": [

            train_rmse,

            valid_rmse,

            test_rmse

        ],

        "MAE": [

            train_mae,

            valid_mae,

            test_mae

        ]

    }

)

print()

print(summary)

summary.to_csv(

    RESULT_DIR / "performance_summary.csv",

    index=False

)
# =============================================================================
# Save Prediction Files
# =============================================================================

train_results = pd.DataFrame(

    {

        "SMILES": train_smiles,

        "Observed": train_y,

        "Predicted": train_pred,

        "Residual": train_y - train_pred

    }

)

valid_results = pd.DataFrame(

    {

        "SMILES": valid_smiles,

        "Observed": valid_y,

        "Predicted": valid_pred,

        "Residual": valid_y - valid_pred

    }

)

test_results = pd.DataFrame(

    {

        "SMILES": test_smiles,

        "Observed": test_y,

        "Predicted": test_pred,

        "Residual": test_y - test_pred

    }

)

train_results.to_csv(

    RESULT_DIR / "train_predictions.csv",

    index=False

)

valid_results.to_csv(

    RESULT_DIR / "validation_predictions.csv",

    index=False

)

test_results.to_csv(

    RESULT_DIR / "test_predictions.csv",

    index=False

)

# =============================================================================
# Save Training History
# =============================================================================

history_df = pd.DataFrame(

    history

)

history_df.to_csv(

    RESULT_DIR / "training_history.csv",

    index=False

)

print()

print("=" * 80)
print("PREDICTIONS SAVED")
print("=" * 80)

print()

print("Files")

print("-" * 80)

print("train_predictions.csv")
print("validation_predictions.csv")
print("test_predictions.csv")
print("training_history.csv")
# =============================================================================
# Loss Curve
# =============================================================================

plt.figure(figsize=(7,5))

plt.plot(

    history["epoch"],

    history["train_loss"],

    label="Train"

)

plt.plot(

    history["epoch"],

    history["validation_loss"],

    label="Validation"

)

plt.xlabel("Epoch")

plt.ylabel("MSE Loss")

plt.title("ChemBERTa Training Loss")

plt.legend()

plt.tight_layout()

plt.savefig(

    RESULT_DIR / "loss_curve.png",

    dpi=300

)

plt.close()

# =============================================================================
# Observed vs Predicted
# =============================================================================

plt.figure(figsize=(6,6))

plt.scatter(

    test_y,

    test_pred,

    alpha=0.7

)

minimum = min(

    test_y.min(),

    test_pred.min()

)

maximum = max(

    test_y.max(),

    test_pred.max()

)

plt.plot(

    [minimum, maximum],

    [minimum, maximum],

    "r--"

)

plt.xlabel("Observed pActivity")

plt.ylabel("Predicted pActivity")

plt.title("Observed vs Predicted")

plt.tight_layout()

plt.savefig(

    RESULT_DIR / "observed_vs_predicted.png",

    dpi=300

)

plt.close()

# =============================================================================
# Residual Plot
# =============================================================================

residuals = test_y - test_pred

plt.figure(figsize=(6,5))

plt.scatter(

    test_pred,

    residuals,

    alpha=0.7

)

plt.axhline(

    0,

    linestyle="--"

)

plt.xlabel("Predicted")

plt.ylabel("Residual")

plt.title("Residual Plot")

plt.tight_layout()

plt.savefig(

    RESULT_DIR / "residual_plot.png",

    dpi=300

)

plt.close()

# =============================================================================
# Save Final Model
# =============================================================================

torch.save(

    model,

    MODEL_DIR / "chemberta_final.pkl"

)

# =============================================================================
# Finish
# =============================================================================

print()

print("=" * 80)
print("CHEMBERTA TRAINING COMPLETED")
print("=" * 80)

print()

print(summary)

print()

print("Best Validation Loss :", round(best_validation_loss,4))

print("Best Epoch           :", best_epoch)

print()

print("Model")

print("-"*80)

print(MODEL_DIR / "chemberta_final.pkl")

print()

print("Results")

print("-"*80)

print(RESULT_DIR)