"""
===============================================================================
Project : BCRABL-AI

File:
    36_train_molformer.py

Purpose:
    Fine-tuning MolFormer for pActivity Prediction
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
from torch.nn.utils import clip_grad_norm_

from transformers import get_linear_schedule_with_warmup

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

from tqdm import tqdm

from molformer_dataset import (
    train_dataset,
    valid_dataset,
    test_dataset,
    data_collator,
    train_smiles,
    valid_smiles,
    test_smiles,
    BATCH_SIZE
)

from molformer_model import (
    model,
    DEVICE
)

# =============================================================================
# Random Seed
# =============================================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

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

MODEL_DIR = Path("04_AI_Models/Models")
RESULT_DIR = Path("07_Results/MolFormer")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("MOLFORMER TRAINING")
print("=" * 80)

print(f"Device             : {DEVICE}")
print(f"Training Samples   : {len(train_dataset):,}")
print(f"Validation Samples : {len(valid_dataset):,}")
print(f"Testing Samples    : {len(test_dataset):,}")
# =============================================================================
# Hyperparameters
# =============================================================================

EPOCHS = 50

LEARNING_RATE = 1e-5

WEIGHT_DECAY = 0.01

WARMUP_RATIO = 0.10

PATIENCE = 7

MAX_GRAD_NORM = 1.0

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
# Scheduler
# =============================================================================

total_steps = len(train_loader) * EPOCHS

warmup_steps = int(

    total_steps * WARMUP_RATIO

)

scheduler = get_linear_schedule_with_warmup(

    optimizer,

    num_warmup_steps=warmup_steps,

    num_training_steps=total_steps

)

# =============================================================================
# Best Model Tracking
# =============================================================================

best_validation_loss = float("inf")

best_epoch = 0

early_counter = 0

history = {

    "epoch": [],

    "train_loss": [],

    "validation_loss": [],

    "train_r2": [],

    "validation_r2": [],

    "learning_rate": []

}

# =============================================================================
# Configuration Summary
# =============================================================================

print()

print("=" * 80)
print("TRAINING CONFIGURATION")
print("=" * 80)

print()

print(f"Epochs            : {EPOCHS}")
print(f"Learning Rate     : {LEARNING_RATE}")
print(f"Weight Decay      : {WEIGHT_DECAY}")
print(f"Warmup Steps      : {warmup_steps}")
print(f"Batch Size        : {BATCH_SIZE}")
print(f"Gradient Clipping : {MAX_GRAD_NORM}")
print("Precision         : FP32")
# =============================================================================
# Train One Epoch
# =============================================================================

def train_one_epoch():

    model.train()

    running_loss = 0.0

    predictions = []

    targets = []

    progress_bar = tqdm(

        train_loader,

        desc="Training",

        leave=False

    )

    for batch in progress_bar:

        input_ids = batch["input_ids"].to(DEVICE)

        attention_mask = batch["attention_mask"].to(DEVICE)

        labels = batch["labels"].float().to(DEVICE)

        optimizer.zero_grad()

        # -------------------------------------------------------------
        # Forward
        # -------------------------------------------------------------

        outputs = model(

            input_ids=input_ids,

            attention_mask=attention_mask

        )

        # -------------------------------------------------------------
        # Numerical Stability Check
        # -------------------------------------------------------------

        if not torch.isfinite(outputs).all():

            raise RuntimeError(

                "Model output contains NaN or Inf."

            )

        loss = criterion(

            outputs,

            labels

        )

        if not torch.isfinite(loss):

            raise RuntimeError(

                "Loss became NaN."

            )

        # -------------------------------------------------------------
        # Backpropagation
        # -------------------------------------------------------------

        loss.backward()

        clip_grad_norm_(

            model.parameters(),

            MAX_GRAD_NORM

        )

        optimizer.step()

        scheduler.step()

        # -------------------------------------------------------------
        # Statistics
        # -------------------------------------------------------------

        running_loss += loss.item()

        predictions.extend(

            outputs.detach().cpu().numpy()

        )

        targets.extend(

            labels.detach().cpu().numpy()

        )

        progress_bar.set_postfix(

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

def validate(loader):

    model.eval()

    running_loss = 0.0

    predictions = []

    targets = []

    progress_bar = tqdm(

        loader,

        desc="Validation",

        leave=False

    )

    for batch in progress_bar:

        input_ids = batch["input_ids"].to(DEVICE)

        attention_mask = batch["attention_mask"].to(DEVICE)

        labels = batch["labels"].float().to(DEVICE)

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

            outputs.detach().cpu().numpy()

        )

        targets.extend(

            labels.detach().cpu().numpy()

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
# Test / Evaluation Function
# =============================================================================

@torch.no_grad()

def evaluate(loader):

    model.eval()

    predictions = []

    targets = []

    for batch in tqdm(

        loader,

        desc="Evaluation",

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

            outputs.detach().cpu().numpy()

        )

        targets.extend(

            labels.detach().cpu().numpy()

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
# Start Training
# =============================================================================

print()

print("=" * 80)
print("START TRAINING")
print("=" * 80)

best_model_state = None

for epoch in range(EPOCHS):

    print()

    print(f"Epoch {epoch + 1}/{EPOCHS}")

    # -------------------------------------------------------------------------
    # Training
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
    ) = validate(valid_loader)

    current_lr = optimizer.param_groups[0]["lr"]

    history["epoch"].append(epoch + 1)
    history["train_loss"].append(train_loss)
    history["validation_loss"].append(valid_loss)
    history["train_r2"].append(train_r2)
    history["validation_r2"].append(valid_r2)
    history["learning_rate"].append(current_lr)

    print(
        f"Train Loss : {train_loss:.5f} | "
        f"Train R² : {train_r2:.4f}"
    )

    print(
        f"Valid Loss : {valid_loss:.5f} | "
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

        best_model_state = copy.deepcopy(
            model.state_dict()
        )

        torch.save(
            best_model_state,
            MODEL_DIR / "best_molformer.pt"
        )

        print("✓ Best model saved.")

    else:

        early_counter += 1

        print(
            f"No improvement ({early_counter}/{PATIENCE})"
        )

    # -------------------------------------------------------------------------
    # Early Stopping
    # -------------------------------------------------------------------------

    if early_counter >= PATIENCE:

        print()

        print("=" * 80)
        print("EARLY STOPPING")
        print("=" * 80)

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

        MODEL_DIR / "best_molformer.pt",

        map_location=DEVICE

    )

)

print(f"Best Epoch : {best_epoch}")
# =============================================================================
# Evaluate Best Model
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

summary = pd.DataFrame({

    "Dataset":[
        "Train",
        "Validation",
        "Test"
    ],

    "R2":[
        train_r2,
        valid_r2,
        test_r2
    ],

    "RMSE":[
        train_rmse,
        valid_rmse,
        test_rmse
    ],

    "MAE":[
        train_mae,
        valid_mae,
        test_mae
    ]

})

summary.to_csv(

    RESULT_DIR / "performance_summary.csv",

    index=False

)

# =============================================================================
# Save Prediction Files
# =============================================================================

pd.DataFrame({

    "SMILES":train_smiles,
    "Observed":train_y,
    "Predicted":train_pred,
    "Residual":train_y-train_pred

}).to_csv(

    RESULT_DIR/"train_predictions.csv",

    index=False

)

pd.DataFrame({

    "SMILES":valid_smiles,
    "Observed":valid_y,
    "Predicted":valid_pred,
    "Residual":valid_y-valid_pred

}).to_csv(

    RESULT_DIR/"validation_predictions.csv",

    index=False

)

pd.DataFrame({

    "SMILES":test_smiles,
    "Observed":test_y,
    "Predicted":test_pred,
    "Residual":test_y-test_pred

}).to_csv(

    RESULT_DIR/"test_predictions.csv",

    index=False

)

# =============================================================================
# Training History
# =============================================================================

history_df = pd.DataFrame(history)

history_df.to_csv(

    RESULT_DIR/"training_history.csv",

    index=False

)

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

plt.ylabel("Loss")

plt.title("MolFormer Training")

plt.legend()

plt.tight_layout()

plt.savefig(

    RESULT_DIR/"loss_curve.png",

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

low = min(

    test_y.min(),

    test_pred.min()

)

high = max(

    test_y.max(),

    test_pred.max()

)

plt.plot(

    [low,high],

    [low,high],

    "r--"

)

plt.xlabel("Observed")

plt.ylabel("Predicted")

plt.tight_layout()

plt.savefig(

    RESULT_DIR/"observed_vs_predicted.png",

    dpi=300

)

plt.close()

# =============================================================================
# Residual Plot
# =============================================================================

plt.figure(figsize=(6,5))

plt.scatter(

    test_pred,

    test_y-test_pred,

    alpha=0.7

)

plt.axhline(

    0,

    linestyle="--"

)

plt.xlabel("Predicted")

plt.ylabel("Residual")

plt.tight_layout()

plt.savefig(

    RESULT_DIR/"residual_plot.png",

    dpi=300

)

plt.close()

# =============================================================================
# Save Final Model
# =============================================================================

torch.save(

    model.state_dict(),

    MODEL_DIR/"molformer_final.pt"

)

# =============================================================================
# Finished
# =============================================================================

print()

print("="*80)
print("MOLFORMER TRAINING COMPLETED")
print("="*80)

print()
print(summary)
print()

print(f"Best Epoch : {best_epoch}")
print(f"Best Validation Loss : {best_validation_loss:.5f}")

print()

print("Results Folder")

print(RESULT_DIR)

print()

print("Model")

print(MODEL_DIR/"molformer_final.pt")
