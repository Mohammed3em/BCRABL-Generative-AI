"""
===============================================================================
Project : BCRABL-AI
Script  : 32_train_gnn.py

Final GIN Training Script

===============================================================================
"""

from pathlib import Path
import random
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tqdm import tqdm

from gnn_dataset import (
    train_loader,
    valid_loader,
    test_loader
)

from gnn_model import GINRegressor

# =============================================================================
# Reproducibility
# =============================================================================

SEED = 42

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():

    torch.cuda.manual_seed(SEED)

    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True

torch.backends.cudnn.benchmark = False

# =============================================================================
# Device
# =============================================================================

DEVICE = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else

    "cpu"

)

print("=" * 80)
print("GIN TRAINING")
print("=" * 80)

print(f"Device : {DEVICE}")

# =============================================================================
# Paths
# =============================================================================

RESULT_DIR = Path("07_Results/GNN")

MODEL_DIR = Path("04_AI_Models/Models")

RESULT_DIR.mkdir(

    parents=True,

    exist_ok=True

)

MODEL_DIR.mkdir(

    parents=True,

    exist_ok=True

)

# =============================================================================
# Hyperparameters
# =============================================================================

EPOCHS = 50

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-5

PATIENCE = 7

model = GINRegressor().to(DEVICE)

criterion = nn.MSELoss()

optimizer = AdamW(

    model.parameters(),

    lr=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY

)

scheduler = ReduceLROnPlateau(

    optimizer,

    mode="min",

    factor=0.5,

    patience=10

)
# =============================================================================
# Train One Epoch
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

        batch = batch.to(DEVICE)

        optimizer.zero_grad()

        output = model(batch)

        loss = criterion(

            output,

            batch.y.view(-1)

        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(

            model.parameters(),

            max_norm=5.0

        )

        optimizer.step()

        running_loss += loss.item()

        predictions.extend(

            output.detach()

                  .cpu()

                  .numpy()

        )

        targets.extend(

            batch.y.view(-1)

                  .cpu()

                  .numpy()

        )

        progress.set_postfix(

            loss=f"{loss.item():.4f}"

        )

    predictions = np.array(predictions)

    targets = np.array(targets)

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

    average_loss = running_loss / len(train_loader)

    return (

        average_loss,

        r2,

        rmse,

        mae,

        predictions,

        targets

    )
# =============================================================================
# Evaluation
# =============================================================================

@torch.no_grad()

def evaluate(loader):

    model.eval()

    running_loss = 0.0

    predictions = []

    targets = []

    for batch in loader:

        batch = batch.to(DEVICE)

        output = model(batch)

        loss = criterion(

            output,

            batch.y.view(-1)

        )

        running_loss += loss.item()

        predictions.extend(

            output.cpu().numpy()

        )

        targets.extend(

            batch.y.view(-1).cpu().numpy()

        )

    predictions = np.array(predictions)

    targets = np.array(targets)

    average_loss = running_loss / len(loader)

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

    return {

        "loss": average_loss,

        "r2": r2,

        "rmse": rmse,

        "mae": mae,

        "predictions": predictions,

        "targets": targets

    }
# =============================================================================
# Early Stopping
# =============================================================================

class EarlyStopping:

    def __init__(

        self,

        patience=30,

        min_delta=0.0

    ):

        self.patience = patience

        self.min_delta = min_delta

        self.counter = 0

        self.best_loss = np.inf

        self.best_epoch = 0

        self.best_state = None

    # -------------------------------------------------------------------------

    def step(

        self,

        validation_loss,

        model,

        epoch

    ):

        if validation_loss < (

            self.best_loss - self.min_delta

        ):

            self.best_loss = validation_loss

            self.best_epoch = epoch

            self.counter = 0

            self.best_state = copy.deepcopy(

                model.state_dict()

            )

            return False

        else:

            self.counter += 1

            return self.counter >= self.patience

    # -------------------------------------------------------------------------

    def restore(

        self,

        model

    ):

        if self.best_state is not None:

            model.load_state_dict(

                self.best_state

            )

# =============================================================================
# Initialize Early Stopping
# =============================================================================

early_stopping = EarlyStopping(

    patience=PATIENCE,

    min_delta=1e-5

)
# =============================================================================
# Training Loop
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
print("START TRAINING")
print("=" * 80)

for epoch in range(1, EPOCHS + 1):

    # -------------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------------

    (
        train_loss,
        train_r2,
        train_rmse,
        train_mae,
        _,
        _
    ) = train_one_epoch()

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    validation = evaluate(valid_loader)

    validation_loss = validation["loss"]

    scheduler.step(validation_loss)

    current_lr = optimizer.param_groups[0]["lr"]

    history["epoch"].append(epoch)

    history["train_loss"].append(train_loss)
    history["validation_loss"].append(validation_loss)

    history["train_r2"].append(train_r2)
    history["validation_r2"].append(validation["r2"])

    history["learning_rate"].append(current_lr)

    print(

        f"Epoch {epoch:03d} | "

        f"Train Loss {train_loss:.4f} | "

        f"Val Loss {validation_loss:.4f} | "

        f"Train R² {train_r2:.4f} | "

        f"Val R² {validation['r2']:.4f} | "

        f"LR {current_lr:.2e}"

    )

    # -------------------------------------------------------------------------
    # Early Stopping
    # -------------------------------------------------------------------------

    stop = early_stopping.step(

        validation_loss,

        model,

        epoch

    )

    if stop:

        print()

        print("=" * 80)
        print("EARLY STOPPING")
        print("=" * 80)

        print(f"Stopped at Epoch : {epoch}")

        break

# =============================================================================
# Restore Best Model
# =============================================================================

early_stopping.restore(model)

print()

print("=" * 80)
print("BEST MODEL RESTORED")
print("=" * 80)

print(f"Best Epoch           : {early_stopping.best_epoch}")
print(f"Best Validation Loss : {early_stopping.best_loss:.6f}")
# =============================================================================
# Final Evaluation
# =============================================================================

print()
print("=" * 80)
print("FINAL EVALUATION")
print("=" * 80)

train_results = evaluate(train_loader)

validation_results = evaluate(valid_loader)

test_results = evaluate(test_loader)

# =============================================================================
# Performance Summary
# =============================================================================

performance = pd.DataFrame({

    "Dataset": [

        "Train",

        "Validation",

        "Test"

    ],

    "R2": [

        train_results["r2"],

        validation_results["r2"],

        test_results["r2"]

    ],

    "RMSE": [

        train_results["rmse"],

        validation_results["rmse"],

        test_results["rmse"]

    ],

    "MAE": [

        train_results["mae"],

        validation_results["mae"],

        test_results["mae"]

    ]

})

performance.to_csv(

    RESULT_DIR / "performance_summary.csv",

    index=False,

    encoding="utf-8-sig"

)

print()
print(performance.round(4))

# =============================================================================
# Save Best Model
# =============================================================================

torch.save(

    {

        "model_state_dict": model.state_dict(),

        "best_epoch": early_stopping.best_epoch,

        "best_validation_loss": early_stopping.best_loss,

        "history": history

    },

    MODEL_DIR / "best_gnn.pt"

)

print()

print("=" * 80)
print("BEST MODEL SAVED")
print("=" * 80)

print(MODEL_DIR / "best_gnn.pt")
# =============================================================================
# Save Predictions
# =============================================================================

datasets = {

    "train": train_results,

    "validation": validation_results,

    "test": test_results

}

for name, result in datasets.items():

    df = pd.DataFrame({

        "Observed": result["targets"],

        "Predicted": result["predictions"],

        "Residual": result["targets"] - result["predictions"]

    })

    df.to_csv(

        RESULT_DIR / f"{name}_predictions.csv",

        index=False,

        encoding="utf-8-sig"

    )

# =============================================================================
# Save Training History
# =============================================================================

history_df = pd.DataFrame(history)

history_df.to_csv(

    RESULT_DIR / "training_history.csv",

    index=False,

    encoding="utf-8-sig"

)

# =============================================================================
# Loss Curve
# =============================================================================

plt.figure(figsize=(8,6))

plt.plot(

    history["epoch"],

    history["train_loss"],

    linewidth=2,

    label="Training"

)

plt.plot(

    history["epoch"],

    history["validation_loss"],

    linewidth=2,

    label="Validation"

)

plt.xlabel("Epoch", fontsize=12, fontweight="bold")

plt.ylabel("Loss", fontsize=12, fontweight="bold")

plt.title("Training Loss", fontsize=15, fontweight="bold")

plt.grid(alpha=0.30, linestyle=":")

plt.legend()

plt.tight_layout()

plt.savefig(

    RESULT_DIR / "loss_curve.png",

    dpi=600,

    bbox_inches="tight"

)

plt.close()

# =============================================================================
# R² Curve
# =============================================================================

plt.figure(figsize=(8,6))

plt.plot(

    history["epoch"],

    history["train_r2"],

    linewidth=2,

    label="Training"

)

plt.plot(

    history["epoch"],

    history["validation_r2"],

    linewidth=2,

    label="Validation"

)

plt.xlabel("Epoch", fontsize=12, fontweight="bold")

plt.ylabel("R²", fontsize=12, fontweight="bold")

plt.title("R² During Training", fontsize=15, fontweight="bold")

plt.grid(alpha=0.30, linestyle=":")

plt.legend()

plt.tight_layout()

plt.savefig(

    RESULT_DIR / "r2_curve.png",

    dpi=600,

    bbox_inches="tight"

)

plt.close()

print()

print("=" * 80)

print("TRAINING HISTORY SAVED")

print("=" * 80)
