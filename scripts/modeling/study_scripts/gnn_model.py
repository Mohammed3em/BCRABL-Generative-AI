"""
===============================================================================
Project : BCRABL-AI
Script  : 31_gnn_model.py

Purpose
-------
GIN Model for Molecular Property Prediction

===============================================================================
"""

import torch
import torch.nn as nn

from torch_geometric.nn import (

    GINConv,

    global_mean_pool,

    BatchNorm

)

# =============================================================================
# MLP Block
# =============================================================================

def build_mlp(

    input_dim,

    hidden_dim

):

    return nn.Sequential(

        nn.Linear(

            input_dim,

            hidden_dim

        ),

        nn.ReLU(),

        nn.Linear(

            hidden_dim,

            hidden_dim

        )

    )
# =============================================================================
# GIN Model
# =============================================================================

class GINRegressor(nn.Module):

    def __init__(

        self,

        node_features=8,

        hidden_dim=128,

        dropout=0.20

    ):

        super().__init__()

        # -------------------------
        # GIN Layer 1
        # -------------------------

        self.conv1 = GINConv(

            build_mlp(

                node_features,

                hidden_dim

            )

        )

        self.bn1 = BatchNorm(hidden_dim)

        # -------------------------
        # GIN Layer 2
        # -------------------------

        self.conv2 = GINConv(

            build_mlp(

                hidden_dim,

                hidden_dim

            )

        )

        self.bn2 = BatchNorm(hidden_dim)

        # -------------------------
        # GIN Layer 3
        # -------------------------

        self.conv3 = GINConv(

            build_mlp(

                hidden_dim,

                hidden_dim

            )

        )

        self.bn3 = BatchNorm(hidden_dim)

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(dropout)

        # -------------------------
        # Regression Head
        # -------------------------

        self.fc = nn.Sequential(

            nn.Linear(

                hidden_dim,

                64

            ),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(

                64,

                1

            )

        )
        # =============================================================================
# Forward
# =============================================================================

    def forward(self, data):

        x = data.x

        edge_index = data.edge_index

        batch = data.batch

        # -------------------------
        # GIN Block 1
        # -------------------------

        x = self.conv1(

            x,

            edge_index

        )

        x = self.bn1(x)

        x = self.relu(x)

        x = self.dropout(x)

        # -------------------------
        # GIN Block 2
        # -------------------------

        x = self.conv2(

            x,

            edge_index

        )

        x = self.bn2(x)

        x = self.relu(x)

        x = self.dropout(x)

        # -------------------------
        # GIN Block 3
        # -------------------------

        x = self.conv3(

            x,

            edge_index

        )

        x = self.bn3(x)

        x = self.relu(x)

        # -------------------------
        # Graph Pooling
        # -------------------------

        x = global_mean_pool(

            x,

            batch

        )

        # -------------------------
        # Regression
        # -------------------------

        x = self.fc(x)

        return x.view(-1)


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":

    model = GINRegressor()

    print("=" * 80)
    print("GIN MODEL")
    print("=" * 80)

    print(model)

    total_params = sum(

        p.numel()

        for p in model.parameters()

    )

    trainable_params = sum(

        p.numel()

        for p in model.parameters()

        if p.requires_grad

    )

    print("\nTotal Parameters :", f"{total_params:,}")
    print("Trainable Parameters :", f"{trainable_params:,}")

    print()
    print("=" * 80)
    print("GIN MODEL READY")
    print("=" * 80)