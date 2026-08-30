"""
===============================================================================
Project : BCRABL-AI

File:
    chemberta_model.py

Purpose:
    ChemBERTa Regression Model

===============================================================================
"""

import torch
import torch.nn as nn

from transformers import AutoModel

# =============================================================================
# Model Configuration
# =============================================================================

MODEL_NAME = "DeepChem/ChemBERTa-77M-MLM"

DROPOUT = 0.20

HIDDEN_DIM = 256

DEVICE = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else

    "cpu"

)


# =============================================================================
# Mean Pooling
# =============================================================================

def mean_pooling(model_output, attention_mask):

    token_embeddings = model_output.last_hidden_state

    input_mask_expanded = attention_mask.unsqueeze(-1).expand(

        token_embeddings.size()

    ).float()

    return torch.sum(

        token_embeddings * input_mask_expanded,

        dim=1

    ) / torch.clamp(

        input_mask_expanded.sum(dim=1),

        min=1e-9

    )

# =============================================================================
# ChemBERTa Regressor
# =============================================================================

class ChemBERTaRegressor(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = AutoModel.from_pretrained(

            MODEL_NAME

        )

        embedding_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(

            DROPOUT

        )

        self.regressor = nn.Sequential(

            nn.Linear(

                embedding_size,

                HIDDEN_DIM

            ),

            nn.GELU(),

            nn.Dropout(

                DROPOUT

            ),

            nn.Linear(

                HIDDEN_DIM,

                1

            )

        )

        self._initialize_weights()

    # -------------------------------------------------------------------------

    def _initialize_weights(self):

        for module in self.regressor:

            if isinstance(

                module,

                nn.Linear

            ):

                nn.init.xavier_uniform_(

                    module.weight

                )

                nn.init.zeros_(

                    module.bias

                )

    # -------------------------------------------------------------------------

    def forward(

        self,

        input_ids,

        attention_mask

    ):

        outputs = self.encoder(

            input_ids=input_ids,

            attention_mask=attention_mask

        )

        pooled = mean_pooling(

            outputs,

            attention_mask

        )

        pooled = self.dropout(

            pooled

        )

        prediction = self.regressor(

            pooled

        )

        return prediction.squeeze(-1)
    # =============================================================================
# Build Model
# =============================================================================

model = ChemBERTaRegressor().to(DEVICE)

# =============================================================================
# Model Summary
# =============================================================================

if __name__ == "__main__":

    print()

    print("=" * 80)
    print("CHEMBERTA MODEL")
    print("=" * 80)

    print()

    print(model)

    print()

    total_params = sum(

        p.numel()

        for p in model.parameters()

    )

    trainable_params = sum(

        p.numel()

        for p in model.parameters()

        if p.requires_grad

    )

    print(f"Total Parameters      : {total_params:,}")

    print(f"Trainable Parameters  : {trainable_params:,}")

    print()

    print("Backbone")

    print("-" * 80)

    print(MODEL_NAME)

    print()

    print("Pooling")

    print("-" * 80)

    print("Mean Pooling")

    print()

    print("Regression Head")

    print("-" * 80)

    print(f"Hidden Layer : {HIDDEN_DIM}")

    print(f"Dropout      : {DROPOUT}")

    print()

    print("=" * 80)
    print("CHEMBERTA MODEL READY")
    print("=" * 80)