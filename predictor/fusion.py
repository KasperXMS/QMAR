"""Fusion module (Section 5.6).

Lightweight MLP fusion: z_i = [v_i, t_i, v_i ⊙ t_i, |v_i - t_i|]
                         h_i = MLP(z_i)
"""

import torch
import torch.nn as nn
from typing import Optional, List


class FusionMLP(nn.Module):
    """MLP fusion of vision and text features (Section 5.6).

    Combines concatenation, element-wise product, and absolute difference
    to capture image-question interaction with low complexity.
    """

    def __init__(
        self,
        vision_dim: int,
        text_dim: int,
        hidden_dims: Optional[List[int]] = None,
        output_dim: int = 512,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        self.vision_dim = vision_dim
        self.text_dim = text_dim

        # Fusion input: [v, t, v⊙t, |v-t|]
        self.fusion_input_dim = vision_dim + text_dim + vision_dim + vision_dim

        if hidden_dims is None:
            hidden_dims = [1024, 512]

        # Build MLP
        layers = []
        in_dim = self.fusion_input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "gelu":
                layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, output_dim))
        self.mlp = nn.Sequential(*layers)
        self.output_dim = output_dim

    def forward(
        self, vision_features: torch.Tensor, text_features: torch.Tensor
    ) -> torch.Tensor:
        """Fuse vision and text features.

        Args:
            vision_features: (B, vision_dim)
            text_features: (B, text_dim)

        Returns:
            h_i: (B, output_dim) request representation
        """
        # Ensure same dtype
        if vision_features.dtype != text_features.dtype:
            text_features = text_features.to(vision_features.dtype)

        # Element-wise interaction terms
        hadamard = vision_features * text_features  # v ⊙ t
        abs_diff = torch.abs(vision_features - text_features)  # |v - t|

        # Concatenate: [v, t, v⊙t, |v-t|]
        z = torch.cat([vision_features, text_features, hadamard, abs_diff], dim=-1)

        # MLP
        h = self.mlp(z)
        return h

    @property
    def dim(self) -> int:
        return self.output_dim
