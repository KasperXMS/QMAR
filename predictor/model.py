"""Multi-task multimodal Predictor (Section 5).

Full model: frozen encoders + fusion MLP + two task heads.

Outputs:
  1. Suitability scores s_i over candidate models (multi-label BCE)
  2. Answer type prediction â_i (3-class softmax: short/medium/long)
"""

import logging
from typing import Optional, List, Dict, Tuple
import torch
import torch.nn as nn

from .encoders import FrozenVisionEncoder, LightweightTextEncoder
from .fusion import FusionMLP

logger = logging.getLogger(__name__)


class SuitabilityHead(nn.Module):
    """Outputs per-model suitability scores s_i = σ(W_s h_i + b_s) (Section 5.7)."""

    def __init__(
        self,
        input_dim: int,
        num_models: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_models = num_models
        if hidden_dim is not None:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_models),
            )
        else:
            self.net = nn.Linear(input_dim, num_models)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Predict suitability scores.

        Args:
            h: Request representation (B, input_dim).

        Returns:
            scores: (B, num_models) raw logits.
        """
        return self.net(h)  # logits, sigmoid applied in loss or inference


class AnswerTypeHead(nn.Module):
    """Outputs answer type prediction â_i = softmax(W_a h_i + b_a) (Section 5.8)."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 3,  # short, medium, long
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dim is not None:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.net = nn.Linear(input_dim, num_classes)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Predict answer type logits.

        Args:
            h: Request representation (B, input_dim).

        Returns:
            logits: (B, num_classes)
        """
        return self.net(h)


class Predictor(nn.Module):
    """Full multi-task multimodal predictor (Section 5.3).

    Architecture:
        Image I_i → Frozen Vision Encoder → v_i
        Text T_i  → Lightweight Text Encoder → t_i
        [v_i, t_i, v_i⊙t_i, |v_i-t_i|] → Fusion MLP → h_i
        h_i → Suitability Head → s_i (over M models)
        h_i → Answer-Type Head → â_i (short/medium/long)
    """

    def __init__(
        self,
        num_models: int,
        vision_encoder_name: str = "openai/clip-vit-base-patch32",
        text_encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        vision_dim: int = 512,
        text_dim: int = 384,
        fusion_hidden_dims: Optional[List[int]] = None,
        fusion_output_dim: int = 512,
        suit_hidden_dim: Optional[int] = None,
        type_hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        device: str = "cpu",
    ):
        super().__init__()
        self.num_models = num_models
        self.device = device

        # Encoders
        self.vision_encoder = FrozenVisionEncoder(
            model_name=vision_encoder_name,
            output_dim=vision_dim,
            device=device,
        )
        self.text_encoder = LightweightTextEncoder(
            model_name=text_encoder_name,
            output_dim=text_dim,
            device=device,
        )

        # Dimension alignment: project text features to match vision dim if needed
        if text_dim != vision_dim:
            self.text_proj = nn.Linear(text_dim, vision_dim)
            fusion_text_dim = vision_dim
        else:
            self.text_proj = nn.Identity()
            fusion_text_dim = text_dim

        # Fusion
        self.fusion = FusionMLP(
            vision_dim=vision_dim,
            text_dim=fusion_text_dim,
            hidden_dims=fusion_hidden_dims,
            output_dim=fusion_output_dim,
            dropout=dropout,
        )

        # Heads
        self.suitability_head = SuitabilityHead(
            input_dim=fusion_output_dim,
            num_models=num_models,
            hidden_dim=suit_hidden_dim,
            dropout=dropout,
        )
        self.answer_type_head = AnswerTypeHead(
            input_dim=fusion_output_dim,
            num_classes=3,
            hidden_dim=type_hidden_dim,
            dropout=dropout,
        )

    def forward(
        self,
        images,
        texts,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            images: Batch of images (paths, PIL Images, or pixel values).
            texts: Batch of question strings.

        Returns:
            dict with:
              - suitability_logits: (B, num_models)
              - suitability_probs: (B, num_models) after sigmoid
              - answer_type_logits: (B, 3)
              - answer_type_probs: (B, 3) after softmax
              - request_repr: (B, fusion_output_dim)
        """
        v = self.vision_encoder(images)
        t = self.text_encoder(texts)
        t = self.text_proj(t)  # align dimensions if needed
        h = self.fusion(v, t)

        suit_logits = self.suitability_head(h)
        type_logits = self.answer_type_head(h)

        return {
            "suitability_logits": suit_logits,
            "suitability_probs": torch.sigmoid(suit_logits),
            "answer_type_logits": type_logits,
            "answer_type_probs": torch.softmax(type_logits, dim=-1),
            "request_repr": h,
        }

    def predict_suitability(
        self, images, texts
    ) -> torch.Tensor:
        """Convenience: predict suitability probabilities only."""
        outputs = self.forward(images, texts)
        return outputs["suitability_probs"]

    def predict_answer_type(
        self, images, texts
    ) -> torch.Tensor:
        """Convenience: predict answer type class indices."""
        outputs = self.forward(images, texts)
        return outputs["answer_type_probs"].argmax(dim=-1)

    def to(self, device):
        """Move model to device (handles frozen encoders correctly)."""
        super().to(device)
        self.device = device
        # Ensure frozen encoders stay on the right device
        if hasattr(self, "vision_encoder"):
            self.vision_encoder = self.vision_encoder.to(device)
        if hasattr(self, "text_encoder"):
            self.text_encoder = self.text_encoder.to(device)
        return self
