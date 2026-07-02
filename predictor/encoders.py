"""Vision and text encoders for the Predictor module (Sections 5.4–5.5).

Vision: Frozen CLIP/SigLIP vision encoder.
Text: Lightweight MiniLM/DistilBERT encoder.

Both are frozen during Predictor training.
"""

import logging
from typing import Optional, Tuple
import torch
import torch.nn as nn
from PIL import Image

logger = logging.getLogger(__name__)


class FrozenVisionEncoder(nn.Module):
    """Frozen vision encoder using CLIP or SigLIP (Section 5.4).

    Extracts visual features: complexity, OCR difficulty, layout, spatial relations.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        output_dim: Optional[int] = None,
        device: str = "cpu",
    ):
        super().__init__()
        self.model_name = model_name
        self.device = device

        try:
            from transformers import CLIPVisionModel, CLIPImageProcessor
            self.encoder = CLIPVisionModel.from_pretrained(model_name)
            self.processor = CLIPImageProcessor.from_pretrained(model_name)
            self.encoder_type = "clip"
            native_dim = self.encoder.config.hidden_size
        except Exception:
            # Fallback: try SigLIP
            from transformers import SiglipVisionModel, SiglipImageProcessor
            self.encoder = SiglipVisionModel.from_pretrained(
                model_name.replace("clip", "siglip")
            )
            self.processor = SiglipImageProcessor.from_pretrained(
                model_name.replace("clip", "siglip")
            )
            self.encoder_type = "siglip"
            native_dim = self.encoder.config.hidden_size

        # Freeze
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

        # Optional projection to desired output_dim
        self.output_dim = output_dim or native_dim
        if self.output_dim != native_dim:
            self.proj = nn.Linear(native_dim, self.output_dim)
        else:
            self.proj = nn.Identity()

        logger.info(
            f"Vision encoder: {self.encoder_type} (frozen), "
            f"native_dim={native_dim}, output_dim={self.output_dim}"
        )

    def forward(self, images) -> torch.Tensor:
        """Extract vision features from a batch of images.

        Args:
            images: Can be:
                - List of PIL Images
                - List of image paths (str)
                - Pre-processed pixel values tensor

        Returns:
            Tensor of shape (batch_size, output_dim)
        """
        if isinstance(images, torch.Tensor):
            pixel_values = images
        elif isinstance(images, list) and len(images) > 0:
            if isinstance(images[0], str):
                # Load from paths
                pil_images = []
                for path in images:
                    try:
                        pil_images.append(Image.open(path).convert("RGB"))
                    except Exception:
                        # Create a blank image as fallback
                        pil_images.append(Image.new("RGB", (224, 224)))
            else:
                pil_images = images

            pixel_values = self.processor(
                images=pil_images, return_tensors="pt"
            ).pixel_values
        else:
            raise ValueError(f"Unsupported image input type: {type(images)}")

        pixel_values = pixel_values.to(next(self.encoder.parameters()).device)

        with torch.no_grad():
            if self.encoder_type == "clip":
                outputs = self.encoder(pixel_values)
                features = outputs.pooler_output  # (B, hidden_size)
            else:
                outputs = self.encoder(pixel_values)
                features = outputs.pooler_output

        features = self.proj(features)
        return features

    @property
    def dim(self) -> int:
        return self.output_dim


class LightweightTextEncoder(nn.Module):
    """Lightweight frozen text encoder (Section 5.5).

    Uses MiniLM or DistilBERT for capturing question intent:
    OCR, counting, yes/no, description, reasoning, chart understanding.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        output_dim: Optional[int] = None,
        device: str = "cpu",
        pooling: str = "mean",
    ):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.pooling = pooling

        from transformers import AutoTokenizer, AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)

        # Freeze
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

        native_dim = self.encoder.config.hidden_size
        self.output_dim = output_dim or native_dim
        if self.output_dim != native_dim:
            self.proj = nn.Linear(native_dim, self.output_dim)
        else:
            self.proj = nn.Identity()

        logger.info(
            f"Text encoder: {model_name} (frozen), "
            f"native_dim={native_dim}, output_dim={self.output_dim}, pooling={pooling}"
        )

    def forward(
        self,
        texts,
        max_length: int = 128,
    ) -> torch.Tensor:
        """Extract text features from a batch of questions.

        Args:
            texts: List of strings.
            max_length: Max token length for tokenizer.

        Returns:
            Tensor of shape (batch_size, output_dim)
        """
        device = next(self.encoder.parameters()).device

        tokens = self.tokenizer(
            list(texts) if not isinstance(texts, str) else [texts],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = self.encoder(**tokens)
            # outputs.last_hidden_state: (B, seq_len, hidden_size)

            if self.pooling == "mean":
                # Mean pooling with attention mask
                mask = tokens["attention_mask"].unsqueeze(-1).float()
                features = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(
                    dim=1
                )
            elif self.pooling == "cls":
                features = outputs.last_hidden_state[:, 0, :]
            else:
                features = outputs.pooler_output

        features = self.proj(features)
        return features

    def encode(self, texts) -> torch.Tensor:
        """Convenience method returning numpy-friendly features."""
        return self.forward(texts)

    @property
    def dim(self) -> int:
        return self.output_dim
