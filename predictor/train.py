"""Training loop for the multi-task multimodal Predictor (Sections 5.7–5.8).

Loss: L = L_suit + α * L_type
  L_suit: multi-label BCE over M models
  L_type: cross-entropy over 3 answer type classes
"""

import logging
import os
from pathlib import Path
from typing import Optional, List, Dict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .model import Predictor

logger = logging.getLogger(__name__)


class PredictorDataset(Dataset):
    """Dataset for Predictor training.

    Each sample provides:
      - image_path or PIL Image
      - question text
      - suitability labels: binary vector of shape (num_models,)
      - answer type label: int ∈ {0, 1, 2}
    """

    def __init__(
        self,
        records,
        model_names: List[str],
        image_loader=None,  # callable: path -> PIL Image
    ):
        self.records = records
        self.model_names = model_names
        self.model_to_idx = {m: i for i, m in enumerate(model_names)}
        self.image_loader = image_loader

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]

        # Suitability labels
        suit_label = torch.zeros(len(self.model_names), dtype=torch.float32)
        for model, correctness in rec.candidate_model_correctness.items():
            if model in self.model_to_idx:
                suit_label[self.model_to_idx[model]] = float(correctness >= 0.5)

        # Answer type label from ground-truth answer
        from data_adapters.common_schema import AnswerType
        answer_text = rec.ground_truth_answer or ""
        words = len(answer_text.split()) if answer_text.strip() else 0
        token_count = max(1, int(words * 1.3)) if words > 0 else 1
        answer_type = AnswerType.from_token_count(token_count)
        type_label = torch.tensor(AnswerType.to_index(answer_type), dtype=torch.long)

        # Load image if image_loader is provided
        image = None
        if self.image_loader is not None:
            try:
                image = self.image_loader(rec)
            except Exception:
                from PIL import Image as PILImage
                image = PILImage.new("RGB", (224, 224))

        return {
            "image": image,
            "image_path": rec.image_path,
            "question_text": rec.question_text,
            "suitability_label": suit_label,
            "answer_type_label": type_label,
            "request_id": rec.request_id,
        }


def collate_fn(batch):
    """Custom collate: keep images and texts as lists (not tensors).

    Images can be PIL Images, paths, or None. The Predictor's forward()
    handles all these types.
    """
    # Prefer loaded images, fall back to paths
    images = []
    for item in batch:
        if item["image"] is not None:
            images.append(item["image"])
        elif item["image_path"] is not None:
            images.append(item["image_path"])
        else:
            # Create blank image as last resort
            from PIL import Image as PILImage
            images.append(PILImage.new("RGB", (224, 224)))

    texts = [item["question_text"] for item in batch]
    suit_labels = torch.stack([item["suitability_label"] for item in batch])
    type_labels = torch.stack([item["answer_type_label"] for item in batch])
    request_ids = [item["request_id"] for item in batch]
    return {
        "images": images,
        "texts": texts,
        "suitability_label": suit_labels,
        "answer_type_label": type_labels,
        "request_id": request_ids,
    }


def train_predictor(
    predictor: Predictor,
    train_records,
    val_records,
    model_names: List[str],
    output_dir: str,
    batch_size: int = 32,
    num_epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    alpha: float = 1.0,
    use_focal: bool = True,
    focal_gamma: float = 2.0,
    device: str = "cuda",
    patience: int = 10,
    num_workers: int = 4,
    save_best: bool = True,
    image_loader=None,
):
    """Train the Predictor model.

    Args:
        predictor: Predictor model instance.
        train_records: List of RequestRecord for training.
        val_records: List of RequestRecord for validation.
        model_names: Ordered list of model names.
        output_dir: Path to save checkpoints.
        batch_size: Training batch size.
        num_epochs: Max training epochs.
        lr: Learning rate.
        weight_decay: Weight decay for AdamW.
        alpha: Weight for answer-type loss term.
        use_focal: Use focal BCE loss for suitability (handles class imbalance).
        focal_gamma: Gamma parameter for focal loss.
        device: Training device.
        patience: Early stopping patience.
        num_workers: DataLoader workers.
        save_best: Save best checkpoint.
        image_loader: Callable record->PIL Image to load images on-the-fly.

    Returns:
        Dict with training history.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Move model to device
    predictor = predictor.to(device)

    # Datasets
    train_dataset = PredictorDataset(train_records, model_names, image_loader=image_loader)
    val_dataset = PredictorDataset(val_records, model_names, image_loader=image_loader)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )

    # Optimizer (only trainable params: fusion MLP + heads)
    trainable_params = [
        p for p in predictor.parameters() if p.requires_grad
    ]
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Loss functions
    if use_focal:
        suit_criterion = FocalBCELoss(gamma=focal_gamma)
    else:
        suit_criterion = nn.BCEWithLogitsLoss()

    type_criterion = nn.CrossEntropyLoss()

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_suit_loss": [],
               "val_suit_loss": [], "train_type_loss": [], "val_type_loss": [],
               "val_type_acc": []}

    for epoch in range(num_epochs):
        # Training
        predictor.train()
        train_losses = []
        train_suit_losses = []
        train_type_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch in pbar:
            # Get suitability label (already binary)
            suit_label = batch["suitability_label"].to(device)

            # Forward
            outputs = predictor(batch["images"], batch["texts"])
            suit_logits = outputs["suitability_logits"]
            type_logits = outputs["answer_type_logits"]

            # Losses
            suit_loss = suit_criterion(suit_logits, suit_label)
            type_loss = type_criterion(
                type_logits, batch["answer_type_label"].to(device)
            )
            loss = suit_loss + alpha * type_loss

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())
            train_suit_losses.append(suit_loss.item())
            train_type_losses.append(type_loss.item())

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "suit": f"{suit_loss.item():.4f}",
                "type": f"{type_loss.item():.4f}",
            })

        # Validation
        predictor.eval()
        val_losses = []
        val_suit_losses = []
        val_type_losses = []
        val_type_correct = 0
        val_type_total = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                suit_label = batch["suitability_label"].to(device)
                outputs = predictor(batch["images"], batch["texts"])

                suit_loss = suit_criterion(outputs["suitability_logits"], suit_label)
                type_loss = type_criterion(
                    outputs["answer_type_logits"],
                    batch["answer_type_label"].to(device),
                )
                loss = suit_loss + alpha * type_loss

                val_losses.append(loss.item())
                val_suit_losses.append(suit_loss.item())
                val_type_losses.append(type_loss.item())

                # Accuracy
                pred_type = outputs["answer_type_probs"].argmax(dim=-1)
                val_type_correct += (pred_type == batch["answer_type_label"].to(device)).sum().item()
                val_type_total += len(pred_type)

        # Logging
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        val_type_acc = val_type_correct / val_type_total if val_type_total > 0 else 0

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_suit_loss"].append(np.mean(train_suit_losses))
        history["val_suit_loss"].append(np.mean(val_suit_losses))
        history["train_type_loss"].append(np.mean(train_type_losses))
        history["val_type_loss"].append(np.mean(val_type_losses))
        history["val_type_acc"].append(val_type_acc)

        logger.info(
            f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, "
            f"val_loss={avg_val_loss:.4f}, val_type_acc={val_type_acc:.4f}"
        )

        scheduler.step()

        # Checkpoint
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": predictor.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": avg_val_loss,
            "model_names": model_names,
            "num_models": len(model_names),
        }

        # Best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            if save_best:
                torch.save(checkpoint, output_dir / "predictor_best.pt")
                logger.info(f"Best model saved (val_loss={avg_val_loss:.4f})")
        else:
            patience_counter += 1

        # Latest model
        torch.save(checkpoint, output_dir / "predictor_latest.pt")

        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    # Save final
    torch.save(checkpoint, output_dir / "predictor_final.pt")
    logger.info(f"Training complete. Best val_loss={best_val_loss:.4f}")

    return history


class FocalBCELoss(nn.Module):
    """Focal Binary Cross Entropy for handling class imbalance (Section 5.7)."""

    def __init__(self, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal BCE loss.

        Args:
            logits: (B, K) raw logits.
            targets: (B, K) binary labels.
        """
        probs = torch.sigmoid(logits)
        # BCE with focal weighting
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
