#!/usr/bin/env python
"""Fast Predictor training using pre-extracted embeddings.

Train the fusion MLP + heads on cached CLIP/text embeddings.
~100x faster than loading + encoding images on-the-fly.

Usage:
    # First extract embeddings:
    python extract_embeddings.py --device cuda

    # Then train fast:
    python train_predictor_fast.py --device cuda --epochs 50
    python train_predictor_fast.py --device cuda --epochs 50 --alpha 2.0
"""

import argparse
import logging
import sys
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from predictor.fusion import FusionMLP
from predictor.model import SuitabilityHead, AnswerTypeHead

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_fast")


class PredictorWithEmbeddings(nn.Module):
    """Predictor that takes pre-extracted embeddings instead of raw images/text."""

    def __init__(self, vision_dim=512, text_dim=384, num_models=17,
                 fusion_hidden=None, fusion_out=512, dropout=0.1,
                 suit_hidden=256, type_hidden=128):
        super().__init__()
        # Align dims
        if text_dim != vision_dim:
            self.text_proj = nn.Linear(text_dim, vision_dim)
            fusion_text_dim = vision_dim
        else:
            self.text_proj = nn.Identity()
            fusion_text_dim = text_dim

        self.fusion = FusionMLP(
            vision_dim=vision_dim, text_dim=fusion_text_dim,
            hidden_dims=fusion_hidden or [1024, 512],
            output_dim=fusion_out, dropout=dropout,
        )
        self.suitability_head = SuitabilityHead(fusion_out, num_models, suit_hidden, dropout)
        self.answer_type_head = AnswerTypeHead(fusion_out, 3, type_hidden, dropout)
        self.num_models = num_models

    def forward(self, v, t):
        t = self.text_proj(t)
        h = self.fusion(v, t)
        return {
            "suitability_logits": self.suitability_head(h),
            "suitability_probs": torch.sigmoid(self.suitability_head(h)),
            "answer_type_logits": self.answer_type_head(h),
            "answer_type_probs": torch.softmax(self.answer_type_head(h), dim=-1),
        }


class FocalBCELoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = probs * targets + (1 - probs) * (1 - targets)
        return ((1 - p_t) ** self.gamma * bce).mean()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--embedding-dir", type=str, default="outputs/embeddings")
    p.add_argument("--output-dir", type=str, default="outputs/fast_run")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--patience", type=int, default=15)
    return p.parse_args()


def main():
    args = parse_args()
    emb_dir = Path(args.embedding_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load cached embeddings ----
    logger.info("Loading cached embeddings...")
    V = np.load(emb_dir / "vision_embeddings.npz")["embeddings"]
    T = np.load(emb_dir / "text_embeddings.npz")["embeddings"]
    Q = np.load(emb_dir / "quality_matrix.npz")
    Y = Q["Y"]
    model_names = list(Q["model_names"])
    with open(emb_dir / "metadata.pkl", "rb") as f:
        meta = pickle.load(f)
    A = meta["answer_types"]

    N, K = Y.shape
    logger.info(f"Records: {N}, Models: {K} ({model_names})")
    logger.info(f"Vision: {V.shape}, Text: {T.shape}, Quality: {Y.shape}")
    logger.info(f"Answer types: short={(A==0).sum()}, medium={(A==1).sum()}, long={(A==2).sum()}")

    # ---- Train/Val split ----
    np.random.seed(42)
    idx = np.random.permutation(N)
    split = int(N * 0.8)
    train_idx = idx[:split]
    val_idx = idx[split:]

    # Convert to tensors
    V_t = torch.FloatTensor(V)
    T_t = torch.FloatTensor(T)
    Y_t = torch.FloatTensor(Y)
    A_t = torch.LongTensor(A)

    train_loader = DataLoader(
        TensorDataset(V_t[train_idx], T_t[train_idx], Y_t[train_idx], A_t[train_idx]),
        batch_size=args.batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(V_t[val_idx], T_t[val_idx], Y_t[val_idx], A_t[val_idx]),
        batch_size=args.batch_size, shuffle=False,
    )
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # ---- Model ----
    model = PredictorWithEmbeddings(
        vision_dim=V.shape[1], text_dim=T.shape[1],
        num_models=K, dropout=0.1,
    ).to(args.device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable params: {n_params:,}")

    # ---- Training ----
    suit_criterion = FocalBCELoss(gamma=2.0)
    type_criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(args.epochs):
        # Train
        model.train()
        train_losses, suit_losses, type_losses = [], [], []

        for v, t, y, a in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]", leave=False):
            v, t, y, a = v.to(args.device), t.to(args.device), y.to(args.device), a.to(args.device)
            outputs = model(v, t)

            suit_loss = suit_criterion(outputs["suitability_logits"], y)
            type_loss = type_criterion(outputs["answer_type_logits"], a)
            loss = suit_loss + args.alpha * type_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            suit_losses.append(suit_loss.item())
            type_losses.append(type_loss.item())

        # Val
        model.eval()
        val_losses, val_suit, val_type = [], [], []
        type_correct, type_total = 0, 0

        with torch.no_grad():
            for v, t, y, a in val_loader:
                v, t, y, a = v.to(args.device), t.to(args.device), y.to(args.device), a.to(args.device)
                outputs = model(v, t)

                suit_loss = suit_criterion(outputs["suitability_logits"], y)
                type_loss = type_criterion(outputs["answer_type_logits"], a)
                loss = suit_loss + args.alpha * type_loss

                val_losses.append(loss.item())
                val_suit.append(suit_loss.item())
                val_type.append(type_loss.item())

                pred_type = outputs["answer_type_probs"].argmax(-1)
                type_correct += (pred_type == a).sum().item()
                type_total += len(a)

        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)
        type_acc = type_correct / type_total

        logger.info(f"Epoch {epoch+1:3d}: train_loss={avg_train:.4f} "
                    f"(s={np.mean(suit_losses):.4f} t={np.mean(type_losses):.4f}) | "
                    f"val_loss={avg_val:.4f} type_acc={type_acc:.4f}")

        scheduler.step()

        # Checkpoint
        ckpt = {
            "epoch": epoch + 1, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": avg_val, "model_names": model_names, "num_models": K,
        }

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            torch.save(ckpt, out_dir / "predictor_best.pt")
            logger.info(f"  Best model saved (val_loss={avg_val:.4f})")
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    torch.save(ckpt, out_dir / "predictor_final.pt")
    logger.info(f"Training complete. Best val_loss={best_val_loss:.4f}")
    logger.info(f"Model saved to {out_dir}/predictor_best.pt")


if __name__ == "__main__":
    main()
