#!/usr/bin/env python
"""Train the multi-task multimodal Predictor on VL-RouterBench data.

Usage:
    conda activate qmar
    python train_predictor.py                          # default: 3 models, 3 datasets
    python train_predictor.py --models all --datasets all  # full scale
    python train_predictor.py --max-samples 5000 --epochs 20 --device cuda
    python train_predictor.py --resume outputs/checkpoints/predictor_best.pt
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
from tqdm import tqdm

from data_adapters.vl_routerbench_adapter import (
    VLRouterBenchAdapter,
    MODEL_DIR_MAP,
    DATASET_TSV_MAP,
)
from data_adapters.common_schema import AnswerType
from profiling.quality_profile import get_model_pool
from profiling.answer_type_labels import build_answer_type_labels
from predictor.model import Predictor
from predictor.train import train_predictor
from predictor.infer import load_predictor, predict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("train")


def parse_args():
    p = argparse.ArgumentParser(description="Train QMAR Predictor on VL-RouterBench")
    # Data
    p.add_argument("--data-root", type=str,
                   default="/home/super/xiaoming/VL-RouterBench/vlm_router_data")
    p.add_argument("--models", type=str, default="3",
                   help="Model count or comma-separated names, or 'all'")
    p.add_argument("--datasets", type=str, default="3",
                   help="Dataset count or comma-separated names, or 'all'")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Max samples per dataset (None=all)")
    # Training
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--alpha", type=float, default=1.0,
                   help="Weight for answer-type loss")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--num-workers", type=int, default=0)
    # Output
    p.add_argument("--output-dir", type=str, default="outputs")
    p.add_argument("--resume", type=str, default=None,
                   help="Resume from checkpoint")
    # Predictor architecture
    p.add_argument("--vision-encoder", type=str,
                   default="openai/clip-vit-base-patch32")
    p.add_argument("--text-encoder", type=str,
                   default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--no-focal", action="store_true",
                   help="Disable focal loss")
    return p.parse_args()


def resolve_models(arg: str):
    """Resolve model selection argument."""
    all_models = list(MODEL_DIR_MAP.values())
    if arg == "all":
        return all_models
    if arg.isdigit():
        n = int(arg)
        return all_models[:n]
    return [m.strip() for m in arg.split(",")]


def resolve_datasets(arg: str):
    """Resolve dataset selection argument."""
    all_datasets = list(DATASET_TSV_MAP.keys())
    if arg == "all":
        return all_datasets
    if arg.isdigit():
        n = int(arg)
        return all_datasets[:n]
    return [d.strip() for d in arg.split(",")]


def main():
    args = parse_args()

    model_names = resolve_models(args.models)
    datasets = resolve_datasets(args.datasets)
    logger.info(f"Models ({len(model_names)}): {model_names}")
    logger.info(f"Datasets ({len(datasets)}): {datasets}")

    # ---- Load data ----
    logger.info("Loading VL-RouterBench data...")
    adapter = VLRouterBenchAdapter(
        data_root=args.data_root,
        model_names=model_names,
        datasets=datasets,
    )
    records = adapter.load_all_records(max_per_dataset=args.max_samples)
    logger.info(f"Loaded {len(records)} records")

    if len(records) == 0:
        logger.error("No records loaded. Check --data-root and model/dataset names.")
        return

    # Filter to records that have at least one correctness label
    records = [r for r in records if r.candidate_model_correctness]
    logger.info(f"Records with correctness labels: {len(records)}")

    # Re-derive model_names from actually loaded data
    model_names = get_model_pool(records)
    logger.info(f"Models in data: {model_names}")

    # ---- Build answer type labels ----
    logger.info("Building answer type labels...")
    at_df = build_answer_type_labels(records)
    logger.info(f"Answer type distribution:\n{at_df['answer_type'].value_counts().to_string()}")

    # ---- Train/Val split ----
    np.random.seed(42)
    indices = np.random.permutation(len(records))
    split = int(len(records) * 0.8)
    train_records = [records[i] for i in indices[:split]]
    val_records = [records[i] for i in indices[split:]]
    logger.info(f"Train: {len(train_records)}, Val: {len(val_records)}")

    # ---- Create Predictor ----
    logger.info("Creating Predictor...")
    if args.resume:
        predictor, loaded_models = load_predictor(args.resume, device=args.device)
        logger.info(f"Resumed from {args.resume}")
        if set(loaded_models) != set(model_names):
            logger.warning(f"Model mismatch: checkpoint has {loaded_models}, data has {model_names}")
    else:
        predictor = Predictor(
            num_models=len(model_names),
            vision_encoder_name=args.vision_encoder,
            text_encoder_name=args.text_encoder,
        )

    # Create image loader from adapter
    def image_loader(rec):
        """Load image for a record from TSV."""
        ds = rec.dataset_name
        idx = rec.extra.get("tsv_index", "0")
        if ds not in DATASET_TSV_MAP:
            for d in DATASET_TSV_MAP:
                if d in rec.request_id:
                    ds = d
                    break
        return adapter.get_image(ds, idx)

    # ---- Train ----
    logger.info(f"Starting training on device={args.device}...")
    logger.info(f"  Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history = train_predictor(
        predictor=predictor,
        train_records=train_records,
        val_records=val_records,
        model_names=model_names,
        output_dir=str(output_dir),
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        alpha=args.alpha,
        use_focal=not args.no_focal,
        device=args.device,
        num_workers=args.num_workers,
        image_loader=image_loader,
    )

    # ---- Final evaluation ----
    logger.info("Running final inference on validation set...")
    pred_results = predict(
        predictor=predictor,
        records=val_records,
        model_names=model_names,
        device=args.device,
        image_loader=image_loader,
    )

    # Compute suitability accuracy
    from sklearn.metrics import roc_auc_score

    # Build ground truth matrix
    n_val = len(val_records)
    y_true = np.zeros((n_val, len(model_names)), dtype=np.float32)
    for i, rec in enumerate(val_records):
        for m, c in rec.candidate_model_correctness.items():
            if m in model_names:
                y_true[i, model_names.index(m)] = float(c >= 0.5)

    suit_preds = pred_results["suitability_scores"]

    # Per-model AUC
    logger.info("Suitability AUC per model:")
    for j, m in enumerate(model_names):
        if y_true[:, j].sum() > 0 and y_true[:, j].sum() < n_val:
            try:
                auc = roc_auc_score(y_true[:, j], suit_preds[:, j])
                logger.info(f"  {m:30s}: AUC={auc:.4f}")
            except ValueError:
                pass

    # Answer type accuracy
    type_true = []
    for rec in val_records:
        at = AnswerType.from_text(rec.ground_truth_answer)
        type_true.append(AnswerType.to_index(at))
    type_true = np.array(type_true)
    type_acc = (pred_results["predicted_answer_types"] == type_true).mean()
    logger.info(f"Answer type accuracy: {type_acc:.4f}")

    logger.info(f"Best checkpoint saved to {output_dir / 'checkpoints' / 'predictor_best.pt'}")
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
