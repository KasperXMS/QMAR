#!/usr/bin/env python
"""Evaluate trained Predictor + QMAR scheduler with all baselines.

Usage:
    python evaluate_qmar.py
    python evaluate_qmar.py --threshold 0.5 --output-dir outputs/eval
"""

import argparse
import logging
import sys
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from train_predictor_fast import PredictorWithEmbeddings
from profiling.latency_table import LatencyTableBuilder
from profiling.communication_profile import build_comm_profile
from scheduler.latency_estimator import LatencyEstimator
from scheduler.qmar import QMARScheduler
from scheduler.baselines import (
    RandomFeasibleBaseline,
    FastestFeasibleBaseline,
    HighestSuitabilityBaseline,
    LatencyOnlyGreedyBaseline,
    QMARWithoutAnswerTypeBaseline,
    OracleBaseline,
)
from evaluation.metrics import evaluate_all, build_summary_table
from evaluation.plots import plot_method_comparison, plot_load_distribution

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--embedding-dir", type=str, default="outputs/embeddings")
    p.add_argument("--checkpoint", type=str, default="outputs/fast_run/predictor_best.pt")
    p.add_argument("--instance-config", type=str, default="configs/instance_pool.yaml")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--output-dir", type=str, default="outputs/eval")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    emb_dir = Path(args.embedding_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load cached data ----
    logger.info("Loading cached embeddings...")
    V = np.load(emb_dir / "vision_embeddings.npz")["embeddings"]
    T = np.load(emb_dir / "text_embeddings.npz")["embeddings"]
    Q = np.load(emb_dir / "quality_matrix.npz")
    Y_gt = Q["Y"]
    model_names = list(Q["model_names"])
    with open(emb_dir / "metadata.pkl", "rb") as f:
        meta = pickle.load(f)

    N, K = Y_gt.shape
    logger.info(f"Records: {N}, Models: {K}")

    # ---- Load predictor ----
    logger.info(f"Loading predictor from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    predictor = PredictorWithEmbeddings(
        vision_dim=V.shape[1], text_dim=T.shape[1],
        num_models=K, dropout=0.1,
    ).to(args.device)
    predictor.load_state_dict(ckpt["model_state_dict"])
    predictor.eval()
    logger.info(f"Loaded predictor (epoch={ckpt.get('epoch')}, val_loss={ckpt.get('val_loss'):.4f})")

    # ---- Inference: get suitability scores + answer types ----
    logger.info("Running inference...")
    V_t = torch.FloatTensor(V).to(args.device)
    T_t = torch.FloatTensor(T).to(args.device)

    batch_size = 512
    suit_probs = []
    type_preds = []

    with torch.no_grad():
        for i in range(0, N, batch_size):
            v_batch = V_t[i : i + batch_size]
            t_batch = T_t[i : i + batch_size]
            out = predictor(v_batch, t_batch)
            suit_probs.append(out["suitability_probs"].cpu().numpy())
            type_preds.append(out["answer_type_probs"].argmax(-1).cpu().numpy())

    suit_scores = np.concatenate(suit_probs, axis=0)  # (N, K)
    type_idx = np.concatenate(type_preds, axis=0)      # (N,)
    idx_to_type = {0: "simple", 1: "moderate", 2: "complex"}
    answer_types = [idx_to_type.get(int(t), "moderate") for t in type_idx]

    logger.info(f"Suitability scores: {suit_scores.shape}")
    logger.info(f"Answer type distribution: short={(type_idx==0).sum()}, "
                f"moderate={(type_idx==1).sum()}, complex={(type_idx==2).sum()}")

    # ---- Setup instance pool ----
    with open(args.instance_config) as f:
        inst_cfg = yaml.safe_load(f)
    instance_config = inst_cfg["instances"]
    instance_ids = [c["instance_id"] for c in instance_config]

    # ---- Build latency table (TOPS-calibrated physical model) ----
    from profiling.latency_table import build_theoretical_table

    latency_table = build_theoretical_table(instance_config)
    # To calibrate with real profiling, replace the line above with:
    #   alpha = calibrate_alpha("Phi-3.5-Vision", "AGX_Orin", "simple", measured=195)
    #   latency_table = build_theoretical_table(instance_config, alpha=alpha)

    comm_profile = build_comm_profile(instance_config)

    # ---- Build latency matrix (simplified: answer-type-based, no comm cost) ----
    K_inst = len(instance_ids)
    L = np.zeros((N, K_inst), dtype=np.float32)
    instance_lat = {}
    for _, row in latency_table.iterrows():
        iid = row["instance_id"]
        instance_lat[iid] = {"simple": row["simple"], "moderate": row["moderate"], "complex": row["complex"]}

    for k, iid in enumerate(instance_ids):
        for i in range(N):
            at = answer_types[i]
            L[i, k] = instance_lat.get(iid, {}).get(at, 500.0)

    logger.info(f"Latency matrix: {L.shape}")

    # Build ground-truth answer type strings for oracle
    gt_idx_to_type = {0: "simple", 1: "moderate", 2: "complex"}
    gt_answer_types_list = [
        gt_idx_to_type.get(int(meta["answer_types"][i]), "moderate")
        for i in range(N)
    ]

    # ---- Run all schedulers ----
    logger.info(f"\nRunning schedulers (τ={args.threshold})...")
    methods = {
        "QMAR_full": QMARScheduler(instance_config, model_names, args.threshold),
        "Random_feasible": RandomFeasibleBaseline(instance_config, model_names, args.threshold),
        "Fastest_feasible": FastestFeasibleBaseline(instance_config, model_names, args.threshold),
        "Highest_suitability": HighestSuitabilityBaseline(instance_config, model_names, args.threshold),
        "Latency_only_greedy": LatencyOnlyGreedyBaseline(instance_config, model_names, args.threshold),
        "QMAR_wo_answer_type": QMARWithoutAnswerTypeBaseline(instance_config, model_names, args.threshold),
        "Oracle": OracleBaseline(
            instance_config, model_names, args.threshold,
            gt_suitability=Y_gt,
            gt_answer_types=gt_answer_types_list,
        ),
    }

    # Create minimal records for scheduler
    class FakeRecord:
        def __init__(self, rid):
            self.request_id = rid
            self.image_path = None
            self.question_text = ""
            self.candidate_model_correctness = {}

    fake_records = [FakeRecord(meta["request_ids"][i]) for i in range(N)]

    all_results = {}
    for name, scheduler in methods.items():
        result = scheduler.schedule(
            records=fake_records,
            suitability_scores=suit_scores,
            latency_matrix=L,
            predicted_answer_types=answer_types,
        )
        metrics = evaluate_all(
            result["assignments"], result["instance_loads"], result["batch_latency"],
        )
        all_results[name] = metrics
        logger.info(f"  {name:25s}: batch_lat={result['batch_latency']:8.0f}ms  "
                    f"fallback={metrics['quality']['fallback_rate']:.4f}  "
                    f"load_cv={metrics['system']['load_cv']:.4f}")

    # ---- Save results ----
    summary = build_summary_table(all_results)
    print("\n" + summary.to_string())
    summary.to_csv(out_dir / "summary_metrics.csv", index=False)

    plot_method_comparison(all_results, output_path=str(out_dir / "method_comparison.png"))

    # Threshold sweep
    logger.info("\nThreshold sweep...")
    sweep_rows = []
    for tau in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        scheduler = QMARScheduler(instance_config, model_names, tau)
        result = scheduler.schedule(fake_records, suit_scores, L, answer_types)
        m = evaluate_all(result["assignments"], result["instance_loads"], result["batch_latency"])
        sweep_rows.append({
            "threshold": tau,
            "batch_latency_ms": m["system"]["batch_latency_ms"],
            "fallback_rate": m["quality"]["fallback_rate"],
            "load_cv": m["system"]["load_cv"],
        })
        logger.info(f"  τ={tau}: batch_lat={m['system']['batch_latency_ms']:.0f}ms  "
                    f"fallback={m['quality']['fallback_rate']:.4f}")

    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(out_dir / "threshold_sweep.csv", index=False)

    from evaluation.plots import plot_threshold_sweep
    plot_threshold_sweep(sweep_df, output_path=str(out_dir / "threshold_sweep.png"))

    logger.info(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()
