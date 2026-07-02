#!/usr/bin/env python
"""Compare QMAR with LatentRouter utilities vs our Predictor.

Loads the pre-trained LatentRouter on VL-RouterBench test split,
uses its predicted utilities as suitability scores in QMAR,
and compares against our own Predictor-driven QMAR.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import yaml
import time
import logging

from latentrouter.routers import BaseRouter
from latentrouter.embedding.store import load_router_bundle

from profiling.latency_table import LatencyTableBuilder
from scheduler.qmar import QMARScheduler
from scheduler.baselines import (
    OracleBaseline, FastestFeasibleBaseline, RandomFeasibleBaseline,
    HighestSuitabilityBaseline, LatencyOnlyGreedyBaseline,
)
from evaluation.metrics import evaluate_all, build_summary_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("compare")


# Model name mapping: LatentRouter dir names → our canonical names
LR_TO_CANONICAL = {
    "deepseek_vl2": "DeepSeek-VL2",
    "deepseek_vl2_tiny": "DeepSeek-VL2-Tiny",
    "GeminiFlash2-5": "Gemini-Flash-2.5",
    "Gemma3-27B": "Gemma3-27B",
    "GPT4o": "GPT-4o",
    "InternVL2_5-78B": "InternVL2_5-78B",
    "Janus-Pro-1B": "Janus-Pro-1B",
    "Janus-Pro-7B": "Janus-Pro-7B",
    "Kimi-VL-A3B-Thinking-2506": "Kimi-VL-A3B-Thinking",
    "llava_next_vicuna_7b": "LLaVA-Next-Vicuna-7B",
    "MiMo-VL-7B-RL": "MiMo-VL-7B-RL",
    "Phi-3.5-Vision": "Phi-3.5-Vision",
    "Pixtral-12B": "Pixtral-12B",
    "Qianfan-VL-8B": "Qianfan-VL-8B",
    "Qwen2.5-VL-32B-Instruct": "Qwen2.5-VL-32B",
    "Qwen2.5-VL-72B-Instruct": "Qwen2.5-VL-72B",
    "SmolVLM2": "SmolVLM2",
}


def main():
    # ── Load LatentRouter ──
    logger.info("Loading LatentRouter model...")
    router = BaseRouter.load(
        "/home/super/xiaoming/LatentRouter/artifacts/models/vl_latentrouter.pkl"
    )
    logger.info(f"Loaded: {type(router).__name__}, setting={router.setting}")

    # ── Load test bundle ──
    logger.info("Loading test bundle...")
    bundle = load_router_bundle(
        "/home/super/xiaoming/LatentRouter/data/processed/vl_routerbench", split="test"
    )
    N, M = bundle.features.shape[0], len(bundle.model_ids)
    logger.info(f"Test samples: {N}, Models: {M}")

    # ── Predict utilities ──
    logger.info("Predicting utilities...")
    t0 = time.time()
    utilities = router.predict_utilities(bundle)  # (N, M), mu_tilde in [0,1]
    logger.info(f"Prediction done in {time.time()-t0:.1f}s, shape={utilities.shape}")

    # ── Map model names ──
    canonical_models = [LR_TO_CANONICAL.get(m, m) for m in bundle.model_ids]
    logger.info(f"Models: {canonical_models}")

    # ── Load instance pool ──
    with open("configs/instance_pool.yaml") as f:
        inst = yaml.safe_load(f)
    instance_config = inst["instances"]
    instance_ids = [c["instance_id"] for c in instance_config]

    # Build latency table
    builder = LatencyTableBuilder()
    for cfg in instance_config:
        builder.add_instance(**cfg)
    lat_table = builder.build()
    instance_lat = {}
    for _, row in lat_table.iterrows():
        iid = row["instance_id"]
        instance_lat[iid] = {"simple": row["simple"], "moderate": row["moderate"],
                             "complex": row["complex"]}

    K_inst = len(instance_ids)

    # ── Build latency matrix: use sample metadata for answer types ──
    sample_frame = bundle.sample_frame
    # Determine answer type per sample from dataset_name
    from data_adapters.common_schema import AnswerType

    class MiniRec:
        def __init__(self, ds_name):
            self.dataset_name = ds_name
            self.task_name = ds_name
            self.question_text = ""
            self.ground_truth_answer = ""
            self.request_id = ""
            self.image_path = None
            self.candidate_model_correctness = {}
            self.extra = {}

    L = np.zeros((N, K_inst), dtype=np.float32)
    atypes = []
    for i in range(N):
        ds = str(sample_frame.iloc[i].get("dataset_name", "MMStar"))
        rec = MiniRec(ds)
        at = AnswerType.classify(rec)
        atypes.append(at)
        for k, iid in enumerate(instance_ids):
            L[i, k] = instance_lat[iid].get(at, 500)

    logger.info(f"Latency matrix: {L.shape}")
    logger.info(f"Answer types: simple={(np.array(atypes)=='simple').sum()}, "
                f"moderate={(np.array(atypes)=='moderate').sum()}, "
                f"complex={(np.array(atypes)=='complex').sum()}")

    # ── Use LatentRouter utilities as suitability scores ──
    # Clip to [0,1] and treat as suitability
    suit_lr = np.clip(utilities, 0, 1).astype(np.float32)
    logger.info(f"LatentRouter suitability: mean={suit_lr.mean():.3f}, "
                f"min={suit_lr.min():.3f}, max={suit_lr.max():.3f}")

    # ── Run QMAR + baselines ──
    class FR:
        def __init__(self, i): self.request_id = str(i); self.image_path = None
        def __getattr__(self, _): return ""

    records = [FR(i) for i in range(N)]
    threshold = 0.5

    methods = {
        "QMAR + LatentRouter": QMARScheduler(instance_config, canonical_models, threshold),
        "Latency_only_greedy": LatencyOnlyGreedyBaseline(instance_config, canonical_models, threshold),
        "Fastest_feasible": FastestFeasibleBaseline(instance_config, canonical_models, threshold),
        "Random_feasible": RandomFeasibleBaseline(instance_config, canonical_models, threshold, seed=42),
        "Highest_suitability": HighestSuitabilityBaseline(instance_config, canonical_models, threshold),
        "Oracle (unconstrained)": OracleBaseline(instance_config, canonical_models, threshold),
    }

    logger.info(f"\nRunning schedulers (τ={threshold})...")
    all_results = {}
    for name, sched in methods.items():
        t0 = time.time()
        r = sched.schedule(records, suit_lr, L, atypes)
        elapsed = (time.time() - t0) * 1000
        metrics = evaluate_all(r["assignments"], r["instance_loads"], r["batch_latency"])
        all_results[name] = metrics
        logger.info(f"  {name:25s}: batch_lat={r['batch_latency']:>10.0f}ms  "
                    f"fallback={metrics['quality']['fallback_rate']:.4f}  "
                    f"load_cv={metrics['system']['load_cv']:.4f}  "
                    f"({elapsed:.0f}ms)")

    # ── Summary ──
    summary = build_summary_table(all_results)
    print("\n" + summary.to_string())

    # ── Compare with our Predictor (from eval2) ──
    print("\n── Comparison: LatentRouter vs Our Predictor (τ=0.5) ──")
    print(f"  {'Method':<30s} {'Batch Lat':>10s} {'Fallback':>10s} {'Load CV':>10s}")
    print(f"  {'─'*30} {'─'*10} {'─'*10} {'─'*10}")
    for name in ["QMAR + LatentRouter", "Latency_only_greedy", "Fastest_feasible",
                 "Oracle (unconstrained)"]:
        if name in all_results:
            m = all_results[name]
            print(f"  {name:<30s} {m['system']['batch_latency_ms']:>10.0f}ms "
                  f"{m['quality']['fallback_rate']:>10.4f} "
                  f"{m['system']['load_cv']:>10.4f}")


if __name__ == "__main__":
    main()
