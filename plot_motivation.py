#!/usr/bin/env python
"""Motivation figure: LatentRouter direct routing vs Oracle vs QMAR.

Shows load distribution of three strategies on the same test data.
Clear visual proof that latency-aware scheduling is essential.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import yaml
import logging

from latentrouter.routers import BaseRouter
from latentrouter.embedding.store import load_router_bundle

from profiling.latency_table import LatencyTableBuilder
from scheduler.qmar import QMARScheduler
from scheduler.baselines import OracleBaseline
from data_adapters.common_schema import AnswerType

logging.basicConfig(level=logging.WARNING)

LR_TO_CANONICAL = {
    "deepseek_vl2": "DeepSeek-VL2", "deepseek_vl2_tiny": "DeepSeek-VL2-Tiny",
    "GeminiFlash2-5": "Gemini-Flash-2.5", "Gemma3-27B": "Gemma3-27B",
    "GPT4o": "GPT-4o", "InternVL2_5-78B": "InternVL2_5-78B",
    "Janus-Pro-1B": "Janus-Pro-1B", "Janus-Pro-7B": "Janus-Pro-7B",
    "Kimi-VL-A3B-Thinking-2506": "Kimi-VL-A3B-Thinking",
    "llava_next_vicuna_7b": "LLaVA-Next-Vicuna-7B", "MiMo-VL-7B-RL": "MiMo-VL-7B-RL",
    "Phi-3.5-Vision": "Phi-3.5-Vision", "Pixtral-12B": "Pixtral-12B",
    "Qianfan-VL-8B": "Qianfan-VL-8B",
    "Qwen2.5-VL-32B-Instruct": "Qwen2.5-VL-32B",
    "Qwen2.5-VL-72B-Instruct": "Qwen2.5-VL-72B", "SmolVLM2": "SmolVLM2",
}


def main():
    # ── Load ──
    print("Loading LatentRouter...", end=" ", flush=True)
    router = BaseRouter.load(
        "/home/super/xiaoming/LatentRouter/artifacts/models/vl_latentrouter.pkl")
    bundle = load_router_bundle(
        "/home/super/xiaoming/LatentRouter/data/processed/vl_routerbench", split="test")
    utilities = np.clip(router.predict_utilities(bundle), 0, 1).astype(np.float32)
    canonical_models = [LR_TO_CANONICAL.get(m, m) for m in bundle.model_ids]
    N, M = utilities.shape
    print(f"{N} samples, {M} models")

    # ── Instances + latency ──
    with open("configs/instance_pool.yaml") as f:
        inst = yaml.safe_load(f)
    instance_config = inst["instances"]
    K = len(instance_config)
    instance_ids = [c["instance_id"] for c in instance_config]

    builder = LatencyTableBuilder()
    for cfg in instance_config:
        builder.add_instance(**cfg)
    lat_table = builder.build()
    instance_lat = {}
    for _, row in lat_table.iterrows():
        iid = row["instance_id"]
        instance_lat[iid] = {"simple": row["simple"], "moderate": row["moderate"],
                             "complex": row["complex"]}

    L = np.zeros((N, K), dtype=np.float32)
    atypes = []
    for i in range(N):
        ds = str(bundle.sample_frame.iloc[i].get("dataset_name", "MMStar"))
        rec = type('R', (), {'dataset_name': ds, 'task_name': ds,
                              'question_text': '', 'ground_truth_answer': ''})()
        at = AnswerType.classify(rec)
        atypes.append(at)
        for k, iid in enumerate(instance_ids):
            L[i, k] = instance_lat[iid][at]

    class FR:
        def __init__(self, i): self.request_id = str(i); self.image_path = None
    records = [FR(i) for i in range(N)]

    # ── Strategy A: LatentRouter direct (pick best model, assign to its instance) ──
    lr_loads = {iid: 0.0 for iid in instance_ids}
    lr_count = {iid: 0 for iid in instance_ids}
    for i in range(N):
        bk = int(np.argmax(utilities[i]))
        # find first instance deploying this model
        best_k = next((k for k, c in enumerate(instance_config)
                       if c["model_name"] == canonical_models[bk]), bk % K)
        lr_loads[instance_ids[best_k]] += L[i, best_k]
        lr_count[instance_ids[best_k]] += 1
    lr_ms = max(lr_loads.values())

    # ── Strategy B: Oracle (unconstrained LPT + local search) ──
    oracle = OracleBaseline(instance_config, canonical_models, 0.5)
    or_result = oracle.schedule(records, utilities, L, atypes)
    or_loads = or_result["instance_loads"]
    or_ms = or_result["batch_latency"]

    # ── Strategy C: QMAR ──
    qmar = QMARScheduler(instance_config, canonical_models, 0.5)
    qm_result = qmar.schedule(records, utilities, L, atypes)
    qm_loads = qm_result["instance_loads"]
    qm_ms = qm_result["batch_latency"]

    # ── Output ──
    def bar(ld, mx, w=40):
        n = int(ld / mx * w) if mx > 0 else 0
        return "█" * n + "·" * max(0, w - n)

    max_load = max(max(lr_loads.values()), max(or_loads.values()),
                   max(qm_loads.values()))

    strategies = [
        ("LatentRouter direct (quality-only routing)", lr_loads, lr_ms, lr_count,
         "每个请求选 utility 最高的模型 → 全部挤到最强模型，其余设备空转"),
        ("Oracle (unconstrained LPT + local search)", or_loads, or_ms,
         {iid: sum(1 for a in or_result["assignments"] if a["assigned_instance"] == iid)
          for iid in instance_ids},
         "无质量约束，纯 makespan 最优 → 理论下界，但质量无保证"),
        ("QMAR (quality-constrained load balancing)", qm_loads, qm_ms,
         {iid: sum(1 for a in qm_result["assignments"] if a["assigned_instance"] == iid)
          for iid in instance_ids},
         "feasible set 约束下负载均衡 → 质量保证 + 近最优 makespan"),
    ]

    for title, loads, ms, counts, desc in strategies:
        print(f"\n{'─'*80}")
        print(f"  {title}")
        print(f"  {desc}")
        print(f"{'─'*80}")
        print(f"  {'Instance':>22s} {'Model':<24s} {'Device':<10s} {'Load':>8s} {'Reqs':>5s}")
        print(f"  {'─'*22} {'─'*24} {'─'*10} {'─'*8} {'─'*5}")
        for inst_id, cfg in zip(instance_ids, instance_config):
            ld = loads[inst_id]
            cnt = counts[inst_id]
            print(f"  {inst_id:>22s} {cfg['model_name']:<24s} {cfg['device_class']:<10s} "
                  f"{ld:>8.0f}ms {bar(ld, max_load)} {cnt:>5d}")
        cv = np.std(list(loads.values())) / np.mean(list(loads.values()))
        print(f"  {'MAKESPAN':>60s} = {ms:.0f}ms,  Load CV = {cv:.3f}")

    print(f"\n{'═'*80}")
    print(f"  Motivation Summary")
    print(f"{'═'*80}")
    print(f"  LatentRouter direct:  {lr_ms:>8.0f}ms  — 纯质量路由，负载极度不均")
    print(f"  Oracle:               {or_ms:>8.0f}ms  — 理论下界，无质量保证")
    print(f"  QMAR:                 {qm_ms:>8.0f}ms  — 质量保证 + 近最优 makespan")
    print(f"")
    print(f"  QMAR vs LatentRouter direct: 快 {(lr_ms-qm_ms)/lr_ms*100:.0f}%  ({(lr_ms-qm_ms)/1000:.0f}s 节省)")
    print(f"  QMAR vs Oracle:              慢 {(qm_ms-or_ms)/or_ms*100:.1f}%  (质量约束的代价)")


if __name__ == "__main__":
    main()
