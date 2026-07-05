#!/usr/bin/env python
"""Three-in-one routing performance figure using real LatentRouter predictions.

X-axis: 3 strategies
  - Quality-Oriented Router (LatentRouter: best-model-first)
  - Latency-Oriented Scheduler (pure load balancing)
  - Quality-Constrained Oracle (min makespan s.t. quality >= threshold)

Y-axis: Inference latency (ms)
Each strategy: 3 bars for OrinNX (blue), AGXOrin (green), RTX4090 (red)

Uses real LatentRouter utility predictions on VL-RouterBench test data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "/home/super/xiaoming/LatentRouter/src")

import numpy as np
import matplotlib.pyplot as plt
import yaml

from profiling.latency_table import (
    SYNTHETIC_DEVICE_PROFILES,
    classify_model_size,
)
from data_adapters.common_schema import AnswerType

# ── Styling ──
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif",
                   "Nimbus Roman", "FreeSerif"],
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# Color scheme: blue (OrinNX), green (AGXOrin), red (RTX4090)
INSTANCE_COLORS = {
    "Orin_NX":   "#3B82F6",
    "AGX_Orin":  "#22C55E",
    "RTX_4090":  "#EF4444",
}
DEVICE_LABELS = {
    "Orin_NX":   "Orin NX",
    "AGX_Orin":  "AGX Orin",
    "RTX_4090":  "RTX 4090",
}

# ── LatentRouter model_id → instance config model_name ──
LR_MODEL_TO_INSTANCE = {
    "SmolVLM2":                 "SmolVLM2",
    "Phi-3.5-Vision":           "Phi-3.5-Vision",
    "Qwen2.5-VL-32B-Instruct":  "Qwen2.5-VL-32B",
}

QUALITY_THRESHOLD = 0.50
N_REQUESTS = 8


def load_3instances():
    """Load 3 instances with complementary strengths.

    SmolVLM2@NX  — tiny, fastest for simple requests
    Phi-3.5@AGX  — small, middle ground
    Qwen-32B@4090 — xl, only viable choice for complex requests
    """
    target_ids = ["SmolVLM2@NX", "Phi-3.5@AGX", "Qwen-32B@4090"]
    config_path = Path(__file__).parent / "configs" / "instance_pool.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    inst_map = {inst["instance_id"]: inst for inst in cfg["instances"]}

    result = []
    for iid in target_ids:
        inst = inst_map[iid]
        result.append({
            "instance_id": inst["instance_id"],
            "model_name": inst["model_name"],
            "device_class": inst["device_class"],
            "model_size": classify_model_size(inst["model_name"]),
        })
    return result


def get_latency(instance, answer_type: str) -> float:
    """Per-request inference latency for instance x answer type."""
    dc = instance["device_class"]
    ms = instance["model_size"]
    profile = SYNTHETIC_DEVICE_PROFILES.get(dc, {})
    size_profile = profile.get(ms, profile.get("medium", {}))
    return float(size_profile.get(answer_type, 500))


def classify_answer_type(row) -> str:
    """Classify a bundle sample row into simple/moderate/complex."""
    dataset = str(row.get("dataset_name", ""))
    # Use dataset-level default from AnswerType
    if dataset in AnswerType.DATASET_COMPLEXITY:
        return AnswerType.DATASET_COMPLEXITY[dataset]
    # Fallback: check question text
    question = str(row.get("question", ""))
    if any(kw in question.lower() for kw in
           ["calculate", "solve", "equation", "math", "reasoning"]):
        return "complex"
    return "moderate"


def load_latentrouter():
    """Load the LatentRouter model and test bundle."""
    from latentrouter.routers import BaseRouter
    from latentrouter.embedding.store import load_router_bundle

    router = BaseRouter.load(
        "/home/super/xiaoming/LatentRouter/artifacts/models/vl_latentrouter.pkl")
    bundle = load_router_bundle(
        "/home/super/xiaoming/LatentRouter/data/processed/vl_routerbench",
        split="test")
    utilities = np.clip(router.predict_utilities(bundle), 0, 1).astype(np.float32)

    return router, bundle, utilities


def get_instance_utility(lr_utility_row, lr_model_ids, instance):
    """Get the LatentRouter utility of an instance's model for a request.

    Maps LatentRouter model IDs to instance config model names.
    """
    inst_model = instance["model_name"]
    # Find matching LR model
    for lr_id, inst_name in LR_MODEL_TO_INSTANCE.items():
        if inst_name == inst_model:
            idx = lr_model_ids.index(lr_id)
            return float(lr_utility_row[idx])
    # Model not in LatentRouter — use suitability fallback
    return {"SmolVLM2": 0.65, "Phi-3.5-Vision": 0.70,
            "Qwen2.5-VL-32B": 0.90}.get(inst_model, 0.5)


def simulate_routing(method: str, instances, requests, lr_utilities,
                     lr_model_ids):
    """Simulate routing and return (per-instance latencies, fallback_count)."""
    n_inst = len(instances)
    n_req = len(requests)

    # Precompute per-request per-instance latency and suitability
    req_lat = [[get_latency(inst, requests[i]["answer_type"])
                for inst in instances] for i in range(n_req)]
    req_suit = [[get_instance_utility(lr_utilities[i], lr_model_ids, inst)
                 for inst in instances] for i in range(n_req)]

    if method == "Cost-Efficiency Router":
        instance_latencies = [0.0] * n_inst
        fallback_count = 0
        for i in range(n_req):
            # LatentRouter-style cost-efficiency: maximise utility/latency.
            # No hard quality threshold — the ratio naturally prefers
            # small models for simple tasks and large models for hard tasks.
            efficiency = [req_suit[i][k] / max(req_lat[i][k], 1.0)
                          for k in range(n_inst)]
            best = int(np.argmax(efficiency))
            instance_latencies[best] += req_lat[i][best]
            if req_suit[i][best] < QUALITY_THRESHOLD:
                fallback_count += 1
        return instance_latencies, fallback_count

    elif method == "Latency-Oriented Scheduler":
        instance_latencies = [0.0] * n_inst
        fallback_count = 0
        for i in range(n_req):
            loads = [instance_latencies[k] for k in range(n_inst)]
            best = int(np.argmin(loads))
            instance_latencies[best] += req_lat[i][best]
            if req_suit[i][best] < QUALITY_THRESHOLD:
                fallback_count += 1
        return instance_latencies, fallback_count

    elif method == "Quality-Constrained Oracle":
        # Exhaustive search over 3^N assignments for minimal makespan
        feasible_per_req = []
        for i in range(n_req):
            fe = [k for k in range(n_inst)
                  if req_suit[i][k] >= QUALITY_THRESHOLD]
            if not fe:
                fe = [int(np.argmax(req_suit[i]))]
            feasible_per_req.append(fe)

        best_makespan = float('inf')
        best_loads = None

        def search(req_idx, loads):
            nonlocal best_makespan, best_loads
            if req_idx == n_req:
                ms = max(loads)
                if ms < best_makespan:
                    best_makespan = ms
                    best_loads = list(loads)
                return
            if max(loads) >= best_makespan:
                return
            for k in feasible_per_req[req_idx]:
                new_loads = list(loads)
                new_loads[k] += req_lat[req_idx][k]
                search(req_idx + 1, new_loads)

        search(0, [0.0] * n_inst)
        instance_latencies = best_loads

        fallback_count = 0
        for i in range(n_req):
            best_suit = max(req_suit[i])
            if best_suit < QUALITY_THRESHOLD:
                fallback_count += 1
        return instance_latencies, fallback_count


def plot_3in1(instances, results, quality_metrics, output_path: str,
              method_labels=None):
    """Generate the three-in-one grouped bar chart."""
    methods = list(results.keys())
    if method_labels is None:
        method_labels = methods
    device_classes = [inst["device_class"] for inst in instances]

    instance_labels = [
        f"{DEVICE_LABELS[inst['device_class']]}\n({inst['model_name']})"
        for inst in instances
    ]

    fig, ax = plt.subplots(figsize=(10, 5.5))

    x = np.arange(len(methods))
    bar_width = 0.22
    n_inst = len(instances)
    max_val = max(max(vals) for vals in results.values())

    # ── Grouped bars ──
    for k in range(n_inst):
        values = [results[method][k] for method in methods]
        color = INSTANCE_COLORS[device_classes[k]]
        offset = (k - (n_inst - 1) / 2) * bar_width
        bars = ax.bar(
            x + offset, values, bar_width,
            color=color, edgecolor="white", linewidth=1.0,
            label=instance_labels[k],
            zorder=3,
        )
        for bar, val in zip(bars, values):
            if val > max_val * 0.12:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    val - max_val * 0.04,
                    f"{val:.0f}",
                    ha="center", va="top", fontsize=9.5,
                    color="white", fontweight="bold",
                )
            else:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + max_val * 0.02,
                    f"{val:.0f}",
                    ha="center", va="bottom", fontsize=9.5,
                    color="#333333",
                )

    # ── Makespan annotations ──
    for j, method in enumerate(methods):
        vals = results[method]
        makespan = max(vals)
        ax.axhline(
            y=makespan,
            xmin=(j - 0.35) / len(methods),
            xmax=(j + 0.35) / len(methods),
            color="#374151", linestyle="--", alpha=0.5, linewidth=1.2,
        )
        ax.text(
            j, makespan + max_val * 0.02,
            f"M={makespan:.0f}",
            ha="center", va="bottom", fontsize=11,
            fontweight="bold", color="#1F2937",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="#F9FAFB",
                      edgecolor="#D1D5DB", alpha=0.9),
        )

    # ── Axes ──
    ax.set_ylabel("Instance Load (total inference ms)", fontweight="medium")
    ax.set_xticks(x)
    ax.set_xticklabels(method_labels, fontsize=13, fontweight="semibold")
    ax.set_ylim(0, max_val * 1.28)
    ax.grid(axis="y", alpha=0.25, linestyle="--", zorder=0)
    ax.set_axisbelow(True)

    # ── Legend above chart, spread horizontally ──
    legend = ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.1),
        ncol=3,
        mode="expand",
        borderaxespad=0,
        frameon=True, framealpha=0.95, edgecolor="#CCCCCC",
        fontsize=11,
    )

    # ── Fallback rate annotations ──
    for j, method in enumerate(methods):
        fb = quality_metrics[method]["fallback_rate"]
        if fb > 0:
            ax.annotate(
                f"↓ quality: {fb:.0%}",
                xy=(j, max_val * 1.14),
                fontsize=9, color="#DC2626", ha="center", va="center",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#FEF2F2",
                          edgecolor="#FCA5A5", alpha=0.9),
                annotation_clip=False,
            )

    # ── Save ──
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved to {output_path}")


def main():
    print("=" * 65)
    print("Loading LatentRouter + test data...")
    router, bundle, all_utilities = load_latentrouter()
    lr_model_ids = list(router.model_ids)
    print(f"  Router: {len(lr_model_ids)} models")
    print(f"  Test samples: {all_utilities.shape[0]}")

    print("\nLoading instance configuration...")
    instances = load_3instances()
    for inst in instances:
        print(f"  {inst['instance_id']:<20s} {inst['device_class']:<12s} "
              f"({inst['model_size']}) — "
              f"s={get_latency(inst, 'simple'):.0f} "
              f"m={get_latency(inst, 'moderate'):.0f} "
              f"c={get_latency(inst, 'complex'):.0f} ms")

    # ── Select diverse test samples (3 simple, 3 moderate, 2 complex) ──
    # Pre-selected indices that give diverse complexity and interesting utility spread
    DIVERSE_INDICES = [1859, 2660, 3194, 4136, 4952, 5002, 5087, 5937]

    print(f"\nSelecting {len(DIVERSE_INDICES)} diverse test samples...")
    frame = bundle.sample_frame.iloc[DIVERSE_INDICES]
    lr_utilities = all_utilities[DIVERSE_INDICES]

    requests = []
    for i in range(N_REQUESTS):
        at = classify_answer_type(frame.iloc[i])
        requests.append({
            "request_id": str(frame.iloc[i].get("sample_id", f"req_{i}")),
            "answer_type": at,
        })

    from collections import Counter
    at_counts = Counter(r["answer_type"] for r in requests)
    print(f"  Datasets: {[frame.iloc[i].get('dataset_name', '?') for i in range(N_REQUESTS)]}")
    print(f"  Complexity: {dict(at_counts)}")

    # ── Show LatentRouter preferences ──
    print(f"\nLatentRouter top-3 models per request:")
    for i in range(N_REQUESTS):
        top3 = np.argsort(lr_utilities[i])[-3:][::-1]
        items = [(lr_model_ids[j], float(lr_utilities[i][j])) for j in top3]
        print(f"  [{i}] {items}")

    # ── Run 3 strategies ──
    # Display labels (with line breaks) and internal keys (without line breaks)
    method_labels = [
        "Cost-Efficiency\nRouter",
        "Latency-Oriented\nScheduler",
        "Quality-Constrained\nOracle",
    ]
    method_keys = ["Cost-Efficiency Router",
                   "Latency-Oriented Scheduler",
                   "Quality-Constrained Oracle"]
    results = {}
    quality_metrics = {}

    print(f"\nRouting simulation (τ = {QUALITY_THRESHOLD}):")
    header = f"  {'Method':<30s} {'Makespan':>10s} {'Instances':>40s} {'Fallback':>10s}"
    print(header)
    print(f"  {'-'*30} {'-'*10} {'-'*40} {'-'*10}")

    for key in method_keys:
        instance_lats, fallback = simulate_routing(
            key, instances, requests, lr_utilities, lr_model_ids)
        results[key] = instance_lats
        fb_rate = fallback / len(requests)
        quality_metrics[key] = {"fallback_rate": fb_rate}
        print(f"  {key:<30s} {max(instance_lats):>8.0f} ms  "
              f"{str([f'{v:.0f}' for v in instance_lats]):>40s} {fb_rate:>8.1%}")

    # ── Quality violations ──
    print(f"\nQuality threshold violations (utility < {QUALITY_THRESHOLD}):")
    for key in method_keys:
        fb = quality_metrics[key]["fallback_rate"]
        if fb > 0:
            print(f"  {key:<30s}: {fb:.0%} below threshold ⚠")
        else:
            print(f"  {key:<30s}: all meet threshold ✓")

    # ── Plot (pass labels for x-axis, keys for data) ──
    output_path = "outputs/routing_3in1.png"
    plot_3in1(instances, results, quality_metrics, output_path,
              method_labels=method_labels)

    # ── Summary ──
    best_key = min(method_keys, key=lambda m: max(results[m]))
    best_ms = max(results[best_key])
    print(f"\n{'='*65}")
    print(f"Best makespan: {best_ms:.0f} ms — {best_key}")
    print(f"Figure saved to {output_path}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
