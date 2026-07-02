"""Visualization for QMAR evaluation results."""

import logging
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def plot_threshold_sweep(
    sweep_results: pd.DataFrame,
    output_path: Optional[str] = None,
):
    """Plot quality-latency trade-off across different thresholds.

    Args:
        sweep_results: DataFrame with columns [threshold, batch_latency_ms, fallback_rate, accuracy]
        output_path: Optional path to save figure.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plot")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Latency vs threshold
    ax = axes[0]
    ax.plot(sweep_results["threshold"], sweep_results["batch_latency_ms"], "b-o", markersize=4)
    ax.set_xlabel("Quality Threshold τ")
    ax.set_ylabel("Batch Latency (ms)")
    ax.set_title("Batch Latency vs Threshold")
    ax.grid(True, alpha=0.3)

    # Fallback rate vs threshold
    ax = axes[1]
    ax.plot(sweep_results["threshold"], sweep_results["fallback_rate"], "r-s", markersize=4)
    ax.set_xlabel("Quality Threshold τ")
    ax.set_ylabel("Fallback Rate")
    ax.set_title("Fallback Rate vs Threshold")
    ax.grid(True, alpha=0.3)

    # Accuracy vs threshold
    ax = axes[2]
    if "accuracy" in sweep_results.columns:
        ax.plot(sweep_results["threshold"], sweep_results["accuracy"], "g-^", markersize=4)
        ax.set_xlabel("Quality Threshold τ")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy vs Threshold")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Threshold sweep plot saved to {output_path}")

    plt.close()


def plot_load_distribution(
    instance_loads: Dict[str, float],
    title: str = "Instance Load Distribution",
    output_path: Optional[str] = None,
):
    """Plot load distribution across instances."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plot")
        return

    instances = list(instance_loads.keys())
    loads = list(instance_loads.values())

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(instances)), loads, color="steelblue", edgecolor="white")
    ax.set_xticks(range(len(instances)))
    ax.set_xticklabels(instances, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Load (ms)")
    ax.set_title(title)

    # Add load values on top of bars
    for bar, load in zip(bars, loads):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(loads) * 0.02,
            f"{load:.0f}",
            ha="center",
            fontsize=7,
        )

    # Horizontal line at max load (makespan)
    ax.axhline(y=max(loads), color="red", linestyle="--", alpha=0.5, label=f"Makespan: {max(loads):.0f}ms")
    ax.legend()

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Load distribution plot saved to {output_path}")

    plt.close()


def plot_method_comparison(
    results: Dict[str, Dict],
    output_path: Optional[str] = None,
):
    """Bar chart comparing batch latency and fallback rate across methods."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    methods = list(results.keys())
    latencies = [results[m].get("system", {}).get("batch_latency_ms", 0) for m in methods]
    fallback_rates = [results[m].get("quality", {}).get("fallback_rate", 0) for m in methods]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))

    ax1.bar(methods, latencies, color=colors, edgecolor="white")
    ax1.set_ylabel("Batch Latency (ms)")
    ax1.set_title("Batch Latency by Method")
    ax1.tick_params(axis="x", rotation=45, labelsize=8)

    ax2.bar(methods, fallback_rates, color=colors, edgecolor="white")
    ax2.set_ylabel("Fallback Rate")
    ax2.set_title("Fallback Rate by Method")
    ax2.tick_params(axis="x", rotation=45, labelsize=8)

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Method comparison plot saved to {output_path}")

    plt.close()
