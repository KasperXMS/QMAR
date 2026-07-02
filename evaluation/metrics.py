"""Evaluation metrics (Section 11).

Three groups:
  1. Quality-side: accuracy, quality satisfaction rate, violation rate, fallback rate
  2. System-side: batch latency, avg/P95 latency, throughput, load imbalance
  3. Routing-side: model distribution, feasible set distribution, margin distribution
"""

import logging
from typing import List, Dict
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_quality_metrics(
    assignments: List[Dict],
    records: List = None,
    ground_truth_correctness: Dict = None,
) -> Dict:
    """Compute quality-side metrics (Section 11.1).

    Args:
        assignments: List of assignment dicts from scheduler.
        records: List of RequestRecord for ground-truth quality lookup.
        ground_truth_correctness: Dict[request_id][model_name] -> correctness.

    Returns:
        Dict of quality metrics.
    """
    total = len(assignments)

    # Fallback rate
    fallback_count = sum(1 for a in assignments if a.get("is_fallback", False))
    fallback_rate = fallback_count / total if total > 0 else 0

    # Quality feasible rate
    feasible_count = sum(1 for a in assignments if a.get("is_quality_feasible", True))
    feasible_rate = feasible_count / total if total > 0 else 0

    # Average predicted suitability
    suitabilities = [
        a.get("predicted_suitability", 0.0) for a in assignments
    ]
    avg_suitability = np.mean(suitabilities) if suitabilities else 0.0

    # If we have ground truth correctness, compute accuracy
    accuracy = None
    violation_rate = None
    if ground_truth_correctness is not None and records is not None:
        correct_count = 0
        violation_count = 0
        rec_map = {r.request_id: r for r in records}

        for a in assignments:
            rid = a["request_id"]
            model = a.get("assigned_model", "")
            rec = rec_map.get(rid)
            if rec:
                is_correct = rec.candidate_model_correctness.get(model, 0) >= 0.5
                if is_correct:
                    correct_count += 1
                elif not a.get("is_fallback", False):
                    # Quality violation: feasible assignment but model was wrong
                    violation_count += 1

        accuracy = correct_count / total if total > 0 else 0
        # Violation rate among feasible (non-fallback) assignments
        feasible_total = total - fallback_count
        violation_rate = violation_count / feasible_total if feasible_total > 0 else 0

    return {
        "total_requests": total,
        "fallback_count": fallback_count,
        "fallback_rate": fallback_rate,
        "quality_feasible_rate": feasible_rate,
        "avg_predicted_suitability": avg_suitability,
        "accuracy": accuracy,
        "threshold_violation_rate": violation_rate,
    }


def compute_system_metrics(
    assignments: List[Dict],
    instance_loads: Dict[str, float],
    batch_latency: float,
) -> Dict:
    """Compute system-side metrics (Section 11.2)."""
    total = len(assignments)

    # Per-request latencies
    latencies = [a.get("estimated_latency", 0.0) for a in assignments]
    avg_latency = np.mean(latencies) if latencies else 0.0
    p50_latency = np.percentile(latencies, 50) if latencies else 0.0
    p95_latency = np.percentile(latencies, 95) if latencies else 0.0
    p99_latency = np.percentile(latencies, 99) if latencies else 0.0

    # Load imbalance: coefficient of variation of instance loads
    loads = list(instance_loads.values())
    mean_load = np.mean(loads) if loads else 0.0
    std_load = np.std(loads) if loads else 0.0
    load_cv = std_load / mean_load if mean_load > 0 else 0.0

    # Throughput: requests per second
    max_load_ms = max(loads) if loads else 1.0
    throughput = total / (max_load_ms / 1000.0) if max_load_ms > 0 else 0.0

    return {
        "batch_latency_ms": batch_latency,
        "avg_request_latency_ms": avg_latency,
        "p50_latency_ms": p50_latency,
        "p95_latency_ms": p95_latency,
        "p99_latency_ms": p99_latency,
        "throughput_req_per_sec": throughput,
        "num_instances": len(instance_loads),
        "load_mean_ms": mean_load,
        "load_std_ms": std_load,
        "load_cv": load_cv,  # coefficient of variation
        "load_imbalance_ratio": max(loads) / mean_load if mean_load > 0 else 1.0,
    }


def compute_routing_metrics(
    assignments: List[Dict],
) -> Dict:
    """Compute routing-side metrics (Section 11.3)."""
    total = len(assignments)

    # Model/instance distribution
    model_counts = {}
    instance_counts = {}
    for a in assignments:
        model = a.get("assigned_model", "unknown")
        instance = a.get("assigned_instance", "unknown")
        model_counts[model] = model_counts.get(model, 0) + 1
        instance_counts[instance] = instance_counts.get(instance, 0) + 1

    # Normalize to probabilities
    model_dist = {m: c / total for m, c in model_counts.items()}
    instance_dist = {iid: c / total for iid, c in instance_counts.items()}

    # Feasible set size stats
    feasible_sizes = [a.get("feasible_set_size", 0) for a in assignments]
    avg_feasible_size = np.mean(feasible_sizes) if feasible_sizes else 0.0

    # Answer type distribution among assignments
    answer_type_counts = {}
    for a in assignments:
        at = a.get("predicted_answer_type", "unknown")
        answer_type_counts[at] = answer_type_counts.get(at, 0) + 1

    return {
        "model_distribution": model_dist,
        "instance_distribution": instance_dist,
        "avg_feasible_set_size": avg_feasible_size,
        "answer_type_distribution": answer_type_counts,
    }


def evaluate_all(
    assignments: List[Dict],
    instance_loads: Dict[str, float],
    batch_latency: float,
    records: List = None,
    ground_truth_correctness: Dict = None,
) -> Dict:
    """Run all evaluation metrics and return combined results.

    Returns:
        Dict with keys: quality, system, routing
    """
    quality = compute_quality_metrics(assignments, records, ground_truth_correctness)
    system = compute_system_metrics(assignments, instance_loads, batch_latency)
    routing = compute_routing_metrics(assignments)

    return {
        "quality": quality,
        "system": system,
        "routing": routing,
    }


def build_summary_table(
    results: Dict[str, Dict],  # method_name -> metrics dict
) -> pd.DataFrame:
    """Build a summary comparison table across methods."""
    rows = []
    for method, metrics in results.items():
        row = {"method": method}
        q = metrics.get("quality", {})
        s = metrics.get("system", {})
        r = metrics.get("routing", {})

        row["fallback_rate"] = q.get("fallback_rate", 0)
        row["accuracy"] = q.get("accuracy", None)
        row["batch_latency_ms"] = s.get("batch_latency_ms", 0)
        row["avg_latency_ms"] = s.get("avg_request_latency_ms", 0)
        row["p95_latency_ms"] = s.get("p95_latency_ms", 0)
        row["throughput_rps"] = s.get("throughput_req_per_sec", 0)
        row["load_cv"] = s.get("load_cv", 0)
        row["avg_feasible_size"] = r.get("avg_feasible_set_size", 0)

        rows.append(row)

    return pd.DataFrame(rows)
