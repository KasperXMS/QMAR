"""QMAR Batch Scheduler (Section 8).

Algorithm:
  1. Feasible set construction: F_i = {u_k | s_{i,m(k)} >= τ_i}
  2. Request ranking: sort by (|F_i|↑, δ_i↑, p_i↓)
  3. Load-balanced greedy assignment: argmin_{u_k ∈ F_i} (L_k + l_{i,k})
  4. Fallback: if F_i = ∅, assign to max-suitability instance
"""

import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd

from data_adapters.common_schema import RequestRecord

logger = logging.getLogger(__name__)


class QMARScheduler:
    """Quality-Margin-Aware Routing scheduler.

    Performs quality-constrained, request-ranking-based greedy assignment
    to minimize batch makespan over heterogeneous VLM instances.
    """

    def __init__(
        self,
        instance_config: List[Dict],
        model_names: List[str],
        quality_threshold: float = 0.5,
        request_specific_thresholds: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            instance_config: List of {instance_id, model_name, device_class} dicts.
            model_names: Ordered list of model names (for suitability score indexing).
            quality_threshold: Global quality threshold τ.
            request_specific_thresholds: Optional per-request thresholds τ_i.
        """
        self.instance_config = instance_config
        self.instance_ids = [cfg["instance_id"] for cfg in instance_config]
        self.model_names = model_names
        self.quality_threshold = quality_threshold
        self.request_specific_thresholds = request_specific_thresholds or {}

        # Map instance -> model index
        self._instance_to_model = {
            cfg["instance_id"]: cfg["model_name"] for cfg in instance_config
        }
        self._model_to_idx = {m: i for i, m in enumerate(model_names)}

    def schedule(
        self,
        records: List[RequestRecord],
        suitability_scores: np.ndarray,  # (N, K_model)
        latency_matrix: np.ndarray,      # (N, K_instance)
        predicted_answer_types: List[str],
    ) -> Dict:
        """Run QMAR scheduling on a batch.

        Args:
            records: List of N RequestRecord.
            suitability_scores: (N, K_model) suitability probabilities s_{i,m}.
            latency_matrix: (N, K_instance) estimated latencies l_{i,k}.
            predicted_answer_types: List of N answer type strings.

        Returns:
            Dict with:
              - assignments: List of {request_id, assigned_instance, ...}
              - batch_latency: max makespan
              - instance_loads: Dict[instance_id, load]
              - fallback_count: number of fallback assignments
              - routing_logs: detailed logs
        """
        N = len(records)
        K_inst = len(self.instance_ids)
        K_model = len(self.model_names)

        # Step 1: Map model suitability to instance suitability
        # s_{i,k} = s_{i,m(k)}
        instance_suitability = np.zeros((N, K_inst), dtype=np.float32)
        for k, cfg in enumerate(self.instance_config):
            model_name = cfg["model_name"]
            if model_name in self._model_to_idx:
                m_idx = self._model_to_idx[model_name]
                instance_suitability[:, k] = suitability_scores[:, m_idx]

        # Step 1: Feasible set construction
        feasible_sets = []  # List of List[int] (instance indices)
        fallback_requests = []

        for i in range(N):
            tau_i = self.request_specific_thresholds.get(
                records[i].request_id, self.quality_threshold
            )
            F_i = [
                k for k in range(K_inst)
                if instance_suitability[i, k] >= tau_i
            ]
            if not F_i:
                fallback_requests.append(i)
                # Use all instances as fallback (will pick max-suitability)
                F_i = list(range(K_inst))
            feasible_sets.append(F_i)

        # Step 2: Request ranking
        # q_i = |F_i|, δ_i = max(s_{i,k} - τ_i), p_i = min(l_{i,k})
        ranking_features = []
        for i in range(N):
            F_i = feasible_sets[i]
            tau_i = self.request_specific_thresholds.get(
                records[i].request_id, self.quality_threshold
            )

            if i in fallback_requests:
                q_i = float("inf")
                delta_i = 0.0
                p_i = float("inf")
            else:
                q_i = len(F_i)
                delta_i = (
                    max(instance_suitability[i, F_i]) - tau_i
                    if F_i else 0.0
                )
                p_i = min(latency_matrix[i, F_i]) if F_i else float("inf")

            ranking_features.append((q_i, delta_i, -p_i, i))

        # Sort by (q_i ascending, delta_i ascending, -p_i ascending = p_i descending)
        ranking_features.sort(key=lambda x: (x[0], x[1], x[2]))
        sorted_indices = [rf[3] for rf in ranking_features]

        # Step 3: Load-balanced greedy assignment
        instance_loads = {iid: 0.0 for iid in self.instance_ids}
        assignments = []
        fallback_count = 0

        for i in sorted_indices:
            F_i = feasible_sets[i]

            if i in fallback_requests:
                # Fallback: pick instance with highest suitability
                best_k = int(np.argmax(instance_suitability[i, :]))
                is_fallback = True
                is_feasible = False
                fallback_count += 1
            else:
                # Greedy: pick feasible instance that minimizes load after assignment
                best_k = F_i[0]
                best_load = instance_loads[self.instance_ids[F_i[0]]] + latency_matrix[i, F_i[0]]
                for k in F_i[1:]:
                    new_load = instance_loads[self.instance_ids[k]] + latency_matrix[i, k]
                    if new_load < best_load:
                        best_load = new_load
                        best_k = k
                is_fallback = False
                is_feasible = True

            iid = self.instance_ids[best_k]
            instance_loads[iid] += latency_matrix[i, best_k]

            assignments.append({
                "request_id": records[i].request_id,
                "assigned_instance": iid,
                "assigned_model": self._instance_to_model[iid],
                "predicted_suitability": float(instance_suitability[i, best_k]),
                "predicted_answer_type": predicted_answer_types[i],
                "estimated_latency": float(latency_matrix[i, best_k]),
                "is_quality_feasible": is_feasible,
                "is_fallback": is_fallback,
                "current_instance_load_after_assignment": float(instance_loads[iid]),
                "feasible_set_size": len([k for k in F_i if k in F_i]),
            })

        batch_latency = max(instance_loads.values()) if instance_loads else 0.0

        # Routing logs
        routing_logs = {
            "feasible_set_sizes": [len(F_i) for F_i in feasible_sets],
            "quality_margins": [
                max(instance_suitability[i, F_i]) - self.quality_threshold
                if F_i else 0.0
                for i, F_i in enumerate(feasible_sets)
            ],
            "fallback_requests": fallback_requests,
            "instance_final_loads": instance_loads,
            "sorted_request_order": sorted_indices,
        }

        logger.info(
            f"QMAR scheduled {N} requests across {K_inst} instances. "
            f"Batch latency: {batch_latency:.1f}ms, "
            f"Fallbacks: {fallback_count}/{N}"
        )

        return {
            "assignments": assignments,
            "batch_latency": batch_latency,
            "instance_loads": instance_loads,
            "fallback_count": fallback_count,
            "routing_logs": routing_logs,
        }


def qmar_schedule(
    records: List[RequestRecord],
    suitability_scores: np.ndarray,
    latency_matrix: np.ndarray,
    predicted_answer_types: List[str],
    instance_config: List[Dict],
    model_names: List[str],
    quality_threshold: float = 0.5,
) -> Dict:
    """Convenience wrapper for QMARScheduler.schedule()."""
    scheduler = QMARScheduler(
        instance_config=instance_config,
        model_names=model_names,
        quality_threshold=quality_threshold,
    )
    return scheduler.schedule(
        records=records,
        suitability_scores=suitability_scores,
        latency_matrix=latency_matrix,
        predicted_answer_types=predicted_answer_types,
    )
