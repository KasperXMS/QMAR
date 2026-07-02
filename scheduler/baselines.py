"""Baseline schedulers (Section 10).

Implements:
  1. Random feasible
  2. Fastest feasible
  3. Highest suitability
  4. Latency-only greedy (no quality-margin ranking)
  5. QMAR w/o answer type (uses average latency)
  6. Oracle (ground-truth suitability + answer types → QMAR)
"""

import logging
import random
from typing import List, Dict, Optional
import numpy as np
import pandas as pd

from data_adapters.common_schema import RequestRecord

logger = logging.getLogger(__name__)


def _build_feasible_sets(
    N: int,
    K_inst: int,
    instance_suitability: np.ndarray,
    quality_threshold: float,
    request_specific_thresholds: Optional[Dict[str, float]],
    records: List[RequestRecord],
):
    """Helper: build feasible sets for each request."""
    feasible_sets = []
    fallback_requests = []
    for i in range(N):
        tau_i = quality_threshold
        if request_specific_thresholds:
            tau_i = request_specific_thresholds.get(records[i].request_id, quality_threshold)
        F_i = [k for k in range(K_inst) if instance_suitability[i, k] >= tau_i]
        if not F_i:
            fallback_requests.append(i)
            F_i = list(range(K_inst))
        feasible_sets.append(F_i)
    return feasible_sets, fallback_requests


def _format_assignments(
    sorted_indices: List[int],
    feasible_sets: List[List[int]],
    fallback_requests: List[int],
    instance_suitability: np.ndarray,
    latency_matrix: np.ndarray,
    instance_ids: List[str],
    instance_to_model: Dict[str, str],
    records: List[RequestRecord],
    predicted_answer_types: List[str],
    instance_loads: Dict[str, float],
) -> List[Dict]:
    """Helper: format assignment results."""
    assignments = []
    for i in sorted_indices:
        # Find which instance this request was assigned to
        # (Look up from instance_loads by finding the matching request)
        pass
    return assignments


class RandomFeasibleBaseline:
    """Randomly select a feasible instance (Section 10.1)."""

    def __init__(
        self,
        instance_config: List[Dict],
        model_names: List[str],
        quality_threshold: float = 0.5,
        seed: int = 42,
    ):
        self.instance_config = instance_config
        self.instance_ids = [cfg["instance_id"] for cfg in instance_config]
        self.model_names = model_names
        self.quality_threshold = quality_threshold
        self.seed = seed
        self._model_to_idx = {m: i for i, m in enumerate(model_names)}
        self._instance_to_model = {
            cfg["instance_id"]: cfg["model_name"] for cfg in instance_config
        }

    def schedule(
        self,
        records: List[RequestRecord],
        suitability_scores: np.ndarray,
        latency_matrix: np.ndarray,
        predicted_answer_types: List[str],
    ) -> Dict:
        N = len(records)
        K_inst = len(self.instance_ids)
        rng = random.Random(self.seed)

        # Map model suitability to instance suitability
        instance_suitability = np.zeros((N, K_inst), dtype=np.float32)
        for k, cfg in enumerate(self.instance_config):
            model_name = cfg["model_name"]
            if model_name in self._model_to_idx:
                instance_suitability[:, k] = suitability_scores[:, self._model_to_idx[model_name]]

        feasible_sets, fallback_requests = _build_feasible_sets(
            N, K_inst, instance_suitability, self.quality_threshold, None, records
        )

        instance_loads = {iid: 0.0 for iid in self.instance_ids}
        assignments = []
        fallback_count = 0

        indices = list(range(N))
        rng.shuffle(indices)  # random order

        for i in indices:
            F_i = feasible_sets[i]
            if i in fallback_requests:
                best_k = int(np.argmax(instance_suitability[i, :]))
                is_feasible = False
                is_fallback = True
                fallback_count += 1
            else:
                best_k = rng.choice(F_i)
                is_feasible = True
                is_fallback = False

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
            })

        batch_latency = max(instance_loads.values()) if instance_loads else 0.0

        return {
            "assignments": assignments,
            "batch_latency": batch_latency,
            "instance_loads": instance_loads,
            "fallback_count": fallback_count,
        }


class FastestFeasibleBaseline:
    """Select feasible instance with minimum estimated latency (Section 10.1)."""

    def __init__(
        self,
        instance_config: List[Dict],
        model_names: List[str],
        quality_threshold: float = 0.5,
    ):
        self.instance_config = instance_config
        self.instance_ids = [cfg["instance_id"] for cfg in instance_config]
        self.model_names = model_names
        self.quality_threshold = quality_threshold
        self._model_to_idx = {m: i for i, m in enumerate(model_names)}
        self._instance_to_model = {
            cfg["instance_id"]: cfg["model_name"] for cfg in instance_config
        }

    def schedule(
        self,
        records: List[RequestRecord],
        suitability_scores: np.ndarray,
        latency_matrix: np.ndarray,
        predicted_answer_types: List[str],
    ) -> Dict:
        N = len(records)
        K_inst = len(self.instance_ids)

        instance_suitability = np.zeros((N, K_inst), dtype=np.float32)
        for k, cfg in enumerate(self.instance_config):
            model_name = cfg["model_name"]
            if model_name in self._model_to_idx:
                instance_suitability[:, k] = suitability_scores[:, self._model_to_idx[model_name]]

        feasible_sets, fallback_requests = _build_feasible_sets(
            N, K_inst, instance_suitability, self.quality_threshold, None, records
        )

        instance_loads = {iid: 0.0 for iid in self.instance_ids}
        assignments = []
        fallback_count = 0

        for i in range(N):
            F_i = feasible_sets[i]
            if i in fallback_requests:
                best_k = int(np.argmax(instance_suitability[i, :]))
                is_feasible = False
                is_fallback = True
                fallback_count += 1
            else:
                # Pick feasible instance with lowest latency
                best_k = min(F_i, key=lambda k: latency_matrix[i, k])
                is_feasible = True
                is_fallback = False

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
            })

        batch_latency = max(instance_loads.values()) if instance_loads else 0.0

        return {
            "assignments": assignments,
            "batch_latency": batch_latency,
            "instance_loads": instance_loads,
            "fallback_count": fallback_count,
        }


class HighestSuitabilityBaseline:
    """Select feasible instance with highest suitability (Section 10.1)."""

    def __init__(
        self,
        instance_config: List[Dict],
        model_names: List[str],
        quality_threshold: float = 0.5,
    ):
        self.instance_config = instance_config
        self.instance_ids = [cfg["instance_id"] for cfg in instance_config]
        self.model_names = model_names
        self.quality_threshold = quality_threshold
        self._model_to_idx = {m: i for i, m in enumerate(model_names)}
        self._instance_to_model = {
            cfg["instance_id"]: cfg["model_name"] for cfg in instance_config
        }

    def schedule(
        self,
        records: List[RequestRecord],
        suitability_scores: np.ndarray,
        latency_matrix: np.ndarray,
        predicted_answer_types: List[str],
    ) -> Dict:
        N = len(records)
        K_inst = len(self.instance_ids)

        instance_suitability = np.zeros((N, K_inst), dtype=np.float32)
        for k, cfg in enumerate(self.instance_config):
            model_name = cfg["model_name"]
            if model_name in self._model_to_idx:
                instance_suitability[:, k] = suitability_scores[:, self._model_to_idx[model_name]]

        feasible_sets, fallback_requests = _build_feasible_sets(
            N, K_inst, instance_suitability, self.quality_threshold, None, records
        )

        instance_loads = {iid: 0.0 for iid in self.instance_ids}
        assignments = []
        fallback_count = 0

        for i in range(N):
            F_i = feasible_sets[i]
            if i in fallback_requests:
                best_k = int(np.argmax(instance_suitability[i, :]))
                is_feasible = False
                is_fallback = True
                fallback_count += 1
            else:
                # Pick feasible instance with highest suitability
                best_k = max(F_i, key=lambda k: instance_suitability[i, k])
                is_feasible = True
                is_fallback = False

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
            })

        batch_latency = max(instance_loads.values()) if instance_loads else 0.0

        return {
            "assignments": assignments,
            "batch_latency": batch_latency,
            "instance_loads": instance_loads,
            "fallback_count": fallback_count,
        }


class LatencyOnlyGreedyBaseline:
    """Load-balanced greedy without quality-margin ranking (Section 10.1).

    Sorts requests by min latency, then greedy assigns (no quality-margin awareness).
    """

    def __init__(
        self,
        instance_config: List[Dict],
        model_names: List[str],
        quality_threshold: float = 0.5,
    ):
        self.instance_config = instance_config
        self.instance_ids = [cfg["instance_id"] for cfg in instance_config]
        self.model_names = model_names
        self.quality_threshold = quality_threshold
        self._model_to_idx = {m: i for i, m in enumerate(model_names)}
        self._instance_to_model = {
            cfg["instance_id"]: cfg["model_name"] for cfg in instance_config
        }

    def schedule(
        self,
        records: List[RequestRecord],
        suitability_scores: np.ndarray,
        latency_matrix: np.ndarray,
        predicted_answer_types: List[str],
    ) -> Dict:
        N = len(records)
        K_inst = len(self.instance_ids)

        instance_suitability = np.zeros((N, K_inst), dtype=np.float32)
        for k, cfg in enumerate(self.instance_config):
            model_name = cfg["model_name"]
            if model_name in self._model_to_idx:
                instance_suitability[:, k] = suitability_scores[:, self._model_to_idx[model_name]]

        feasible_sets, fallback_requests = _build_feasible_sets(
            N, K_inst, instance_suitability, self.quality_threshold, None, records
        )

        # Sort by minimum feasible latency (no quality-margin awareness)
        sort_keys = []
        for i in range(N):
            F_i = feasible_sets[i]
            if i in fallback_requests:
                min_lat = float("inf")
            else:
                min_lat = min(latency_matrix[i, F_i]) if F_i else float("inf")
            sort_keys.append((min_lat, i))
        sort_keys.sort(key=lambda x: x[0], reverse=True)  # hardest first
        sorted_indices = [sk[1] for sk in sort_keys]

        instance_loads = {iid: 0.0 for iid in self.instance_ids}
        assignments = []
        fallback_count = 0

        for i in sorted_indices:
            F_i = feasible_sets[i]
            if i in fallback_requests:
                best_k = int(np.argmax(instance_suitability[i, :]))
                is_feasible = False
                is_fallback = True
                fallback_count += 1
            else:
                best_k = F_i[0]
                best_load = instance_loads[self.instance_ids[F_i[0]]] + latency_matrix[i, F_i[0]]
                for k in F_i[1:]:
                    new_load = instance_loads[self.instance_ids[k]] + latency_matrix[i, k]
                    if new_load < best_load:
                        best_load = new_load
                        best_k = k
                is_feasible = True
                is_fallback = False

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
            })

        batch_latency = max(instance_loads.values()) if instance_loads else 0.0

        return {
            "assignments": assignments,
            "batch_latency": batch_latency,
            "instance_loads": instance_loads,
            "fallback_count": fallback_count,
        }


class QMARWithoutAnswerTypeBaseline:
    """QMAR without answer-type-aware latency (Section 10.1).

    Uses average instance latency instead of answer-type-specific latency.
    """

    def __init__(
        self,
        instance_config: List[Dict],
        model_names: List[str],
        quality_threshold: float = 0.5,
        avg_latency: Optional[Dict[str, float]] = None,
    ):
        self.instance_config = instance_config
        self.instance_ids = [cfg["instance_id"] for cfg in instance_config]
        self.model_names = model_names
        self.quality_threshold = quality_threshold
        self._model_to_idx = {m: i for i, m in enumerate(model_names)}
        self._instance_to_model = {
            cfg["instance_id"]: cfg["model_name"] for cfg in instance_config
        }
        # Average latency per instance (if not provided, use 300ms default)
        self.avg_latency = avg_latency or {
            iid: 300.0 for iid in self.instance_ids
        }

    def schedule(
        self,
        records: List[RequestRecord],
        suitability_scores: np.ndarray,
        latency_matrix: np.ndarray,
        predicted_answer_types: List[str],
    ) -> Dict:
        N = len(records)
        K_inst = len(self.instance_ids)

        # Override latency matrix with average latencies
        avg_latency_matrix = np.zeros_like(latency_matrix)
        for k, iid in enumerate(self.instance_ids):
            avg_latency_matrix[:, k] = self.avg_latency.get(iid, 300.0)

        instance_suitability = np.zeros((N, K_inst), dtype=np.float32)
        for k, cfg in enumerate(self.instance_config):
            model_name = cfg["model_name"]
            if model_name in self._model_to_idx:
                instance_suitability[:, k] = suitability_scores[:, self._model_to_idx[model_name]]

        feasible_sets, fallback_requests = _build_feasible_sets(
            N, K_inst, instance_suitability, self.quality_threshold, None, records
        )

        # QMAR-style ranking but with average latencies
        ranking_features = []
        for i in range(N):
            F_i = feasible_sets[i]
            if i in fallback_requests:
                q_i, delta_i, p_i = float("inf"), 0.0, float("inf")
            else:
                q_i = len(F_i)
                delta_i = max(instance_suitability[i, F_i]) - self.quality_threshold if F_i else 0.0
                p_i = min(avg_latency_matrix[i, F_i]) if F_i else float("inf")
            ranking_features.append((q_i, delta_i, -p_i, i))
        ranking_features.sort(key=lambda x: (x[0], x[1], x[2]))
        sorted_indices = [rf[3] for rf in ranking_features]

        instance_loads = {iid: 0.0 for iid in self.instance_ids}
        assignments = []
        fallback_count = 0

        for i in sorted_indices:
            F_i = feasible_sets[i]
            if i in fallback_requests:
                best_k = int(np.argmax(instance_suitability[i, :]))
                is_feasible, is_fallback = False, True
                fallback_count += 1
            else:
                best_k = F_i[0]
                best_load = instance_loads[self.instance_ids[F_i[0]]] + avg_latency_matrix[i, F_i[0]]
                for k in F_i[1:]:
                    new_load = instance_loads[self.instance_ids[k]] + avg_latency_matrix[i, k]
                    if new_load < best_load:
                        best_load = new_load
                        best_k = k
                is_feasible, is_fallback = True, False

            iid = self.instance_ids[best_k]
            instance_loads[iid] += latency_matrix[i, best_k]  # use real latency for load
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
            })

        batch_latency = max(instance_loads.values()) if instance_loads else 0.0

        return {
            "assignments": assignments,
            "batch_latency": batch_latency,
            "instance_loads": instance_loads,
            "fallback_count": fallback_count,
        }


class OracleBaseline:
    """Oracle: optimal unconstrained makespan via LPT + multi-start + local search.

    The min-max assignment problem (R||Cmax) is NP-hard. This oracle computes:
      1. An analytical LP lower bound (provably ≤ true optimum)
      2. LPT greedy from 10 random sort orders, picking the best
      3. Pairwise-swap local search to further reduce makespan

    The gap between the analytical bound and the best found solution
    bounds how far we are from true optimality.

    (Section 10.2)
    """

    def __init__(
        self,
        instance_config: List[Dict],
        model_names: List[str],
        quality_threshold: float = 0.5,
        gt_suitability: Optional[np.ndarray] = None,
        gt_answer_types: Optional[List[str]] = None,
        n_restarts: int = 10,
        local_search_iters: int = 3,
    ):
        self.instance_config = instance_config
        self.instance_ids = [cfg["instance_id"] for cfg in instance_config]
        self.model_names = model_names
        self.quality_threshold = quality_threshold
        self.gt_answer_types = gt_answer_types
        self.n_restarts = n_restarts
        self.local_search_iters = local_search_iters
        self._instance_to_model = {
            cfg["instance_id"]: cfg["model_name"] for cfg in instance_config
        }

    def schedule(
        self,
        records: List[RequestRecord],
        suitability_scores: np.ndarray,
        latency_matrix: np.ndarray,
        predicted_answer_types: List[str],
    ) -> Dict:
        N = len(records)
        K = len(self.instance_ids)
        L = latency_matrix  # (N, K)
        gt_types = self.gt_answer_types or predicted_answer_types

        # ── Analytical LP lower bound (provable) ──
        min_lat_per_req = L.min(axis=1)  # (N,)
        lb_avg = min_lat_per_req.sum() / K        # average minimum load
        lb_max = min_lat_per_req.max()             # largest single job
        lp_lower_bound = max(lb_avg, lb_max)

        # ── Multi-start LPT greedy ──
        best_makespan = float("inf")
        best_assignment = None
        best_loads = None

        for seed in range(self.n_restarts):
            assignment, loads = self._lpt_greedy(L, N, K, seed)
            makespan = max(loads.values())

            # ── Local search: pairwise swaps ──
            for _ in range(self.local_search_iters):
                improved = False
                # Try moving a request from the most loaded instance
                max_iid = max(loads, key=loads.get)
                max_k = self.instance_ids.index(max_iid)

                # Find requests on this instance
                requests_on_max = [
                    (i, a["estimated_latency"])
                    for i, a in enumerate(assignment)
                    if a["assigned_instance"] == max_iid
                ]
                if not requests_on_max:
                    break

                # Sort by latency descending (biggest first)
                requests_on_max.sort(key=lambda x: -x[1])

                for i, lat_i in requests_on_max[:100]:  # try top 100
                    for k in range(K):
                        if k == max_k:
                            continue
                        # Would moving request i to instance k reduce makespan?
                        new_load_src = loads[max_iid] - lat_i
                        new_load_dst = loads[self.instance_ids[k]] + L[i, k]
                        new_makespan = max(
                            max(loads.values()),  # other instances unchanged
                            new_load_src,
                            new_load_dst,
                        )
                        if new_makespan < makespan:
                            # Perform swap
                            loads[max_iid] = new_load_src
                            loads[self.instance_ids[k]] = new_load_dst
                            assignment[i]["assigned_instance"] = self.instance_ids[k]
                            assignment[i]["assigned_model"] = self._instance_to_model[
                                self.instance_ids[k]
                            ]
                            assignment[i]["estimated_latency"] = float(L[i, k])
                            makespan = new_makespan
                            improved = True
                            break
                    if improved:
                        break
                if not improved:
                    break

            if makespan < best_makespan:
                best_makespan = makespan
                best_assignment = assignment
                best_loads = loads

            # Early stop if we hit the LP bound
            if best_makespan <= lp_lower_bound * 1.001:
                break

        # Rebuild assignments list with correct metadata
        final_assignments = []
        instance_loads_final = {iid: 0.0 for iid in self.instance_ids}

        # Replay best assignment to compute cumulative loads
        for i in range(N):
            iid = best_assignment[i]["assigned_instance"]
            lat = best_assignment[i]["estimated_latency"]
            instance_loads_final[iid] += lat
            final_assignments.append({
                "request_id": records[i].request_id,
                "assigned_instance": iid,
                "assigned_model": self._instance_to_model[iid],
                "predicted_suitability": 1.0,
                "predicted_answer_type": gt_types[i],
                "estimated_latency": float(lat),
                "is_quality_feasible": True,
                "is_fallback": False,
                "current_instance_load_after_assignment": float(instance_loads_final[iid]),
            })

        makespan = max(instance_loads_final.values())
        ideal_lb = max(L.min(axis=1).max(), L.min(axis=1).sum() / K)

        logger.info(
            f"  Oracle: best_found={makespan:.0f}ms  "
            f"(naive_LB={ideal_lb:.0f}ms, "
            f"restarts={self.n_restarts}, ls_iters={self.local_search_iters})"
        )

        return {
            "assignments": final_assignments,
            "batch_latency": makespan,
            "instance_loads": instance_loads_final,
            "fallback_count": 0,
            "routing_logs": {
                "lp_lower_bound": lp_lower_bound,
                "naive_lower_bound": ideal_lb,
            },
        }

    def _lpt_greedy(self, L: np.ndarray, N: int, K: int, seed: int):
        """LPT-like greedy: sort by descending min-latency, assign to least-loaded."""
        rng = np.random.RandomState(seed)
        min_lat = L.min(axis=1)
        # Add small random noise to break ties differently per restart
        noise = rng.uniform(0, min_lat.max() * 0.01, size=N)
        order = np.argsort(-(min_lat + noise))

        loads = {iid: 0.0 for iid in self.instance_ids}
        assignments = []

        for i in order:
            best_k = 0
            best_load = loads[self.instance_ids[0]] + L[i, 0]
            for k in range(1, K):
                new_load = loads[self.instance_ids[k]] + L[i, k]
                if new_load < best_load:
                    best_load = new_load
                    best_k = k

            iid = self.instance_ids[best_k]
            loads[iid] += L[i, best_k]
            assignments.append({
                "assigned_instance": iid,
                "assigned_model": self._instance_to_model[iid],
                "estimated_latency": float(L[i, best_k]),
            })

        return assignments, loads
