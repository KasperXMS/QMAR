"""Request-specific latency estimator (Section 6).

l_{i,k} = l̄_{k, â_i} + l^{comm}_{i,k}

Combines answer-type-aware instance latency with communication cost.
"""

import logging
from typing import List, Optional, Dict
import numpy as np
import pandas as pd

from profiling.latency_table import LatencyTableBuilder
from profiling.communication_profile import compute_comm_cost
from data_adapters.common_schema import RequestRecord, AnswerType

logger = logging.getLogger(__name__)


class LatencyEstimator:
    """Estimate request-instance latency l_{i,k}.

    Formula (Section 6.1):
      l_{i,k} = l̄_{k, â_i} + l^{comm}_{i,k}

    where:
      - l̄_{k, â_i}: answer-type-based instance latency from latency table
      - l^{comm}_{i,k}: communication cost (RTT + data/BW)
    """

    def __init__(
        self,
        latency_table: pd.DataFrame,
        comm_profile: pd.DataFrame,
        use_comm_cost: bool = True,
    ):
        """
        Args:
            latency_table: DataFrame with columns [instance_id, short, medium, long].
            comm_profile: DataFrame with columns [instance_id, RTT_ms, BW_mbps].
            use_comm_cost: Whether to add communication cost.
        """
        self.latency_table = latency_table
        self.comm_profile = comm_profile
        self.use_comm_cost = use_comm_cost

        # Build lookup dicts for fast access
        self._instance_latency = {}
        for _, row in latency_table.iterrows():
            iid = row["instance_id"]
            self._instance_latency[iid] = {
                "simple": float(row["simple"]),
                "moderate": float(row["moderate"]),
                "complex": float(row["complex"]),
            }

    def estimate(
        self,
        record: RequestRecord,
        predicted_answer_type: str,
        instance_ids: List[str],
    ) -> Dict[str, float]:
        """Estimate latency for one request against all instances.

        Args:
            record: RequestRecord with image_path and question_text.
            predicted_answer_type: One of 'short', 'medium', 'long'.
            instance_ids: List of instance identifiers.

        Returns:
            Dict mapping instance_id -> estimated latency in ms.
        """
        latencies = {}
        for iid in instance_ids:
            # Base instance latency from answer type
            base_lat = self._instance_latency.get(iid, {}).get(
                predicted_answer_type, 500.0
            )

            # Communication cost
            if self.use_comm_cost:
                comm_lat = compute_comm_cost(
                    iid,
                    record.image_path,
                    record.question_text,
                    self.comm_profile,
                )
            else:
                comm_lat = 0.0

            latencies[iid] = base_lat + comm_lat

        return latencies

    def estimate_batch(
        self,
        records: List[RequestRecord],
        predicted_answer_types: List[str],
        instance_ids: List[str],
    ) -> np.ndarray:
        """Estimate latency for a batch of requests.

        Args:
            records: List of RequestRecord.
            predicted_answer_types: List of predicted answer type strings.
            instance_ids: Ordered list of instance IDs.

        Returns:
            L: (N, K) array where L[i,k] = l_{i,k} in ms.
        """
        N = len(records)
        K = len(instance_ids)
        L = np.zeros((N, K), dtype=np.float32)

        for i, (rec, at) in enumerate(zip(records, predicted_answer_types)):
            lat_dict = self.estimate(rec, at, instance_ids)
            for k, iid in enumerate(instance_ids):
                L[i, k] = lat_dict.get(iid, 500.0)

        return L

    def get_instance_ids(self) -> List[str]:
        """Return list of instance IDs from latency table."""
        return list(self._instance_latency.keys())
