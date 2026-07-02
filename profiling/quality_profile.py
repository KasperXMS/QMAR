"""Quality profile builder (Section 4.1).

Constructs correctness labels from benchmark records and builds quality matrix Y.
"""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from data_adapters.common_schema import RequestRecord

logger = logging.getLogger(__name__)


def build_quality_profile(
    records: List[RequestRecord],
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """Build quality profile table from RequestRecords.

    Output columns:
      request_id, model_name, correctness_label, score
    """
    rows = []
    for rec in tqdm(records, desc="Building quality profile"):
        for model_name, correctness in rec.candidate_model_correctness.items():
            rows.append({
                "request_id": rec.request_id,
                "model_name": model_name,
                "correctness_label": int(correctness >= 0.5),
                "score": correctness,
            })

    df = pd.DataFrame(rows)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Quality profile saved to {output_path} ({len(df)} rows)")

    return df


def build_quality_matrix(
    records: List[RequestRecord],
    model_names: List[str],
) -> np.ndarray:
    """Build binary quality matrix Y of shape (N_samples, K_models)."""
    N = len(records)
    K = len(model_names)
    Y = np.zeros((N, K), dtype=np.int32)
    model_to_idx = {m: k for k, m in enumerate(model_names)}

    for i, rec in enumerate(records):
        for model, correctness in rec.candidate_model_correctness.items():
            if model in model_to_idx:
                Y[i, model_to_idx[model]] = int(correctness >= 0.5)

    return Y


def get_model_pool(records: List[RequestRecord]) -> List[str]:
    """Extract sorted set of candidate models from records."""
    models = set()
    for rec in records:
        models.update(rec.model_names)
    return sorted(models)
