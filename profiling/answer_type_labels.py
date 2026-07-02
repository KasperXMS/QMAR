"""Answer type label builder (Section 2.3.2, revised).

Classifies requests by task complexity for latency estimation:
  - simple:   Direct visual perception (object recognition, scene description, OCR)
  - moderate: Structured understanding (charts, documents, spatial relations, diagrams)
  - complex:  Multi-step reasoning (math, logic, hallucination detection)

Uses (dataset, task_category) mapping from AnswerType.classify(),
NOT ground-truth answer token length.
"""

import logging
from pathlib import Path
from typing import List, Optional
import pandas as pd
from tqdm import tqdm

from data_adapters.common_schema import RequestRecord, AnswerType

logger = logging.getLogger(__name__)


def build_answer_type_labels(
    records: List[RequestRecord],
    tokenizer=None,  # deprecated, kept for API compatibility
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """Build task-complexity-based answer type labels.

    Uses AnswerType.classify() which maps (dataset, task_category) to
    simple/moderate/complex. Falls back to question text heuristics.

    Args:
        records: List of RequestRecord with dataset_name and task_name populated.
        tokenizer: Ignored (kept for backward compatibility).
        output_path: If provided, save CSV.

    Returns:
        DataFrame with columns: request_id, answer_type, dataset, task_name
    """
    rows = []
    for rec in tqdm(records, desc="Building answer type labels"):
        answer_type = AnswerType.classify(rec)
        rows.append({
            "request_id": rec.request_id,
            "answer_type": answer_type,
            "dataset": rec.dataset_name,
            "task_name": rec.task_name,
            "question_preview": rec.question_text[:120],
        })

    df = pd.DataFrame(rows)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Answer type labels saved to {output_path}")

    # Log per-dataset distribution
    logger.info("Answer type distribution:")
    dist = df["answer_type"].value_counts()
    logger.info(f"  Overall: {dist.to_dict()}")

    # Per-dataset breakdown
    for ds in sorted(df["dataset"].unique()):
        ds_dist = df[df["dataset"] == ds]["answer_type"].value_counts().to_dict()
        logger.info(f"  {ds}: {ds_dist}")

    return df
