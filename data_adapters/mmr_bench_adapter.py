"""Adapter for MMR-Bench data → unified RequestRecord schema.

MMR-Bench stores data in a merged CSV:
  data/MMR_Bench.csv (~1.57M rows, 47 columns)
  - model predictions: <Model>_prediction, <Model>_correct, <Model>_token
  - model costs: <Model>_cost
  - metadata: dataset_idx, question, answer

This adapter reads the merged CSV and converts to RequestRecord.
"""

import logging
from pathlib import Path
from typing import Optional, List

import pandas as pd
from tqdm import tqdm

from .common_schema import RequestRecord

logger = logging.getLogger(__name__)

# 11 models in MMR-Bench
MMR_MODELS = [
    "Qwen2.5-VL-3B-Instruct", "Qwen2.5-VL-72B-Instruct",
    "InternVL3-78B", "Gemma3-4B", "Qwen2.5-VL-7B-Instruct",
    "gpt-5-minimal-2025-08-07", "Claude3-7V_Sonnet",
    "gpt-5-nano-2025-08-07", "GeminiPro2-5",
    "gpt-5-2025-08-07", "GeminiFlash2-5",
]


class MMRBenchAdapter:
    """Convert MMR-Bench merged CSV data to unified RequestRecord schema."""

    def __init__(
        self,
        csv_path: str,
        image_root: Optional[str] = None,
        model_names: Optional[List[str]] = None,
    ):
        self.csv_path = Path(csv_path)
        self.image_root = Path(image_root) if image_root else self.csv_path.parent
        self.model_names = model_names or MMR_MODELS

    def _discover_models(self, df: pd.DataFrame) -> List[str]:
        """Discover model names from column naming convention."""
        models = set()
        for col in df.columns:
            if col.endswith("_correct"):
                model = col[:-len("_correct")]
                models.add(model)
        return sorted(models)

    def load_all_records(
        self,
        max_rows: Optional[int] = None,
        filter_datasets: Optional[List[str]] = None,
    ) -> List[RequestRecord]:
        """Load records from MMR-Bench CSV into unified schema.

        Args:
            max_rows: If set, limit the number of rows to load (for quick experiments).
            filter_datasets: If set, only load records from these datasets.
        """
        logger.info(f"Loading MMR-Bench data from {self.csv_path}")
        df = pd.read_csv(self.csv_path, nrows=max_rows)

        # Discover models from column names
        if self.model_names is None:
            self.model_names = self._discover_models(df)

        # Validate that all expected model columns exist
        available_models = []
        for model in self.model_names:
            correct_col = f"{model}_correct"
            if correct_col in df.columns:
                available_models.append(model)
            else:
                logger.debug(f"Model {model} not found in CSV columns, skipping")

        self.model_names = available_models
        logger.info(f"Found {len(self.model_names)} models in CSV")

        records = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Building records"):
            dataset_idx = str(row.get("dataset_idx", "unknown"))
            # Parse dataset name from dataset_idx (e.g., "MMStar_42" -> "MMStar")
            dataset_name = self._parse_dataset_name(dataset_idx)

            if filter_datasets and dataset_name not in filter_datasets:
                continue

            # Resolve image path
            image_path = row.get("image_path") or row.get("img_path")
            if image_path is None or pd.isna(image_path):
                # Construct from dataset_idx convention
                image_path = self._resolve_image_path(dataset_idx)

            record = RequestRecord(
                request_id=f"mmr_{dataset_idx}",
                dataset_name=dataset_name,
                task_name=dataset_name,
                image_path=str(image_path) if image_path else None,
                image_id=dataset_idx,
                question_text=str(row.get("question", "")),
                ground_truth_answer=str(row.get("answer", "")),
            )

            # Add correctness and outputs for each model
            for model in self.model_names:
                correct_col = f"{model}_correct"
                pred_col = f"{model}_prediction"
                token_col = f"{model}_token"
                cost_col = f"{model}_cost"

                if correct_col in df.columns:
                    val = row[correct_col]
                    if pd.notna(val):
                        if isinstance(val, bool):
                            record.candidate_model_correctness[model] = 1.0 if val else 0.0
                        else:
                            record.candidate_model_correctness[model] = float(val)

                if pred_col in df.columns and pd.notna(row[pred_col]):
                    record.candidate_model_outputs[model] = str(row[pred_col])

                # Store token count and cost in extra metadata
                if token_col in df.columns and pd.notna(row[token_col]):
                    record.extra.setdefault("token_counts", {})[model] = int(row[token_col])
                if cost_col in df.columns and pd.notna(row[cost_col]):
                    record.extra.setdefault("costs", {})[model] = float(row[cost_col])

            if record.candidate_model_correctness:
                records.append(record)

        logger.info(f"Loaded {len(records)} records from MMR-Bench")
        return records

    def _parse_dataset_name(self, dataset_idx: str) -> str:
        """Parse dataset name from dataset_idx.

        Examples:
            MMStar_42 -> MMStar
            MathVerse_1000 -> MathVerse
            SEEDBench2_Plus_0 -> SEEDBench2_Plus
        """
        # Try to match known dataset prefixes
        known_prefixes = [
            "SEEDBench2_Plus", "MathVision", "MathVerse",
            "MathVista", "MMStar", "OCRBench", "RealWorldQA",
        ]
        for prefix in known_prefixes:
            if dataset_idx.startswith(prefix):
                return prefix
        # Fallback: split on last underscore-number pattern
        parts = dataset_idx.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            return parts[0]
        return dataset_idx

    def _resolve_image_path(self, dataset_idx: str) -> Optional[str]:
        """Resolve image path from dataset_idx convention."""
        dataset_name = self._parse_dataset_name(dataset_idx)
        # MMR-Bench images are stored as <dataset>/<index>.jpg
        image_filename = f"{dataset_idx}.jpg"
        candidate = self.image_root / dataset_name / image_filename
        if candidate.exists():
            return str(candidate)

        # Try with just the numeric index
        parts = dataset_idx.rsplit("_", 1)
        if len(parts) == 2:
            image_filename = f"{parts[1]}.jpg"
            candidate = self.image_root / dataset_name / image_filename
            if candidate.exists():
                return str(candidate)

        return None
