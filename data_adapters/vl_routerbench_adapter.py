"""Adapter for VL-RouterBench data → unified RequestRecord schema.

Handles:
  - xlsx evaluation files (17 models, 14 datasets) under VLMEvalKit_evaluation/
  - TSV image files under TSV_images/ (base64-encoded)
  - Fuzzy filename matching for various xlsx suffixes
"""

import base64
import io
import logging
import os
import re
from pathlib import Path
from typing import Optional, List

import pandas as pd
from PIL import Image
from tqdm import tqdm

from .common_schema import RequestRecord

logger = logging.getLogger(__name__)

# Actual model directory names → canonical names
MODEL_DIR_MAP = {
    "GPT4o": "GPT-4o",
    "GeminiFlash2-5": "Gemini-Flash-2.5",
    "Gemma3-27B": "Gemma3-27B",
    "InternVL2_5-78B": "InternVL2_5-78B",
    "Janus-Pro-1B": "Janus-Pro-1B",
    "Janus-Pro-7B": "Janus-Pro-7B",
    "Kimi-VL-A3B-Thinking-2506": "Kimi-VL-A3B-Thinking",
    "MiMo-VL-7B-RL": "MiMo-VL-7B-RL",
    "Phi-3.5-Vision": "Phi-3.5-Vision",
    "Pixtral-12B": "Pixtral-12B",
    "Qianfan-VL-8B": "Qianfan-VL-8B",
    "Qwen2.5-VL-32B-Instruct": "Qwen2.5-VL-32B",
    "Qwen2.5-VL-72B-Instruct": "Qwen2.5-VL-72B",
    "SmolVLM2": "SmolVLM2",
    "deepseek_vl2": "DeepSeek-VL2",
    "deepseek_vl2_tiny": "DeepSeek-VL2-Tiny",
    "llava_next_vicuna_7b": "LLaVA-Next-Vicuna-7B",
}

# Reverse map
CANONICAL_TO_DIR = {v: k for k, v in MODEL_DIR_MAP.items()}

# TSV file mapping: dataset name → TSV filename
DATASET_TSV_MAP = {
    "AI2D_TEST": "AI2D_TEST.tsv",
    "ChartQA_TEST": "ChartQA_TEST.tsv",
    "DocVQA_VAL": "DocVQA_VAL.tsv",
    "HallusionBench": "HallusionBench.tsv",
    "InfoVQA_VAL": "InfoVQA_VAL.tsv",
    "MathVerse_MINI": "MathVerse_MINI.tsv",
    "MathVision_MINI": "MathVision_MINI.tsv",
    "MathVista_MINI": "MathVista_MINI.tsv",
    "MMBench_DEV_EN_V11": "MMBench_DEV_EN_V11.tsv",
    "MMMU_DEV_VAL": "MMMU_DEV_VAL.tsv",
    "MMStar": "MMStar.tsv",
    "OCRBench": "OCRBench.tsv",
    "RealWorldQA": "RealWorldQA.tsv",
    "TextVQA_VAL": "TextVQA_VAL.tsv",
}


class TSVImageLoader:
    """Load images from VL-RouterBench TSV files.

    TSV files contain base64-encoded JPEG/PNG images in an 'image' column.
    Images are loaded and decoded on demand.
    """

    def __init__(self, tsv_dir: str):
        self.tsv_dir = Path(tsv_dir)
        self._cache = {}  # (dataset, index) -> PIL Image

    def load_image(self, dataset: str, index) -> Image.Image:
        """Load a single image from TSV by dataset and index.

        Reads on-the-fly without caching to avoid memory explosion.
        """
        tsv_filename = DATASET_TSV_MAP.get(dataset)
        if tsv_filename is None:
            raise FileNotFoundError(f"No TSV mapping for dataset: {dataset}")

        tsv_path = self.tsv_dir / tsv_filename
        if not tsv_path.exists():
            raise FileNotFoundError(f"TSV not found: {tsv_path}")

        # Lazy-load TSV once per dataset (but don't cache images)
        if dataset not in self._cache:
            self._cache[dataset] = pd.read_csv(tsv_path, sep="\t")

        df = self._cache[dataset]
        idx_str = str(index)
        mask = df["index"].astype(str) == idx_str
        if mask.sum() == 0:
            raise ValueError(f"Index {index} not found in {tsv_filename}")

        img_b64 = df.loc[mask, "image"].iloc[0]
        img_bytes = base64.b64decode(img_b64)
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")

    def load_images_batch(self, datasets: List[str], indices: List[int]) -> List[Image.Image]:
        """Load a batch of images."""
        images = []
        for ds, idx in zip(datasets, indices):
            try:
                img = self.load_image(ds, idx)
            except Exception as e:
                logger.warning(f"Failed to load image {ds}/{idx}: {e}")
                img = Image.new("RGB", (224, 224))
            images.append(img)
        return images

    def get_image_count(self, dataset: str) -> int:
        """Get the number of images in a dataset TSV."""
        tsv_filename = DATASET_TSV_MAP.get(dataset)
        if tsv_filename is None:
            return 0
        tsv_path = self.tsv_dir / tsv_filename
        if not tsv_path.exists():
            return 0
        # Quick count using file size estimation or read just the header
        df = pd.read_csv(tsv_path, sep="\t", nrows=1)
        # Count total rows (expensive but accurate)
        # For a quicker estimate, count lines
        with open(tsv_path, "r") as f:
            return sum(1 for _ in f) - 1  # subtract header


class VLRouterBenchAdapter:
    """Convert VL-RouterBench evaluation data to unified RequestRecord schema."""

    def __init__(
        self,
        data_root: str,
        model_names: Optional[List[str]] = None,
        datasets: Optional[List[str]] = None,
        image_loader: Optional[TSVImageLoader] = None,
    ):
        self.data_root = Path(data_root)
        self.eval_dir = self.data_root / "VLMEvalKit_evaluation"
        self.tsv_dir = self.data_root / "TSV_images"
        self.image_loader = image_loader or TSVImageLoader(str(self.tsv_dir))

        # Use canonical model names; resolve to directory names internally
        if model_names is None:
            self.model_names = list(MODEL_DIR_MAP.values())
        else:
            self.model_names = model_names

        self.datasets = datasets or list(DATASET_TSV_MAP.keys())

    def _find_xlsx(
        self, model_dir_name: str, dataset: str
    ) -> Optional[Path]:
        """Find the evaluation xlsx file for a model directory + dataset.

        Handles various filename suffixes: _openai_result.xlsx, _results.xlsx,
        _auxmatch.xlsx, _gpt-4o-mini.xlsx, or just .xlsx.
        """
        model_dir = self.eval_dir / model_dir_name
        if not model_dir.exists():
            return None

        # Look for any xlsx file matching <model>_<dataset>_*.xlsx
        for f in sorted(model_dir.iterdir()):
            if not f.suffix == ".xlsx":
                continue
            fname = f.stem
            # Must contain both model dir name and dataset
            if model_dir_name in fname and dataset in fname:
                return f

        return None

    def _load_and_normalize(self, path: Path) -> pd.DataFrame:
        """Load an evaluation xlsx file and normalize columns."""
        df = pd.read_excel(path)

        rename = {}
        for col in df.columns:
            cl = col.lower().strip()
            if "correct" in cl and "is_correct" not in df.columns:
                rename[col] = "is_correct"
            elif cl in ("prediction", "predict"):
                rename[col] = "prediction"
            elif cl == "answer":
                rename[col] = "answer"
            elif cl == "question":
                rename[col] = "question"
            elif cl == "index":
                rename[col] = "index"

        if rename:
            df = df.rename(columns=rename)
        return df

    def load_all_records(
        self,
        max_per_dataset: Optional[int] = None,
    ) -> List[RequestRecord]:
        """Load all records from VL-RouterBench into unified schema.

        Strategy: For each dataset, load the TSV to get images + questions + GT answers.
        Then merge with evaluation xlsx files for model correctness.

        Args:
            max_per_dataset: Limit records per dataset (for quick experiments).
        """
        all_records = []

        for dataset in tqdm(self.datasets, desc="Loading datasets"):
            # Load TSV for this dataset (has images, questions, answers)
            tsv_filename = DATASET_TSV_MAP.get(dataset)
            if tsv_filename is None:
                continue
            tsv_path = self.tsv_dir / tsv_filename
            if not tsv_path.exists():
                logger.warning(f"TSV not found: {tsv_path}, skipping {dataset}")
                continue

            try:
                tsv_df = pd.read_csv(tsv_path, sep="\t")
            except Exception as e:
                logger.error(f"Failed to read {tsv_path}: {e}")
                continue

            if max_per_dataset and len(tsv_df) > max_per_dataset:
                tsv_df = tsv_df.head(max_per_dataset)

            # Build base records from TSV
            records_map = {}  # index -> RequestRecord
            for _, row in tsv_df.iterrows():
                idx_raw = row["index"]
                idx = str(idx_raw)  # Handle both int and string indices
                rec = RequestRecord(
                    request_id=f"vlr_{dataset}_{idx}",
                    dataset_name=dataset,
                    task_name=str(row.get("category", dataset)),
                    image_id=f"{dataset}_{idx}",
                    question_text=str(row.get("question", "")),
                    ground_truth_answer=str(row.get("answer", "")),
                    extra={
                        "category": str(row.get("category", "")),
                        "l2_category": str(row.get("l2_category", "")),
                        "bench": str(row.get("bench", "")),
                        "tsv_index": idx,
                    },
                )
                records_map[idx] = rec

            # Merge model correctness from xlsx files
            for canonical_name in self.model_names:
                dir_name = CANONICAL_TO_DIR.get(canonical_name)
                if dir_name is None:
                    continue

                xlsx_path = self._find_xlsx(dir_name, dataset)
                if xlsx_path is None:
                    continue

                try:
                    xdf = self._load_and_normalize(xlsx_path)
                except Exception as e:
                    logger.warning(f"Failed to load {xlsx_path}: {e}")
                    continue

                for _, row in xdf.iterrows():
                    idx = str(row.get("index", -1))
                    if idx not in records_map:
                        continue

                    is_correct = row.get("is_correct", 0)
                    if isinstance(is_correct, (int, float)):
                        correctness = float(is_correct)
                    elif isinstance(is_correct, str):
                        correctness = 1.0 if is_correct.lower() in ("true", "1", "yes") else 0.0
                    elif isinstance(is_correct, bool):
                        correctness = 1.0 if is_correct else 0.0
                    else:
                        correctness = 0.0

                    records_map[idx].candidate_model_correctness[canonical_name] = correctness
                    records_map[idx].candidate_model_outputs[canonical_name] = str(
                        row.get("prediction", "")
                    )

            all_records.extend(records_map.values())
            logger.debug(f"  {dataset}: {len(records_map)} records")

        logger.info(f"Loaded {len(all_records)} total records from VL-RouterBench")
        return all_records

    def get_image(self, dataset: str, index) -> Image.Image:
        """Load an image for a given dataset and index (int or str)."""
        return self.image_loader.load_image(dataset, index)

    def get_images_batch(self, records: List[RequestRecord]) -> List[Image.Image]:
        """Load images for a batch of records."""
        datasets = []
        indices = []
        for rec in records:
            ds = rec.dataset_name
            idx = rec.extra.get("tsv_index", 0)
            if ds not in DATASET_TSV_MAP:
                ds = self._guess_dataset(rec)
            datasets.append(ds)
            indices.append(int(idx))

        return self.image_loader.load_images_batch(datasets, indices)

    def _guess_dataset(self, record: RequestRecord) -> str:
        """Try to guess the TSV dataset name from record info."""
        rid = record.request_id
        for ds in DATASET_TSV_MAP:
            if ds in rid:
                return ds
        return "MMStar"  # fallback
