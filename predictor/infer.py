"""Inference for the Predictor module.

Given a checkpoint and request data, output:
  - suitability scores s_i per model
  - predicted answer type â_i
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import torch
import numpy as np
from tqdm import tqdm

from .model import Predictor

logger = logging.getLogger(__name__)


def load_predictor(
    checkpoint_path: str,
    device: str = "cpu",
) -> Tuple[Predictor, List[str]]:
    """Load a trained Predictor from checkpoint.

    Returns:
        (predictor, model_names)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_names = checkpoint.get("model_names", [f"model_{i}" for i in range(checkpoint["num_models"])])
    num_models = len(model_names)

    predictor = Predictor(num_models=num_models)
    predictor.load_state_dict(checkpoint["model_state_dict"])
    predictor = predictor.to(device)
    predictor.eval()

    logger.info(
        f"Loaded Predictor from {checkpoint_path} "
        f"(epoch={checkpoint.get('epoch', '?')}, "
        f"val_loss={checkpoint.get('val_loss', '?'):.4f}, "
        f"num_models={num_models})"
    )
    return predictor, model_names


def predict(
    predictor: Predictor,
    records,
    model_names: Optional[List[str]] = None,
    batch_size: int = 32,
    device: str = "cpu",
    image_loader=None,  # callable: record -> PIL Image
) -> Dict[str, np.ndarray]:
    """Run inference on a list of RequestRecords.

    Args:
        predictor: Trained Predictor model.
        records: List of RequestRecord.
        model_names: Ordered list of model names (for column order).
        batch_size: Batch size for inference.
        device: Device to run on.

    Returns:
        dict with:
          - suitability_scores: (N, K) float32 array
          - answer_type_probs: (N, 3) float32 array
          - predicted_answer_types: (N,) int32 array (0=short, 1=medium, 2=long)
          - request_ids: (N,) string array
    """
    predictor = predictor.to(device)
    predictor.eval()

    from data_adapters.common_schema import AnswerType

    all_suit_probs = []
    all_type_probs = []
    all_type_preds = []
    all_request_ids = []

    with torch.no_grad():
        for i in tqdm(range(0, len(records), batch_size), desc="Predicting"):
            batch_records = records[i : i + batch_size]

            # Load images: use image_loader if provided, else try image_path
            if image_loader is not None:
                images = []
                for r in batch_records:
                    try:
                        images.append(image_loader(r))
                    except Exception:
                        from PIL import Image as PILImage
                        images.append(PILImage.new("RGB", (224, 224)))
            else:
                images = [r.image_path for r in batch_records]

            texts = [r.question_text for r in batch_records]

            try:
                outputs = predictor(images, texts)

                suit_probs = outputs["suitability_probs"].cpu().numpy()
                type_probs = outputs["answer_type_probs"].cpu().numpy()
                type_preds = type_probs.argmax(axis=-1)

                all_suit_probs.append(suit_probs)
                all_type_probs.append(type_probs)
                all_type_preds.append(type_preds)
                all_request_ids.extend([r.request_id for r in batch_records])

            except Exception as e:
                logger.warning(f"Batch {i//batch_size} failed: {e}")
                # Fill with NaN
                n = len(batch_records)
                K = predictor.num_models
                all_suit_probs.append(np.full((n, K), np.nan, dtype=np.float32))
                all_type_probs.append(np.full((n, 3), np.nan, dtype=np.float32))
                all_type_preds.append(np.full((n,), -1, dtype=np.int32))
                all_request_ids.extend([r.request_id for r in batch_records])

    result = {
        "suitability_scores": np.concatenate(all_suit_probs, axis=0),
        "answer_type_probs": np.concatenate(all_type_probs, axis=0),
        "predicted_answer_types": np.concatenate(all_type_preds, axis=0),
        "request_ids": np.array(all_request_ids),
    }

    # Convert answer type indices to string labels
    idx_to_type = {0: AnswerType.SHORT, 1: AnswerType.MEDIUM, 2: AnswerType.LONG}
    result["predicted_answer_type_labels"] = np.array([
        idx_to_type.get(int(t), "moderate") for t in result["predicted_answer_types"]
    ])

    logger.info(f"Prediction complete: {len(records)} requests, "
                f"suitability shape={result['suitability_scores'].shape}")

    return result
