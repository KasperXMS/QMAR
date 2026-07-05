"""Latency table builder (Section 4.2).

Constructs answer-type-based latency table: l̄_{k,a}
  - instance k × answer type a → average latency

Since benchmarks don't provide real edge latency, this module:
  1. Uses synthetic profiles based on model size + device class
  2. Can import real profiling data if available
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# Synthetic latency profiles (milliseconds) for common edge/cloud devices
# These are approximate baselines for VLM inference
SYNTHETIC_DEVICE_PROFILES = {
    # device_class -> {model_size -> {answer_type -> base_latency_ms}}
    "Orin_Nano": {
        "tiny":   {"simple": 150, "moderate": 400, "complex": 1200},
        "small":  {"simple": 300, "moderate": 800, "complex": 2500},
        "medium": {"simple": 600, "moderate": 1600, "complex": 5000},
    },
    "Orin_NX": {
        "tiny":   {"simple": 100, "moderate": 280, "complex": 800},
        "small":  {"simple": 200, "moderate": 550, "complex": 1600},
        "medium": {"simple": 400, "moderate": 1100, "complex": 3200},
        "large":  {"simple": 800, "moderate": 2200, "complex": 6000},
    },
    # AGX Orin 32GB: ~1.6–2× faster than NX (2048 GPU cores vs 1024, 204 GB/s mem vs 102 GB/s)
    "AGX_Orin": {
        "tiny":   {"simple": 60,  "moderate": 160, "complex": 450},
        "small":  {"simple": 120, "moderate": 320, "complex": 900},
        "medium": {"simple": 240, "moderate": 650, "complex": 1800},
        "large":  {"simple": 480, "moderate": 1300, "complex": 3600},
    },
    "RTX_4090": {
        "tiny":   {"simple": 20, "moderate": 60, "complex": 180},
        "small":  {"simple": 40, "moderate": 110, "complex": 320},
        "medium": {"simple": 80, "moderate": 220, "complex": 650},
        "large":  {"simple": 150, "moderate": 450, "complex": 1300},
        "xl":     {"simple": 300, "moderate": 900, "complex": 2600},
    },
}


# Approximate model size classification
def classify_model_size(model_name: str) -> str:
    """Classify model into size bucket based on name heuristics.

    Buckets are aligned with the edge instance pool:
      tiny  — 1B (Janus-1B)
      small — 2–4B (SmolVLM2, Phi-3.5-Vision)
      medium — 7–8B (Qwen2.5-VL-7B, LLaVA-Next-Vicuna-7B)
      large — 12–13B (Pixtral-12B)
      xl    — 32B+ (Qwen2.5-VL-32B)
    """
    name_lower = model_name.lower()
    # Check larger/more-specific sizes first to avoid partial matches
    # (e.g. "12b" contains "2b", "32b" contains "2b")
    if any(x in name_lower for x in ["27b", "32b", "gpt", "claude", "gemini", "sonnet", "flash", "pro"]):
        return "xl"
    elif any(x in name_lower for x in ["12b", "13b"]):
        return "large"
    elif any(x in name_lower for x in ["7b", "8b"]):
        return "medium"
    elif any(x in name_lower for x in ["1b", "tiny", "nano", "smolvlm"]):
        return "tiny"
    elif any(x in name_lower for x in ["2b", "3b", "4b", "5b", "mini", "small", "phi-3.5"]):
        return "small"
    else:
        return "medium"  # default


class LatencyTableBuilder:
    """Build instance-level latency table."""

    def __init__(
        self,
        instance_config: Optional[List[Dict]] = None,
        use_synthetic: bool = True,
    ):
        """
        Args:
            instance_config: List of instance dicts, each with:
                - instance_id: str
                - model_name: str (which VLM is deployed)
                - device_class: str (Orin_NX, RTX_3090, etc.)
            use_synthetic: If True, use synthetic profiles. Otherwise require real data.
        """
        self.instance_config = instance_config or []
        self.use_synthetic = use_synthetic

    def add_instance(self, instance_id: str, model_name: str, device_class: str):
        """Add an instance configuration."""
        self.instance_config.append({
            "instance_id": instance_id,
            "model_name": model_name,
            "device_class": device_class,
        })

    def build(
        self,
        answer_types: List[str] = None,
        output_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """Build latency table: rows=instances, columns=answer_types.

        Returns DataFrame with columns:
          instance_id, model_name, device_class, simple, moderate, complex
        """
        if answer_types is None:
            answer_types = ["simple", "moderate", "complex"]

        rows = []
        for cfg in self.instance_config:
            model_size = classify_model_size(cfg["model_name"])
            device_class = cfg["device_class"]

            row = {
                "instance_id": cfg["instance_id"],
                "model_name": cfg["model_name"],
                "device_class": device_class,
            }

            if self.use_synthetic:
                profile = SYNTHETIC_DEVICE_PROFILES.get(device_class, {})
                size_profile = profile.get(model_size, profile.get("moderate", {}))
                for at in answer_types:
                    row[at] = size_profile.get(at, 500)  # default 500ms
            else:
                # Placeholder for real profiling data
                for at in answer_types:
                    row[at] = 100  # dummy

            rows.append(row)

        df = pd.DataFrame(rows)

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            logger.info(f"Latency table saved to {output_path}")

        return df

    def get_latency(
        self, instance_id: str, answer_type: str, latency_table: pd.DataFrame = None
    ) -> float:
        """Look up latency l̄_{k,a} for a specific instance and answer type.

        Args:
            instance_id: Instance identifier.
            answer_type: One of 'simple', 'moderate', 'complex'.
            latency_table: Pre-built latency DataFrame.

        Returns:
            Latency in milliseconds.
        """
        if latency_table is None:
            return 500.0  # default
        row = latency_table[latency_table["instance_id"] == instance_id]
        if len(row) == 0:
            return 500.0
        return float(row.iloc[0][answer_type])
