"""Latency table builder (Section 4.2).

Constructs latency table l̄_{k,a} from a TOPS-calibrated physical model:

    l̄_{k,a} = α_ref × (P_k / D_{dev(k)}) × C_a × φ_{dev(k)}

where:
  P_k      — model parameter count (billions)
  D_dev    — device compute (INT8 TOPS)
  C_a      — task complexity factor (simple=1, moderate=3, complex=8)
  φ_dev    — device efficiency factor (Jetson penalty vs dGPU)
  α_ref    — calibration constant from real profiling

The table can be refined by replacing any cell with real profiling data
(see scripts/profile_instances.py). A single reference measurement per
device class is sufficient to calibrate all models on that device.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Device compute specs
# ═══════════════════════════════════════════════════════════════════

# INT8 TOPS (manufacturer quoted, sparsity off)
DEVICE_TOPS: Dict[str, float] = {
    "Orin_NX":    72,     # Orin NX 16GB — 1024 GPU cores, 32 Tensor
    "AGX_Orin":  248,     # AGX Orin 32GB — 2048 GPU cores, 64 Tensor
    "RTX_4090":  1321,    # RTX 4090 — 512 Tensor cores (Ada)
}

# Memory bandwidth GB/s — matters for decode-heavy requests
DEVICE_BANDWIDTH_GBS: Dict[str, float] = {
    "Orin_NX":   102,     # LPDDR5 128-bit
    "AGX_Orin":  204,     # LPDDR5 256-bit
    "RTX_4090":  1008,    # GDDR6X 384-bit
}

# Device efficiency factor (empirical: Jetson achieves lower effective
# utilisation than dGPU due to thermal limits, shared memory, etc.)
DEVICE_EFFICIENCY: Dict[str, float] = {
    "Orin_NX":   0.85,
    "AGX_Orin":  0.90,
    "RTX_4090":  1.00,    # reference
}


# ═══════════════════════════════════════════════════════════════════
# Model parameter counts (billions)
# ═══════════════════════════════════════════════════════════════════

MODEL_PARAMS_B: Dict[str, float] = {
    "Janus-Pro-1B":           1.0,
    "SmolVLM2":               2.2,
    "Phi-3.5-Vision":         4.2,
    "Qwen2.5-VL-7B":          8.3,
    "LLaVA-Next-Vicuna-7B":   7.6,
    "Pixtral-12B":           12.0,
    "Qwen2.5-VL-32B":        32.0,
    # fallback
    "_default":                7.0,
}


# ═══════════════════════════════════════════════════════════════════
# Task complexity factors
# ═══════════════════════════════════════════════════════════════════

COMPLEXITY_FACTORS: Dict[str, float] = {
    "simple":    1.0,
    "moderate":  3.0,
    "complex":   8.0,
}


# ═══════════════════════════════════════════════════════════════════
# Calibration
# ═══════════════════════════════════════════════════════════════════

# Calibration constant α_ref: derived from one real profiling point.
# Default is tuned so that Phi-3.5 (4.2B) on AGX Orin (248 TOPS)
# yields simple ≈ 200 ms:
#   α_ref = 200 / (4.2 / 248 × 0.90 × 1.0) ≈ 13122
# Adjust this after running real profiling on any instance.
ALPHA_REF: float = 13122.0


# ═══════════════════════════════════════════════════════════════════
# Legacy synthetic table (fallback)
# ═══════════════════════════════════════════════════════════════════

SYNTHETIC_DEVICE_PROFILES = {
    "Orin_NX": {
        "tiny":   {"simple": 140, "moderate": 400, "complex": 1100},
        "small":  {"simple": 280, "moderate": 800, "complex": 2200},
        "medium": {"simple": 550, "moderate": 1600, "complex": 4400},
        "large":  {"simple": 1100, "moderate": 3100, "complex": 8300},
    },
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


# ═══════════════════════════════════════════════════════════════════
# TOPS-based theoretical latency estimation
# ═══════════════════════════════════════════════════════════════════

def estimate_latency_theoretical(
    model_name: str,
    device_class: str,
    complexity: str,
    alpha: float = ALPHA_REF,
) -> float:
    """Estimate VLM inference latency from device TOPS and model params.

    Formula:
      l̄ = α × (P / D) / φ × C
    where α is calibrated from a real measurement.

    Args:
        model_name: canonical model name (key in MODEL_PARAMS_B).
        device_class: device class (key in DEVICE_TOPS).
        complexity: "simple" | "moderate" | "complex".
        alpha: calibration constant.

    Returns:
        Estimated latency in milliseconds.
    """
    P = MODEL_PARAMS_B.get(model_name, MODEL_PARAMS_B["_default"])
    D = DEVICE_TOPS.get(device_class, 248)
    C = COMPLEXITY_FACTORS.get(complexity, 3.0)
    phi = DEVICE_EFFICIENCY.get(device_class, 1.0)

    latency = alpha * (P / D) / phi * C
    return round(latency)


def calibrate_alpha(
    model_name: str,
    device_class: str,
    complexity: str,
    measured_latency_ms: float,
) -> float:
    """Compute calibration constant from a single real measurement.

    Usage:
      alpha = calibrate_alpha("Phi-3.5-Vision", "AGX_Orin", "simple", 195.0)
      # alpha ≈ 12794

    Then use this alpha for all other estimates on any device.
    """
    P = MODEL_PARAMS_B.get(model_name, MODEL_PARAMS_B["_default"])
    D = DEVICE_TOPS.get(device_class, 248)
    C = COMPLEXITY_FACTORS.get(complexity, 1.0)
    phi = DEVICE_EFFICIENCY.get(device_class, 1.0)

    alpha = measured_latency_ms / ((P / D) / phi * C)
    logger.info(
        f"Calibration: {model_name} on {device_class}, {complexity}={measured_latency_ms:.0f}ms "
        f"→ α = {alpha:.0f}"
    )
    return alpha


def build_theoretical_table(
    instance_config: List[Dict],
    alpha: Optional[float] = None,
) -> pd.DataFrame:
    """Build latency table from TOPS-calibrated physical model.

    Args:
        instance_config: List of {instance_id, model_name, device_class}.
        alpha: Calibration constant. Uses ALPHA_REF if not provided.

    Returns:
        DataFrame: instance_id, model_name, device_class, simple, moderate, complex.
    """
    a = alpha if alpha is not None else ALPHA_REF

    rows = []
    for cfg in instance_config:
        row = {
            "instance_id": cfg["instance_id"],
            "model_name": cfg["model_name"],
            "device_class": cfg["device_class"],
        }
        for c in ["simple", "moderate", "complex"]:
            row[c] = estimate_latency_theoretical(
                cfg["model_name"], cfg["device_class"], c, alpha=a
            )
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df[["instance_id", "model_name", "device_class", "simple", "moderate", "complex"]]

    _log_table(df, a)
    return df


# ═══════════════════════════════════════════════════════════════════
# Hybrid builder — theoretical with profiling overrides
# ═══════════════════════════════════════════════════════════════════

def build_latency_table_hybrid(
    instance_config: List[Dict],
    profiling_results: Optional[Dict[str, Dict[str, float]]] = None,
    alpha: Optional[float] = None,
) -> pd.DataFrame:
    """Build latency table: theoretical base, real profiling overrides.

    profiling_results format:
      {"instance_id": {"simple": 195, "moderate": 580, "complex": 1720}, ...}

    Cells not in profiling_results fall back to theoretical estimates.
    """
    df = build_theoretical_table(instance_config, alpha=alpha)

    if profiling_results:
        for iid, vals in profiling_results.items():
            mask = df["instance_id"] == iid
            if mask.any():
                for c in ["simple", "moderate", "complex"]:
                    if c in vals and vals[c] > 0:
                        df.loc[mask, c] = vals[c]

    return df


# ═══════════════════════════════════════════════════════════════════
# Utility
# ═══════════════════════════════════════════════════════════════════

def _log_table(df: pd.DataFrame, alpha: float) -> None:
    """Print a formatted latency table."""
    logger.info(f"Latency table (α={alpha:.0f}):")
    logger.info(f"{'Instance':<25s} {'Simple':>8s} {'Mod':>8s} {'Complex':>8s}")
    logger.info("-" * 52)
    for _, row in df.iterrows():
        logger.info(
            f"{row['instance_id']:<25s} "
            f"{row['simple']:>6.0f}ms "
            f"{row['moderate']:>6.0f}ms "
            f"{row['complex']:>6.0f}ms"
        )


# ═══════════════════════════════════════════════════════════════════
# Legacy builder (kept for backward compatibility)
# ═══════════════════════════════════════════════════════════════════

def classify_model_size(model_name: str) -> str:
    """Classify model into size bucket (legacy, used by LatencyTableBuilder)."""
    name_lower = model_name.lower()
    if any(x in name_lower for x in ["27b", "32b"]):
        return "xl"
    elif any(x in name_lower for x in ["12b", "13b"]):
        return "large"
    elif any(x in name_lower for x in ["7b", "8b"]):
        return "medium"
    elif any(x in name_lower for x in ["1b", "tiny", "nano", "smolvlm"]):
        return "tiny"
    elif any(x in name_lower for x in ["2b", "3b", "4b", "5b", "mini", "small", "phi-3.5"]):
        return "small"
    return "medium"


class LatencyTableBuilder:
    """Legacy builder using coarse size-bucket synthetic profiles.

    Prefer build_theoretical_table() or build_latency_table_hybrid() for
    TOPS-calibrated estimates.
    """

    def __init__(
        self,
        instance_config: Optional[List[Dict]] = None,
        use_synthetic: bool = True,
    ):
        self.instance_config = instance_config or []
        self.use_synthetic = use_synthetic

    def add_instance(self, instance_id: str, model_name: str, device_class: str):
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
                size_profile = profile.get(model_size, profile.get("medium", {}))
                for at in answer_types:
                    row[at] = size_profile.get(at, 500)
            else:
                for at in answer_types:
                    row[at] = 100

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
        if latency_table is None:
            return 500.0
        row = latency_table[latency_table["instance_id"] == instance_id]
        if len(row) == 0:
            return 500.0
        return float(row.iloc[0][answer_type])
