"""Communication profile builder (Section 4.3).

Simple model: l^{comm}_{i,k} = RTT_k + (size(I_i) + size(T_i)) / BW_k

If no real network data, uses synthetic RTT and bandwidth profiles.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Synthetic communication profiles for edge devices
SYNTHETIC_COMM_PROFILES = {
    # device_class -> {RTT_ms, bandwidth_mbps}
    "Orin_NX":    {"RTT_ms": 2, "BW_mbps": 150},    # local edge (WiFi/LAN)
    "AGX_Orin":   {"RTT_ms": 2, "BW_mbps": 200},    # local edge (WiFi/LAN)
    "RTX_4090":   {"RTT_ms": 3, "BW_mbps": 800},    # local server (LAN)
}


def estimate_image_size(image_path: str) -> float:
    """Estimate image size in bits from file."""
    if image_path and os.path.exists(image_path):
        return os.path.getsize(image_path) * 8  # bytes -> bits
    # Default: assume ~200KB image
    return 200 * 1024 * 8  # 1,638,400 bits


def estimate_text_size(text: str) -> float:
    """Estimate text size in bits (UTF-8 encoding)."""
    return len(text.encode("utf-8")) * 8


def build_comm_profile(
    instance_config: List[Dict],
    records: List = None,
    output_path: Optional[str] = None,
    use_synthetic: bool = True,
) -> pd.DataFrame:
    """Build communication cost matrix.

    Args:
        instance_config: List of {instance_id, device_class} dicts.
        records: Optional list of RequestRecord for per-request image sizes.
        output_path: Optional CSV output path.
        use_synthetic: Use synthetic RTT/BW if True.

    Returns:
        DataFrame with columns: instance_id, device_class, RTT_ms, BW_mbps
    """
    rows = []
    for cfg in instance_config:
        device_class = cfg.get("device_class", "RTX_3090")
        if use_synthetic:
            profile = SYNTHETIC_COMM_PROFILES.get(
                device_class,
                {"RTT_ms": 10, "BW_mbps": 200},
            )
            rows.append({
                "instance_id": cfg["instance_id"],
                "device_class": device_class,
                "RTT_ms": profile["RTT_ms"],
                "BW_mbps": profile["BW_mbps"],
            })
        else:
            rows.append({
                "instance_id": cfg["instance_id"],
                "device_class": device_class,
                "RTT_ms": 10,
                "BW_mbps": 200,
            })

    df = pd.DataFrame(rows)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Communication profile saved to {output_path}")

    return df


def compute_comm_cost(
    instance_id: str,
    image_path: Optional[str],
    question_text: str,
    comm_profile: pd.DataFrame,
) -> float:
    """Compute communication cost for a specific request-instance pair.

    l^{comm}_{i,k} = RTT_k + (size(I_i) + size(T_i)) / BW_k

    Returns latency contribution in milliseconds.
    """
    row = comm_profile[comm_profile["instance_id"] == instance_id]
    if len(row) == 0:
        return 100.0  # default 100ms

    rtt_ms = float(row.iloc[0]["RTT_ms"])
    bw_mbps = float(row.iloc[0]["BW_mbps"])  # megabits per second

    # Data sizes in bits
    image_bits = estimate_image_size(image_path) if image_path else 0
    text_bits = estimate_text_size(question_text)
    total_mbits = (image_bits + text_bits) / 1_000_000  # bits -> megabits

    # Transmission time in ms
    trans_ms = (total_mbits / bw_mbps) * 1000

    return rtt_ms + trans_ms
