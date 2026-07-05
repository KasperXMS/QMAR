#!/usr/bin/env python
"""Profile VLM instances on real edge hardware.

Run this script on EACH device (Orin NX, AGX Orin, RTX 4090) separately.
It profiles all instances whose device_class matches the current machine.

Workflow (one device at a time):
  1. On a machine with GPU/benchmark data:
     python scripts/sample_requests.py --output-dir outputs/profiling

  2. Copy outputs/profiling/ to each edge device.

  3. On EACH device, run:
     python scripts/profile_instances.py \
       --device-class AGX_Orin \           # which device class to profile
       --requests outputs/profiling/profiling_requests.json \
       --iterations 3 --warmup 5

  4. Collect all outputs/profiling/*_raw.csv + *_latency.json back to
     the main machine.

  5. Build final latency table:
     python scripts/profile_instances.py --merge \\
       --results-dir outputs/profiling \\
       --output outputs/profiling/latency_table.csv

Features:
  - Crash-resilient: skips instances with existing raw CSV
  - Thermal management: cooldown between requests for Jetson devices
  - Progress logging: per-request timing with ETA
  - Merge mode: combines per-device results into final latency table
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from profiling.instance_profiler import (
    InstanceProfiler,
    InstanceProfilingResult,
    ProfilingRequest,
    load_profiling_requests,
    build_latency_table_from_results,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("profile_instances")


def parse_args():
    p = argparse.ArgumentParser(
        description="Profile VLM instances on real edge hardware"
    )

    # Mode selection
    p.add_argument(
        "--merge", action="store_true",
        help="Merge mode: combine per-instance results into final latency table",
    )

    # Profiling mode args
    p.add_argument(
        "--instance-config", type=str, default="configs/instance_pool.yaml",
        help="Path to instance pool YAML",
    )
    p.add_argument(
        "--device-class", type=str, default=None,
        help="Which device class to profile (Orin_NX, AGX_Orin, RTX_4090). "
             "If not set, auto-detects from hostname/GPU info.",
    )
    p.add_argument(
        "--instances", type=str, default=None,
        help="Comma-separated instance IDs to profile (overrides device-class filter)",
    )
    p.add_argument(
        "--requests", type=str, default="outputs/profiling/profiling_requests.json",
        help="Path to profiling requests JSON",
    )
    p.add_argument(
        "--output-dir", type=str, default="outputs/profiling",
        help="Output directory for profiling results",
    )
    p.add_argument(
        "--iterations", type=int, default=3,
        help="Number of measurement iterations per request",
    )
    p.add_argument(
        "--warmup", type=int, default=5,
        help="Number of warmup inference runs",
    )
    p.add_argument(
        "--cooldown", type=float, default=2.0,
        help="Cooldown seconds between requests (higher for Jetson devices)",
    )
    p.add_argument(
        "--device", type=str, default="cuda",
        help="Torch device: 'cuda', 'cuda:0', or 'cpu'",
    )
    p.add_argument(
        "--custom-model-id", type=str, default=None,
        help="Override HF model ID for all instances (JSON file or direct string)",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Re-profile even if results already exist",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Show which instances would be profiled, then exit",
    )

    # Merge mode args
    p.add_argument(
        "--results-dir", type=str, default="outputs/profiling",
        help="Directory containing per-instance *_latency.json files",
    )
    p.add_argument(
        "--output", type=str, default="outputs/profiling/latency_table.csv",
        help="Output path for merged latency table CSV",
    )

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════
# Device detection
# ═══════════════════════════════════════════════════════════════════


def detect_device_class() -> str:
    """Auto-detect device class from system info.

    Heuristics:
      - /sys/firmware/devicetree/base/model contains "Orin NX" → Orin_NX
      - /sys/firmware/devicetree/base/model contains "Orin" → AGX_Orin
      - nvidia-smi shows "RTX 4090" → RTX_4090
    """
    # Check Jetson device tree
    try:
        model_path = "/sys/firmware/devicetree/base/model"
        if os.path.exists(model_path):
            with open(model_path) as f:
                model = f.read().strip("\x00").strip()
            model_lower = model.lower()
            if "orin nx" in model_lower:
                return "Orin_NX"
            elif "orin" in model_lower:
                return "AGX_Orin"
    except Exception:
        pass

    # Check nvidia-smi
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        gpu_name = result.stdout.strip()
        if "4090" in gpu_name:
            return "RTX_4090"
    except Exception:
        pass

    # Check torch
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            if "4090" in gpu_name:
                return "RTX_4090"
    except Exception:
        pass

    logger.warning(
        "Could not auto-detect device class. "
        "Use --device-class to specify manually."
    )
    return "unknown"


# ═══════════════════════════════════════════════════════════════════
# Profiling mode
# ═══════════════════════════════════════════════════════════════════


def load_instance_config(config_path: str) -> List[Dict]:
    """Load instance pool config."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg["instances"]


def filter_instances(
    instances: List[Dict],
    device_class: Optional[str] = None,
    instance_ids: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    force: bool = False,
) -> List[Dict]:
    """Filter instances to profile on this device.

    Skips already-profiled instances unless --force is used.
    """
    if instance_ids:
        id_set = set(instance_ids)
        filtered = [i for i in instances if i["instance_id"] in id_set]
    elif device_class:
        filtered = [i for i in instances if i["device_class"] == device_class]
    else:
        filtered = instances

    # Check for existing results
    if output_dir and not force:
        out = Path(output_dir)
        remaining = []
        for inst in filtered:
            safe_name = inst["instance_id"].replace("@", "_").replace("/", "_")
            raw_path = out / f"{safe_name}_raw.csv"
            if raw_path.exists():
                logger.info(f"  SKIP {inst['instance_id']}: already profiled ({raw_path})")
            else:
                remaining.append(inst)
        return remaining

    return filtered


def preflight_check() -> dict:
    """Check environment and report version info.

    Returns a dict of {package: version} for diagnostics.
    """
    import sys
    import torch
    import transformers

    info = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)

    logger.info("── Environment ──")
    for k, v in info.items():
        logger.info(f"  {k}: {v}")

    # Compatibility warnings
    py_major, py_minor = map(int, info["python"].split(".")[:2])
    tf_major, tf_minor = map(int, info["transformers"].split(".")[:2])

    issues = []

    if (py_major, py_minor) < (3, 9):
        issues.append(
            "Python < 3.9 — some models may need trust_remote_code=True. "
            "Consider: conda install python=3.10"
        )

    if (tf_major, tf_minor) < (4, 40):
        issues.append(
            f"transformers {info['transformers']} is old — Qwen2.5-VL / SmolVLM "
            "may fail. Run: pip install --upgrade transformers"
        )

    # Check protobuf
    try:
        import google.protobuf
        pb_ver = google.protobuf.__version__
        if pb_ver.startswith("4."):
            issues.append(
                f"protobuf {pb_ver} conflicts with sentencepiece. "
                "Run: pip install protobuf==3.20.3"
            )
    except ImportError:
        pass

    if issues:
        logger.warning("── Known issues ──")
        for issue in issues:
            logger.warning(f"  ⚠ {issue}")
    else:
        logger.info("  ✓ No known compatibility issues")

    return info


def run_profiling(args) -> None:
    """Main profiling workflow."""
    # Pre-flight check
    preflight_check()

    # Load config
    instances = load_instance_config(args.instance_config)
    logger.info(f"Loaded {len(instances)} instances from {args.instance_config}")

    # Determine device class
    device_class = args.device_class or detect_device_class()
    logger.info(f"Device class: {device_class}")

    # Parse instance filter
    instance_ids = None
    if args.instances:
        instance_ids = [i.strip() for i in args.instances.split(",")]

    # Filter
    to_profile = filter_instances(
        instances, device_class, instance_ids, args.output_dir, args.force
    )

    if not to_profile:
        logger.info("No instances to profile on this device.")
        return

    logger.info(f"Will profile {len(to_profile)} instance(s):")
    for inst in to_profile:
        logger.info(f"  {inst['instance_id']} ({inst['model_name']})")

    if args.dry_run:
        return

    # Load requests
    logger.info(f"Loading profiling requests from {args.requests}")
    requests = load_profiling_requests(args.requests)
    logger.info(
        f"Loaded {len(requests)} requests"
    )
    # Count by complexity
    from collections import Counter
    cc = Counter(r.complexity_class for r in requests)
    logger.info(f"  Breakdown: {dict(cc)}")

    # Profile each instance
    results: List[InstanceProfilingResult] = []
    t_start = time.time()

    for idx, inst in enumerate(to_profile):
        logger.info(f"\n{'='*60}")
        logger.info(
            f"[{idx+1}/{len(to_profile)}] Profiling {inst['instance_id']} "
            f"({inst['model_name']} on {inst['device_class']})"
        )
        logger.info(f"{'='*60}")

        custom_id = None
        if args.custom_model_id:
            if os.path.exists(args.custom_model_id):
                with open(args.custom_model_id) as f:
                    id_map = json.load(f)
                custom_id = id_map.get(inst["model_name"])
            else:
                custom_id = args.custom_model_id

        profiler = InstanceProfiler(
            instance_id=inst["instance_id"],
            model_name=inst["model_name"],
            device_class=inst["device_class"],
            device=args.device,
            output_dir=args.output_dir,
            custom_model_id=custom_id,
        )

        try:
            profiler.load()
            result = profiler.profile(
                requests=requests,
                n_iterations=args.iterations,
                n_warmup=args.warmup,
                cooldown_seconds=args.cooldown,
            )
            results.append(result)

        except Exception as e:
            logger.error(f"FAILED profiling {inst['instance_id']}: {e}", exc_info=True)

        finally:
            try:
                profiler.unload()
            except Exception:
                pass

    # Summary
    elapsed = time.time() - t_start
    logger.info(f"\n{'='*60}")
    logger.info(f"Profiling complete. {len(results)}/{len(to_profile)} instances succeeded.")
    logger.info(f"Total wall time: {elapsed/60:.1f} minutes")

    if results:
        # Print summary table
        logger.info(f"\n{'Instance':<25s} {'Simple':>10s} {'Moderate':>10s} {'Complex':>10s}")
        logger.info("-" * 55)
        for r in results:
            row = r.latency_table_row()
            logger.info(
                f"{r.instance_id:<25s} "
                f"{row['simple']:>8.0f}ms "
                f"{row['moderate']:>8.0f}ms "
                f"{row['complex']:>8.0f}ms"
            )


# ═══════════════════════════════════════════════════════════════════
# Merge mode
# ═══════════════════════════════════════════════════════════════════


def run_merge(args) -> None:
    """Merge per-instance latency JSONs into final latency table."""
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        sys.exit(1)

    json_files = sorted(results_dir.glob("*_latency.json"))
    if not json_files:
        logger.error(f"No *_latency.json files found in {results_dir}")
        sys.exit(1)

    logger.info(f"Found {len(json_files)} instance latency files in {results_dir}")

    results = []
    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)

        # Reconstruct InstanceProfilingResult from JSON
        result = InstanceProfilingResult(
            instance_id=data["instance_id"],
            model_name=data["model_name"],
            device_class=data["device_class"],
        )
        results.append(result)

        logger.info(
            f"  {data['instance_id']:<25s} "
            f"s={data.get('simple', 0):.0f}ms "
            f"m={data.get('moderate', 0):.0f}ms "
            f"c={data.get('complex', 0):.0f}ms"
        )

    # However, `latency_table_row()` reads from self.measurements, which is empty here.
    # So read directly from the JSON.
    rows = []
    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)
        rows.append({
            "instance_id": data["instance_id"],
            "model_name": data["model_name"],
            "device_class": data["device_class"],
            "simple": data.get("simple", 0),
            "moderate": data.get("moderate", 0),
            "complex": data.get("complex", 0),
        })

    import pandas as pd
    df = pd.DataFrame(rows)
    df = df[["instance_id", "model_name", "device_class", "simple", "moderate", "complex"]]
    df = df.sort_values(["device_class", "model_name"])

    # Save
    output_path = args.output
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info(f"\nFinal latency table ({len(df)} instances):")
    logger.info(f"\n{df.to_string(index=False)}")
    logger.info(f"\nSaved to {output_path}")

    # Also generate a Python module fragment for latency_table.py
    py_path = str(Path(output_path).with_suffix(".py"))
    with open(py_path, "w") as f:
        f.write("# Auto-generated from real device profiling\n")
        f.write("# Run: python scripts/profile_instances.py --merge\n")
        f.write("PROFILED_LATENCY_TABLE = {\n")
        for _, row in df.iterrows():
            f.write(
                f"    \"{row['instance_id']}\": {{"
                f"\"simple\": {row['simple']:.0f}, "
                f"\"moderate\": {row['moderate']:.0f}, "
                f"\"complex\": {row['complex']:.0f}"
                f"}},\n"
            )
        f.write("}\n")
    logger.info(f"Python snippet → {py_path}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════


def main():
    args = parse_args()

    if args.merge:
        run_merge(args)
    else:
        run_profiling(args)


if __name__ == "__main__":
    main()
