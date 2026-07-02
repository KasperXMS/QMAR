#!/usr/bin/env python
"""Sample representative requests from benchmark datasets for device profiling.

Selects balanced requests across complexity classes (simple/moderate/complex),
extracts images from TSV/base64, and saves everything as a self-contained
JSON + image directory that the profiler can consume.

Usage:
    # Sample 40 requests (default: 15 simple, 15 moderate, 10 complex)
    python scripts/sample_requests.py --n-simple 15 --n-moderate 15 --n-complex 10

    # Sample from VL-RouterBench only
    python scripts/sample_requests.py --source vl_routerbench --output-dir outputs/profiling

    # With custom dataset priority
    python scripts/sample_requests.py --datasets MMBench_DEV_EN_V11,MMStar,MathVista_MINI
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_adapters.common_schema import RequestRecord, AnswerType
from data_adapters.vl_routerbench_adapter import (
    VLRouterBenchAdapter,
    TSVImageLoader,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("sample")


# Dataset → complexity mapping (mirrors AnswerType.DATASET_COMPLEXITY)
COMPLEXITY_DATASETS = {
    "simple": [
        "MMBench_DEV_EN_V11",
        "MMStar",
        "RealWorldQA",
    ],
    "moderate": [
        "TextVQA_VAL",
        "OCRBench",
        "DocVQA_VAL",
        "ChartQA_TEST",
        "InfoVQA_VAL",
        "AI2D_TEST",
    ],
    "complex": [
        "MathVista_MINI",
        "MathVision_MINI",
        "MathVerse_MINI",
        "HallusionBench",
        "MMMU_DEV_VAL",
    ],
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Sample requests for device profiling"
    )
    p.add_argument(
        "--source", type=str, default="vl_routerbench",
        choices=["vl_routerbench", "mmr_bench"],
        help="Benchmark source",
    )
    p.add_argument(
        "--data-root", type=str,
        default="../VL-RouterBench/vlm_router_data",
        help="Path to VL-RouterBench data root",
    )
    p.add_argument(
        "--datasets", type=str, default=None,
        help="Comma-separated dataset whitelist (default: all available)",
    )
    p.add_argument(
        "--n-simple", type=int, default=15,
        help="Number of simple requests to sample",
    )
    p.add_argument(
        "--n-moderate", type=int, default=15,
        help="Number of moderate requests to sample",
    )
    p.add_argument(
        "--n-complex", type=int, default=10,
        help="Number of complex requests to sample",
    )
    p.add_argument(
        "--output-dir", type=str, default="outputs/profiling",
        help="Output directory for JSON + images",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Only print statistics, don't extract images",
    )
    return p.parse_args()


def load_records_from_vlb(
    data_root: str,
    datasets: Optional[List[str]] = None,
) -> List[RequestRecord]:
    """Load RequestRecords from VL-RouterBench.

    Uses a focused set of datasets suitable for edge-VLM evaluation.
    """
    from data_adapters.vl_routerbench_adapter import MODEL_DIR_MAP

    # Only load models that are in our edge instance pool
    edge_models = [
        "SmolVLM2",
        "Janus-Pro-1B",
        "Phi-3.5-Vision",
        "Qwen2.5-VL-32B",  # also covers 7B (same family, correctness similar)
        "LLaVA-Next-Vicuna-7B",
        "Pixtral-12B",
    ]

    if datasets is None:
        # All datasets that have complexity labels
        datasets = list(AnswerType.DATASET_COMPLEXITY.keys())

    logger.info(f"Loading VL-RouterBench records from {data_root}")
    logger.info(f"  Models: {edge_models}")
    logger.info(f"  Datasets: {datasets}")

    adapter = VLRouterBenchAdapter(
        data_root=data_root,
        model_names=edge_models,
        datasets=datasets,
    )

    records = adapter.load_all_records()
    logger.info(f"  Loaded {len(records)} records")
    return records


def classify_and_group(
    records: List[RequestRecord],
) -> dict:
    """Classify records by complexity. Returns {complexity: [records]}."""
    groups = {"simple": [], "moderate": [], "complex": []}

    for rec in records:
        c = AnswerType.classify(rec)
        if c in groups:
            groups[c].append(rec)

    for c, recs in groups.items():
        datasets_in_group = sorted(set(r.dataset_name for r in recs))
        logger.info(f"  {c}: {len(recs)} records from {datasets_in_group}")

    return groups


def extract_image(
    rec: RequestRecord,
    tsv_loader: TSVImageLoader,
    output_dir: Path,
) -> Optional[str]:
    """Extract a single image from TSV and save as PNG.

    Returns the saved image path, or None on failure.
    """
    try:
        # Find the row index from the request_id
        # RequestRecord stores index or we need to look it up
        # Use the record's extra metadata or image_path
        if rec.image_path and Path(rec.image_path).exists():
            # Already a file path — just copy/link
            return rec.image_path

        # Extract from TSV
        # The request_id format is typically {dataset}_{index}
        # or stored in rec.image_id / extra
        img_index = rec.extra.get("index", None) or rec.image_id
        if img_index is None:
            # Try parsing from request_id
            parts = rec.request_id.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                img_index = int(parts[1])
            else:
                logger.warning(f"Cannot determine image index for {rec.request_id}")
                return None

        img = tsv_loader.load_image(rec.dataset_name, img_index)

        # Save
        safe_name = rec.request_id.replace("/", "_").replace("\\", "_")
        img_path = output_dir / f"{safe_name}.png"
        img.save(img_path, "PNG")
        return str(img_path)

    except Exception as e:
        logger.warning(f"Failed to extract image for {rec.request_id}: {e}")
        return None


def main():
    args = parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # Parse dataset whitelist
    dataset_list = None
    if args.datasets:
        dataset_list = [d.strip() for d in args.datasets.split(",")]

    # Load records
    if args.source == "vl_routerbench":
        records = load_records_from_vlb(args.data_root, dataset_list)
    else:
        logger.error("MMR-Bench sampling not yet implemented. Use --source vl_routerbench")
        sys.exit(1)

    # Classify and group
    groups = classify_and_group(records)

    # Check we have enough
    target = {
        "simple": args.n_simple,
        "moderate": args.n_moderate,
        "complex": args.n_complex,
    }
    for c in ["simple", "moderate", "complex"]:
        if len(groups[c]) < target[c]:
            logger.warning(
                f"Only {len(groups[c])} {c} records available "
                f"(need {target[c]}). Will use all available."
            )
            target[c] = min(target[c], len(groups[c]))

    # Sample
    import random
    rng = random.Random(args.seed)

    sampled = {}
    for c in ["simple", "moderate", "complex"]:
        sampled[c] = rng.sample(groups[c], target[c])
        ds_counts = {}
        for r in sampled[c]:
            ds_counts[r.dataset_name] = ds_counts.get(r.dataset_name, 0) + 1
        logger.info(f"  Sampled {len(sampled[c])} {c}: {ds_counts}")

    if args.dry_run:
        logger.info("Dry run complete. Use without --dry-run to extract images.")
        return

    # Extract images
    tsv_dir = Path(args.data_root) / "TSV_images"
    tsv_loader = TSVImageLoader(str(tsv_dir))

    profiling_requests = []
    total = sum(len(v) for v in sampled.values())

    for c, recs in sampled.items():
        for rec in recs:
            img_path = extract_image(rec, tsv_loader, img_dir)

            profiling_requests.append({
                "request_id": rec.request_id,
                "image_path": img_path or "",
                "question": rec.question_text,
                "complexity_class": c,
                "dataset": rec.dataset_name,
                "task_name": rec.task_name,
            })

    # Filter out failed extractions
    valid = [r for r in profiling_requests if r["image_path"]]
    failed = len(profiling_requests) - len(valid)
    if failed > 0:
        logger.warning(f"Failed to extract images for {failed}/{total} requests")

    # Save JSON
    output_path = out_dir / "profiling_requests.json"
    with open(output_path, "w") as f:
        json.dump(valid, f, indent=2, ensure_ascii=False)

    # Print per-complexity breakdown
    complexity_counts = {"simple": 0, "moderate": 0, "complex": 0}
    for r in valid:
        complexity_counts[r["complexity_class"]] += 1

    logger.info(
        f"\n✓ Saved {len(valid)} profiling requests to {output_path}"
        f"\n  Images: {img_dir}/"
        f"\n  Breakdown: simple={complexity_counts['simple']}, "
        f"moderate={complexity_counts['moderate']}, "
        f"complex={complexity_counts['complex']}"
    )


if __name__ == "__main__":
    main()
