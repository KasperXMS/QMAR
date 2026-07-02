#!/usr/bin/env python
"""Pre-extract CLIP vision + text embeddings from VL-RouterBench.

Memory-efficient: writes embeddings incrementally via memmap, no accumulation.

Usage:
    python extract_embeddings.py --device cuda
    python extract_embeddings.py --device cpu --max-samples 500
"""

import argparse
import logging
import sys
import pickle
import gc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

from data_adapters.vl_routerbench_adapter import VLRouterBenchAdapter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("extract")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str,
                   default="/home/super/xiaoming/VL-RouterBench/vlm_router_data")
    p.add_argument("--vision-encoder", type=str, default="openai/clip-vit-base-patch32")
    p.add_argument("--text-encoder", type=str,
                   default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--models", type=str, default="all")
    p.add_argument("--datasets", type=str, default="all")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output-dir", type=str, default="outputs/embeddings")
    return p.parse_args()


def main():
    args = parse_args()

    from data_adapters.vl_routerbench_adapter import MODEL_DIR_MAP, DATASET_TSV_MAP
    all_models = list(MODEL_DIR_MAP.values())
    all_datasets = list(DATASET_TSV_MAP.keys())

    if args.models == "all":
        model_names = all_models
    elif args.models.isdigit():
        model_names = all_models[:int(args.models)]
    else:
        model_names = [m.strip() for m in args.models.split(",")]

    if args.datasets == "all":
        datasets = all_datasets
    elif args.datasets.isdigit():
        datasets = all_datasets[:int(args.datasets)]
    else:
        datasets = [d.strip() for d in args.datasets.split(",")]

    logger.info(f"Models: {len(model_names)}, Datasets: {len(datasets)}")

    # ---- Load data (adapter keeps TSV DataFrames, but not images) ----
    adapter = VLRouterBenchAdapter(
        data_root=args.data_root,
        model_names=model_names,
        datasets=datasets,
    )
    records = adapter.load_all_records(max_per_dataset=args.max_samples)
    records = [r for r in records if r.candidate_model_correctness]
    N = len(records)
    logger.info(f"Loaded {N} records with correctness labels")

    # ---- Build answer type labels (task-complexity-based) ----
    from data_adapters.common_schema import AnswerType
    answer_types = np.zeros(N, dtype=np.int32)
    for i, rec in enumerate(records):
        at = AnswerType.classify(rec)
        answer_types[i] = AnswerType.to_index(at)
    logger.info(f"Answer types: simple={(answer_types==0).sum()}, "
                f"moderate={(answer_types==1).sum()}, complex={(answer_types==2).sum()}")

    # ---- Build quality matrix (lightweight) ----
    from profiling.quality_profile import get_model_pool
    model_names_final = get_model_pool(records)
    K = len(model_names_final)
    model_to_idx = {m: i for i, m in enumerate(model_names_final)}
    Y = np.zeros((N, K), dtype=np.float32)
    for i, rec in enumerate(records):
        for m, c in rec.candidate_model_correctness.items():
            if m in model_to_idx:
                Y[i, model_to_idx[m]] = float(c >= 0.5)
    logger.info(f"Quality matrix: {Y.shape}, pos_rate={Y.mean():.3f}")

    # ---- Load encoders ----
    logger.info("Loading vision encoder...")
    from predictor.encoders import FrozenVisionEncoder, LightweightTextEncoder
    vision_encoder = FrozenVisionEncoder(
        model_name=args.vision_encoder, output_dim=512, device=args.device,
    ).to(args.device)
    vision_encoder.eval()

    logger.info("Loading text encoder...")
    text_encoder = LightweightTextEncoder(
        model_name=args.text_encoder, output_dim=384, device=args.device,
    ).to(args.device)
    text_encoder.eval()

    # ---- Create memmap files for incremental writing ----
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    V_dim = vision_encoder.dim
    T_dim = text_encoder.dim

    V_path = output_dir / "vision_embeddings.dat"
    T_path = output_dir / "text_embeddings.dat"

    V_mem = np.memmap(str(V_path), dtype=np.float32, mode="w+", shape=(N, V_dim))
    T_mem = np.memmap(str(T_path), dtype=np.float32, mode="w+", shape=(N, T_dim))
    request_ids = []

    # ---- Extract in batches, write incrementally ----
    logger.info(f"Extracting {N} records, batch_size={args.batch_size}...")
    idx = 0
    for i in tqdm(range(0, N, args.batch_size), desc="Extracting"):
        batch = records[i : i + args.batch_size]
        bsize = len(batch)

        # Load images
        images = []
        texts = []
        for rec in batch:
            try:
                img = adapter.get_image(rec.dataset_name, rec.extra.get("tsv_index", "0"))
                images.append(img)
            except Exception:
                images.append(Image.new("RGB", (224, 224)))
            texts.append(rec.question_text)
            request_ids.append(rec.request_id)

        # Encode
        with torch.no_grad():
            v = vision_encoder(images).cpu().numpy()
            t = text_encoder(texts).cpu().numpy()

        # Write to memmap
        V_mem[idx : idx + bsize] = v
        T_mem[idx : idx + bsize] = t
        idx += bsize

        # Aggressive cleanup
        del images, v, t
        if i % (args.batch_size * 10) == 0:
            gc.collect()
            if args.device == "cuda":
                torch.cuda.empty_cache()

    # Flush memmap
    V_mem.flush()
    T_mem.flush()

    logger.info(f"Vision embeddings: ({N}, {V_dim})")
    logger.info(f"Text embeddings:   ({N}, {T_dim})")

    # ---- Save metadata and quality matrix ----
    np.savez_compressed(output_dir / "quality_matrix.npz",
                        Y=Y, model_names=np.array(model_names_final))
    np.savez_compressed(output_dir / "answer_types.npz", answer_types=answer_types)

    metadata = {
        "request_ids": np.array(request_ids),
        "model_names": model_names_final,
        "answer_types": answer_types,
        "num_records": N,
        "vision_dim": V_dim,
        "text_dim": T_dim,
    }
    with open(output_dir / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    # Also save as npz for easy loading
    np.savez_compressed(output_dir / "vision_embeddings.npz",
                        embeddings=np.array(V_mem))
    np.savez_compressed(output_dir / "text_embeddings.npz",
                        embeddings=np.array(T_mem))

    logger.info(f"Saved to {output_dir}/")
    logger.info(f"  vision_embeddings.npz + .dat")
    logger.info(f"  text_embeddings.npz + .dat")
    logger.info(f"  quality_matrix.npz  ({K} models)")
    logger.info(f"  metadata.pkl")


if __name__ == "__main__":
    main()
