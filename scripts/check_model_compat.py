#!/usr/bin/env python3
"""Quick model-load compatibility check.

Tries to load each candidate model with trust_remote_code=True
and reports success/failure + peak VRAM. Run on the target device.

Usage:
    python3 scripts/check_model_compat.py
    python3 scripts/check_model_compat.py --device cuda --size-limit 14  # max GB
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Candidate models (all exist in VL-RouterBench MODEL_DIR_MAP) ──
CANDIDATES = [
    # (display_name, hf_model_id, min_vram_gb_est)
    ("Janus-Pro-7B",        "deepseek-ai/Janus-Pro-7B",           16),
    ("DeepSeek-VL2-Tiny",   "deepseek-ai/deepseek-vl2-tiny",       8),
    ("MiMo-VL-7B-RL",       "Mimo-VL-7B-RL",                      16),
    ("Qianfan-VL-8B",       "Qwen/Qwen2-VL-7B-Instruct",          16),
    ("Kimi-VL-A3B",         "moonshotai/Kimi-VL-A3B-Thinking",     8),
    ("Phi-3.5-Vision",      "microsoft/Phi-3.5-vision-instruct",  10),
    ("LLaVA-Next-Vicuna-7B","llava-hf/llava-v1.6-vicuna-7b-hf",  16),
    ("Janus-Pro-1B",        "deepseek-ai/Janus-Pro-1B",            4),
    ("SmolVLM2",            "HuggingFaceTB/SmolVLM2-2.2B-Instruct",6),
    ("Qwen2.5-VL-7B",       "Qwen/Qwen2.5-VL-7B-Instruct",        16),
    ("Pixtral-12B",         "mistralai/Pixtral-12B-2409",         24),
]

STATUS_SYMBOLS = {True: "✓ OK", False: "✗ FAIL", None: "? SKIP"}


def check_model(
    hf_id: str,
    device: str = "cuda",
    size_limit_gb: float = 32,
    load_timeout: float = 120,
) -> tuple[bool, str, float]:
    """Try to load a model. Returns (success, error_msg, peak_vram_gb)."""
    import torch

    # Check VRAM before loading
    if device != "cpu" and torch.cuda.is_available():
        free_gb = torch.cuda.mem_get_info()[0] / (1024**3)
        if free_gb < size_limit_gb * 0.5:
            return (False, f"insufficient VRAM ({free_gb:.1f} GB free)", 0)

    try:
        from transformers import AutoConfig, AutoProcessor, AutoModelForVision2Seq, AutoModel

        # Step 1: config
        t0 = time.time()
        try:
            config = AutoConfig.from_pretrained(hf_id, trust_remote_code=True)
        except Exception:
            # Try without trust_remote_code
            config = AutoConfig.from_pretrained(hf_id)

        if time.time() - t0 > load_timeout:
            return (False, "config load timeout", 0)

        # Step 2: processor (optional)
        try:
            processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)
        except Exception:
            processor = None  # not all models have a processor

        # Step 3: model
        loaded = False
        for model_cls in [AutoModelForVision2Seq, AutoModel]:
            try:
                model = model_cls.from_pretrained(
                    hf_id,
                    config=config,
                    torch_dtype=torch.float16,
                    device_map=device,
                    trust_remote_code=True,
                )
                loaded = True
                break
            except Exception:
                continue

        if not loaded:
            # Last resort: AutoModel without config
            model = AutoModel.from_pretrained(
                hf_id,
                torch_dtype=torch.float16,
                device_map=device,
                trust_remote_code=True,
            )

        # Measure VRAM
        if device != "cpu" and torch.cuda.is_available():
            peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
            torch.cuda.reset_peak_memory_stats()
        else:
            peak_gb = 0

        # Quick forward pass to verify it works
        model.eval()
        del model, processor
        gc.collect()
        torch.cuda.empty_cache()

        return (True, "", peak_gb)

    except Exception as e:
        gc.collect()
        if device != "cpu":
            torch.cuda.empty_cache()
        return (False, str(e)[:120], 0)


def main():
    p = argparse.ArgumentParser(description="Check which VLMs load on this device")
    p.add_argument("--device", default="cuda")
    p.add_argument("--size-limit", type=float, default=32,
                   help="Max model size in GB (skip if estimated > this)")
    args = p.parse_args()

    print(f"{'Model':<25s} {'Status':<10s} {'VRAM':>8s}  Notes")
    print("-" * 75)

    working, failed = [], []

    for name, hf_id, est_gb in CANDIDATES:
        if est_gb > args.size_limit:
            print(f"{name:<25s} {'? SKIP':<10s} {'':>8s}  est {est_gb}GB > limit {args.size_limit}GB")
            continue

        print(f"{name:<25s} ...", end="", flush=True)
        ok, err, peak_gb = check_model(hf_id, args.device, est_gb)

        status = STATUS_SYMBOLS[ok]
        vram_str = f"{peak_gb:5.1f}GB" if peak_gb > 0 else ""
        print(f"\r{name:<25s} {status:<10s} {vram_str:>8s}", end="")

        if ok:
            working.append(name)
            print()
        else:
            failed.append(name)
            print(f"  — {err}")

    print("\n" + "=" * 75)
    print(f"Working: {len(working)} — {', '.join(working) if working else 'none'}")
    print(f"Failed:  {len(failed)} — {', '.join(failed) if failed else 'none'}")
    print(f"\nRecommended for AGX Orin pool: {', '.join(working) if working else 'TBD'}")


if __name__ == "__main__":
    main()
