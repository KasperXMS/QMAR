# QMAR — Quality-Margin-Aware Routing for Edge VLM Serving

## Environment Setup

### Prerequisites

- **CUDA 11.8+** (recommended: 12.1 for RTX 4090 / AGX Orin)
- **Python 3.10+**

### Option A: Conda (recommended)

```bash
# Create environment with Python 3.10
conda create -n qmar python=3.10 -y
conda activate qmar

# Install PyTorch with CUDA 12.1 support
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install the rest
pip install -r requirements.txt
```

**Jetson devices (Orin NX / AGX Orin):** Conda is not officially supported on aarch64. Use system Python + pip instead:

```bash
# Jetson comes with PyTorch pre-installed via JetPack.
# Verify:
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

# Then install remaining dependencies:
pip install -r requirements.txt
```

### Option B: venv (any platform)

```bash
python3 -m venv .venv
source .venv/bin/activate

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Or CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
```

### Verify

```bash
python3 -c "
import torch
import transformers
print(f'Python:  3.10+')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA:    {torch.cuda.is_available()} ({torch.version.cuda})')
print(f'GPU:     {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

Expected output on each device:

| Device | GPU |
|---|---|
| RTX 4090 | `NVIDIA GeForce RTX 4090` |
| AGX Orin | `Orin (integrated)` |
| Orin NX | `Orin (integrated)` |

---

## Project Structure

```
QMAR/
├── configs/
│   ├── instance_pool.yaml          # 10 edge instances across 3 device classes
│   └── qmar.yaml                   # Pipeline config (predictor, training, scheduler)
├── data_adapters/                  # VL-RouterBench & MMR-Bench → unified schema
├── predictor/                      # Multi-task predictor (suitability + complexity)
├── scheduler/                      # QMAR scheduler + 7 baselines
├── profiling/                      # Latency table, communication profiles, profiler
├── evaluation/                     # Metrics computation & plots
├── scripts/
│   ├── sample_requests.py          # Step 1: sample benchmark requests for profiling
│   ├── profile_instances.py        # Step 2: profile VLM instances on real hardware
│   ├── extract_embeddings.py       # Step 3a: extract CLIP + text embeddings
│   ├── train_predictor_fast.py     # Step 3b: train predictor on cached embeddings
│   └── evaluate_qmar.py            # Step 4: run QMAR + baselines, report metrics
├── outputs/                        # Generated artifacts (embeddings, checkpoints, eval)
└── requirements.txt
```

---

## Full Pipeline

### Step 1 — Sample profiling requests

```bash
python3 scripts/sample_requests.py \
  --source vl_routerbench \
  --data-root ../VL-RouterBench/vlm_router_data \
  --n-simple 15 --n-moderate 15 --n-complex 10 \
  --output-dir outputs/profiling
```

Produces `outputs/profiling/profiling_requests.json` and extracted images.

### Step 2 — Profile instances on real hardware

Copy `outputs/profiling/` to each edge device, then run:

```bash
# On each device — auto-detects device class:
python3 scripts/profile_instances.py \
  --device cuda --iterations 3 --warmup 5

# Or specify device class explicitly:
python3 scripts/profile_instances.py \
  --device-class Orin_NX --device cuda --cooldown 3.0

# RTX 4090 — one instance at a time to manage GPU memory:
python3 scripts/profile_instances.py \
  --instances Qwen-7B@4090 --device cuda
```

Collect all `*_latency.json` back to the main machine:

```bash
python3 scripts/profile_instances.py --merge \
  --results-dir outputs/profiling \
  --output outputs/profiling/latency_table.csv
```

### Step 3 — Train the predictor

```bash
# 3a. Extract embeddings (one-time, ~5-10 min on GPU)
python3 extract_embeddings.py --device cuda

# 3b. Train on cached embeddings (~30s on GPU)
python3 train_predictor_fast.py --device cuda --epochs 50
```

### Step 4 — Evaluate QMAR

```bash
python3 evaluate_qmar.py \
  --checkpoint outputs/fast_run/predictor_best.pt \
  --threshold 0.5 \
  --output-dir outputs/eval
```

Produces:
- `outputs/eval/summary_metrics.csv` — QMAR vs baselines
- `outputs/eval/threshold_sweep.png` — τ sensitivity
- `outputs/eval/method_comparison.png` — latency + fallback comparison

---

## Instances

| Instance | Model | Device | RAM |
|---|---|---|---|
| SmolVLM2@NX | SmolVLM2 (~2B) | Orin NX 16GB | 16 GB |
| Janus-1B@NX | Janus-Pro-1B | Orin NX 16GB | 16 GB |
| Phi-3.5@NX | Phi-3.5-Vision (~4B) | Orin NX 16GB | 16 GB |
| SmolVLM2@AGX | SmolVLM2 (~2B) | AGX Orin 32GB | 32 GB |
| Phi-3.5@AGX | Phi-3.5-Vision (~4B) | AGX Orin 32GB | 32 GB |
| Qwen-7B@AGX | Qwen2.5-VL-7B | AGX Orin 32GB | 32 GB |
| LLaVA-7B@AGX | LLaVA-Next-Vicuna-7B | AGX Orin 32GB | 32 GB |
| Qwen-7B@4090 | Qwen2.5-VL-7B | RTX 4090 24GB | 24 GB |
| Pixtral-12B@4090 | Pixtral-12B | RTX 4090 24GB | 24 GB |
| Qwen-32B@4090 | Qwen2.5-VL-32B | RTX 4090 24GB | 24 GB |

---

## Key Configuration

Edit `configs/qmar.yaml` to adjust:

| Parameter | Default | Description |
|---|---|---|
| `scheduler.quality_threshold` | 0.5 | Global quality threshold τ |
| `predictor.vision_encoder` | `openai/clip-vit-base-patch32` | Frozen vision backbone |
| `predictor.text_encoder` | `sentence-transformers/all-MiniLM-L6-v2` | Frozen text backbone |
| `training.num_epochs` | 50 | Max training epochs |
| `training.patience` | 15 | Early stopping patience |
| `experiment.threshold_sweep` | [0.3–0.9] | τ values for sensitivity analysis |
