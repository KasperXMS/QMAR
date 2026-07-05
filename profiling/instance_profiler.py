"""Real-device VLM instance profiler.

Measures per-request inference latency on actual edge hardware.
Outputs raw per-request measurements and aggregated latency tables
that can directly replace the synthetic profiles in latency_table.py.

Architecture:
  ModelAdapter (per model family)
    → InstanceProfiler (orchestrates warmup + measurement)
      → CLI (profile_instances.py)

Usage (per device):
  python scripts/profile_instances.py \
    --instance "Qwen-7B@4090" \
    --requests outputs/profiling_requests.json \
    --output-dir outputs/profiling \
    --iterations 3 --warmup 5

Supported model families:
  - Qwen2.5-VL (7B, 32B)  — transformers Qwen2_5VL
  - LLaVA-Next-Vicuna-7B  — transformers Llava
  - SmolVLM2              — transformers Idefics3
  - Phi-3.5-Vision        — transformers Phi3V
  - Pixtral-12B           — transformers Pixtral
  - Janus-Pro-1B          — custom (deepseek-ai/Janus)
"""

from __future__ import annotations

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("profiler")


# ═══════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════


@dataclass
class ProfilingRequest:
    """A single request to profile."""
    request_id: str
    image_path: str
    question: str
    complexity_class: str  # "simple" | "moderate" | "complex"
    dataset: str = ""
    task_name: str = ""


@dataclass
class SingleMeasurement:
    """One measurement for one request on one instance."""
    request_id: str
    instance_id: str
    complexity_class: str
    latency_ms: float
    num_tokens: int
    iteration: int


@dataclass
class InstanceProfilingResult:
    """Aggregated profiling result for one instance."""
    instance_id: str
    model_name: str
    device_class: str
    measurements: List[SingleMeasurement] = field(default_factory=list)

    def latency_table_row(self) -> Dict[str, float]:
        """Aggregate measurements into {simple, moderate, complex} medians."""
        by_complexity = {"simple": [], "moderate": [], "complex": []}
        for m in self.measurements:
            if m.complexity_class in by_complexity:
                by_complexity[m.complexity_class].append(m.latency_ms)

        return {
            c: float(np.median(vals)) if vals else 0.0
            for c, vals in by_complexity.items()
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Raw per-measurement DataFrame."""
        return pd.DataFrame([{
            "request_id": m.request_id,
            "instance_id": m.instance_id,
            "complexity_class": m.complexity_class,
            "latency_ms": m.latency_ms,
            "num_tokens": m.num_tokens,
            "iteration": m.iteration,
        } for m in self.measurements])


# ═══════════════════════════════════════════════════════════════════
# Model Adapters — one per VLM family
# ═══════════════════════════════════════════════════════════════════


class ModelAdapter(ABC):
    """Abstract adapter: knows how to load and run one VLM family."""

    @abstractmethod
    def load(self, device: str, model_id: Optional[str] = None) -> None:
        """Load model + processor onto the given device."""
        ...

    @abstractmethod
    def generate(
        self,
        image_path: str,
        question: str,
        max_new_tokens: int = 256,
    ) -> Tuple[str, float, int]:
        """Run inference. Returns (output_text, elapsed_ms, num_tokens)."""
        ...

    @abstractmethod
    def unload(self) -> None:
        """Free GPU memory."""
        ...

    def _load_model(self, device: str, mid: str) -> None:
        """Load a VLM via AutoModel — works across transformers 4.x and 5.x.

        AutoModelForVision2Seq was removed in transformers 5.x; AutoModel is
        the stable API that resolves to the correct architecture in all versions.
        """
        import torch
        from transformers import AutoModel

        self._device = device
        self.model = AutoModel.from_pretrained(
            mid,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            device_map=device if device != "cpu" else None,
            trust_remote_code=True,
        )
        if device == "cpu":
            self.model = self.model.to(device)
        self.model.eval()


class QwenVLAdapter(ModelAdapter):
    """Qwen2.5-VL family (7B, 32B).

    Uses transformers Qwen2_5VLForConditionalGeneration.
    HF IDs: Qwen/Qwen2.5-VL-7B-Instruct, Qwen/Qwen2.5-VL-32B-Instruct
    """

    DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

    def __init__(self):
        self.model = None
        self.processor = None
        self._device = None

    def load(self, device: str, model_id: Optional[str] = None) -> None:
        import torch
        from transformers import AutoProcessor

        mid = model_id or self.DEFAULT_MODEL_ID
        logger.info(f"Loading Qwen2.5-VL: {mid}")

        self._device = device
        self.processor = AutoProcessor.from_pretrained(mid, trust_remote_code=True)
        self._load_model(device, mid)
        logger.info(f"  Loaded. Device: {next(self.model.parameters()).device}")

    def generate(
        self,
        image_path: str,
        question: str,
        max_new_tokens: int = 256,
    ) -> Tuple[str, float]:
        import torch
        from PIL import Image
        from qwen_vl_utils import process_vision_info

        image = Image.open(image_path).convert("RGB")

        # Build Qwen2.5-VL message format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
        ).to(self._device)

        # Time the generation only (not preprocessing)
        if self._device != "cpu":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        if self._device != "cpu":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Trim input tokens from output
        generated_ids = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        num_tokens = len(generated_ids[0])

        return output_text, elapsed_ms, num_tokens

    def unload(self) -> None:
        import torch
        if self.model is not None:
            del self.model
        if self.processor is not None:
            del self.processor
        torch.cuda.empty_cache()
        self.model = None
        self.processor = None


class LlavaAdapter(ModelAdapter):
    """LLaVA-Next-Vicuna-7B.

    HF ID: llava-hf/llava-v1.6-vicuna-7b-hf
    """

    DEFAULT_MODEL_ID = "llava-hf/llava-v1.6-vicuna-7b-hf"

    def __init__(self):
        self.model = None
        self.processor = None
        self._device = None

    def load(self, device: str, model_id: Optional[str] = None) -> None:
        import os
        import torch
        from transformers import LlavaForConditionalGeneration, AutoProcessor

        # Fix protobuf/sentencepiece conflict on older environments
        os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

        mid = model_id or self.DEFAULT_MODEL_ID
        logger.info(f"Loading LLaVA: {mid}")

        self._device = device
        self.processor = AutoProcessor.from_pretrained(mid)
        self._load_model(device, mid)
        logger.info(f"  Loaded. Device: {next(self.model.parameters()).device}")

    def generate(
        self,
        image_path: str,
        question: str,
        max_new_tokens: int = 256,
    ) -> Tuple[str, float]:
        import torch
        from PIL import Image

        image = Image.open(image_path).convert("RGB")

        # LLaVA chat format
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image"},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )

        inputs = self.processor(
            images=image, text=prompt, return_tensors="pt"
        ).to(self._device)

        if self._device != "cpu":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        if self._device != "cpu":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        generated_ids = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        num_tokens = len(generated_ids[0])

        return output_text, elapsed_ms, num_tokens

    def unload(self) -> None:
        import torch
        if self.model is not None:
            del self.model
        if self.processor is not None:
            del self.processor
        torch.cuda.empty_cache()
        self.model = None
        self.processor = None


class SmolVLMAdapter(ModelAdapter):
    """SmolVLM2 (Idefics3-based, ~2.2B).

    HF ID: HuggingFaceTB/SmolVLM2-2.2B-Instruct
    """

    DEFAULT_MODEL_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"

    def __init__(self):
        self.model = None
        self.processor = None
        self._device = None

    def load(self, device: str, model_id: Optional[str] = None) -> None:
        mid = model_id or self.DEFAULT_MODEL_ID
        logger.info(f"Loading SmolVLM2: {mid}")

        self._device = device
        try:
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(mid, trust_remote_code=True)
        except Exception:
            self.processor = None
        self._load_model(device, mid)
        logger.info(f"  Loaded. Device: {next(self.model.parameters()).device}")

    def generate(
        self,
        image_path: str,
        question: str,
        max_new_tokens: int = 256,
    ) -> Tuple[str, float]:
        import torch
        from PIL import Image

        image = Image.open(image_path).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )

        inputs = self.processor(
            text=prompt, images=[image], return_tensors="pt"
        ).to(self._device)

        if self._device != "cpu":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        if self._device != "cpu":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        generated_ids = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        num_tokens = len(generated_ids[0])

        return output_text, elapsed_ms, num_tokens

    def unload(self) -> None:
        import torch
        if self.model is not None:
            del self.model
        if self.processor is not None:
            del self.processor
        torch.cuda.empty_cache()
        self.model = None
        self.processor = None


class PhiVisionAdapter(ModelAdapter):
    """Phi-3.5-Vision (~4B).

    HF ID: microsoft/Phi-3.5-vision-instruct
    """

    DEFAULT_MODEL_ID = "microsoft/Phi-3.5-vision-instruct"

    def __init__(self):
        self.model = None
        self.processor = None
        self._device = None

    def load(self, device: str, model_id: Optional[str] = None) -> None:
        import torch
        from transformers import AutoProcessor, AutoModelForCausalLM

        mid = model_id or self.DEFAULT_MODEL_ID
        logger.info(f"Loading Phi-3.5-Vision: {mid}")

        self._device = device
        self.processor = AutoProcessor.from_pretrained(mid, trust_remote_code=True)
        self._load_model(device, mid)
        logger.info(f"  Loaded. Device: {next(self.model.parameters()).device}")

    def generate(
        self,
        image_path: str,
        question: str,
        max_new_tokens: int = 256,
    ) -> Tuple[str, float]:
        import torch
        from PIL import Image

        image = Image.open(image_path).convert("RGB")

        # Phi-3.5 uses a specific chat format with <|user|> / <|assistant|> tokens
        messages = [
            {"role": "user", "content": f"<|image_1|>\n{question}"},
        ]
        prompt = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(prompt, [image], return_tensors="pt").to(self._device)

        if self._device != "cpu":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        if self._device != "cpu":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        generated_ids = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        num_tokens = len(generated_ids[0])

        return output_text, elapsed_ms, num_tokens

    def unload(self) -> None:
        import torch
        if self.model is not None:
            del self.model
        if self.processor is not None:
            del self.processor
        torch.cuda.empty_cache()
        self.model = None
        self.processor = None


class PixtralAdapter(ModelAdapter):
    """Pixtral-12B (Mistral).

    HF ID: mistralai/Pixtral-12B-2409
    Uses mistral_infer or transformers Llava-like interface.
    """

    DEFAULT_MODEL_ID = "mistralai/Pixtral-12B-2409"

    def __init__(self):
        self.model = None
        self.processor = None
        self._device = None

    def load(self, device: str, model_id: Optional[str] = None) -> None:
        mid = model_id or self.DEFAULT_MODEL_ID
        logger.info(f"Loading Pixtral: {mid}")

        self._device = device

        # Pixtral may not have a unified AutoProcessor
        try:
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(mid, trust_remote_code=True)
        except (ValueError, ImportError):
            logger.info("  No unified processor — loading tokenizer + image processor separately")
            self.processor = None
            from transformers import AutoTokenizer, AutoImageProcessor
            self._tokenizer = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
            self._image_processor = AutoImageProcessor.from_pretrained(mid, trust_remote_code=True)

        self._load_model(device, mid)
        logger.info(f"  Loaded. Device: {next(self.model.parameters()).device}")

    def generate(
        self,
        image_path: str,
        question: str,
        max_new_tokens: int = 256,
    ) -> Tuple[str, float]:
        import torch
        from PIL import Image

        image = Image.open(image_path).convert("RGB")

        if self.processor is not None:
            # Unified processor path
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ]}
            ]
            prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=prompt, images=[image], return_tensors="pt"
            ).to(self._device)
        else:
            # Separate tokenizer + image processor path
            img_inputs = self._image_processor(images=[image], return_tensors="pt")
            img_inputs = {k: v.to(self._device) for k, v in img_inputs.items()}
            text_inputs = self._tokenizer(
                question, return_tensors="pt", padding=True
            )
            text_inputs = {k: v.to(self._device) for k, v in text_inputs.items()}
            inputs = {**img_inputs, **text_inputs}

        if self._device != "cpu":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        if self._device != "cpu":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        generated_ids = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.get("input_ids", inputs.get("pixel_values")), generated_ids)
        ]
        if self.processor is not None:
            output_text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]
        else:
            output_text = self._tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]
        num_tokens = len(generated_ids[0])

        return output_text, elapsed_ms, num_tokens

    def unload(self) -> None:
        import torch
        if self.model is not None:
            del self.model
        if self.processor is not None:
            del self.processor
        torch.cuda.empty_cache()
        self.model = None
        self.processor = None


class JanusAdapter(ModelAdapter):
    """Janus-Pro-1B (DeepSeek).

    HF ID: deepseek-ai/Janus-Pro-1B
    Uses custom modeling code — requires trust_remote_code=True.
    """

    DEFAULT_MODEL_ID = "deepseek-ai/Janus-Pro-1B"

    def __init__(self):
        self.model = None
        self.processor = None
        self._device = None

    def load(self, device: str, model_id: Optional[str] = None) -> None:
        import torch
        from transformers import AutoModelForCausalLM

        mid = model_id or self.DEFAULT_MODEL_ID
        logger.info(f"Loading Janus: {mid}")

        self._device = device
        self.processor = None  # Janus uses custom generate(), no HF processor needed
        self._load_model(device, mid)
        logger.info(f"  Loaded. Device: {next(self.model.parameters()).device}")

    def generate(
        self,
        image_path: str,
        question: str,
        max_new_tokens: int = 256,
    ) -> Tuple[str, float]:
        import torch
        from PIL import Image

        image = Image.open(image_path).convert("RGB")

        if self._device != "cpu":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        # Janus uses a multimodal generate method
        with torch.no_grad():
            output_text = self.model.generate(
                image=image,
                question=question,
                max_new_tokens=max_new_tokens,
            )

        if self._device != "cpu":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Rough token count from output text
        num_tokens = len(output_text.split()) if output_text else 0

        return output_text, elapsed_ms, num_tokens

    def unload(self) -> None:
        import torch
        if self.model is not None:
            del self.model
        torch.cuda.empty_cache()
        self.model = None


# ═══════════════════════════════════════════════════════════════════
# Model registry — maps model_name → (adapter_class, default_hf_id)
# ═══════════════════════════════════════════════════════════════════

MODEL_REGISTRY: Dict[str, Tuple[type, str]] = {
    "Qwen2.5-VL-7B":   (QwenVLAdapter,    "Qwen/Qwen2.5-VL-7B-Instruct"),
    "Qwen2.5-VL-32B":  (QwenVLAdapter,    "Qwen/Qwen2.5-VL-32B-Instruct"),
    "LLaVA-Next-Vicuna-7B": (LlavaAdapter, "llava-hf/llava-v1.6-vicuna-7b-hf"),
    "SmolVLM2":         (SmolVLMAdapter,   "HuggingFaceTB/SmolVLM2-2.2B-Instruct"),
    "Phi-3.5-Vision":   (PhiVisionAdapter, "microsoft/Phi-3.5-vision-instruct"),
    "Pixtral-12B":      (PixtralAdapter,   "mistralai/Pixtral-12B-2409"),
    "Janus-Pro-1B":     (JanusAdapter,     "deepseek-ai/Janus-Pro-1B"),
}


def get_adapter(model_name: str, custom_model_id: Optional[str] = None) -> ModelAdapter:
    """Factory: return a ModelAdapter for the given model_name."""
    if model_name in MODEL_REGISTRY:
        adapter_cls, default_hf_id = MODEL_REGISTRY[model_name]
        adapter = adapter_cls()
        # Use registry's HF ID unless overridden
        adapter.DEFAULT_MODEL_ID = custom_model_id or default_hf_id
        return adapter

    # Fallback: try as a generic transformers VLM
    logger.warning(
        f"Model '{model_name}' not in registry. "
        f"Trying generic AutoModel. Results may vary."
    )
    return _GenericVLAdapter(model_name)


class _GenericVLAdapter(ModelAdapter):
    """Fallback for models not in the registry."""

    def __init__(self, model_name: str):
        self._model_name = model_name
        self.model = None
        self.processor = None
        self._device = None
        self.DEFAULT_MODEL_ID = model_name

    def load(self, device: str, model_id: Optional[str] = None) -> None:
        from transformers import AutoProcessor

        mid = model_id or self._model_name
        self._device = device
        self.processor = AutoProcessor.from_pretrained(mid, trust_remote_code=True)
        self._load_model(device, mid)

    def generate(self, image_path, question, max_new_tokens=256):
        import torch
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": question}
        ]}]
        prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = self.processor(
            text=prompt, images=[image], return_tensors="pt"
        ).to(self._device)

        if self._device != "cpu":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )

        if self._device != "cpu":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        generated_ids = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        num_tokens = len(generated_ids[0])
        return output_text, elapsed_ms, num_tokens

    def unload(self) -> None:
        import torch
        if self.model is not None:
            del self.model
        if self.processor is not None:
            del self.processor
        torch.cuda.empty_cache()
        self.model = None
        self.processor = None


# ═══════════════════════════════════════════════════════════════════
# Instance Profiler
# ═══════════════════════════════════════════════════════════════════


class InstanceProfiler:
    """Orchestrates profiling for one VLM instance.

    Pipeline:
      1. Load model via adapter
      2. Warmup (N iterations, results discarded)
      3. Measure (M iterations per request, results recorded)
      4. Save raw results + aggregated latency table
      5. Unload model
    """

    def __init__(
        self,
        instance_id: str,
        model_name: str,
        device_class: str,
        device: str = "cuda",
        output_dir: str = "outputs/profiling",
        max_new_tokens_map: Optional[Dict[str, int]] = None,
        custom_model_id: Optional[str] = None,
    ):
        """
        Args:
            instance_id: e.g. "Qwen-7B@AGX"
            model_name: e.g. "Qwen2.5-VL-7B"
            device_class: e.g. "AGX_Orin"
            device: torch device string ("cuda", "cuda:0", "cpu")
            output_dir: where to save profiling results
            max_new_tokens_map: per-complexity max tokens override
            custom_model_id: override default HF model ID
        """
        self.instance_id = instance_id
        self.model_name = model_name
        self.device_class = device_class
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.max_new_tokens = max_new_tokens_map or {
            "simple": 128,
            "moderate": 256,
            "complex": 512,
        }

        self.adapter = get_adapter(model_name, custom_model_id)
        self._measurements: List[SingleMeasurement] = []

    def load(self) -> None:
        """Load the model onto the target device."""
        logger.info(f"[{self.instance_id}] Loading {self.model_name} on {self.device}...")
        self.adapter.load(self.device)
        logger.info(f"[{self.instance_id}] Model loaded.")

    def unload(self) -> None:
        """Unload the model to free memory."""
        logger.info(f"[{self.instance_id}] Unloading model...")
        self.adapter.unload()
        logger.info(f"[{self.instance_id}] Model unloaded.")

    def profile(
        self,
        requests: List[ProfilingRequest],
        n_iterations: int = 3,
        n_warmup: int = 5,
        cooldown_seconds: float = 2.0,
    ) -> InstanceProfilingResult:
        """Profile this instance on a set of requests.

        Args:
            requests: List of ProfilingRequest to measure.
            n_iterations: Number of repeated measurements per request.
            n_warmup: Number of warmup inference runs before timing.
            cooldown_seconds: Pause between requests (thermal management).

        Returns:
            InstanceProfilingResult with all measurements.
        """
        result = InstanceProfilingResult(
            instance_id=self.instance_id,
            model_name=self.model_name,
            device_class=self.device_class,
        )

        # ── Warmup ──
        logger.info(
            f"[{self.instance_id}] Warmup: {n_warmup} runs "
            f"on {min(n_warmup, len(requests))} requests..."
        )
        warmup_requests = requests[: min(n_warmup, len(requests))]
        for req in warmup_requests:
            max_tok = self.max_new_tokens.get(req.complexity_class, 256)
            try:
                self.adapter.generate(req.image_path, req.question, max_tok)
            except Exception as e:
                logger.warning(
                    f"[{self.instance_id}] Warmup failed for {req.request_id}: {e}"
                )
        logger.info(f"[{self.instance_id}] Warmup complete.")

        # ── Measurement ──
        n_total = len(requests) * n_iterations
        logger.info(
            f"[{self.instance_id}] Measuring: {len(requests)} requests "
            f"× {n_iterations} iterations = {n_total} runs"
        )

        for req_idx, req in enumerate(requests):
            max_tok = self.max_new_tokens.get(req.complexity_class, 256)

            for it in range(n_iterations):
                try:
                    output_text, elapsed_ms, num_tokens = self.adapter.generate(
                        req.image_path, req.question, max_tok
                    )

                    measurement = SingleMeasurement(
                        request_id=req.request_id,
                        instance_id=self.instance_id,
                        complexity_class=req.complexity_class,
                        latency_ms=elapsed_ms,
                        num_tokens=num_tokens,
                        iteration=it + 1,
                    )
                    self._measurements.append(measurement)
                    result.measurements.append(measurement)

                    logger.debug(
                        f"[{self.instance_id}] {req.request_id} "
                        f"({req.complexity_class}) it={it+1}: "
                        f"{elapsed_ms:.0f}ms, {num_tokens} tokens"
                    )

                except Exception as e:
                    logger.error(
                        f"[{self.instance_id}] FAILED {req.request_id} it={it+1}: {e}"
                    )

            # Cooldown between requests (especially for Jetson thermal throttling)
            if cooldown_seconds > 0 and req_idx < len(requests) - 1:
                time.sleep(cooldown_seconds)

        # ── Save ──
        self._save_result(result)
        self._print_summary(result)

        return result

    def _save_result(self, result: InstanceProfilingResult) -> None:
        """Save raw measurements and aggregated table to disk."""
        safe_name = self.instance_id.replace("@", "_").replace("/", "_")

        # Raw CSV
        raw_path = self.output_dir / f"{safe_name}_raw.csv"
        result.to_dataframe().to_csv(raw_path, index=False)
        logger.info(f"[{self.instance_id}] Raw data → {raw_path}")

        # Aggregated latency row (one row per instance × complexity)
        agg_path = self.output_dir / f"{safe_name}_latency.json"
        agg_row = {
            "instance_id": self.instance_id,
            "model_name": self.model_name,
            "device_class": self.device_class,
            **result.latency_table_row(),
        }
        with open(agg_path, "w") as f:
            json.dump(agg_row, f, indent=2)
        logger.info(f"[{self.instance_id}] Aggregated → {agg_path}")

    def _print_summary(self, result: InstanceProfilingResult) -> None:
        """Print a one-line summary per complexity class."""
        row = result.latency_table_row()
        parts = [
            f"{c}: {row[c]:.0f}ms" for c in ["simple", "moderate", "complex"]
        ]
        logger.info(
            f"[{self.instance_id}] Summary — {' | '.join(parts)} "
            f"(from {len(result.measurements)} measurements)"
        )


# ═══════════════════════════════════════════════════════════════════
# Utility: build full latency table from profiling results
# ═══════════════════════════════════════════════════════════════════


def build_latency_table_from_results(
    results: List[InstanceProfilingResult],
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """Combine multiple InstanceProfilingResults into a latency table.

    Output format matches what LatencyTableBuilder produces:
      instance_id, model_name, device_class, simple, moderate, complex
    """
    rows = []
    for r in results:
        row = {
            "instance_id": r.instance_id,
            "model_name": r.model_name,
            "device_class": r.device_class,
        }
        row.update(r.latency_table_row())
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df[["instance_id", "model_name", "device_class", "simple", "moderate", "complex"]]

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Full latency table → {output_path}")

    return df


def load_profiling_requests(json_path: str) -> List[ProfilingRequest]:
    """Load profiling requests from a JSON file.

    Expected format:
    [
      {
        "request_id": "...",
        "image_path": "/path/to/image.jpg",
        "question": "...",
        "complexity_class": "simple"
      },
      ...
    ]
    """
    with open(json_path) as f:
        data = json.load(f)

    return [
        ProfilingRequest(
            request_id=r["request_id"],
            image_path=r["image_path"],
            question=r["question"],
            complexity_class=r.get("complexity_class", "moderate"),
            dataset=r.get("dataset", ""),
            task_name=r.get("task_name", ""),
        )
        for r in data
    ]
