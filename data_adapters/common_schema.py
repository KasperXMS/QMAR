"""Unified request-level schema for QMAR pipeline.

All data from VL-RouterBench and MMR-Bench is converted to this schema
before entering the Predictor + QMAR pipeline.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List


@dataclass
class RequestRecord:
    """Unified request record following the technical spec (Section 2.2)."""

    request_id: str
    dataset_name: str
    task_name: str  # or subset_name
    image_path: Optional[str] = None
    image_id: Optional[str] = None
    question_text: str = ""
    ground_truth_answer: str = ""

    # Model outputs and correctness
    candidate_model_outputs: Dict[str, str] = field(default_factory=dict)
    # model_name -> correctness (0/1 or float score)
    candidate_model_correctness: Dict[str, float] = field(default_factory=dict)

    # Optional fields
    optional_original_utility: Optional[Dict[str, float]] = None
    optional_task_metric_label: Optional[str] = None

    # Additional metadata for flexibility
    extra: Dict = field(default_factory=dict)

    @property
    def model_names(self) -> List[str]:
        return list(self.candidate_model_correctness.keys())

    @property
    def correct_models(self) -> List[str]:
        return [
            m for m, c in self.candidate_model_correctness.items() if c >= 0.5
        ]

    def get_suitability_label(self, model_name: str) -> int:
        """Binary suitability label y_{i,m} ∈ {0,1}."""
        return 1 if self.candidate_model_correctness.get(model_name, 0) >= 0.5 else 0


# ── Task-complexity-based Answer Type ──────────────────────────────
#
# VLM latency depends on how much "thinking" the model needs to do,
# NOT on how many tokens the final answer contains.
#
# A math reasoning question whose answer is "42" requires chain-of-thought
# and is far more expensive than an OCR question whose answer is "copenhagen".
#
# Classification is based on (dataset, task_category), falling back to
# question text analysis when metadata is insufficient.
# ────────────────────────────────────────────────────────────────────

class AnswerType:
    """Task-complexity-based answer type for latency estimation.

    Replaces the original length-based short/medium/long with
    complexity-based simple/moderate/complex.
    """

    SIMPLE = "simple"      # Direct visual perception: "what is this?", OCR, color, count
    MODERATE = "moderate"  # Structured understanding: chart, document, spatial, diagram
    COMPLEX = "complex"    # Multi-step reasoning: math, logic, hallucination detection

    # ── Dataset → complexity mapping ──
    # Derived from VL-RouterBench benchmark taxonomy (Section 2)
    DATASET_COMPLEXITY = {
        # Direct visual QA — multiple choice, no reasoning needed
        "MMBench_DEV_EN_V11": "simple",
        "MMStar": "simple",
        "RealWorldQA": "simple",

        # OCR & document understanding — structured but not deep reasoning
        "TextVQA_VAL": "moderate",
        "OCRBench": "moderate",
        "DocVQA_VAL": "moderate",
        "ChartQA_TEST": "moderate",
        "InfoVQA_VAL": "moderate",

        # STEM reasoning — needs chain-of-thought
        "MathVista_MINI": "complex",
        "MathVision_MINI": "complex",
        "MathVerse_MINI": "complex",

        # Diagrams & science — moderate (structured) by default, tuned by sub-task
        "AI2D_TEST": "moderate",

        # Hallucination detection — requires comparing image vs text claims
        "HallusionBench": "complex",

        # College-level multi-discipline — complex
        "MMMU_DEV_VAL": "complex",
    }

    # ── Per-dataset task category overrides ──
    # Finer-grained tuning within each dataset
    TASK_OVERRIDES = {
        # MMStar: logical/math are complex, rest simple
        "MMStar": {
            "logical reasoning": "complex",
            "math": "complex",
        },
        # MMBench: reasoning-heavy categories → moderate/complex
        "MMBench_DEV_EN_V11": {
            "future_prediction": "complex",
            "nature_reasoning": "complex",
            "function_reasoning": "complex",
            "identity_reasoning": "complex",
            "social_relation": "complex",
            "physical_property_reasoning": "moderate",
            "attribute_comparison": "moderate",
            "spatial_relationship": "moderate",
            "structuralized_imagetext_understanding": "moderate",
        },
        # AI2D: food chain / life cycle / moon phases are diagram reasoning
        "AI2D_TEST": {
            "foodChainsWebs": "complex",
            "lifeCycles": "complex",
            "moonPhaseEquinox": "complex",
        },
        # OCRBench: handwritten math → complex
        "OCRBench": {
            "Handwritten Mathematical Expression Recognition": "complex",
        },
        # MathVista: math-targeted → complex, general-vqa → moderate
        "MathVista_MINI": {
            "math-targeted-vqa": "complex",
            "general-vqa": "moderate",
        },
        # MathVision: logic & solid geometry are the hardest
        "MathVision_MINI": {
            "logic": "complex",
            "solid geometry": "complex",
            "combinatorial geometry": "complex",
        },
    }

    @classmethod
    def classify(cls, record: "RequestRecord") -> str:
        """Classify a request into simple/moderate/complex based on task metadata.

        Hierarchy:
          1. Per-dataset task override (most specific)
          2. Dataset-level default
          3. Question text heuristic (fallback)
        """
        dataset = record.dataset_name
        task = record.task_name or ""
        question = record.question_text or ""

        # 1. Check per-dataset task override
        overrides = cls.TASK_OVERRIDES.get(dataset, {})
        if task in overrides:
            return overrides[task]

        # 2. Dataset-level default
        if dataset in cls.DATASET_COMPLEXITY:
            return cls.DATASET_COMPLEXITY[dataset]

        # 3. Fallback: question text heuristics
        q_lower = question.lower()
        # Math keywords → complex
        math_keywords = ["hint:", "calculate", "solve", "equation", "formula",
                         "math", "reasoning", "why", "explain"]
        if any(kw in q_lower for kw in math_keywords):
            return cls.COMPLEX
        # Multi-choice with reasoning indicators → moderate
        if any(opt in question for opt in ["A.", "B.", "C.", "D."]):
            return cls.SIMPLE

        return cls.MODERATE  # safest default

    @classmethod
    def to_index(cls, answer_type: str) -> int:
        mapping = {cls.SIMPLE: 0, cls.MODERATE: 1, cls.COMPLEX: 2}
        return mapping.get(answer_type, 1)  # default moderate

    @classmethod
    def from_index(cls, idx: int) -> str:
        mapping = {0: cls.SIMPLE, 1: cls.MODERATE, 2: cls.COMPLEX}
        return mapping.get(idx, cls.MODERATE)

    @classmethod
    def num_classes(cls) -> int:
        return 3
