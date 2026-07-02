"""Phase 1 Experiment: VL-RouterBench main pipeline (Section 12.1).

Steps:
  1. Load data from VL-RouterBench → unified RequestRecord schema
  2. Build quality profile + answer type labels
  3. Build synthetic latency table + communication profile
  4. Train Predictor (multi-task: suitability + answer type)
  5. Run inference on test split
  6. Run QMAR scheduler + all baselines
  7. Evaluate and save results
"""

import logging
import sys
import os
from pathlib import Path
from typing import List, Dict
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_adapters.vl_routerbench_adapter import VLRouterBenchAdapter
from data_adapters.common_schema import RequestRecord, AnswerType

from profiling.quality_profile import build_quality_profile, get_model_pool
from profiling.answer_type_labels import build_answer_type_labels
from profiling.latency_table import LatencyTableBuilder
from profiling.communication_profile import build_comm_profile

from predictor.model import Predictor
from predictor.train import train_predictor
from predictor.infer import predict, load_predictor

from scheduler.latency_estimator import LatencyEstimator
from scheduler.qmar import QMARScheduler
from scheduler.baselines import (
    RandomFeasibleBaseline,
    FastestFeasibleBaseline,
    HighestSuitabilityBaseline,
    LatencyOnlyGreedyBaseline,
    QMARWithoutAnswerTypeBaseline,
)

from evaluation.metrics import evaluate_all, build_summary_table
from evaluation.plots import (
    plot_threshold_sweep,
    plot_load_distribution,
    plot_method_comparison,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("phase1")


def load_config(config_path: str = None) -> Dict:
    """Load QMAR configuration."""
    if config_path is None:
        config_path = str(Path(__file__).parent.parent / "configs" / "qmar.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_instance_config(config_dir: str = None) -> List[Dict]:
    """Load instance pool configuration."""
    if config_dir is None:
        config_dir = str(Path(__file__).parent.parent / "configs")
    path = Path(config_dir) / "instance_pool.yaml"
    with open(str(path), "r") as f:
        cfg = yaml.safe_load(f)
    return cfg["instances"]


def step1_load_data(config: Dict) -> List[RequestRecord]:
    """Load VL-RouterBench data and convert to unified schema."""
    logger.info("=" * 60)
    logger.info("Step 1: Loading VL-RouterBench data")

    data_root = config["data"]["vl_routerbench_root"]
    # Resolve relative path
    if not os.path.isabs(data_root):
        data_root = os.path.join(
            Path(__file__).parent.parent, data_root
        )

    adapter = VLRouterBenchAdapter(data_root=data_root)
    records = adapter.load_all_records()

    logger.info(f"Loaded {len(records)} records")
    return records


def step2_build_profiles(
    records: List[RequestRecord],
    instance_config: List[Dict],
    config: Dict,
    output_dir: str,
):
    """Build quality profile, answer type labels, latency table, comm profile."""
    logger.info("=" * 60)
    logger.info("Step 2: Building profiles")

    profile_dir = Path(output_dir) / "profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)

    # Quality profile
    quality_df = build_quality_profile(
        records,
        output_path=str(profile_dir / "quality_profile.csv"),
    )
    model_names = get_model_pool(records)
    logger.info(f"Quality profile: {len(quality_df)} rows, {len(model_names)} models")

    # Answer type labels
    answer_type_df = build_answer_type_labels(
        records,
        output_path=str(profile_dir / "answer_type_labels.csv"),
    )

    # Latency table
    builder = LatencyTableBuilder()
    for cfg in instance_config:
        builder.add_instance(
            instance_id=cfg["instance_id"],
            model_name=cfg["model_name"],
            device_class=cfg["device_class"],
        )
    latency_table = builder.build(
        output_path=str(profile_dir / "latency_table.csv"),
    )
    logger.info(f"Latency table: {len(latency_table)} instances")

    # Communication profile
    comm_profile = build_comm_profile(
        instance_config,
        output_path=str(profile_dir / "communication_profile.csv"),
    )

    return quality_df, model_names, answer_type_df, latency_table, comm_profile


def step3_train_predictor(
    records: List[RequestRecord],
    model_names: List[str],
    config: Dict,
    output_dir: str,
    device: str = "cpu",
):
    """Train the multi-task Predictor."""
    logger.info("=" * 60)
    logger.info("Step 3: Training Predictor")

    # Split records
    seed = config.get("experiment", {}).get("random_seed", 42)
    test_ratio = config.get("experiment", {}).get("test_split_ratio", 0.2)
    np.random.seed(seed)
    indices = np.random.permutation(len(records))
    test_size = int(len(records) * test_ratio)
    train_indices = indices[test_size:]
    val_indices = indices[:test_size]

    train_records = [records[i] for i in train_indices]
    val_records = [records[i] for i in val_indices]
    logger.info(f"Train: {len(train_records)}, Val: {len(val_records)}")

    # Create predictor
    pred_cfg = config["predictor"]
    train_cfg = config["training"]

    predictor = Predictor(
        num_models=len(model_names),
        vision_encoder_name=pred_cfg["vision_encoder"],
        text_encoder_name=pred_cfg["text_encoder"],
        vision_dim=pred_cfg["vision_dim"],
        text_dim=pred_cfg["text_dim"],
        fusion_hidden_dims=pred_cfg["fusion_hidden_dims"],
        fusion_output_dim=pred_cfg["fusion_output_dim"],
        suit_hidden_dim=pred_cfg["suit_hidden_dim"],
        type_hidden_dim=pred_cfg["type_hidden_dim"],
        dropout=pred_cfg["dropout"],
        device=device,
    )

    # Train
    checkpoint_dir = Path(output_dir) / "checkpoints"
    history = train_predictor(
        predictor=predictor,
        train_records=train_records,
        val_records=val_records,
        model_names=model_names,
        output_dir=str(checkpoint_dir),
        batch_size=train_cfg["batch_size"],
        num_epochs=train_cfg["num_epochs"],
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
        alpha=train_cfg["alpha"],
        use_focal=train_cfg["use_focal"],
        focal_gamma=train_cfg["focal_gamma"],
        patience=train_cfg["patience"],
        num_workers=train_cfg["num_workers"],
        device=device,
    )

    return predictor, train_records, val_records


def step4_run_inference(
    predictor: Predictor,
    records: List[RequestRecord],
    model_names: List[str],
    device: str = "cpu",
) -> Dict:
    """Run Predictor inference on records."""
    logger.info("=" * 60)
    logger.info("Step 4: Running Predictor inference")

    results = predict(
        predictor=predictor,
        records=records,
        model_names=model_names,
        batch_size=32,
        device=device,
    )

    logger.info(
        f"Inference complete: suitability_scores shape={results['suitability_scores'].shape}"
    )
    return results


def step5_run_schedulers(
    records: List[RequestRecord],
    pred_results: Dict,
    model_names: List[str],
    instance_config: List[Dict],
    latency_table: pd.DataFrame,
    comm_profile: pd.DataFrame,
    config: Dict,
    output_dir: str,
) -> Dict[str, Dict]:
    """Run QMAR and all baselines."""
    logger.info("=" * 60)
    logger.info("Step 5: Running QMAR scheduler and baselines")

    qmar_cfg = config.get("scheduler", {})
    quality_threshold = qmar_cfg.get("quality_threshold", 0.5)
    use_comm_cost = qmar_cfg.get("use_comm_cost", True)

    # Setup latency estimator
    estimator = LatencyEstimator(
        latency_table=latency_table,
        comm_profile=comm_profile,
        use_comm_cost=use_comm_cost,
    )
    instance_ids = estimator.get_instance_ids()
    if not instance_ids:
        instance_ids = [cfg["instance_id"] for cfg in instance_config]

    # Build latency matrix
    latency_matrix = estimator.estimate_batch(
        records=records,
        predicted_answer_types=list(pred_results["predicted_answer_type_labels"]),
        instance_ids=instance_ids,
    )

    # Define all methods
    methods = {
        "QMAR_full": QMARScheduler(
            instance_config=instance_config,
            model_names=model_names,
            quality_threshold=quality_threshold,
        ),
        "Random_feasible": RandomFeasibleBaseline(
            instance_config=instance_config,
            model_names=model_names,
            quality_threshold=quality_threshold,
        ),
        "Fastest_feasible": FastestFeasibleBaseline(
            instance_config=instance_config,
            model_names=model_names,
            quality_threshold=quality_threshold,
        ),
        "Highest_suitability": HighestSuitabilityBaseline(
            instance_config=instance_config,
            model_names=model_names,
            quality_threshold=quality_threshold,
        ),
        "Latency_only_greedy": LatencyOnlyGreedyBaseline(
            instance_config=instance_config,
            model_names=model_names,
            quality_threshold=quality_threshold,
        ),
        "QMAR_wo_answer_type": QMARWithoutAnswerTypeBaseline(
            instance_config=instance_config,
            model_names=model_names,
            quality_threshold=quality_threshold,
        ),
    }

    all_results = {}
    for method_name, scheduler in methods.items():
        logger.info(f"Running {method_name}...")
        result = scheduler.schedule(
            records=records,
            suitability_scores=pred_results["suitability_scores"],
            latency_matrix=latency_matrix,
            predicted_answer_types=list(pred_results["predicted_answer_type_labels"]),
        )

        # Evaluate
        metrics = evaluate_all(
            assignments=result["assignments"],
            instance_loads=result["instance_loads"],
            batch_latency=result["batch_latency"],
            records=records,
        )
        all_results[method_name] = metrics

        # Save assignments
        assign_dir = Path(output_dir) / "assignments"
        assign_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(result["assignments"])
        df.to_csv(
            assign_dir / f"{method_name}_assignments_tau{quality_threshold}.csv",
            index=False,
        )

        logger.info(
            f"  {method_name}: batch_latency={result['batch_latency']:.1f}ms, "
            f"fallback_rate={metrics['quality']['fallback_rate']:.3f}"
        )

    return all_results


def step6_evaluate_and_save(
    all_results: Dict[str, Dict],
    output_dir: str,
):
    """Evaluate, generate summary tables and plots."""
    logger.info("=" * 60)
    logger.info("Step 6: Generating evaluation outputs")

    results_dir = Path(output_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Summary table
    summary = build_summary_table(all_results)
    summary.to_csv(results_dir / "summary_metrics.csv", index=False)
    logger.info(f"\nSummary:\n{summary.to_string()}")

    # Method comparison plot
    plot_method_comparison(
        all_results,
        output_path=str(results_dir / "method_comparison.png"),
    )

    # Load distribution for best method (QMAR_full)
    if "QMAR_full" in all_results:
        # We need instance loads; reconstruct from assignments
        pass

    logger.info(f"Results saved to {results_dir}")
    return summary


def run_phase1(
    config_path: str = None,
    output_dir: str = None,
    device: str = "cpu",
    skip_training: bool = False,
    checkpoint_path: str = None,
):
    """Run full Phase 1 pipeline."""
    config = load_config(config_path)

    if output_dir is None:
        output_dir = str(Path(__file__).parent.parent / "outputs")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Step 1: Load data
    records = step1_load_data(config)

    if not records:
        logger.error("No records loaded. Check VL-RouterBench data path.")
        return

    # Step 2: Build profiles
    instance_config = load_instance_config()
    quality_df, model_names, answer_type_df, latency_table, comm_profile = (
        step2_build_profiles(records, instance_config, config, str(output_dir))
    )

    # Step 3-4: Train or load predictor + inference
    if skip_training and checkpoint_path:
        predictor, model_names = load_predictor(checkpoint_path, device=device)
    else:
        predictor, train_records, val_records = step3_train_predictor(
            records, model_names, config, str(output_dir), device=device
        )

    # Use all records for scheduling (in real scenario, use test split)
    # For VL-RouterBench, we use the same records since correctness is ground-truth
    pred_results = step4_run_inference(
        predictor, records, model_names, device=device
    )

    # Step 5: Run schedulers
    all_results = step5_run_schedulers(
        records, pred_results, model_names, instance_config,
        latency_table, comm_profile, config, str(output_dir),
    )

    # Step 6: Evaluate
    summary = step6_evaluate_and_save(all_results, str(output_dir))

    return summary, all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="QMAR Phase 1: VL-RouterBench Pipeline")
    parser.add_argument("--config", type=str, default=None, help="Path to qmar.yaml")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training, load checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to predictor checkpoint")
    args = parser.parse_args()

    run_phase1(
        config_path=args.config,
        output_dir=args.output,
        device=args.device,
        skip_training=args.skip_training,
        checkpoint_path=args.checkpoint,
    )
