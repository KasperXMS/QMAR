#!/usr/bin/env python
"""Small-scale optimality validation: compare QMAR vs true ILP optimum.

For N ≤ 200, solves the exact R||Cmax ILP via pulp/PuLP to get the
provably optimal makespan. Compares Oracle, QMAR, and baselines against it.

Usage:
    python validate_optimality.py                    # N=50, 100, 200
    python validate_optimality.py --n 80 --seed 42   # single run
"""

import argparse
import logging
import sys
import pickle
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import yaml
import pulp

from profiling.latency_table import LatencyTableBuilder
from profiling.communication_profile import build_comm_profile
from scheduler.qmar import QMARScheduler
from scheduler.baselines import (
    RandomFeasibleBaseline,
    FastestFeasibleBaseline,
    HighestSuitabilityBaseline,
    LatencyOnlyGreedyBaseline,
    QMARWithoutAnswerTypeBaseline,
    OracleBaseline,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("validate")


def solve_ilp_optimal(L: np.ndarray, time_limit: int = 300) -> dict:
    """Solve R||Cmax exactly via ILP: minimize makespan T.

    Formulation:
        min T
        s.t.  Σ_k x[i,k] = 1           ∀i
              Σ_i x[i,k] · l[i,k] ≤ T  ∀k
              x[i,k] ∈ {0, 1}

    Args:
        L: (N, K) latency matrix.
        time_limit: max seconds for solver.

    Returns:
        dict with optimal_makespan, assignments, solve_time, status.
    """
    N, K = L.shape

    # Binary variables x[i,k]
    x = {(i, k): pulp.LpVariable(f"x_{i}_{k}", cat="Binary")
         for i in range(N) for k in range(K)}

    # Continuous makespan variable
    T = pulp.LpVariable("T", lowBound=0, cat="Continuous")

    # Problem
    prob = pulp.LpProblem("R_Cmax", pulp.LpMinimize)
    prob += T

    # Assignment constraints
    for i in range(N):
        prob += pulp.lpSum([x[i, k] for k in range(K)]) == 1

    # Load constraints
    for k in range(K):
        prob += pulp.lpSum([x[i, k] * L[i, k] for i in range(N)]) <= T

    # Solve
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit)
    start = time.time()
    status = prob.solve(solver)
    elapsed = time.time() - start

    status_str = pulp.LpStatus[status]

    if status == pulp.LpStatusOptimal:
        optimal_makespan = pulp.value(T)
        assignments = np.zeros(N, dtype=int)
        for i in range(N):
            for k in range(K):
                if pulp.value(x[i, k]) > 0.5:
                    assignments[i] = k
                    break
    else:
        optimal_makespan = None
        assignments = None

    return {
        "optimal_makespan": optimal_makespan,
        "assignments": assignments,
        "solve_time": elapsed,
        "status": status_str,
    }


def run_methods(L, N, K, instance_config, model_names, threshold):
    """Run all scheduling methods on the same latency matrix."""
    class FR:
        def __init__(self, i):
            self.request_id = f"r{i}"
            self.image_path = None
            self.question_text = ""
            self.candidate_model_correctness = {}

    records = [FR(i) for i in range(N)]
    # Dummy suitability: all 1.0 (no quality constraints)
    suit = np.ones((N, len(model_names)), dtype=np.float32)
    # Dummy answer types: all "moderate"
    atypes = ["moderate"] * N

    methods = {
        "Oracle (LPT+LS)":        OracleBaseline(instance_config, model_names, threshold),
        "QMAR_full":              QMARScheduler(instance_config, model_names, threshold),
        "QMAR_wo_answer_type":    QMARWithoutAnswerTypeBaseline(instance_config, model_names, threshold),
        "Latency_only_greedy":    LatencyOnlyGreedyBaseline(instance_config, model_names, threshold),
        "Fastest_feasible":       FastestFeasibleBaseline(instance_config, model_names, threshold),
        "Random_feasible":        RandomFeasibleBaseline(instance_config, model_names, threshold, seed=42),
        "Highest_suitability":   HighestSuitabilityBaseline(instance_config, model_names, threshold),
    }

    results = {}
    for name, sched in methods.items():
        t0 = time.time()
        r = sched.schedule(records, suit, L, atypes)
        t = (time.time() - t0) * 1000
        results[name] = {
            "makespan": r["batch_latency"],
            "fallback": r["fallback_count"],
            "load_cv": (
                np.std(list(r["instance_loads"].values())) /
                np.mean(list(r["instance_loads"].values()))
                if np.mean(list(r["instance_loads"].values())) > 0 else 0
            ),
            "time_ms": t,
        }

    return results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=None,
                   help="Single N to run (default: sweep 50,100,200)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--instance-config", type=str, default="configs/instance_pool.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    with open(args.instance_config) as f:
        inst = yaml.safe_load(f)
    instance_config = inst["instances"]
    instance_ids = [c["instance_id"] for c in instance_config]
    model_names = list(set(c["model_name"] for c in instance_config))
    K = len(instance_ids)

    # Build latency table for answer-type-aware latencies
    builder = LatencyTableBuilder()
    for cfg in instance_config:
        builder.add_instance(**cfg)
    latency_table = builder.build()
    instance_lat = {}
    for _, row in latency_table.iterrows():
        iid = row["instance_id"]
        instance_lat[iid] = {"simple": row["simple"], "moderate": row["moderate"],
                             "complex": row["complex"]}

    Ns = [args.n] if args.n else [50, 100, 200]
    seeds = [42, 123, 456]

    for seed in seeds:
        np.random.seed(seed)

        print(f"\n{'='*120}")
        print(f"  SEED = {seed}")
        print(f"{'='*120}")
        print(f"{'N':>4} | {'ILP Optimal':>12} | "
              f"{'Oracle':>12} | {'QMAR_full':>12} | {'QMAR_noAT':>12} | "
              f"{'LatGreedy':>12} | {'Fastest':>12} | {'Random':>12} | {'HighSuit':>12} | "
              f"{'ILP Time':>8}")
        print("-" * 135)

        for N in Ns:
            # Generate synthetic latency matrix
            L = np.zeros((N, K), dtype=np.float32)
            for k, iid in enumerate(instance_ids):
                lat_row = instance_lat[iid]
                for i in range(N):
                    at = np.random.choice(["simple", "moderate", "complex"], p=[0.1, 0.6, 0.3])
                    L[i, k] = lat_row[at] * np.random.uniform(0.7, 1.3)

            # True optimal via ILP
            ilp_result = solve_ilp_optimal(L, time_limit=300)
            ilp_opt = ilp_result["optimal_makespan"]

            # All methods
            mr = run_methods(L, N, K, instance_config, model_names, threshold=0.5)

            if ilp_opt is not None:
                def g(name):
                    return (mr[name]["makespan"] - ilp_opt) / ilp_opt * 100

                print(f"{N:4d} | {ilp_opt:>10.0f}ms | "
                      f"{mr['Oracle (LPT+LS)']['makespan']:>10.0f}ms ({g('Oracle (LPT+LS)'):+5.1f}%) | "
                      f"{mr['QMAR_full']['makespan']:>10.0f}ms ({g('QMAR_full'):+5.1f}%) | "
                      f"{mr['QMAR_wo_answer_type']['makespan']:>10.0f}ms ({g('QMAR_wo_answer_type'):+5.1f}%) | "
                      f"{mr['Latency_only_greedy']['makespan']:>10.0f}ms ({g('Latency_only_greedy'):+5.1f}%) | "
                      f"{mr['Fastest_feasible']['makespan']:>10.0f}ms ({g('Fastest_feasible'):+5.1f}%) | "
                      f"{mr['Random_feasible']['makespan']:>10.0f}ms ({g('Random_feasible'):+5.1f}%) | "
                      f"{mr['Highest_suitability']['makespan']:>10.0f}ms ({g('Highest_suitability'):+5.1f}%) | "
                      f"{ilp_result['solve_time']:>6.1f}s")
            else:
                print(f"{N:4d} | {'FAILED':>12} | ... ({ilp_result['status']})")

        # Load CV table
        print(f"\n  Load CV (coefficient of variation, lower = more balanced):")
        print(f"  {'N':>4} | {'Oracle':>10} | {'QMAR_full':>10} | {'QMAR_noAT':>10} | "
              f"{'LatGreedy':>10} | {'Fastest':>10} | {'Random':>10} | {'HighSuit':>10}")
        print(f"  " + "-"*95)
        for N in Ns:
            L = np.zeros((N, K), dtype=np.float32)
            for k, iid in enumerate(instance_ids):
                lat_row = instance_lat[iid]
                for i in range(N):
                    at = np.random.choice(["simple", "moderate", "complex"], p=[0.1, 0.6, 0.3])
                    L[i, k] = lat_row[at] * np.random.uniform(0.7, 1.3)
            mr = run_methods(L, N, K, instance_config, model_names, threshold=0.5)
            print(f"  {N:4d} | "
                  f"{mr['Oracle (LPT+LS)']['load_cv']:>10.4f} | "
                  f"{mr['QMAR_full']['load_cv']:>10.4f} | "
                  f"{mr['QMAR_wo_answer_type']['load_cv']:>10.4f} | "
                  f"{mr['Latency_only_greedy']['load_cv']:>10.4f} | "
                  f"{mr['Fastest_feasible']['load_cv']:>10.4f} | "
                  f"{mr['Random_feasible']['load_cv']:>10.4f} | "
                  f"{mr['Highest_suitability']['load_cv']:>10.4f}")

    print()
    print("All gaps are vs true ILP optimum. Negative gap = result better than optimum (noise/rounding).")


if __name__ == "__main__":
    main()
