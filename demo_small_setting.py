#!/usr/bin/env python
"""QMAR concrete demo: 8 requests, 5 heterogeneous instances.

A fully worked-through example showing every step of the pipeline.
All numbers are realistic synthetic values consistent with the latency
table and model capabilities.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════
# Step 0: Define the setting
# ═══════════════════════════════════════════════════════════════

# 5 heterogeneous instances: (model, device)
INSTANCES = [
    {"id": "U1", "model": "SmolVLM2",         "device": "Orin NX",   "class": "edge-tiny"},
    {"id": "U2", "model": "Phi-3.5-Vision",   "device": "Orin NX",   "class": "edge-small"},
    {"id": "U3", "model": "Qwen2.5-VL-7B",    "device": "RTX 3090",  "class": "desktop-mid"},
    {"id": "U4", "model": "Qwen2.5-VL-32B",   "device": "RTX 4090",  "class": "server-large"},
    {"id": "U5", "model": "GPT-4o",            "device": "Cloud API", "class": "cloud-xl"},
]

# 8 requests: (id, description, task_type, complexity)
REQUESTS = [
    {"id": "R1", "desc": "What color is the car?",                    "task": "object recognition", "complexity": "simple"},
    {"id": "R2", "desc": "How many people are in the image?",          "task": "counting",           "complexity": "simple"},
    {"id": "R3", "desc": "Read the street sign text.",                "task": "OCR",                "complexity": "moderate"},
    {"id": "R4", "desc": "What is the brand of this laptop?",         "task": "OCR + recognition",  "complexity": "moderate"},
    {"id": "R5", "desc": "Explain the food chain shown in the diagram.","task": "diagram reasoning", "complexity": "moderate"},
    {"id": "R6", "desc": "Is this image real or AI-generated? Why?",  "task": "hallucination det.", "complexity": "complex"},
    {"id": "R7", "desc": "Solve: if x²+y²=25 and xy=12, find x+y.",  "task": "math reasoning",     "complexity": "complex"},
    {"id": "R8", "desc": "Calculate the area of the shaded region.",  "task": "geometry",           "complexity": "complex"},
]

# Latency table (ms): rows=instances, cols=answer_types
# From profiling/latency_table.py synthetic profiles
LATENCY_TABLE = {
    "U1": {"simple": 250, "moderate": 600, "complex": 2000},
    "U2": {"simple": 350, "moderate": 850, "complex": 2800},
    "U3": {"simple":  80, "moderate": 350, "complex": 1200},
    "U4": {"simple":  50, "moderate": 200, "complex":  700},
    "U5": {"simple": 300, "moderate": 600, "complex": 1800},  # cloud: RTT-dominated for simple, competitive for complex
}

# ═══════════════════════════════════════════════════════════════
# Step 1: Predictor outputs (simulated)
# ═══════════════════════════════════════════════════════════════
# Suitability s[i,m] = probability model m answers request i correctly
# Higher = more capable model; complex tasks widen the gap

np.random.seed(42)

MODEL_NAMES = [inst["model"] for inst in INSTANCES]
N = len(REQUESTS)
K = len(INSTANCES)

# Realistic suitability pattern:
# - Simple tasks: all models do well (small models near 0.8, large near 1.0)
# - Moderate tasks: small models drop to 0.4-0.6, large stay high
# - Complex tasks: only large/cloud models reliable (>0.7), small drop to 0.2-0.4
suitability = np.array([
    # U1(Smol)  U2(Phi)  U3(Qwen7B) U4(Qwen32B) U5(GPT4o)
    [  0.82,     0.88,     0.95,       0.98,       0.99  ],  # R1: simple - color
    [  0.75,     0.82,     0.92,       0.96,       0.98  ],  # R2: simple - counting
    [  0.35,     0.48,     0.78,       0.88,       0.92  ],  # R3: moderate - OCR
    [  0.40,     0.52,     0.80,       0.90,       0.94  ],  # R4: moderate - OCR+recog
    [  0.25,     0.38,     0.65,       0.82,       0.90  ],  # R5: moderate - diagram
    [  0.20,     0.30,     0.55,       0.78,       0.85  ],  # R6: complex - hallucination
    [  0.15,     0.22,     0.45,       0.72,       0.88  ],  # R7: complex - math
    [  0.18,     0.25,     0.48,       0.75,       0.85  ],  # R8: complex - geometry
])

# Predicted answer types
predicted_answer_types = [
    "simple",   # R1
    "simple",   # R2
    "moderate", # R3
    "moderate", # R4
    "moderate", # R5
    "complex",  # R6
    "complex",  # R7
    "complex",  # R8
]

# Build latency matrix L[i,k]
L = np.zeros((N, K))
for i in range(N):
    at = predicted_answer_types[i]
    for k, inst in enumerate(INSTANCES):
        L[i, k] = LATENCY_TABLE[inst["id"]][at]

# ═══════════════════════════════════════════════════════════════
# Step 2: Feasible set construction (τ = 0.5)
# ═══════════════════════════════════════════════════════════════
tau = 0.5
feasible_sets = []
for i in range(N):
    F_i = [k for k in range(K) if suitability[i, k] >= tau]
    feasible_sets.append(F_i)

# ═══════════════════════════════════════════════════════════════
# Step 3: Request ranking (QMAR lexicographic)
# ═══════════════════════════════════════════════════════════════
ranking = []
for i in range(N):
    F_i = feasible_sets[i]
    q_i = len(F_i)                                    # feasible set size
    delta_i = max(suitability[i, F_i]) - tau if F_i else 0  # quality margin
    p_i = min(L[i, F_i]) if F_i else float("inf")     # min feasible latency
    ranking.append({
        "request": REQUESTS[i]["id"],
        "desc": REQUESTS[i]["desc"],
        "feasible_set": [INSTANCES[k]["id"] for k in F_i],
        "q": q_i,
        "delta": round(delta_i, 3),
        "p_min": p_i,
        "complexity": REQUESTS[i]["complexity"],
    })

# Sort: q ascending, delta ascending, p descending
ranking.sort(key=lambda x: (x["q"], x["delta"], -x["p_min"]))

# ═══════════════════════════════════════════════════════════════
# Step 4: Greedy load-balanced assignment
# ═══════════════════════════════════════════════════════════════
instance_loads = {inst["id"]: 0.0 for inst in INSTANCES}
assignments = []

for item in ranking:
    i = REQUESTS.index(next(r for r in REQUESTS if r["id"] == item["request"]))
    F_i = feasible_sets[i]

    # Greedy: pick feasible k that minimizes resulting load
    best_k = F_i[0]
    best_load = instance_loads[INSTANCES[best_k]["id"]] + L[i, best_k]
    for k in F_i[1:]:
        new_load = instance_loads[INSTANCES[k]["id"]] + L[i, k]
        if new_load < best_load:
            best_load = new_load
            best_k = k

    inst = INSTANCES[best_k]
    lat = L[i, best_k]
    instance_loads[inst["id"]] += lat
    assignments.append({
        "request": REQUESTS[i]["id"],
        "desc": REQUESTS[i]["desc"],
        "assigned_to": inst["id"],
        "model": inst["model"],
        "device": inst["device"],
        "latency_ms": lat,
        "suitability": round(suitability[i, best_k], 2),
        "cumulative_load": round(instance_loads[inst["id"]]),
        "feasible_options": [INSTANCES[k]["id"] for k in F_i],
    })

makespan = max(instance_loads.values())

# ═══════════════════════════════════════════════════════════════
# Print report
# ═══════════════════════════════════════════════════════════════
print("=" * 88)
print("  QMAR DEMO — 8 requests, 5 heterogeneous instances")
print("=" * 88)

print("\n── Instances ──")
print(f"{'ID':>4s} {'Model':<22s} {'Device':<12s} {'simple':>8s} {'moderate':>8s} {'complex':>8s}")
print("-" * 68)
for inst in INSTANCES:
    lt = LATENCY_TABLE[inst["id"]]
    print(f"{inst['id']:>4s} {inst['model']:<22s} {inst['device']:<12s} "
          f"{lt['simple']:>6d}ms {lt['moderate']:>6d}ms {lt['complex']:>6d}ms")

print("\n── Requests ──")
print(f"{'ID':>4s} {'Description':<48s} {'Complexity':>10s} {'Predicted':>10s}")
print("-" * 78)
for req, at in zip(REQUESTS, predicted_answer_types):
    print(f"{req['id']:>4s} {req['desc']:<48s} {req['complexity']:>10s} {at:>10s}")

print("\n── Suitability Matrix s[i,k] (τ=0.5, bold = feasible) ──")
header = f"{'Req':>4s}"
for inst in INSTANCES:
    header += f" {inst['id']:>8s}"
print(header)
print("-" * (4 + 9 * K))
for i, req in enumerate(REQUESTS):
    row = f"{req['id']:>4s}"
    for k in range(K):
        s = suitability[i, k]
        if s >= tau:
            row += f" \033[1m{s:>7.2f}\033[0m"  # bold for feasible
        else:
            row += f" {s:>8.2f}"
    print(row)

print(f"\n── Latency Matrix L[i,k] (ms) ──")
header = f"{'Req':>4s}"
for inst in INSTANCES:
    header += f" {inst['id']:>8s}"
print(header)
print("-" * (4 + 9 * K))
for i, req in enumerate(REQUESTS):
    row = f"{req['id']:>4s}"
    for k in range(K):
        row += f" {L[i, k]:>8.0f}"
    print(row)

print(f"\n── Step 1: Feasible Sets (τ={tau}) ──")
for i, req in enumerate(REQUESTS):
    F_i = [INSTANCES[k]["id"] for k in feasible_sets[i]]
    print(f"  {req['id']}: F = {{{', '.join(F_i)}}}  (|F|={len(F_i)})")

print(f"\n── Step 2: QMAR Lexicographic Ranking ──")
print(f"  Sort key: ρ = (|F|, quality_margin, -min_latency)")
print(f"  {'Rank':>4s} {'Req':>4s} {'|F|':>4s} {'δ':>8s} {'p_min':>8s}  {'Description'}")
print(f"  {'─'*4} {'─'*4} {'─'*4} {'─'*8} {'─'*8}  {'─'*40}")
for rank, item in enumerate(ranking):
    print(f"  {rank+1:>4d} {item['request']:>4s} {item['q']:>4d} "
          f"{item['delta']:>8.3f} {item['p_min']:>8.0f}  {item['desc']}")

print(f"\n── Step 3: Greedy Assignment ──")
print(f"  {'Req':>4s} → {'Inst':>4s} {'Model':<22s} {'Device':<12s} "
      f"{'Lat':>6s} {'Suit':>6s} {'CumLoad':>8s}  {'Feasible options'}")
print(f"  {'─'*4} → {'─'*4} {'─'*22} {'─'*12} {'─'*6} {'─'*6} {'─'*8}  {'─'*30}")
for a in assignments:
    print(f"  {a['request']:>4s} → {a['assigned_to']:>4s} {a['model']:<22s} "
          f"{a['device']:<12s} {a['latency_ms']:>6.0f} {a['suitability']:>6.2f} "
          f"{a['cumulative_load']:>8.0f}  {{{', '.join(a['feasible_options'])}}}")

print(f"\n── Final Loads ──")
for inst in INSTANCES:
    load = instance_loads[inst["id"]]
    bar = "█" * int(load / makespan * 40)
    print(f"  {inst['id']:>4s} ({inst['model']:<22s}): {load:>8.0f}ms  {bar}")

print(f"\n  MAKESPAN = {makespan:.0f}ms")
print(f"  Load CV   = {np.std(list(instance_loads.values())) / np.mean(list(instance_loads.values())):.4f}")

# ── Bonus: compare with Fastest_feasible ──
print(f"\n── Baseline Comparison ──")
print(f"  {'Method':<25s} {'Makespan':>10s} {'Load CV':>10s}")
print(f"  {'─'*25} {'─'*10} {'─'*10}")

# Fastest feasible
ff_loads = {inst["id"]: 0.0 for inst in INSTANCES}
for i in range(N):
    F_i = feasible_sets[i]
    best_k = min(F_i, key=lambda k: L[i, k])
    ff_loads[INSTANCES[best_k]["id"]] += L[i, best_k]
ff_makespan = max(ff_loads.values())
ff_cv = np.std(list(ff_loads.values())) / np.mean(list(ff_loads.values()))
print(f"  {'Fastest feasible':<25s} {ff_makespan:>10.0f}ms {ff_cv:>10.4f}")

# Random feasible
np.random.seed(0)
rf_loads = {inst["id"]: 0.0 for inst in INSTANCES}
for i in range(N):
    F_i = feasible_sets[i]
    best_k = np.random.choice(F_i)
    rf_loads[INSTANCES[best_k]["id"]] += L[i, best_k]
rf_makespan = max(rf_loads.values())
rf_cv = np.std(list(rf_loads.values())) / np.mean(list(rf_loads.values()))
print(f"  {'Random feasible':<25s} {rf_makespan:>10.0f}ms {rf_cv:>10.4f}")

# Oracle (unconstrained, LPT greedy)
oracle_loads = {inst["id"]: 0.0 for inst in INSTANCES}
order = np.argsort(-L.min(axis=1))  # descending min-latency
for i in order:
    best_k = 0
    best_load = oracle_loads[INSTANCES[0]["id"]] + L[i, 0]
    for k in range(1, K):
        nl = oracle_loads[INSTANCES[k]["id"]] + L[i, k]
        if nl < best_load:
            best_load = nl
            best_k = k
    oracle_loads[INSTANCES[best_k]["id"]] += L[i, best_k]
or_makespan = max(oracle_loads.values())
or_cv = np.std(list(oracle_loads.values())) / np.mean(list(oracle_loads.values()))
print(f"  {'Oracle (unconstrained)':<25s} {or_makespan:>10.0f}ms {or_cv:>10.4f}")

print(f"  {'QMAR (this run)':<25s} {makespan:>10.0f}ms "
      f"{np.std(list(instance_loads.values())) / np.mean(list(instance_loads.values())):>10.4f}")

print()
print("─" * 88)
print("  QMAR achieves quality-constrained scheduling with near-optimal makespan.")
print("  Fastest_feasible has NO load balancing → overloads fastest instance.")
print("  Oracle ignores quality → lower bound on makespan, but no quality guarantee.")
print("─" * 88)
