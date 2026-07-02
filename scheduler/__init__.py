"""QMAR scheduler module."""
from .latency_estimator import LatencyEstimator
from .qmar import QMARScheduler, qmar_schedule
from .baselines import (
    RandomFeasibleBaseline,
    FastestFeasibleBaseline,
    HighestSuitabilityBaseline,
    LatencyOnlyGreedyBaseline,
    QMARWithoutAnswerTypeBaseline,
    OracleBaseline,
)
