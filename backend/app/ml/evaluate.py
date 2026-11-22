"""Evaluation helpers shared by training and the benchmark script."""
import numpy as np
from sklearn.metrics import roc_auc_score

from app.algorithms.hungarian import allocate_hungarian
from app.algorithms.ml_dispatcher import allocate_online
from app.ml.features import FEATURE_NAMES
from app.ml.scorer import Scorer
from app.models import AllocationResult, Scenario, Weights

COST_RANK = FEATURE_NAMES.index("cost_rank")
N_FEASIBLE = FEATURE_NAMES.index("n_feasible")
KEY_METRICS = [
    "fulfillment_rate", "priority_weighted_fulfillment", "high_priority_fulfillment",
    "avg_travel_km", "avg_idle_min", "utilization", "workload_std", "runtime_ms",
]


def _segments(decision_ids: np.ndarray) -> list[tuple[int, int]]:
    cuts = np.flatnonzero(np.diff(decision_ids)) + 1
    starts = np.concatenate([[0], cuts])
    ends = np.concatenate([cuts, [len(decision_ids)]])
    return list(zip(starts, ends))


def top1_accuracy(scores: np.ndarray, y: np.ndarray, decision_ids: np.ndarray, X: np.ndarray) -> float:
    """Share of non-trivial decisions (>=2 candidates) where the top-scored candidate is the teacher's pick."""
    hits, total = 0, 0
    for s, e in _segments(decision_ids):
        if X[s, N_FEASIBLE] < 2:
            continue
        total += 1
        hits += int(y[s + int(np.argmax(scores[s:e]))] == 1)
    return hits / total if total else 0.0


def ranking_metrics(scores: np.ndarray, y: np.ndarray, decision_ids: np.ndarray, X: np.ndarray) -> dict:
    return {
        "auc": float(roc_auc_score(y, scores)),
        "top1_accuracy": top1_accuracy(scores, y, decision_ids, X),
        "cost_baseline_top1": top1_accuracy(-X[:, COST_RANK], y, decision_ids, X),
    }


def agreement(a: AllocationResult, b: AllocationResult) -> float:
    pa = {(x.job_id, x.technician_id) for x in a.assignments}
    pb = {(x.job_id, x.technician_id) for x in b.assignments}
    return len(pa & pb) / len(pa) if pa else 0.0


def business_eval(scenarios: list[Scenario], weights: Weights, scorer: Scorer) -> dict:
    """Mean outcome metrics of each strategy across scenarios."""
    rows: dict[str, list[dict]] = {"hungarian": [], "ml": []}
    agree = []
    for sc in scenarios:
        h = allocate_hungarian(sc, weights)
        m = allocate_online(sc, weights, scorer)
        rows["hungarian"].append(h.metrics)
        rows["ml"].append(m.metrics)
        agree.append(agreement(h, m))
    out = {name: {k: float(np.mean([r[k] for r in rs])) for k in KEY_METRICS} for name, rs in rows.items()}
    out["ml_agreement_with_hungarian"] = float(np.mean(agree))
    return out
