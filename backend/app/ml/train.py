"""Train the online dispatcher's scorer by imitating Hungarian (batch-optimal) decisions.

Usage: python -m app.ml.train [--n-scenarios 200] [--seed 10000]
"""
import argparse
import json
import time
from datetime import datetime, timezone

import joblib
import numpy as np
from lightgbm import LGBMClassifier, LGBMRanker
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.ml.dataset import Dataset, build_dataset, random_scenarios
from app.ml.evaluate import business_eval, ranking_metrics
from app.ml.features import FEATURE_NAMES
from app.ml.registry import CARD_PATH, MODEL_PATH
from app.ml.scorer import Scorer
from app.models import Weights

MODEL_KINDS = ["logreg", "lgbm_classifier", "lgbm_ranker"]


def make_model(kind: str):
    if kind == "logreg":
        return Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))])
    if kind == "lgbm_classifier":
        return LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.8, random_state=0, verbose=-1)
    if kind == "lgbm_ranker":
        return LGBMRanker(objective="lambdarank", n_estimators=300, learning_rate=0.05, num_leaves=31,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0, verbose=-1)
    raise ValueError(kind)


def fit(kind: str, X: np.ndarray, y: np.ndarray, decision_ids: np.ndarray) -> Scorer:
    model = make_model(kind)
    if kind == "lgbm_ranker":
        # decision_ids are contiguous and increasing, so unique() counts follow row order
        _, group_sizes = np.unique(decision_ids, return_counts=True)
        model.fit(X, y, group=group_sizes)
    else:
        model.fit(X, y)
    return Scorer(kind, model, FEATURE_NAMES)


def _subset(ds: Dataset, mask: np.ndarray) -> tuple:
    return ds.X[mask], ds.y[mask], ds.decision_ids[mask]


def train(n_scenarios: int = 200, seed: int = 10_000, verbose: bool = True) -> dict:
    log = print if verbose else (lambda *a, **k: None)
    t0 = time.perf_counter()
    weights = Weights()

    scenarios = random_scenarios(n_scenarios, seed)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_scenarios)
    n_test = max(1, n_scenarios // 5)
    test_sids, train_sids = set(order[:n_test].tolist()), set(order[n_test:].tolist())

    log(f"Building dataset from {n_scenarios} scenarios (Hungarian teacher)...")
    ds = build_dataset(scenarios, weights)
    is_test = np.isin(ds.scenario_ids, list(test_sids))
    assert not set(ds.scenario_ids[is_test]) & set(ds.scenario_ids[~is_test]), "scenario leakage"
    log(f"  rows={len(ds.y)} decisions={len(np.unique(ds.decision_ids))} "
        f"positives={int(ds.y.sum())} teacher_coverage={ds.coverage:.1%}")

    X_tr, y_tr, d_tr = _subset(ds, ~is_test)
    g_tr = ds.scenario_ids[~is_test]
    cv: dict[str, dict] = {}
    for kind in MODEL_KINDS:
        folds = []
        for tr, va in GroupKFold(n_splits=3).split(X_tr, y_tr, g_tr):
            scorer = fit(kind, X_tr[tr], y_tr[tr], d_tr[tr])
            folds.append(ranking_metrics(scorer.score(X_tr[va]), y_tr[va], d_tr[va], X_tr[va]))
        cv[kind] = {k: float(np.mean([f[k] for f in folds])) for k in folds[0]}
        log(f"  CV {kind:16s} auc={cv[kind]['auc']:.3f} top1={cv[kind]['top1_accuracy']:.3f}")

    best = max(MODEL_KINDS, key=lambda k: cv[k]["top1_accuracy"])
    scorer = fit(best, X_tr, y_tr, d_tr)
    X_te, y_te, d_te = _subset(ds, is_test)
    test_metrics = ranking_metrics(scorer.score(X_te), y_te, d_te, X_te)
    log(f"Selected {best}: test auc={test_metrics['auc']:.3f} top1={test_metrics['top1_accuracy']:.3f} "
        f"(cost-only baseline top1={test_metrics['cost_baseline_top1']:.3f})")

    log("Business evaluation on held-out scenarios...")
    business = business_eval([scenarios[i] for i in sorted(test_sids)], weights, scorer)

    card = {
        "model_kind": best,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "training_seconds": round(time.perf_counter() - t0, 1),
        "data": {
            "n_scenarios": n_scenarios, "seed": seed, "train_scenarios": len(train_sids),
            "test_scenarios": len(test_sids), "rows": int(len(ds.y)),
            "decisions": int(len(np.unique(ds.decision_ids))), "teacher_coverage": ds.coverage,
            "split": "by scenario (no row-level leakage)",
        },
        "cv": cv,
        "test": test_metrics,
        "business_eval_holdout": business,
        "feature_importance": scorer.feature_importance(),
        "features": FEATURE_NAMES,
        "intended_use": "Rank feasible technicians for one request at a time (online dispatch). "
                        "Hard constraints are enforced before scoring; the model never sees infeasible pairs.",
        "limitations": "Trained on synthetic scenarios; quality depends on how closely the generator "
                       "matches real demand. Teacher is round-based Hungarian, not a full VRP optimum.",
    }
    scorer.meta = card
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scorer, MODEL_PATH)
    CARD_PATH.write_text(json.dumps(card, indent=2))
    log(f"Saved {MODEL_PATH} ({card['training_seconds']}s)")
    return card


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-scenarios", type=int, default=200)
    p.add_argument("--seed", type=int, default=10_000)
    a = p.parse_args()
    train(a.n_scenarios, a.seed)
