import numpy as np
import pytest

from app.algorithms.ml_dispatcher import allocate_online
from app.core.state import init_states
from app.core.validation import validate_assignments
from app.data.generator import PRESETS, generate_preset
from app.ml.dataset import build_dataset, random_scenarios
from app.ml.evaluate import _segments, ranking_metrics
from app.ml.features import FEATURE_NAMES, FeatureContext, candidate_features, feasible_candidates
from tests.factories import job, tech


def test_feature_matrix_shape_and_cost_rank(weights):
    techs = [tech(f"T{i}", lat=12.90 + 0.03 * i) for i in range(3)]
    j = job()
    cands, _ = feasible_candidates(init_states(techs), j, weights)
    X = candidate_features(j, cands, FeatureContext(techs))
    assert X.shape == (3, len(FEATURE_NAMES))
    rank = X[:, FEATURE_NAMES.index("cost_rank")]
    costs = [c.cost for c in cands]
    assert rank[int(np.argmin(costs))] == 0
    assert sorted(rank) == [0, 1, 2]


def test_dataset_has_one_label_per_decision(weights):
    ds = build_dataset(random_scenarios(3, base_seed=900), weights)
    for s, e in _segments(ds.decision_ids):
        assert ds.y[s:e].sum() == 1
        assert len(set(ds.scenario_ids[s:e])) == 1
    assert 0.5 < ds.coverage <= 1.0


def test_model_beats_cost_ranking_on_unseen_scenarios(small_scorer, weights):
    ds = build_dataset(random_scenarios(10, base_seed=7000), weights)
    m = ranking_metrics(small_scorer.score(ds.X), ds.y, ds.decision_ids, ds.X)
    assert m["top1_accuracy"] > m["cost_baseline_top1"]


@pytest.mark.parametrize("preset", [p for p in PRESETS if p != "large"])
def test_ml_dispatcher_respects_hard_constraints(preset, small_scorer, weights):
    sc = generate_preset(preset, seed=11)
    result = allocate_online(sc, weights, small_scorer)
    assert result.algorithm == "ml"
    assert validate_assignments(sc, result.assignments) == []


def test_ml_explanations(small_scorer, weights):
    result = allocate_online(generate_preset("balanced"), weights, small_scorer)
    exp = result.assignments[0].explanation
    assert len(exp["top_factors"]) == 5
    assert {f["feature"] for f in exp["top_factors"]} <= set(FEATURE_NAMES)
    assert 0 < exp["confidence"] <= 1
    assert exp["summary"]


def test_online_dispatch_follows_request_order(small_scorer, weights):
    sc = generate_preset("balanced")
    result = allocate_online(sc, weights, small_scorer)
    requested = {j.id: j.requested_at for j in sc.jobs}
    decided = [requested[a.job_id] for a in result.assignments]
    assert decided == sorted(decided)
