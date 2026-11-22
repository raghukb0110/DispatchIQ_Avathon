from itertools import permutations

import pytest

from app.algorithms.hungarian import allocate_hungarian
from app.core.constraints import evaluate_pair
from app.core.cost import pair_cost
from app.core.state import init_states
from app.core.validation import validate_assignments
from app.data.generator import PRESETS, generate_preset
from app.models import Scenario
from tests.factories import job, tech


@pytest.mark.parametrize("preset", list(PRESETS))
def test_hungarian_respects_all_hard_constraints(preset, weights):
    sc = generate_preset(preset, seed=3)
    result = allocate_hungarian(sc, weights)
    assert validate_assignments(sc, result.assignments) == []
    assert len(result.assignments) + len(result.unassigned) == len(sc.jobs)


def test_hungarian_matches_brute_force_on_single_round(weights):
    techs = [tech(f"T{i}", lat=12.90 + 0.04 * i, lng=77.55 + 0.03 * i, max_jobs=1) for i in range(4)]
    jobs = [job(f"J{k}", lat=12.92 + 0.05 * k, lng=77.62 - 0.02 * k, we=900) for k in range(4)]
    sc = Scenario(technicians=techs, jobs=jobs)

    states = init_states(techs)
    cost = {(t.id, j.id): pair_cost(states[t.id], j, evaluate_pair(states[t.id], j), weights)[0]
            for t in techs for j in jobs}
    brute = min(sum(cost[(t.id, j.id)] for t, j in zip(techs, perm)) for perm in permutations(jobs))

    result = allocate_hungarian(sc, weights)
    assert len(result.assignments) == 4
    assert sum(a.cost for a in result.assignments) == pytest.approx(brute, abs=1e-6)


def test_batch_sacrifices_local_cost_for_global_optimum(weights):
    # T1 is cheapest for J1 but is the only HVAC tech; taking J1 greedily would strand J2.
    techs = [
        tech("T1", lat=12.980, lng=77.600, skills=("electrical", "hvac"), max_jobs=1),
        tech("T2", lat=12.940, lng=77.560, skills=("electrical",), max_jobs=1),
    ]
    jobs = [
        job("J1", lat=12.981, lng=77.601, skill="electrical", requested=480, ws=540, we=700),
        job("J2", lat=13.000, lng=77.620, skill="hvac", requested=500, ws=600, we=800),
    ]
    batch = allocate_hungarian(Scenario(technicians=techs, jobs=jobs), weights)
    assert {a.job_id: a.technician_id for a in batch.assignments} == {"J1": "T2", "J2": "T1"}
    j1 = next(a for a in batch.assignments if a.job_id == "J1")
    assert j1.explanation["rank_by_own_cost"] == 2


def test_unassigned_job_gets_reason(weights):
    sc = Scenario(technicians=[tech(skills=("plumbing",))], jobs=[job(skill="network")])
    result = allocate_hungarian(sc, weights)
    assert result.assignments == []
    assert "network" in result.unassigned[0].reason


def test_capacity_respected_across_rounds(weights):
    techs = [tech("T1", max_jobs=2)]
    jobs = [job(f"J{k}", ws=500 + 90 * k, we=1000, duration=30) for k in range(5)]
    result = allocate_hungarian(Scenario(technicians=techs, jobs=jobs), weights)
    assert len(result.assignments) == 2
    assert result.metrics["fulfillment_rate"] == pytest.approx(40.0)
