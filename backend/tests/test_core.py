import pytest

from app.core.constraints import evaluate_pair
from app.core.cost import pair_cost
from app.core.geo import haversine_km, travel_minutes
from app.core.state import init_states
from app.data.generator import generate_scenario
from tests.factories import job, tech


def test_haversine_one_degree_latitude():
    assert haversine_km(0, 0, 1, 0) == pytest.approx(111.19, abs=0.05)
    assert haversine_km(12.97, 77.59, 12.97, 77.59) == 0


def test_travel_minutes_uses_city_speed():
    assert travel_minutes(25) == pytest.approx(60)


def _eval(t, j, jobs_done=0):
    state = init_states([t])[t.id]
    state.jobs_done = jobs_done
    return evaluate_pair(state, j)


@pytest.mark.parametrize("t,j,jobs_done,code", [
    (tech(skills=("plumbing",)), job(skill="electrical"), 0, "skill"),
    (tech(level=1), job(level=3), 0, "level"),
    (tech(max_jobs=2), job(), 2, "capacity"),
    (tech(lat=12.86, lng=77.48), job(lat=13.08, lng=77.72, ws=480, we=490), 0, "late"),
    (tech(end=600), job(ws=560, we=600, duration=90), 0, "shift"),
])
def test_hard_constraints_reject_with_reason(t, j, jobs_done, code):
    ev = _eval(t, j, jobs_done)
    assert not ev.feasible
    assert ev.code == code
    assert t.id in ev.reason


def test_feasible_pair_timing():
    ev = _eval(tech(), job(ws=540, we=660, duration=60))
    assert ev.feasible
    assert ev.arrival == pytest.approx(480 + ev.travel_min)
    assert ev.start == 540  # arrives early, waits for window
    assert ev.end == 600
    assert ev.idle_min == pytest.approx(540 - 480 - ev.travel_min)


def test_cannot_depart_before_request():
    ev = _eval(tech(), job(requested=600, ws=540, we=700))
    assert ev.arrival == pytest.approx(600 + ev.travel_min)


def test_cost_breakdown_sums_and_rewards_priority(weights):
    t, j = tech(level=3), job(level=1, priority=5)
    state = init_states([t])[t.id]
    ev = evaluate_pair(state, j)
    total, bd = pair_cost(state, j, ev, weights)
    assert total == pytest.approx(sum(bd.values()), abs=1e-2)
    assert bd["priority"] == -weights.priority * 5
    assert bd["overskill"] == weights.overskill * 2


def test_generator_is_deterministic():
    a, b = generate_scenario(seed=7), generate_scenario(seed=7)
    assert a.model_dump() == b.model_dump()
    assert generate_scenario(seed=8).model_dump() != a.model_dump()
    assert len(a.technicians) == 15 and len(a.jobs) == 60
    assert all(j.requested_at <= j.window_start < j.window_end for j in a.jobs)
