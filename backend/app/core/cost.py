"""Soft constraints: weighted cost of a feasible pair (lower is better)."""
from app.core.constraints import PairEval
from app.core.state import TechState
from app.models import Job, Weights


def pair_cost(state: TechState, job: Job, ev: PairEval, w: Weights) -> tuple[float, dict[str, float]]:
    breakdown = {
        "distance": w.distance * ev.travel_km,
        "idle": w.wait * ev.idle_min,
        "priority": -w.priority * job.priority,
        "overskill": w.overskill * (state.tech.skill_level - job.min_skill_level),
        "load": w.load * state.jobs_done,
    }
    breakdown = {k: round(v, 3) for k, v in breakdown.items()}
    return round(sum(breakdown.values()), 3), breakdown
