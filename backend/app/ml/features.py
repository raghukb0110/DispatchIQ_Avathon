"""Pair features shared by dataset building (training) and the online dispatcher (inference)."""
from dataclasses import dataclass

import numpy as np

from app.core.constraints import PairEval, evaluate_pair
from app.core.cost import pair_cost
from app.core.state import TechState
from app.models import SKILLS, Job, Technician, Weights

FEATURE_NAMES = [
    "travel_km",
    "idle_min",
    "slack_min",
    "overskill",
    "priority",
    "tech_jobs_done",
    "tech_capacity_left",
    "tech_shift_left_min",
    "n_feasible",
    "cost_rank",
    "cost_gap_to_best",
    "cost",
    "job_skill_supply",
    "tech_n_skills",
    "tech_rarity",
    "tech_skill_level",
    "job_duration",
    "window_len",
    "start_hour",
]


@dataclass
class Candidate:
    state: TechState
    ev: PairEval
    cost: float
    breakdown: dict[str, float]


class FeatureContext:
    """Static, scenario-level facts known before any request arrives."""

    def __init__(self, techs: list[Technician]):
        n = max(len(techs), 1)
        self.supply = {
            (s, lvl): sum(1 for t in techs if s in t.skills and t.skill_level >= lvl) / n
            for s in SKILLS for lvl in (1, 2, 3)
        }
        skill_frac = {s: sum(1 for t in techs if s in t.skills) / n for s in SKILLS}
        # High when a technician holds skills few others have - worth saving for jobs only they can do.
        self.tech_rarity = {t.id: sum(1 - skill_frac.get(s, 0.0) for s in t.skills) for t in techs}


def feasible_candidates(
    states: dict[str, TechState], job: Job, weights: Weights
) -> tuple[list[Candidate], list[PairEval]]:
    cands, rejected = [], []
    for s in states.values():
        ev = evaluate_pair(s, job)
        if ev.feasible:
            c, bd = pair_cost(s, job, ev, weights)
            cands.append(Candidate(s, ev, c, bd))
        else:
            rejected.append(ev)
    return cands, rejected


def candidate_features(job: Job, cands: list[Candidate], ctx: FeatureContext) -> np.ndarray:
    costs = np.array([c.cost for c in cands])
    ranks = costs.argsort(kind="stable").argsort()
    best = costs.min()
    rows = []
    for c, rank in zip(cands, ranks):
        s, t, ev = c.state, c.state.tech, c.ev
        rows.append([
            ev.travel_km,
            ev.idle_min,
            job.window_end - ev.start,
            t.skill_level - job.min_skill_level,
            job.priority,
            s.jobs_done,
            t.max_jobs - s.jobs_done,
            t.shift_end - ev.end,
            len(cands),
            rank,
            c.cost - best,
            c.cost,
            ctx.supply.get((job.required_skill, job.min_skill_level), 0.0),
            len(t.skills),
            ctx.tech_rarity.get(t.id, 0.0),
            t.skill_level,
            job.duration,
            job.window_end - job.window_start,
            ev.start / 60.0,
        ])
    return np.asarray(rows, dtype=np.float64)
