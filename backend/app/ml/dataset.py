"""Imitation-learning dataset: replay each scenario online and label the candidate the
batch-optimal Hungarian solution picked for that job (teacher forcing)."""
from dataclasses import dataclass

import numpy as np

from app.algorithms.hungarian import allocate_hungarian
from app.core.state import commit, init_states
from app.data.generator import generate_scenario
from app.ml.features import FeatureContext, candidate_features, feasible_candidates
from app.models import Scenario, Weights


@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray
    scenario_ids: np.ndarray  # split unit - prevents leakage between correlated rows
    decision_ids: np.ndarray  # one candidate set per (scenario, job)
    coverage: float  # share of teacher assignments reproducible during online replay


def random_scenarios(n: int, base_seed: int = 10_000) -> list[Scenario]:
    rng = np.random.default_rng(base_seed)
    out = []
    for i in range(n):
        n_techs = int(rng.integers(6, 31))
        out.append(generate_scenario(
            seed=base_seed + i,
            n_techs=n_techs,
            n_jobs=int(n_techs * rng.uniform(3, 8)),
            clusters=int(rng.choice([0, 0, 2, 3, 5])),
            rare_skill_boost=float(rng.choice([0.0, 0.0, 0.3, 0.5])),
            name=f"train_{i}",
        ))
    return out


def build_dataset(scenarios: list[Scenario], weights: Weights) -> Dataset:
    X_parts, y_parts, sid_parts, did_parts = [], [], [], []
    decision, teacher_total, replayed = 0, 0, 0

    for sid, sc in enumerate(scenarios):
        teacher = {a.job_id: a.technician_id for a in allocate_hungarian(sc, weights).assignments}
        teacher_total += len(teacher)
        states = init_states(sc.technicians)
        ctx = FeatureContext(sc.technicians)
        for job in sorted(sc.jobs, key=lambda j: (j.requested_at, j.id)):
            target = teacher.get(job.id)
            if target is None:
                continue
            cands, _ = feasible_candidates(states, job, weights)
            ids = [c.state.tech.id for c in cands]
            if target not in ids:
                continue  # replay order diverged from the teacher's route; no valid label
            X = candidate_features(job, cands, ctx)
            X_parts.append(X)
            y_parts.append(np.array([tid == target for tid in ids], dtype=np.int8))
            sid_parts.append(np.full(len(ids), sid))
            did_parts.append(np.full(len(ids), decision))
            decision += 1
            replayed += 1
            c = cands[ids.index(target)]
            commit(c.state, job, c.ev.end, c.ev.travel_min)

    return Dataset(
        X=np.vstack(X_parts), y=np.concatenate(y_parts),
        scenario_ids=np.concatenate(sid_parts), decision_ids=np.concatenate(did_parts),
        coverage=replayed / teacher_total if teacher_total else 0.0,
    )
