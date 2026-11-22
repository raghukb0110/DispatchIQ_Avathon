"""Online ML dispatch: requests are handled one at a time in arrival order and committed immediately.

A learned scorer (LambdaRank, trained to imitate the batch Hungarian solution) ranks the technicians
that already passed every hard constraint, so the model can only choose between valid options.
"""
import time

import numpy as np

from app.algorithms.common import finalize, make_assignment
from app.core.explain import rejection_summary
from app.core.geo import fmt_time
from app.core.state import commit, init_states
from app.ml.features import FEATURE_NAMES, FeatureContext, candidate_features, feasible_candidates
from app.ml.scorer import Scorer
from app.models import AllocationResult, Scenario, Weights


def allocate_online(scenario: Scenario, weights: Weights, scorer: Scorer) -> AllocationResult:
    t0 = time.perf_counter()
    states = init_states(scenario.technicians)
    ctx = FeatureContext(scenario.technicians)
    assignments = []

    for job in sorted(scenario.jobs, key=lambda j: (j.requested_at, j.id)):
        cands, rejected = feasible_candidates(states, job, weights)
        if not cands:
            continue

        X = candidate_features(job, cands, ctx)
        scores = scorer.score(X)
        pick = int(scores.argmax())
        cost_pick = int(np.argmin([c.cost for c in cands]))
        conf = scorer.confidence(scores)
        contrib = scorer.contributions(X[pick:pick + 1])[0]
        top = sorted(range(len(FEATURE_NAMES)), key=lambda i: -abs(contrib[i]))[:5]

        summary = (f"Decided at {fmt_time(job.requested_at)}: model ranked {cands[pick].state.tech.id} first "
                   f"of {len(cands)} feasible ({conf[pick]:.0%} confidence).")
        if pick != cost_pick:
            summary += (f" Overrode the cheapest option {cands[cost_pick].state.tech.id} - "
                        f"main driver: {FEATURE_NAMES[top[0]]}.")
        explanation = {
            "method": "online dispatch (ML ranker)",
            "decided_at": fmt_time(job.requested_at),
            "feasible_candidates": len(cands),
            "rejected": rejection_summary(rejected),
            "confidence": round(float(conf[pick]), 3),
            "model": scorer.kind,
            "top_factors": [
                {"feature": FEATURE_NAMES[i], "value": round(float(X[pick, i]), 2),
                 "contribution": round(float(contrib[i]), 3)}
                for i in top
            ],
            "ranking": [
                {"technician_id": cands[i].state.tech.id, "confidence": round(float(conf[i]), 3),
                 "cost": cands[i].cost}
                for i in np.argsort(-scores)[:4]
            ],
            "cost_only_choice": cands[cost_pick].state.tech.id,
            "summary": summary,
        }

        c = cands[pick]
        assignments.append(make_assignment(c.state, job, c.ev, c.cost, c.breakdown, explanation,
                                           round(float(scores[pick]), 4)))
        commit(c.state, job, c.ev.end, c.ev.travel_min)

    return finalize("ml", scenario, assignments, states, t0)
