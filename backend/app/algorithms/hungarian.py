"""Batch allocation: Hungarian (Kuhn-Munkres) solved in rounds.

Each round optimally pairs every technician with at most one pending job given the
technicians' current position/time; committed jobs extend their routes and the next
round re-solves for the remaining jobs. Rounds stop when no feasible pair remains.
"""
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

from app.algorithms.common import finalize, make_assignment
from app.core.constraints import evaluate_pair
from app.core.cost import pair_cost
from app.core.explain import rejection_summary
from app.core.state import commit, init_states
from app.models import AllocationResult, Scenario, Weights

INFEASIBLE = 1e9


def allocate_hungarian(scenario: Scenario, weights: Weights) -> AllocationResult:
    t0 = time.perf_counter()
    states = init_states(scenario.technicians)
    pending = list(scenario.jobs)
    assignments = []
    round_no = 0

    while pending:
        round_no += 1
        techs = [s for s in states.values() if s.jobs_done < s.tech.max_jobs]
        if not techs:
            break

        cost = np.full((len(techs), len(pending)), INFEASIBLE)
        feasible: dict[tuple[int, int], tuple] = {}
        rejections: dict[int, list] = {k: [] for k in range(len(pending))}
        for i, s in enumerate(techs):
            for k, job in enumerate(pending):
                ev = evaluate_pair(s, job)
                if ev.feasible:
                    c, bd = pair_cost(s, job, ev, weights)
                    cost[i, k] = c
                    feasible[(i, k)] = (ev, c, bd)
                else:
                    rejections[k].append(ev)
        if not feasible:
            break

        rows, cols = linear_sum_assignment(cost)
        chosen = [(i, k) for i, k in zip(rows, cols) if (i, k) in feasible]
        if not chosen:
            break

        for i, k in chosen:
            state, job = techs[i], pending[k]
            ev, c, bd = feasible[(i, k)]
            options = sorted((feasible[(r, k)][1], techs[r].tech.id) for r in range(len(techs)) if (r, k) in feasible)
            rank = [tid for _, tid in options].index(state.tech.id) + 1
            explanation = {
                "method": "batch optimisation (Hungarian)",
                "round": round_no,
                "feasible_candidates": len(options),
                "rank_by_own_cost": rank,
                "alternatives": [{"technician_id": tid, "cost": oc} for oc, tid in options[:4] if tid != state.tech.id][:3],
                "rejected": rejection_summary(rejections[k]),
                "summary": _summary(round_no, state.tech.id, c, len(options), rank, options),
            }
            assignments.append(make_assignment(state, job, ev, c, bd, explanation))
            commit(state, job, ev.end, ev.travel_min)

        done = {pending[k].id for _, k in chosen}
        pending = [j for j in pending if j.id not in done]

    return finalize("hungarian", scenario, assignments, states, t0)


def _summary(round_no: int, tid: str, cost: float, n: int, rank: int, options: list) -> str:
    text = f"Round {round_no}: {tid} chosen (cost {cost:.1f}) from {n} feasible technician(s)."
    if rank > 1:
        text += (f" Not the cheapest for this job ({options[0][1]} costs {options[0][0]:.1f}) - "
                 "the global optimum used that technician elsewhere.")
    return text
