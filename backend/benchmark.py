"""Compare strategies across presets and seeds. Usage: python benchmark.py [--seeds 5]"""
import argparse
import json
from pathlib import Path

import numpy as np

from app.algorithms.hungarian import allocate_hungarian
from app.algorithms.ml_dispatcher import allocate_online
from app.data.generator import PRESETS, generate_preset
from app.ml.evaluate import agreement
from app.ml.registry import get_scorer
from app.models import Weights

COLUMNS = ["fulfillment_rate", "priority_weighted_fulfillment", "high_priority_fulfillment",
           "avg_travel_km", "avg_idle_min", "workload_std", "runtime_ms"]


def main(n_seeds: int) -> None:
    weights, scorer = Weights(), get_scorer()
    report: dict = {}
    for preset in PRESETS:
        rows: dict[str, list] = {"hungarian": [], "ml": []}
        agree = []
        for seed in range(1, n_seeds + 1):
            sc = generate_preset(preset, seed)
            h = allocate_hungarian(sc, weights)
            m = allocate_online(sc, weights, scorer)
            for r in (h, m):
                rows[r.algorithm].append({**r.metrics, "ms_per_job": r.runtime_ms / len(sc.jobs)})
            agree.append(agreement(h, m))
        report[preset] = {
            name: {k: float(np.mean([r[k] for r in rs])) for k in COLUMNS + ["ms_per_job"]}
            for name, rs in rows.items()
        }
        report[preset]["agreement"] = float(np.mean(agree))

    header = "| preset | algorithm | fulfil % | prio-wtd % | high-prio % | km/job | idle min | load std | ms |"
    print(header + "\n|" + "---|" * 9)
    for preset, res in report.items():
        for name in ("hungarian", "ml"):
            v = res[name]
            print(f"| {preset} | {name} | {v['fulfillment_rate']:.1f} | {v['priority_weighted_fulfillment']:.1f} | "
                  f"{v['high_priority_fulfillment']:.1f} | {v['avg_travel_km']:.2f} | {v['avg_idle_min']:.1f} | "
                  f"{v['workload_std']:.2f} | {v['runtime_ms']:.0f} |")
    print("\nML / Hungarian agreement:", {p: round(r["agreement"], 2) for p, r in report.items()})

    out = Path(__file__).parent / "results" / "benchmark.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5)
    main(p.parse_args().seeds)
