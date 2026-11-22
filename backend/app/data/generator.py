"""Seeded synthetic scenarios over Bengaluru."""
import numpy as np

from app.models import SKILLS, Job, Scenario, Technician

BBOX = (12.86, 13.08, 77.48, 77.72)  # lat_min, lat_max, lng_min, lng_max
SKILL_P = np.array([0.30, 0.30, 0.15, 0.15, 0.10])
RARE_SKILL_P = np.array([0.0, 0.0, 0.5, 0.0, 0.5])  # hvac + network

PRESETS: dict[str, dict] = {
    "balanced": {"n_techs": 15, "n_jobs": 60, "clusters": 0, "rare_skill_boost": 0.0,
                 "description": "Enough technicians for the demand, uniform job locations"},
    "scarce": {"n_techs": 10, "n_jobs": 90, "clusters": 0, "rare_skill_boost": 0.0,
               "description": "Demand far exceeds capacity - contention for technicians"},
    "clustered": {"n_techs": 15, "n_jobs": 80, "clusters": 4, "rare_skill_boost": 0.0,
                  "description": "Jobs concentrated in a few hotspots, technicians spread out"},
    "rare_skills": {"n_techs": 15, "n_jobs": 70, "clusters": 0, "rare_skill_boost": 0.5,
                    "description": "Many jobs need skills few technicians have (hvac, network)"},
    "large": {"n_techs": 80, "n_jobs": 600, "clusters": 3, "rare_skill_boost": 0.2,
              "description": "City-scale day for runtime comparison"},
}


def _points(rng: np.random.Generator, n: int, clusters: int) -> np.ndarray:
    lat0, lat1, lng0, lng1 = BBOX
    if clusters > 0:
        centers = np.column_stack([
            rng.uniform(lat0 + 0.03, lat1 - 0.03, clusters),
            rng.uniform(lng0 + 0.03, lng1 - 0.03, clusters),
        ])
        pts = centers[rng.integers(0, clusters, n)] + rng.normal(0, 0.012, (n, 2))
    else:
        pts = np.column_stack([rng.uniform(lat0, lat1, n), rng.uniform(lng0, lng1, n)])
    pts[:, 0] = pts[:, 0].clip(lat0, lat1)
    pts[:, 1] = pts[:, 1].clip(lng0, lng1)
    return pts


def generate_scenario(
    seed: int = 42, n_techs: int = 15, n_jobs: int = 60, clusters: int = 0,
    rare_skill_boost: float = 0.0, name: str = "custom",
) -> Scenario:
    rng = np.random.default_rng(seed)

    techs = []
    for i, (lat, lng) in enumerate(_points(rng, n_techs, 0)):
        n_skills = int(rng.integers(1, 4))
        skills = rng.choice(SKILLS, size=n_skills, replace=False, p=SKILL_P)
        shift_start = int(rng.choice([480, 540, 600]))
        techs.append(Technician(
            id=f"T{i:02d}", name=f"Technician {i:02d}", lat=float(lat), lng=float(lng),
            skills=[str(s) for s in skills],
            skill_level=int(rng.choice([1, 2, 3], p=[0.4, 0.4, 0.2])),
            shift_start=shift_start, shift_end=shift_start + 540, max_jobs=6,
        ))

    job_p = (1 - rare_skill_boost) * SKILL_P + rare_skill_boost * RARE_SKILL_P
    jobs = []
    for i, (lat, lng) in enumerate(_points(rng, n_jobs, clusters)):
        window_start = int(rng.integers(34, 69)) * 15  # 08:30 - 17:00
        jobs.append(Job(
            id=f"J{i:03d}", lat=float(lat), lng=float(lng),
            required_skill=str(rng.choice(SKILLS, p=job_p)),
            min_skill_level=int(rng.choice([1, 2, 3], p=[0.5, 0.35, 0.15])),
            requested_at=max(420, window_start - int(rng.integers(4, 13)) * 15),
            window_start=window_start,
            window_end=window_start + int(rng.choice([60, 120, 180])),
            duration=int(rng.integers(2, 9)) * 15,
            priority=int(rng.choice([1, 2, 3, 4, 5], p=[0.3, 0.3, 0.2, 0.15, 0.05])),
        ))
    return Scenario(name=name, seed=seed, technicians=techs, jobs=jobs)


def generate_preset(preset: str, seed: int = 42) -> Scenario:
    cfg = {k: v for k, v in PRESETS[preset].items() if k != "description"}
    return generate_scenario(seed=seed, name=preset, **cfg)
