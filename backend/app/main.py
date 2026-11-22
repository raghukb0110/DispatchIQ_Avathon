from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.algorithms.hungarian import allocate_hungarian
from app.algorithms.ml_dispatcher import allocate_online
from app.data.generator import PRESETS, generate_preset, generate_scenario
from app.ml.evaluate import agreement
from app.ml.registry import get_scorer, model_card
from app.models import (
    AllocateRequest, AllocationResult, CompareRequest, GenerateRequest, Scenario, Weights,
)

app = FastAPI(title="Resource Allocation Engine", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def run(name: str, scenario: Scenario, weights: Weights) -> AllocationResult:
    if name == "hungarian":
        return allocate_hungarian(scenario, weights)
    if name == "ml":
        return allocate_online(scenario, weights, get_scorer())
    raise HTTPException(400, f"Unknown algorithm '{name}'")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/presets")
def presets() -> dict:
    return PRESETS


@app.post("/api/scenario/generate", response_model=Scenario)
def generate(req: GenerateRequest) -> Scenario:
    if req.preset:
        if req.preset not in PRESETS:
            raise HTTPException(404, f"Unknown preset '{req.preset}'")
        return generate_preset(req.preset, req.seed)
    return generate_scenario(req.seed, req.n_techs, req.n_jobs, req.clusters, req.rare_skill_boost)


@app.post("/api/allocate", response_model=AllocationResult)
def allocate(req: AllocateRequest) -> AllocationResult:
    return run(req.algorithm, req.scenario, req.weights)


@app.post("/api/compare")
def compare(req: CompareRequest) -> dict:
    results = {n: run(n, req.scenario, req.weights) for n in ("hungarian", "ml")}
    h, m = results["hungarian"].metrics, results["ml"].metrics
    return {
        "results": results,
        "summary": {
            "agreement": agreement(results["hungarian"], results["ml"]),
            "fulfillment_gap": m["fulfillment_rate"] - h["fulfillment_rate"],
            "priority_fulfillment_gap": m["priority_weighted_fulfillment"] - h["priority_weighted_fulfillment"],
            "travel_km_gap": m["avg_travel_km"] - h["avg_travel_km"],
            "speedup": h["runtime_ms"] / m["runtime_ms"] if m["runtime_ms"] else None,
        },
    }


@app.get("/api/model")
def model() -> dict:
    return model_card()
