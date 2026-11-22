"""Domain schemas. All times are minutes since midnight."""
from typing import Literal

from pydantic import BaseModel, Field

SKILLS = ["electrical", "plumbing", "hvac", "appliance", "network"]


class Technician(BaseModel):
    id: str
    name: str
    lat: float
    lng: float
    skills: list[str]
    skill_level: int = Field(ge=1, le=3)
    shift_start: int
    shift_end: int
    max_jobs: int = Field(ge=1)


class Job(BaseModel):
    id: str
    lat: float
    lng: float
    required_skill: str
    min_skill_level: int = Field(ge=1, le=3)
    requested_at: int  # when the request reaches dispatch (drives online order)
    window_start: int
    window_end: int  # latest allowed start time
    duration: int
    priority: int = Field(ge=1, le=5)


class Scenario(BaseModel):
    name: str = "custom"
    seed: int | None = None
    technicians: list[Technician]
    jobs: list[Job]


class Weights(BaseModel):
    distance: float = 1.0  # per km
    wait: float = 0.1  # per idle minute
    priority: float = 4.0  # bonus per priority point
    overskill: float = 2.0  # per level above requirement
    load: float = 1.5  # per job already on the technician


class Assignment(BaseModel):
    technician_id: str
    job_id: str
    arrival: float
    start: float
    end: float
    travel_km: float
    wait_min: float
    cost: float
    cost_breakdown: dict[str, float]
    score: float | None = None  # model probability for the ML dispatcher
    explanation: dict


class UnassignedJob(BaseModel):
    job_id: str
    reason: str


class AllocationResult(BaseModel):
    algorithm: str
    assignments: list[Assignment]
    unassigned: list[UnassignedJob]
    metrics: dict[str, float]
    runtime_ms: float


AlgorithmName = Literal["hungarian", "ml"]


class GenerateRequest(BaseModel):
    preset: str | None = None
    seed: int = 42
    n_techs: int = Field(default=15, ge=1, le=500)
    n_jobs: int = Field(default=60, ge=1, le=3000)
    clusters: int = Field(default=0, ge=0, le=20)
    rare_skill_boost: float = Field(default=0.0, ge=0.0, le=1.0)


class AllocateRequest(BaseModel):
    scenario: Scenario
    algorithm: AlgorithmName = "hungarian"
    weights: Weights = Weights()


class CompareRequest(BaseModel):
    scenario: Scenario
    weights: Weights = Weights()
