export interface Technician {
  id: string;
  name: string;
  lat: number;
  lng: number;
  skills: string[];
  skill_level: number;
  shift_start: number;
  shift_end: number;
  max_jobs: number;
}

export interface Job {
  id: string;
  lat: number;
  lng: number;
  required_skill: string;
  min_skill_level: number;
  requested_at: number;
  window_start: number;
  window_end: number;
  duration: number;
  priority: number;
}

export interface Scenario {
  name: string;
  seed: number | null;
  technicians: Technician[];
  jobs: Job[];
}

export interface Weights {
  distance: number;
  wait: number;
  priority: number;
  overskill: number;
  load: number;
}

export interface Factor {
  feature: string;
  value: number;
  contribution: number;
}

export interface Explanation {
  method: string;
  summary: string;
  feasible_candidates: number;
  rejected: Record<string, number>;
  round?: number;
  rank_by_own_cost?: number;
  alternatives?: { technician_id: string; cost: number }[];
  decided_at?: string;
  confidence?: number;
  top_factors?: Factor[];
  ranking?: { technician_id: string; confidence: number; cost: number }[];
  cost_only_choice?: string;
}

export interface Assignment {
  technician_id: string;
  job_id: string;
  arrival: number;
  start: number;
  end: number;
  travel_km: number;
  wait_min: number;
  cost: number;
  cost_breakdown: Record<string, number>;
  score: number | null;
  explanation: Explanation;
}

export interface AllocationResult {
  algorithm: string;
  assignments: Assignment[];
  unassigned: { job_id: string; reason: string }[];
  metrics: Record<string, number>;
  runtime_ms: number;
}

export interface CompareResponse {
  results: Record<string, AllocationResult>;
  summary: {
    agreement: number;
    fulfillment_gap: number;
    priority_fulfillment_gap: number;
    travel_km_gap: number;
    speedup: number | null;
  };
}

export interface Preset {
  n_techs: number;
  n_jobs: number;
  clusters: number;
  rare_skill_boost: number;
  description: string;
}

export interface GenerateParams {
  preset?: string;
  seed: number;
  n_techs?: number;
  n_jobs?: number;
  clusters?: number;
  rare_skill_boost?: number;
}

export type ModelCard = Record<string, any>;
