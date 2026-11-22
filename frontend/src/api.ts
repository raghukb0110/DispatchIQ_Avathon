import type { CompareResponse, GenerateParams, ModelCard, Preset, Scenario, Weights } from "./types";

async function request<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(`/api${path}`, {
    method: body === undefined ? "GET" : "POST",
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
  return res.json() as Promise<T>;
}

export const api = {
  presets: () => request<Record<string, Preset>>("/presets"),
  generate: (params: GenerateParams) => request<Scenario>("/scenario/generate", params),
  compare: (scenario: Scenario, weights: Weights) => request<CompareResponse>("/compare", { scenario, weights }),
  model: () => request<ModelCard>("/model"),
};
