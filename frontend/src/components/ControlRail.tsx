import type { GenerateParams, Preset, Weights } from "../types";

interface Props {
  presets: Record<string, Preset>;
  params: GenerateParams;
  setParams: (p: GenerateParams) => void;
  weights: Weights;
  setWeights: (w: Weights) => void;
  defaultWeights: Weights;
  onGenerate: () => void;
  onRerun: () => void;
  busy: boolean;
  canRerun: boolean;
}

const PRESET_ICONS: Record<string, string> = {
  balanced: "⚖️",
  scarce: "🔥",
  clustered: "📍",
  rare_skills: "🧩",
  large: "🏙️",
};

const WEIGHT_FIELDS: { key: keyof Weights; label: string; max: number; step: number; hint: string }[] = [
  { key: "distance", label: "Distance", max: 5, step: 0.1, hint: "cost per km driven" },
  { key: "wait", label: "Idle time", max: 1, step: 0.01, hint: "cost per idle minute" },
  { key: "priority", label: "Priority bonus", max: 15, step: 0.5, hint: "reward per priority point" },
  { key: "overskill", label: "Over-qualification", max: 10, step: 0.5, hint: "senior tech on simple job" },
  { key: "load", label: "Workload balance", max: 10, step: 0.5, hint: "cost per job already done" },
];

export default function ControlRail(p: Props) {
  const custom = !p.params.preset;
  const setNum = (key: keyof GenerateParams, v: string) => p.setParams({ ...p.params, [key]: Number(v) });
  const weightsChanged = WEIGHT_FIELDS.some((f) => p.weights[f.key] !== p.defaultWeights[f.key]);

  return (
    <aside className="rail">
      <section>
        <div className="rail-title">
          <span className="step">1</span> Pick a day
        </div>
        <div className="preset-grid">
          {Object.entries(p.presets).map(([name, cfg]) => (
            <button
              key={name}
              className={`preset ${p.params.preset === name ? "selected" : ""}`}
              onClick={() => p.setParams({ ...p.params, preset: name })}
              title={cfg.description}
            >
              <span className="preset-icon">{PRESET_ICONS[name] ?? "•"}</span>
              <span className="preset-name">{name.replace("_", " ")}</span>
              <span className="preset-stats">
                {cfg.n_techs} techs · {cfg.n_jobs} jobs
              </span>
            </button>
          ))}
          <button
            className={`preset ${custom ? "selected" : ""}`}
            onClick={() => p.setParams({ ...p.params, preset: undefined })}
          >
            <span className="preset-icon">🛠️</span>
            <span className="preset-name">custom</span>
            <span className="preset-stats">your own mix</span>
          </button>
        </div>
        {p.params.preset && <p className="hint">{p.presets[p.params.preset]?.description}</p>}

        {custom && (
          <div className="custom-grid">
            <label>
              Technicians
              <input type="number" min={1} max={500} value={p.params.n_techs ?? 15} onChange={(e) => setNum("n_techs", e.target.value)} />
            </label>
            <label>
              Jobs
              <input type="number" min={1} max={3000} value={p.params.n_jobs ?? 60} onChange={(e) => setNum("n_jobs", e.target.value)} />
            </label>
            <label>
              Hotspots
              <input type="number" min={0} max={20} value={p.params.clusters ?? 0} onChange={(e) => setNum("clusters", e.target.value)} />
            </label>
            <label>
              Rare-skill demand {(p.params.rare_skill_boost ?? 0).toFixed(2)}
              <input type="range" min={0} max={1} step={0.05} value={p.params.rare_skill_boost ?? 0} onChange={(e) => setNum("rare_skill_boost", e.target.value)} />
            </label>
          </div>
        )}

        <div className="seed-row">
          <label>
            Seed
            <input type="number" value={p.params.seed} onChange={(e) => setNum("seed", e.target.value)} />
          </label>
          <button
            className="icon-btn"
            title="Random seed"
            onClick={() => p.setParams({ ...p.params, seed: Math.floor(Math.random() * 10_000) })}
          >
            🎲
          </button>
        </div>
        <button className="cta" onClick={p.onGenerate} disabled={p.busy}>
          {p.busy ? "Dispatching…" : "Generate & dispatch"}
        </button>
      </section>

      <section>
        <div className="rail-title">
          <span className="step">2</span> Tune the cost function
        </div>
        {WEIGHT_FIELDS.map((f) => (
          <label key={f.key} className="slider">
            <span className="slider-head">
              <span>{f.label}</span>
              <b>{p.weights[f.key]}</b>
            </span>
            <input
              type="range"
              min={0}
              max={f.max}
              step={f.step}
              value={p.weights[f.key]}
              onChange={(e) => p.setWeights({ ...p.weights, [f.key]: Number(e.target.value) })}
            />
            <span className="hint">{f.hint}</span>
          </label>
        ))}
        <div className="row">
          <button className="ghost" onClick={() => p.setWeights(p.defaultWeights)} disabled={!weightsChanged}>
            Reset
          </button>
          <button className="ghost strong" onClick={p.onRerun} disabled={p.busy || !p.canRerun}>
            Re-run same day
          </button>
        </div>
        {weightsChanged && (
          <p className="hint warn">The ML model was trained with the default weights; custom weights change what “optimal” means.</p>
        )}
      </section>
    </aside>
  );
}
