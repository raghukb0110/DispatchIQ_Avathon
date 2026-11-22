import { useEffect, useRef, useState } from "react";
import { api } from "./api";
import ControlRail from "./components/ControlRail";
import JobDrawer from "./components/JobDrawer";
import MapView, { type MapGroup } from "./components/MapView";
import ModelLab from "./components/ModelLab";
import ScheduleView from "./components/ScheduleView";
import Scoreboard from "./components/Scoreboard";
import Timeline from "./components/Timeline";
import type { CompareResponse, GenerateParams, ModelCard, Preset, Scenario, Weights } from "./types";
import { ALGO_META, ALGOS, DAY_END } from "./utils";

const DEFAULT_WEIGHTS: Weights = { distance: 1, wait: 0.1, priority: 4, overskill: 2, load: 1.5 };
type Tab = "map" | "schedule" | "score" | "model";
const TABS: { id: Tab; label: string; icon: string }[] = [
  { id: "map", label: "Live map", icon: "🗺️" },
  { id: "schedule", label: "Schedules", icon: "📅" },
  { id: "score", label: "Scoreboard", icon: "🏁" },
  { id: "model", label: "Model lab", icon: "🧠" },
];

export default function App() {
  const [presets, setPresets] = useState<Record<string, Preset>>({});
  const [params, setParams] = useState<GenerateParams>({ preset: "balanced", seed: 42 });
  const [weights, setWeights] = useState<Weights>(DEFAULT_WEIGHTS);
  const [scenario, setScenario] = useState<Scenario | null>(null);
  const [comparison, setComparison] = useState<CompareResponse | null>(null);
  const [card, setCard] = useState<ModelCard | null>(null);
  const [selectedJob, setSelectedJob] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("map");
  const [time, setTime] = useState(DAY_END);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(3);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const mapGroup = useRef<MapGroup>({ maps: new Set(), syncing: false }).current;

  const guard = async (fn: () => Promise<void>) => {
    setBusy(true);
    setError(null);
    try {
      await fn();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const dispatch = (sc: Scenario) =>
    guard(async () => {
      setComparison(await api.compare(sc, weights));
    });

  const generate = () =>
    guard(async () => {
      const sc = await api.generate(params);
      setScenario(sc);
      setSelectedJob(null);
      setComparison(null);
      setComparison(await api.compare(sc, weights));
      setPlaying(false);
      setTime(DAY_END);
    });

  useEffect(() => {
    api.presets().then(setPresets).catch((e) => setError(String(e)));
    api.model().then(setCard).catch(() => setCard(null));
    generate();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!playing) return;
    const id = window.setInterval(() => {
      setTime((t) => {
        const next = t + 2 * speed;
        if (next >= DAY_END) {
          setPlaying(false);
          return DAY_END;
        }
        return next;
      });
    }, 60);
    return () => window.clearInterval(id);
  }, [playing, speed]);

  const h = comparison?.results.hungarian.metrics;
  const m = comparison?.results.ml.metrics;

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <div className="logo">◆</div>
          <div>
            <div className="brand-name">Dispatch<span>IQ</span></div>
            <div className="brand-sub">Resource Allocation Engine · field-service technicians → repair jobs · Bengaluru</div>
          </div>
        </div>
        <div className="versus">
          <span className="vs-side" style={{ color: ALGO_META.hungarian.color }}>
            {ALGO_META.hungarian.label} <small>batch optimiser</small>
          </span>
          <span className="vs">VS</span>
          <span className="vs-side" style={{ color: ALGO_META.ml.color }}>
            {ALGO_META.ml.label} <small>online · LambdaRank</small>
          </span>
        </div>
      </header>

      <div className="body">
        <ControlRail
          presets={presets}
          params={params}
          setParams={setParams}
          weights={weights}
          setWeights={setWeights}
          defaultWeights={DEFAULT_WEIGHTS}
          onGenerate={generate}
          onRerun={() => scenario && dispatch(scenario)}
          busy={busy}
          canRerun={!!scenario}
        />

        <main className="stage">
          {error && <div className="error">⚠ {error}</div>}

          {scenario && h && m && comparison && (
            <div className="kpis">
              <div className="kpi">
                <div className="kpi-label">Scenario</div>
                <div className="kpi-value small">{scenario.name.replace("_", " ")} <span className="muted">#{scenario.seed}</span></div>
                <div className="muted">{scenario.technicians.length} technicians · {scenario.jobs.length} jobs</div>
              </div>
              <Duel label="Jobs fulfilled" h={h.fulfillment_rate} m={m.fulfillment_rate} unit="%" />
              <Duel label="High-priority served" h={h.high_priority_fulfillment} m={m.high_priority_fulfillment} unit="%" />
              <Duel label="Travel per job" h={h.avg_travel_km} m={m.avg_travel_km} unit="km" lowerBetter digits={1} />
              <div className="kpi">
                <div className="kpi-label">Same technician chosen</div>
                <div className="kpi-value">{(comparison.summary.agreement * 100).toFixed(0)}%</div>
                <div className="agree-bar"><span style={{ width: `${comparison.summary.agreement * 100}%` }} /></div>
              </div>
            </div>
          )}

          <nav className="tabs">
            {TABS.map((t) => (
              <button key={t.id} className={tab === t.id ? "active" : ""} onClick={() => setTab(t.id)}>
                <span>{t.icon}</span> {t.label}
              </button>
            ))}
          </nav>

          {busy && !comparison && <div className="loading">Dispatching technicians…</div>}

          {tab === "map" && scenario && (
            <>
              <div className="maps">
                {ALGOS.map((a) => (
                  <MapView
                    key={a}
                    algo={a}
                    scenario={scenario}
                    result={comparison?.results[a]}
                    time={time}
                    selectedJob={selectedJob}
                    onSelectJob={setSelectedJob}
                    group={mapGroup}
                  />
                ))}
              </div>
              <Timeline time={time} setTime={setTime} playing={playing} setPlaying={setPlaying} speed={speed} setSpeed={setSpeed} />
              <div className="legend">
                <span><i className="lg pin" /> technician (number = id, glow = on a job)</span>
                <span><i className="lg ring" /> requested, waiting</span>
                <span><i className="lg faint" /> planned</span>
                <span><i className="lg solid" /> in progress / done</span>
                <span><i className="lg red" /> window closed, unserved</span>
                <span className="muted">Dot size = priority · maps pan together · click any job</span>
              </div>
            </>
          )}

          {tab === "schedule" && scenario && comparison && (
            <ScheduleView scenario={scenario} results={comparison.results} time={time} selectedJob={selectedJob} onSelectJob={setSelectedJob} />
          )}
          {tab === "score" && comparison && <Scoreboard data={comparison} />}
          {tab === "model" && <ModelLab card={card} />}
        </main>

        {selectedJob && scenario && comparison && (
          <JobDrawer scenario={scenario} jobId={selectedJob} results={comparison.results} onClose={() => setSelectedJob(null)} />
        )}
      </div>
    </div>
  );
}

function Duel({ label, h, m, unit, lowerBetter = false, digits = 0 }: {
  label: string; h: number; m: number; unit: string; lowerBetter?: boolean; digits?: number;
}) {
  const hWins = lowerBetter ? h < m : h > m;
  const tie = h === m;
  return (
    <div className="kpi duel">
      <div className="kpi-label">{label}</div>
      <div className="duel-values">
        <span className={!tie && hWins ? "win" : ""} style={{ color: ALGO_META.hungarian.color }}>{h.toFixed(digits)}</span>
        <span className="muted">{unit}</span>
        <span className={!tie && !hWins ? "win" : ""} style={{ color: ALGO_META.ml.color }}>{m.toFixed(digits)}</span>
      </div>
      <div className="duel-bar">
        <span style={{ flex: Math.max(h, 0.01), background: ALGO_META.hungarian.color }} />
        <span style={{ flex: Math.max(m, 0.01), background: ALGO_META.ml.color }} />
      </div>
    </div>
  );
}
