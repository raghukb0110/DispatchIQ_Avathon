import L from "leaflet";
import { useEffect, useMemo } from "react";
import { CircleMarker, MapContainer, Marker, Polyline, TileLayer, Tooltip, useMap } from "react-leaflet";
import type { AllocationResult, Job, Scenario } from "../types";
import { ALGO_META, type Algo, fmtTime, jobStatus, routesByTech, techColor, techPosition } from "../utils";

/** Maps registered in one group pan and zoom together. */
export interface MapGroup {
  maps: Set<L.Map>;
  syncing: boolean;
}

interface Props {
  algo: Algo;
  scenario: Scenario;
  result?: AllocationResult;
  time: number;
  selectedJob: string | null;
  onSelectJob: (id: string) => void;
  group: MapGroup;
}

const CENTER: [number, number] = [12.97, 77.6];

function SyncMaps({ group }: { group: MapGroup }) {
  const map = useMap();
  useEffect(() => {
    group.maps.add(map);
    const onMove = () => {
      if (group.syncing) return;
      group.syncing = true;
      group.maps.forEach((m) => m !== map && m.setView(map.getCenter(), map.getZoom(), { animate: false }));
      group.syncing = false;
    };
    map.on("move", onMove);
    return () => {
      map.off("move", onMove);
      group.maps.delete(map);
    };
  }, [map, group]);
  return null;
}

const iconCache = new Map<string, L.DivIcon>();
function techIcon(id: string, busy: boolean) {
  const key = `${id}-${busy}`;
  if (!iconCache.has(key)) {
    iconCache.set(
      key,
      L.divIcon({
        className: "",
        html: `<div class="tech-pin ${busy ? "busy" : ""}" style="--c:${techColor(id)}">${id.slice(1)}</div>`,
        iconSize: [24, 24],
        iconAnchor: [12, 12],
      }),
    );
  }
  return iconCache.get(key)!;
}

export default function MapView({ algo, scenario, result, time, selectedJob, onSelectJob, group }: Props) {
  const meta = ALGO_META[algo];
  const jobs = useMemo(() => new Map<string, Job>(scenario.jobs.map((j) => [j.id, j])), [scenario]);
  const byJob = useMemo(() => new Map((result?.assignments ?? []).map((a) => [a.job_id, a])), [result]);
  const routes = useMemo(() => routesByTech(result?.assignments ?? []), [result]);

  const counts = { done: 0, active: 0, waiting: 0 };
  for (const j of scenario.jobs) {
    const s = jobStatus(j, byJob.get(j.id), time);
    if (s === "done") counts.done++;
    else if (s === "active") counts.active++;
    else if (s === "waiting" || s === "unassigned") counts.waiting++;
  }

  return (
    <div className="map-card" style={{ ["--accent" as string]: meta.color }}>
      <div className="map-head">
        <div>
          <span className="algo-dot" />
          <b>{meta.label}</b> <span className="chip">{meta.short}</span>
          <span className="muted"> · {meta.tag}</span>
        </div>
        <div className="map-counts">
          <span title="completed">✔ {counts.done}</span>
          <span title="in progress">⚡ {counts.active}</span>
          <span title="requested, no technician">⏳ {counts.waiting}</span>
        </div>
      </div>
      <MapContainer center={CENTER} zoom={11} preferCanvas className="map" zoomControl={false}>
        <SyncMaps group={group} />
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        {scenario.technicians.map((t) => {
          const route = routes.get(t.id) ?? [];
          const driven = route.filter((a) => a.arrival - (a.travel_km / 25) * 60 <= time);
          if (!driven.length) return null;
          const path: [number, number][] = [[t.lat, t.lng]];
          driven.forEach((a) => {
            const j = jobs.get(a.job_id)!;
            path.push(a.arrival <= time ? [j.lat, j.lng] : techPosition(t, route, jobs, time));
          });
          return <Polyline key={t.id} positions={path} pathOptions={{ color: techColor(t.id), weight: 2.5, opacity: 0.85 }} />;
        })}

        {scenario.jobs.map((j) => {
          const a = byJob.get(j.id);
          const status = jobStatus(j, a, time);
          if (status === "hidden") return null;
          const selected = j.id === selectedJob;
          const color = a ? techColor(a.technician_id) : "#94a3b8";
          const style = {
            waiting: { fillOpacity: 0, dashArray: "3 3", weight: 1.5 },
            unassigned: { fillOpacity: 0, dashArray: "2 3", weight: 1.5, color: "#ef4444" },
            planned: { fillOpacity: 0.2, weight: 1.5 },
            active: { fillOpacity: 1, weight: 3 },
            done: { fillOpacity: 0.85, weight: 1 },
          }[status];
          return (
            <CircleMarker
              key={j.id}
              center={[j.lat, j.lng]}
              radius={selected ? 11 : 3 + j.priority * (status === "active" ? 1.4 : 1)}
              pathOptions={{ color: selected ? "#fff" : color, fillColor: color, ...style, ...(selected ? { weight: 3 } : {}) }}
              eventHandlers={{ click: () => onSelectJob(j.id) }}
            >
              <Tooltip direction="top">
                <b>{j.id}</b> · {j.required_skill} L{j.min_skill_level} · P{j.priority}
                <br />
                window {fmtTime(j.window_start)}–{fmtTime(j.window_end)}
                <br />
                {a ? `${a.technician_id} · ${fmtTime(a.start)}–${fmtTime(a.end)} (${status})` : status}
              </Tooltip>
            </CircleMarker>
          );
        })}

        {scenario.technicians.map((t) => {
          const route = routes.get(t.id) ?? [];
          const busy = route.some((a) => a.start <= time && time < a.end);
          return (
            <Marker key={t.id} position={techPosition(t, route, jobs, time)} icon={techIcon(t.id, busy)}>
              <Tooltip direction="top">
                <b>{t.name}</b>
                <br />
                {t.skills.join(", ")} · level {t.skill_level}
                <br />
                shift {fmtTime(t.shift_start)}–{fmtTime(t.shift_end)} · {route.length} job(s)
              </Tooltip>
            </Marker>
          );
        })}
      </MapContainer>
    </div>
  );
}
