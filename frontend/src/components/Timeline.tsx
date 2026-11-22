import { DAY_END, DAY_START, fmtTime } from "../utils";

interface Props {
  time: number;
  setTime: (t: number) => void;
  playing: boolean;
  setPlaying: (p: boolean) => void;
  speed: number;
  setSpeed: (s: number) => void;
}

const TICKS = Array.from({ length: (DAY_END - DAY_START) / 60 + 1 }, (_, i) => DAY_START + i * 60);

export default function Timeline({ time, setTime, playing, setPlaying, speed, setSpeed }: Props) {
  const pctPos = ((time - DAY_START) / (DAY_END - DAY_START)) * 100;
  return (
    <div className="timeline">
      <button
        className="play"
        onClick={() => {
          if (!playing && time >= DAY_END) setTime(DAY_START);
          setPlaying(!playing);
        }}
        title={playing ? "Pause" : "Play the day"}
      >
        {playing ? "❚❚" : "▶"}
      </button>
      <div className="clock">{fmtTime(time)}</div>
      <div className="track">
        <input
          type="range"
          min={DAY_START}
          max={DAY_END}
          step={5}
          value={time}
          onChange={(e) => {
            setPlaying(false);
            setTime(Number(e.target.value));
          }}
        />
        <div className="fill" style={{ width: `${pctPos}%` }} />
        <div className="ticks">
          {TICKS.map((t) => (
            <span key={t} style={{ left: `${((t - DAY_START) / (DAY_END - DAY_START)) * 100}%` }}>
              {String(t / 60).padStart(2, "0")}
            </span>
          ))}
        </div>
      </div>
      <div className="speed">
        {[1, 3, 6].map((s) => (
          <button key={s} className={speed === s ? "on" : ""} onClick={() => setSpeed(s)}>
            {s}×
          </button>
        ))}
        <button onClick={() => { setPlaying(false); setTime(DAY_END); }} title="Show full day">
          ⏭
        </button>
      </div>
    </div>
  );
}
