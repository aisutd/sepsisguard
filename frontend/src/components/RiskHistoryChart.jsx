import { useEffect, useState } from "react";
import AlertBadge from "./AlertBadge";

// Line chart of one horizon's TFT score for every hour up to the snapshot
// (detail.scores.history), with a dashed line at that horizon's cutoff.
// Plain SVG, so no chart library is needed.
//
// Play button: replays the stay one hour at a time, so you can watch the score
// rise and see the hour the alert level changes. It only replays hours the
// backend sent, which stop at the snapshot hour, so it never shows the future.
//
// The y axis runs from 0 to a bit above the highest value (score or cutoff),
// not 0 to 1: real scores are small (e.g. 0.04) and would look flat on 0-1.
// The axes are fixed for the whole stay, so they don't jump while playing.

// Drawing area in SVG units. The SVG stretches to the panel's width.
const W = 600;
const H = 170;
const PAD = { left: 56, right: 12, top: 10, bottom: 24 };

// Time between hours while playing (ms). 85 hours take about 10 seconds.
const STEP_MS = 120;

function RiskHistoryChart({ scores, alert, horizon = "24h" }) {
  const title = `${horizon} risk over the ICU stay`;
  const history = scores?.available ? scores.history : [];
  const total = history.length;

  // How many hours are drawn. Starts at all of them (the full trend).
  const [shown, setShown] = useState(total);
  const [playing, setPlaying] = useState(false);

  // While playing, add one hour every STEP_MS, and stop at the last hour.
  // React re-runs this after every step; the cleanup cancels the old timer.
  useEffect(() => {
    if (!playing) return;
    if (shown >= total) {
      setPlaying(false);
      return;
    }
    const timer = setTimeout(() => setShown((n) => n + 1), STEP_MS);
    return () => clearTimeout(timer);
  }, [playing, shown, total]);

  // Some patients have only 1 scored hour (e.g. p000619): nothing to draw.
  if (total < 2) {
    return (
      <section className="panel">
        <h2>{title}</h2>
        <p className="muted">Not enough scored hours to draw a trend for this patient.</p>
      </section>
    );
  }

  const points = history.map((h) => ({ x: h.icu_hour, y: h[horizon] }));
  const cutoff = alert?.cutoffs?.[horizon];

  // Axes cover the whole stay up to the snapshot, even while only part is drawn
  const xMin = points[0].x;
  const xMax = points[total - 1].x;
  const yMax = Math.max(...points.map((p) => p.y), cutoff ?? 0) * 1.15;

  // Turn an ICU hour / score into a position inside the SVG
  const sx = (x) => PAD.left + ((x - xMin) / (xMax - xMin)) * (W - PAD.left - PAD.right);
  const sy = (y) => H - PAD.bottom - (y / yMax) * (H - PAD.top - PAD.bottom);

  // Only the hours replayed so far
  const visible = points.slice(0, shown);
  const line = visible.map((p) => `${sx(p.x)},${sy(p.y)}`).join(" ");
  const current = visible[visible.length - 1];
  const currentHour = history[shown - 1];

  // Play from the first hour (or from where it was paused); Pause stops it.
  function togglePlay() {
    if (playing) {
      setPlaying(false);
      return;
    }
    if (shown >= total) setShown(1); // at the end: start over
    setPlaying(true);
  }

  const buttonText = playing ? "❚❚ Pause" : shown >= total ? "▶ Replay" : "▶ Play";

  return (
    <section className="panel">
      <div className="panel-head">
        <h2>{title}</h2>
        <div className="legend small">
          <span><span className="legend-line" />Score</span>
          {cutoff !== undefined && <span><span className="legend-line cutoff" />Cutoff</span>}
        </div>
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} className="chart" role="img"
           aria-label={`${horizon} score from ICU hour ${xMin} to ${current.x}`}>
        {/* Horizontal grid lines with their values */}
        {[0, yMax / 2, yMax].map((v) => (
          <g key={v}>
            <line className="grid" x1={PAD.left} x2={W - PAD.right} y1={sy(v)} y2={sy(v)} />
            <text className="axis" x={PAD.left - 6} y={sy(v) + 4} textAnchor="end">{v.toFixed(3)}</text>
          </g>
        ))}

        {/* First and last hour on the x axis */}
        <text className="axis" x={PAD.left} y={H - 6}>hour {xMin}</text>
        <text className="axis" x={W - PAD.right} y={H - 6} textAnchor="end">hour {xMax} (snapshot)</text>

        {cutoff !== undefined && (
          <line className="cutoff-line" x1={PAD.left} x2={W - PAD.right} y1={sy(cutoff)} y2={sy(cutoff)} />
        )}
        <polyline className="score-line" points={line} />
        <circle className="score-dot" cx={sx(current.x)} cy={sy(current.y)} r="4" />
      </svg>

      {/* Playback controls */}
      <div className="playback">
        <button className="btn" onClick={togglePlay}>{buttonText}</button>
        {/* Drag to any hour; dragging pauses playback */}
        <input
          type="range"
          className="playback-slider"
          min={1}
          max={total}
          value={shown}
          onChange={(e) => {
            setPlaying(false);
            setShown(Number(e.target.value));
          }}
          aria-label="ICU hour to show"
        />
        <span className="mono small">ICU hour {current.x} of {xMax}</span>
        {/* The alert level the backend computed for this hour */}
        {currentHour.alert_level && <AlertBadge level={currentHour.alert_level} />}
      </div>

      <p className="panel-foot">
        {"Hours up to the snapshot only, so the screen never shows the future. Cutoff from alert_thresholds.json."}
      </p>
    </section>
  );
}

export default RiskHistoryChart;
