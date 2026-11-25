import { useState } from "react";
import AlertBadge from "./AlertBadge";
import { REPLAY_START, isAlert } from "../replay";
import { ONSET_NOTE } from "../disclaimers";

// "Risk over this patient's stay": all three TFT horizons on one chart, drawn
// up to the shared replay clock's hour (App.jsx). Nothing after the clock is drawn.
//
// How to read it:
//   - Each score is divided by its own horizon's cutoff. The horizons have
//     different cutoffs, so this puts them on one scale: the dashed line at 1×
//     is the alert threshold for all three.
//   - Solid = 6h, dashed = 12h, dotted = 24h. Line style and color both differ,
//     so the chart reads without color.
//   - The ribbon under the chart is the alert level hour by hour (from the backend).
//   - Blue marker: the first hour with any alert, looking only up to the clock.
//   - Red marker: sepsis onset as recorded in the dataset. Shown only once the
//     clock passes that hour, with the warning gap, or "No alert before this".
//   - Patients without sepsis get "Stay ended · no sepsis recorded" at the end
//     of their stay, and nothing before it.
//   - Hover for that hour's three scores and level.
// The y axis is fixed for the whole stay, so it doesn't jump while the clock runs.

const HORIZONS = [
  { key: "6h", name: "6 hour", cls: "h6" },
  { key: "12h", name: "12 hour", cls: "h12" },
  { key: "24h", name: "24 hour", cls: "h24" },
];

// Ribbon color class for each level (App.css)
const LEVEL_FILL = { "Urgent": "urgent", "High risk": "high", "Elevated risk": "elevated" };

// Drawing area in SVG units. The SVG stretches to the panel's width.
const W = 1000;
const H = 200;
const PAD = { left: 44, right: 16, top: 38 };  // top leaves room for 2 rows of marker labels
const PLOT_BOTTOM = H - 46;                     // bottom of the lines area
const RIBBON_Y = PLOT_BOTTOM + 8;               // level ribbon under the lines
const RIBBON_H = 10;
const AXIS_Y = H - 8;                           // x axis labels

function RiskHistoryChart({ scores, alert, hour, onsetHour = null, stayEnd = null, noSepsis = false, onHowToRead }) {
  const [hoverHour, setHoverHour] = useState(null);

  const title = "Risk over this patient's stay";
  const history = scores?.available ? scores.history : [];
  const cutoffs = alert?.cutoffs;

  if (history.length === 0 || !cutoffs) {
    return (
      <section className="panel">
        <h2>{title}</h2>
        <p className="muted">Scores not available for this patient.</p>
      </section>
    );
  }

  // Score divided by its horizon's cutoff (1 = at the alert threshold)
  const ratio = (h, key) => h[key] / cutoffs[key];

  const lastScored = history[history.length - 1].icu_hour;
  const onsetShown = onsetHour !== null && hour >= onsetHour;

  // x axis: ICU hour 25 to the last scored hour. It grows to the onset hour
  // only once the onset marker appears (p111054: scores stop at 35, onset 38).
  const xMin = REPLAY_START;
  const xMax = Math.max(lastScored, onsetShown ? onsetHour : 0, xMin + 1);
  const yMax = Math.max(1, ...history.flatMap((h) => HORIZONS.map(({ key }) => ratio(h, key)))) * 1.15;

  const plotW = W - PAD.left - PAD.right;
  const sx = (x) => PAD.left + ((x - xMin) / (xMax - xMin)) * plotW;
  const sy = (y) => PLOT_BOTTOM - (y / yMax) * (PLOT_BOTTOM - PAD.top);

  // Only hours up to the clock
  const visible = history.filter((h) => h.icu_hour <= hour);
  const last = visible[visible.length - 1];
  const firstAlert = visible.find((h) => isAlert(h.alert_level))?.icu_hour ?? null;
  const stayEnded = noSepsis && stayEnd !== null && hour >= stayEnd;
  const hovered = visible.find((h) => h.icu_hour === hoverHour) ?? null;

  // Labels near the right edge are written leftwards so they stay inside the chart
  const nearRight = (x) => x > W * 0.7;
  const labelX = (x) => (nearRight(x) ? x - 6 : x + 6);
  const labelAnchor = (x) => (nearRight(x) ? "end" : "start");
  const plural = (n) => `${n} hour${n === 1 ? "" : "s"}`;

  // Mouse position → nearest ICU hour (for the tooltip)
  function handleMove(e) {
    const box = e.currentTarget.getBoundingClientRect();
    const svgX = ((e.clientX - box.left) / box.width) * W;
    setHoverHour(Math.round(xMin + ((svgX - PAD.left) / plotW) * (xMax - xMin)));
  }

  // Keep the tooltip inside the panel near the edges
  const tipPos = hovered ? sx(hovered.icu_hour) / W : 0;
  const tipEdge = tipPos > 0.8 ? " right-edge" : tipPos < 0.2 ? " left-edge" : "";

  return (
    <section className="panel">
      <div className="panel-head">
        <h2>{title}</h2>
        <div className="legend small">
          {HORIZONS.map(({ key, name, cls }) => (
            <span key={key}>
              <svg width="24" height="8" aria-hidden="true">
                <line className={`risk-line ${cls}`} x1="0" x2="24" y1="4" y2="4" />
              </svg>{" "}
              {name}
            </span>

          ))}
          <span><span className="legend-line cutoff" />1× alert threshold</span>
          <button className="link-btn" onClick={onHowToRead}>How to read this</button>

        </div>
      </div>

      <div className="risk-chart">
        <svg viewBox={`0 0 ${W} ${H}`} className="chart" role="img"
             aria-label={`Risk from ICU hour ${xMin} to ${Math.min(hour, xMax)}, each score divided by its cutoff`}
             onMouseMove={handleMove} onMouseLeave={() => setHoverHour(null)}>

          {/* y axis: bottom and top values */}
          <text className="axis" x={PAD.left - 6} y={sy(0) + 4} textAnchor="end">0×</text>
          <text className="axis" x={PAD.left - 6} y={sy(yMax) + 10} textAnchor="end">{yMax.toFixed(1)}×</text>

          {/* Alert threshold: 1× for every horizon */}
          <line className="cutoff-line" x1={PAD.left} x2={W - PAD.right} y1={sy(1)} y2={sy(1)} />
          <text className="axis" x={PAD.left + 4} y={sy(1) - 5}>Alert threshold (1×)</text>

          {/* One line per horizon, plus a dot at the latest hour
              (so a patient with a single scored hour still shows) */}
          {HORIZONS.map(({ key, cls }) => (
            <g key={key}>
              <polyline className={`risk-line ${cls}`}
                        points={visible.map((h) => `${sx(h.icu_hour)},${sy(ratio(h, key))}`).join(" ")} />
              {last && <circle className={`risk-dot ${cls}`} cx={sx(last.icu_hour)} cy={sy(ratio(last, key))} r="3" />}
            </g>
          ))}

          {visible.length === 0 && (
            <text className="axis" x={W / 2} y={(PAD.top + PLOT_BOTTOM) / 2} textAnchor="middle">
              Not scored yet: the first score is at hour {history[0].icu_hour}
            </text>
          )}

          {/* Level ribbon: one cell per scored hour up to the clock */}
          <rect className="ribbon-bg" x={PAD.left} y={RIBBON_Y} width={plotW} height={RIBBON_H} />
          {visible.map((h) => {
            const x0 = Math.max(sx(h.icu_hour - 0.5), PAD.left);
            const x1 = Math.min(sx(h.icu_hour + 0.5), W - PAD.right);
            return (
              <rect key={h.icu_hour} className={`ribbon-cell ${LEVEL_FILL[h.alert_level] ?? "none"}`}
                    x={x0} y={RIBBON_Y} width={x1 - x0} height={RIBBON_H} />
            );
          })}

          {/* x axis in ICU hours, starting at 25 */}
          <text className="axis" x={PAD.left} y={AXIS_Y}>ICU hour {xMin}</text>
          <text className="axis" x={W - PAD.right} y={AXIS_Y} textAnchor="end">hour {xMax}</text>


          {/* First alert (up to the clock) */}
          {firstAlert !== null && (
            <g className="marker-alert">
              <line x1={sx(firstAlert)} x2={sx(firstAlert)} y1={4} y2={RIBBON_Y + RIBBON_H} />
              <circle cx={sx(firstAlert)} cy={sy(1)} r="4" />
              <text x={labelX(sx(firstAlert))} y={12} textAnchor={labelAnchor(sx(firstAlert))}>
                First alert · hour {firstAlert}
              </text>
            </g>
          )}

          {/* Sepsis onset: only once the clock has passed it */}
          {onsetShown && (
            <g className="marker-onset">
              <line x1={sx(onsetHour)} x2={sx(onsetHour)} y1={18} y2={RIBBON_Y + RIBBON_H} />
              <text x={labelX(sx(onsetHour))} y={30} textAnchor={labelAnchor(sx(onsetHour))}>
                Sepsis onset (recorded in dataset) · hour {onsetHour}
              </text>
            </g>
          )}

          {/* Warning gap from first alert to onset, or a missed patient */}
          {onsetShown && firstAlert !== null && firstAlert < onsetHour && (
            <g className="warning-gap">
              <line x1={sx(firstAlert)} x2={sx(onsetHour)} y1={PLOT_BOTTOM - 8} y2={PLOT_BOTTOM - 8} />
              <line x1={sx(firstAlert)} x2={sx(firstAlert)} y1={PLOT_BOTTOM - 13} y2={PLOT_BOTTOM - 3} />
              <line x1={sx(onsetHour)} x2={sx(onsetHour)} y1={PLOT_BOTTOM - 13} y2={PLOT_BOTTOM - 3} />
              <text x={(sx(firstAlert) + sx(onsetHour)) / 2} y={PLOT_BOTTOM - 14} textAnchor="middle">
                {plural(onsetHour - firstAlert)} of warning
              </text>
            </g>
          )}
          {onsetShown && (firstAlert === null || firstAlert >= onsetHour) && (
            <text className="missed-note" x={sx(onsetHour) - 6} y={PLOT_BOTTOM - 14} textAnchor="end">
              No alert before this
            </text>
          )}

          {/* No sepsis: a note only once their stay has ended */}
          {stayEnded && (
            <text className="end-note" x={sx(Math.min(stayEnd, xMax))} y={30} textAnchor="end">
              Stay ended · no sepsis recorded
            </text>
          )}

          {hovered && (
            <line className="hover-line" x1={sx(hovered.icu_hour)} x2={sx(hovered.icu_hour)}
                  y1={PAD.top} y2={RIBBON_Y + RIBBON_H} />
          )}
        </svg>

        {/* Tooltip: the hour, the three raw scores (and × cutoff), and the level */}
        {hovered && (
          <div className={`chart-tooltip${tipEdge}`} style={{ left: `${tipPos * 100}%` }}>
            <div className="tooltip-hour">ICU hour {hovered.icu_hour}</div>
            {HORIZONS.map(({ key }) => (
              <div key={key} className="tooltip-row mono">
                <span>{key}</span>
                <span>{hovered[key].toFixed(4)}</span>
                <span className="muted">{ratio(hovered, key).toFixed(1)}×</span>
              </div>
            ))}
            <AlertBadge level={hovered.alert_level} />
          </div>
        )}
      </div>
      {/* Short caveat, only while the onset marker is on screen */}
      {onsetShown && <p className="panel-foot">{ONSET_NOTE}</p>}

    </section>
  );
}

export default RiskHistoryChart;
