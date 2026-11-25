import { REPLAY_START, SPEEDS } from "../replay";

// Replay controls in the top bar: Play/Pause, speed, hour slider,
// "Jump to next alert" and Reset. They control the one shared clock
// (useReplayClock in replay.js), so the Ward, Patient Review and Alerts
// all move together.
// Props: clock (from useReplayClock), nextAlert (hour or null).
function ReplayBar({ clock, nextAlert }) {
  const playText = clock.playing ? "❚❚ Pause" : clock.hour >= clock.end ? "▶ Replay" : "▶ Play";

  return (
    <div className="replay-bar">
      <button className="btn btn-primary replay-play" onClick={clock.playing ? clock.pause : clock.play}>
        {playText}
      </button>

      <select
        className="replay-speed"
        value={clock.speed}
        onChange={(e) => clock.setSpeed(Number(e.target.value))}
        aria-label="Replay speed"
      >
        {SPEEDS.map((s) => <option key={s} value={s}>{s} h/s</option>)}
      </select>

      {/* Drag to any hour; dragging pauses */}
      <input
        type="range"
        className="replay-slider"
        min={REPLAY_START}
        max={clock.end}
        value={clock.hour}
        onChange={(e) => clock.goTo(Number(e.target.value))}
        aria-label="ICU hour"
      />
      <span className="replay-hour mono">ICU hour {clock.hour}</span>

      <button className="btn" disabled={nextAlert === null} onClick={() => clock.goTo(nextAlert)}>
        Jump to next alert
      </button>
      <button className="btn" onClick={clock.reset}>Reset</button>
    </div>
  );
}

export default ReplayBar;
