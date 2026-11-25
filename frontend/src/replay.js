// replay.js
// The shared replay clock, and helpers that read one patient's hourly scores
// at a given ICU hour. Used by the Ward, Patient Review and Alerts (App.jsx).
//
// Rules:
//   - The clock runs from ICU hour 25 (the TFT needs 24 hours of data before
//     its first score) to the latest snapshot hour among the demo patients.
//   - Each hour's alert level comes from the backend
//     (scores.history[].alert_level in backend/main.py, using Harish's
//     cutoffs). Nothing here recalculates it.
//   - Only hours up to the clock are shown, so the screen never shows the future.
import { useEffect, useState } from "react";
import { ALERT_LEVELS } from "./alertLevels";

export const REPLAY_START = 25;
export const SPEEDS = [1, 2, 4, 8];   // ICU hours per second

// true for Urgent / High risk / Elevated risk
export const isAlert = (level) => level !== "No alert" && ALERT_LEVELS.includes(level);

// The clock. `end` is the last hour it can reach.
export function useReplayClock(end) {
  const [hour, setHour] = useState(REPLAY_START);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(4);   // about 4 hours per second

  // Stops by itself at the end. Worked out here rather than stored,
  // so no extra state update is needed.
  const isPlaying = playing && hour < end;

  // While playing, move forward one hour every 1/speed seconds
  useEffect(() => {
    if (!isPlaying) return;
    const timer = setInterval(() => setHour((h) => Math.min(h + 1, end)), 1000 / speed);
    return () => clearInterval(timer);
  }, [isPlaying, speed, end]);

  return {
    hour,
    end,
    speed,
    playing: isPlaying,
    setSpeed,
    // At the end, Play starts over from hour 25
    play: () => {
      if (hour >= end) setHour(REPLAY_START);
      setPlaying(true);
    },
    pause: () => setPlaying(false),
    // Slider, "Jump to next alert" and "Go to hour X": move there and pause
    goTo: (h) => {
      setPlaying(false);
      setHour(Math.max(REPLAY_START, Math.min(h, end)));
    },
    reset: () => {
      setPlaying(false);
      setHour(REPLAY_START);
    },
  };
}

// Where one patient is at `hour`:
//   status      "waiting" (no score yet), "live", or "ended" (past their last
//               scored hour: frozen on that score)
//   entry       the latest scored hour at or before `hour` (null while waiting)
//   level       that hour's alert level
//   lastHour    their last scored hour (the snapshot)
//   firstAlert  the first hour with an alert, looking only up to `hour`
export function patientAt(history = [], hour) {
  const seen = history.filter((h) => h.icu_hour <= hour);
  const entry = seen.length > 0 ? seen[seen.length - 1] : null;
  const lastHour = history.length > 0 ? history[history.length - 1].icu_hour : null;
  const first = seen.find((h) => isAlert(h.alert_level));
  return {
    status: !entry ? "waiting" : hour > lastHour ? "ended" : "live",
    entry,
    level: entry ? entry.alert_level : "Not scored yet",
    lastHour,
    firstAlert: first ? first.icu_hour : null,
  };
}

// The score that sets the alert level, compared with its own threshold
// (ratio = score ÷ cutoff, 1 = at the threshold). Follows the alert rule:
// 6h, then 12h, then 24h, and the first at or above its threshold is the one
// returned. If none is, the closest one (highest ratio) is returned, for the bar.
// Returns { horizon, ratio } or null. A ratio, never a percentage.
export function levelRatio(entry, cutoffs) {
  if (!entry || !cutoffs) return null;
  const all = ["6h", "12h", "24h"].map((h) => ({ horizon: h, ratio: entry[h] / cutoffs[h] }));
  return all.find((r) => r.ratio >= 1) ?? all.reduce((a, b) => (b.ratio > a.ratio ? b : a));
}


// For "Jump to next alert": the first hour after `hour` when any patient goes
// from no alert to an alert. null if there are no more.
export function nextAlertHour(histories, hour) {
  let next = null;
  for (const history of Object.values(histories)) {
    history.forEach((h, i) => {
      const startsAlert = isAlert(h.alert_level) && (i === 0 || !isAlert(history[i - 1].alert_level));
      if (h.icu_hour > hour && startsAlert && (next === null || h.icu_hour < next)) {
        next = h.icu_hour;
      }
    });
  }
  return next;
}
