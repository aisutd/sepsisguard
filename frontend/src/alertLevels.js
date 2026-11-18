// alertLevels.js
// The 4 alert levels the backend returns (ALERT_LEVELS and NO_ALERT in
// backend/main.py), from most to least serious. The backend decides each
// patient's level; this file only lists the names for sorting and coloring.
export const ALERT_LEVELS = ["Urgent", "High risk", "Elevated risk", "No alert"];

// CSS class name for each level. The colors are in index.css and App.css.
const LEVEL_CLASS = {
  "Urgent": "urgent",
  "High risk": "high",
  "Elevated risk": "elevated",
  "No alert": "none",
};

// Class for a level. Anything unexpected (e.g. "not available") is grey.
export function levelClass(level) {
  return LEVEL_CLASS[level] ?? "none";
}

// Sort patients most serious first. Unknown levels go last.
// Returns a new list and leaves the original alone.
export function sortByAlert(patients) {
  const rank = (p) => {
    const i = ALERT_LEVELS.indexOf(p.alert_level);
    return i === -1 ? ALERT_LEVELS.length : i;
  };
  return [...patients].sort((a, b) => rank(a) - rank(b));
}
