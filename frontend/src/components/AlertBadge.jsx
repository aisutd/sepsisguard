import { levelClass } from "../alertLevels";

// Colored pill with a dot AND the level written out, so the level can be
// read without relying on color (projectors, color-blind viewers).
function AlertBadge({ level }) {
  return (
    <span className={`badge badge-${levelClass(level)}`}>
      <span className="badge-dot" aria-hidden="true" />
      {level}
    </span>
  );
}

export default AlertBadge;
