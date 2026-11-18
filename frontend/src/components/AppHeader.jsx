// Top bar from the Figma design.
// Props:
//   apiOk         true / false from checkHealth(), or null while still checking
//   onRefresh     reloads the data from the backend
//   onShowAlerts  opens the "How alerts work" popup
function AppHeader({ apiOk, onRefresh, onShowAlerts }) {
  const apiState = apiOk === null ? "checking" : apiOk ? "ok" : "down";
  const apiText = {
    checking: "Checking API…",
    ok: "API connected",
    down: "API not running",
  }[apiState];

  return (
    <header className="app-header">
      <div className="brand">
        {/* Heartbeat logo, drawn as a small inline picture */}
        <div className="brand-logo" aria-hidden="true">
          <svg viewBox="0 0 24 24" width="18" height="18">
            <polyline
              points="2,12 7,12 10,5 14,19 17,12 22,12"
              fill="none" stroke="white" strokeWidth="2.2"
              strokeLinecap="round" strokeLinejoin="round"
            />
          </svg>
        </div>
        <div>
          <div className="brand-name">SepsisGuard</div>
          <div className="brand-sub">ICU sepsis early warning</div>
        </div>
      </div>

      <div className="header-actions">
        <button className="btn" onClick={onShowAlerts}>How alerts work</button>
        <span className={`api-pill ${apiState}`}>
          <span className="api-dot" aria-hidden="true" />
          {apiText}
        </span>
        <button className="btn" onClick={onRefresh}>Refresh</button>
      </div>
    </header>
  );
}

export default AppHeader;
