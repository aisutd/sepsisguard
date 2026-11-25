import { PAGES } from "../pages";

// Left navigation for the whole app: the SepsisGuard title at the top,
// then one link per page. The current page is highlighted.
// Props:
//   page        id of the page on screen (see pages.js)
//   onNavigate  called with a page id when a link is clicked
function NavSidebar({ page, onNavigate }) {
  return (
    <nav className="nav-sidebar" aria-label="Main">
      <div className="nav-brand">
        {/* Heartbeat logo, same as the design */}
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
          <div className="nav-title">SepsisGuard</div>
          <div className="brand-sub">ICU sepsis early warning</div>
        </div>
      </div>

      <ul className="nav-list">
        {PAGES.map((p) => {
          const active = p.id === page;
          return (
            <li key={p.id}>
              <button
                className={`nav-item${active ? " active" : ""}`}
                onClick={() => onNavigate(p.id)}
                aria-current={active ? "page" : undefined}
              >
                {p.label}
              </button>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}

export default NavSidebar;
