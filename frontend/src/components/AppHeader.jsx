// Top bar: the page name, then the replay controls on pages that use the
// clock (passed in by App.jsx as children).
// (The SepsisGuard title is in the left navigation, NavSidebar.jsx.)
// No developer status here: if the backend is down, the page itself says so
// and offers "Try again" (StatusMessage.jsx).
function AppHeader({ title, children }) {
  return (
    <header className="app-header">
      <h1 className="header-title">{title}</h1>
      <div className="header-left">{children}</div>
    </header>
  );
}

export default AppHeader;
