// Stand-in for pages whose content isn't built yet (Alerts, Track Record,
// Data and safety). Replace with the real page component when it's ready.
function PlaceholderPage({ title }) {
  return (
    <>
      <div className="title-row">
        <h1>{title}</h1>
      </div>
      <section className="panel status">
        <p className="muted">Content for this page is coming soon.</p>
      </section>
    </>
  );
}

export default PlaceholderPage;
