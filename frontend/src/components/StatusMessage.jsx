// One box for the loading and error screens in the design
// (State / Loading, State / API unreachable, State / Patient not found).
// onRetry is optional: when given, a "Try again" button is shown.
function StatusMessage({ title, message, onRetry }) {
  return (
    <section className="panel status">
      <h2>{title}</h2>
      {message && <p className="muted">{message}</p>}
      {onRetry && <button className="btn" onClick={onRetry}>Try again</button>}
    </section>
  );
}

export default StatusMessage;
