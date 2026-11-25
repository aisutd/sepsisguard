import { BANNER } from "../disclaimers";

// Required on every page (Round 2 checklist). The sentence comes from
// disclaimers.js; "Details" opens the Data and safety page for the full
// explanation, so other screens only need a short line.
// Props: onDetails, called when "Details" is clicked
function DisclaimerBar({ onDetails }) {
  return (
    <div className="disclaimer" role="note">
      {BANNER}{" "}
      <button className="disclaimer-link" onClick={onDetails}>Details</button>
    </div>
  );
}

export default DisclaimerBar;
