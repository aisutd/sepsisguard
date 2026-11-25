// disclaimers.js
// Every safety message on screen, written once. Components import these
// instead of writing their own wording, so a message appears exactly where
// it's needed and is changed in one place.
//
// Where each one is used:
//   BANNER        top banner on every page (DisclaimerBar.jsx). Approved
//                 change from the Round 2 wording; the "relative risk" part
//                 is now in "How to read this" (HowToRead.jsx).
//   SCORES_NOTE   under the horizon scores (RiskScore.jsx)
//   ONSET_NOTE    under the risk chart, only once the onset marker is shown
//                 (RiskHistoryChart.jsx)
//   REASONS_NOTE  under the reasons (ShapChart.jsx)
// Everything longer is on the Data and safety page (DataSafetyPage.jsx),
// which the banner's "Details" link opens.

export const BANNER =
  "Research system · historical data · not for clinical use";

export const SCORES_NOTE =
  "Scores show how high a patient ranks, not the chance of getting sepsis.";

export const ONSET_NOTE =
  "Onset is from the dataset's records. The model didn't see it when scoring.";

export const REASONS_NOTE =
  "Reasons from our baseline model. They show what it weighed, not a diagnosis.";
