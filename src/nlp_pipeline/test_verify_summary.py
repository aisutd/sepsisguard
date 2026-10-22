"""Tests for generate_summary.py and verify_summary.py.

Run from the repo root (standard library only, no extra packages):
    python -m unittest discover -s src/nlp_pipeline -v

The "bad summary" cases imitate the mistakes a language model makes when
rewriting the input: adding a feature, swapping a direction, using a synonym
for a feature that isn't there, or turning a risk estimate into a diagnosis.
"""

import json
import unittest
from pathlib import Path

from generate_summary import CAVEAT, generate_summary, parse_top_features
from verify_summary import MOCK_INPUT, find_mentions, verify_summary

# Lactate, ICULOS and HR push risk up; Temp and Age push it down.
PATIENT = {
    "patient_id": "test",
    "top_features": [
        {"feature": "Lactate", "shap_value": 0.42},
        {"feature": "ICULOS", "shap_value": 0.31},
        {"feature": "HR", "shap_value": 0.18},
        {"feature": "Temp", "shap_value": -0.12},
        {"feature": "Age", "shap_value": -0.05},
    ],
}


class TemplateTests(unittest.TestCase):
    def test_every_mock_patient_passes(self):
        """The template is grounded by construction, so all mock patients pass."""
        for entry in json.loads(Path(MOCK_INPUT).read_text()):
            summary = generate_summary(entry)
            result = verify_summary(summary, entry)
            self.assertTrue(result["verified"], (entry["patient_id"], result))
            self.assertEqual(result["warnings"], [], entry["patient_id"])

    def test_groups_by_direction_strongest_first(self):
        self.assertEqual(
            generate_summary(PATIENT),
            "Lactate raised this patient's predicted sepsis risk the most, "
            "followed by time in the ICU and heart rate. Temperature and age "
            "lowered it. These reflect what the model weighed, not measured "
            "values or a diagnosis.",
        )

    def test_strongest_negative_feature_leads(self):
        entry = {"patient_id": "x", "top_features": [
            {"feature": "Temp", "shap_value": -0.5},
            {"feature": "Age", "shap_value": -0.2},
            {"feature": "HR", "shap_value": 0.3}]}
        self.assertEqual(
            generate_summary(entry),
            "Temperature lowered this patient's predicted sepsis risk the most, "
            "followed by age. Heart rate raised it. " + CAVEAT,
        )

    def test_only_negative_features(self):
        entry = {"patient_id": "x", "top_features": [{"feature": "HR", "shap_value": -0.2}]}
        self.assertTrue(generate_summary(entry).startswith(
            "Heart rate lowered this patient's predicted sepsis risk."))

    def test_unsorted_input_with_extra_features_is_trimmed_to_top5(self):
        entry = {"patient_id": "x", "top_features": [
            {"feature": f, "shap_value": v} for f, v in
            [("Age", 0.01), ("HR", 0.5), ("Temp", -0.4), ("Resp", 0.3),
             ("WBC", 0.2), ("Lactate", -0.1)]]}
        self.assertEqual([f for f, _ in parse_top_features(entry)],
                         ["HR", "Temp", "Resp", "WBC", "Lactate"])

    def test_zero_shap_features_are_dropped(self):
        entry = {"patient_id": "x", "top_features": [
            {"feature": "HR", "shap_value": 0.3}, {"feature": "Age", "shap_value": 0}]}
        self.assertNotIn("age", generate_summary(entry).lower())

    def test_bad_input_is_rejected(self):
        for bad in [
            {"patient_id": "x", "top_features": [{"feature": "Nope", "shap_value": 0.1}]},
            {"patient_id": "x", "top_features": [{"feature": "HR", "shap_value": "0.1"}]},
            {"patient_id": "x", "top_features": [{"feature": "HR", "shap_value": True}]},
            {"patient_id": "x", "top_features": [{"feature": "HR", "shap_value": float("nan")}]},
            {"patient_id": "x", "top_features": [{"feature": "HR", "shap_value": 0}]},
            {"patient_id": "x"},
        ]:
            with self.assertRaises(ValueError, msg=bad):
                generate_summary(bad)


class VerifierCatchesBadSummaries(unittest.TestCase):
    def assertFails(self, summary, expected_reason):
        """The summary (with a valid caveat added) must fail for this reason."""
        result = verify_summary(summary + " " + CAVEAT, PATIENT)
        self.assertFalse(result["verified"], summary)
        self.assertTrue(
            any(expected_reason in p for p in result["problems"]),
            f"expected '{expected_reason}' in {result['problems']}",
        )

    def test_extra_feature(self):
        self.assertFails("Lactate and white blood cell count raised the risk.",
                         "white blood cell count")

    def test_extra_feature_written_as_synonym(self):
        self.assertFails("Lactate and blood pressure raised the risk.", "blood pressure")
        self.assertFails("Lactic acid and leukocytes raised the risk.", "leukocytes")

    def test_extra_feature_written_as_code(self):
        self.assertFails("Lactate and MAP raised the risk.", "MAP")

    def test_direction_flipped(self):
        self.assertFails("Temperature raised this patient's predicted risk.",
                         "says 'Temperature' raised")
        self.assertFails("Lactate lowered the risk.", "says 'Lactate' lowered")

    def test_direction_flipped_with_synonym_verbs(self):
        self.assertFails("Temperature increased the risk.", "says 'Temperature' raised")
        self.assertFails("Lactate decreased the risk.", "says 'Lactate' lowered")

    def test_unrecognized_direction_wording_fails_closed(self):
        # Temp has a negative SHAP value, so this is wrong, but no direction
        # word we know appears. It must not slip through.
        self.assertFails("Temperature shifted the risk upward.", "without saying")
        self.assertFails("Temperature shifted the risk upward.", "shifted")

    def test_mixed_direction_sentence_fails(self):
        # Both directions are reversed here, and the sentence is ambiguous.
        self.assertFails("Lactate lowered the risk while temperature raised it.",
                         "both raised and lowered")

    def test_diagnosis_in_any_wording_fails(self):
        for claim in ["so the patient has sepsis", "so the patient may be septic",
                      "and sepsis is likely", "but the patient does not have sepsis"]:
            self.assertFails(f"Lactate raised the risk, {claim}.", "wording the template never uses")

    def test_reason_from_outside_the_model(self):
        self.assertFails("Lactate raised the risk, suggesting an infection.", "infection")
        self.assertFails("Lactate and possible kidney damage raised the risk.", "kidney")
        self.assertFails("Lactate raised the risk due to poor perfusion.", "perfusion")

    def test_claim_about_a_measurement(self):
        # SHAP says nothing about whether the measured value was high.
        self.assertFails("Lactate was high and raised the risk.", "high")

    def test_numbers_are_not_accepted_yet(self):
        self.assertFails("Lactate 5.2 raised the risk.", "contains a number")

    def test_no_features(self):
        self.assertFails("The model made a prediction.", "names no features")

    def test_missing_caveat(self):
        result = verify_summary("Lactate raised the risk.", PATIENT)
        self.assertFalse(result["verified"])
        self.assertIn("is missing the caveat sentence", result["problems"])

    def test_damaged_caveat_is_reported_once(self):
        # DistilBART turned "a diagnosis" into "diagnosis". That is one problem,
        # not ten unfamiliar words.
        damaged = CAVEAT.replace("a diagnosis", "diagnosis")
        result = verify_summary("Lactate raised the risk. " + damaged, PATIENT)
        self.assertEqual(result["problems"], ["is missing the caveat sentence"])

    def test_caveat_cannot_hide_other_text_inside_it(self):
        altered = CAVEAT.replace("model weighed", "model weighed, due to infection")
        result = verify_summary("Lactate raised the risk. " + altered, PATIENT)
        self.assertFalse(result["verified"])


class VerifierAcceptsGoodSummaries(unittest.TestCase):
    def test_paraphrase_with_synonyms_passes(self):
        result = verify_summary(
            "Lactic acid, ICU length of stay and pulse increased the estimated risk." + " " + CAVEAT,
            PATIENT)
        self.assertTrue(result["verified"], result)

    def test_partial_summary_passes_with_warning(self):
        result = verify_summary("Lactate raised the predicted risk." + " " + CAVEAT, PATIENT)
        self.assertTrue(result["verified"])
        self.assertIn("leaves out: ICULOS, HR, Temp, Age", result["warnings"])


class MentionMatching(unittest.TestCase):
    def test_ordinary_words_are_not_features(self):
        # "map", "ph" and "average" should not count as MAP, pH or Age.
        self.assertEqual(find_mentions("We map the average ph of the cohort."), [])

    def test_longest_phrase_wins(self):
        # One mention of SBP, not an extra generic "blood pressure" mention.
        mentions = find_mentions("Systolic blood pressure raised it.")
        self.assertEqual([c for _, c in mentions], [{"SBP"}])

    def test_group_term_is_grounded_by_any_member(self):
        entry = {"patient_id": "x", "top_features": [{"feature": "MAP", "shap_value": 0.2}]}
        self.assertTrue(verify_summary("Blood pressure raised the risk." + " " + CAVEAT, entry)["verified"])


if __name__ == "__main__":
    unittest.main()
