"""Tests for generate_summary.py and verify_summary.py.

Run from the repo root (standard library only, no extra packages):
    python -m unittest discover -s src/nlp_pipeline -v

The "bad summary" cases imitate the mistakes a language model makes when
rewriting the input: adding a feature, swapping a direction, quoting the wrong
number, using a synonym for a feature that isn't there, or turning a risk
estimate into a diagnosis.
"""

import json
import unittest
from pathlib import Path
from unittest import mock

import generate_summary as generator
from generate_summary import (
    CAVEAT,
    format_number,
    generate_summary,
    parse_top_features,
    parse_values,
)
from verify_summary import (
    MOCK_INPUT,
    VALUES_INPUT,
    find_mentions,
    summarize_and_verify,
    verify_summary,
)

# Lactate, ICULOS and HR push risk up; Temp and Age push it down. The value
# is the number the model was given for that feature.
PATIENT = {
    "patient_id": "test",
    "top_features": [
        {"feature": "Lactate", "shap_value": 0.42, "value": 3.1},
        {"feature": "ICULOS", "shap_value": 0.31, "value": 42},
        {"feature": "HR", "shap_value": 0.18, "value": 118},
        {"feature": "Temp", "shap_value": -0.12, "value": 36.2},
        {"feature": "Age", "shap_value": -0.05, "value": 71},
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
            "Lactate (3.1) raised this patient's predicted sepsis risk the most, "
            "followed by time in the ICU (42) and heart rate (118). Temperature "
            "(36.2) and age (71) lowered it. " + CAVEAT,
        )

    def test_every_real_patient_passes(self):
        """All 50 real patients (SHAP computed on real values) pass, with values quoted."""
        entries = json.loads(Path(VALUES_INPUT).read_text())
        self.assertGreaterEqual(len(entries), 20)
        for entry in entries:
            result = verify_summary(generate_summary(entry), entry)
            self.assertTrue(result["verified"], (entry["patient_id"], result["problems"]))

    def test_values_are_quoted_but_not_for_yes_no_features(self):
        entry = {"patient_id": "x", "top_features": [
            {"feature": "Lactate", "shap_value": 0.4, "value": 3.14159},
            {"feature": "Unit1", "shap_value": -0.2, "value": 1.0}]}
        self.assertTrue(generate_summary(entry).startswith(
            "Lactate (3.14) raised this patient's predicted sepsis risk. "
            "Being in a medical ICU lowered it. "))

    def test_missing_value_is_left_out(self):
        entry = {"patient_id": "x", "top_features": [
            {"feature": "HR", "shap_value": 0.4, "value": 118},
            {"feature": "Age", "shap_value": 0.2}]}
        self.assertIn("Heart rate (118) raised this patient's predicted sepsis risk "
                      "the most, followed by age.", generate_summary(entry))

    def test_number_formatting(self):
        self.assertEqual(
            [format_number(v) for v in [110.0, 3.1, -0.031, -0.001, 0.0, 52.014, 100]],
            ["110", "3.1", "-0.03", "0", "0", "52.01", "100"])

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
            {"patient_id": "x", "top_features": [{"feature": "HR", "shap_value": 0.1, "value": "98"}]},
            {"patient_id": "x", "top_features": [{"feature": "HR", "shap_value": 0.1, "value": float("inf")}]},
            {"patient_id": "x"},
        ]:
            with self.assertRaises(ValueError, msg=bad):
                generate_summary(bad)

    def test_parse_values_skips_features_without_one(self):
        entry = {"patient_id": "x", "top_features": [
            {"feature": "HR", "shap_value": 0.4, "value": 118},
            {"feature": "Age", "shap_value": 0.2}]}
        self.assertEqual(parse_values(entry), {"HR": 118.0})


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

    def test_wrong_number(self):
        self.assertFails("Lactate (5.2) raised the risk.", "says 'Lactate' is 5.2, but the input value is 3.1")

    def test_right_number_on_the_wrong_feature(self):
        # 118 is the heart rate; attaching it to lactate is still wrong.
        self.assertFails("Lactate (118) and heart rate (3.1) raised the risk.",
                         "says 'Lactate' is 118")
        self.assertFails("Lactate (118) and heart rate (3.1) raised the risk.",
                         "says 'heart rate' is 3.1")

    def test_number_for_a_feature_with_no_input_value(self):
        entry = {"patient_id": "x", "top_features": [{"feature": "HR", "shap_value": 0.4}]}
        result = verify_summary("Heart rate (118) raised the risk. " + CAVEAT, entry)
        self.assertEqual(result["failure_types"], ["wrong_number"])
        self.assertIn("the input has no value for it", result["problems"][0])

    def test_number_with_no_feature_before_it(self):
        self.assertFails("About 118 raised the risk, and lactate did too.", "without a feature name before it")

    def test_number_in_a_sentence_with_no_feature_fails(self):
        # A bare number has no words for the wording check to object to.
        self.assertFails("Lactate (3.1) raised the risk. 42.", "without a feature name before it")

    def test_quoting_the_shap_value_instead_of_the_value_fails(self):
        self.assertFails("Lactate (0.42) raised the risk.", "says 'Lactate' is 0.42")

    def test_unit_or_judgement_next_to_a_number_fails(self):
        self.assertFails("Lactate (3.1 mmol) raised the risk.", "mmol")
        self.assertFails("Heart rate (118, high) raised the risk.", "high")

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

    def test_numbers_match_to_the_decimals_written(self):
        # 3.1 and 36.2 as given; "118" and "36" are roundings of 118 and 36.2.
        for text in ["Lactate (3.1) raised the risk.", "Lactate (3.10) raised the risk.",
                     "Heart rate (118) raised the risk.", "Temperature (36) lowered the risk."]:
            result = verify_summary(text + " " + CAVEAT, PATIENT)
            self.assertTrue(result["verified"], (text, result["problems"]))

    def test_digits_inside_a_feature_name_are_not_numbers(self):
        entry = {"patient_id": "x", "top_features": [{"feature": "O2Sat", "shap_value": 0.4, "value": 96}]}
        result = verify_summary("Oxygen saturation (96) raised the risk. " + CAVEAT, entry)
        self.assertTrue(result["verified"], result["problems"])
        result = verify_summary("O2 sat (96) raised the risk. " + CAVEAT, entry)
        self.assertTrue(result["verified"], result["problems"])

    def test_partial_summary_passes_with_warning(self):
        result = verify_summary("Lactate raised the predicted risk." + " " + CAVEAT, PATIENT)
        self.assertTrue(result["verified"])
        self.assertIn("leaves out: ICULOS, HR, Temp, Age", result["warnings"])


class DistilbartBackend(unittest.TestCase):
    def test_caveat_is_added_by_code_not_left_to_the_model(self):
        # Whatever the model returns, the summary ends with the exact caveat,
        # and a missing final full stop does not glue it to the last sentence.
        for model_text in ["Lactate (3.1) raised the risk.", "Lactate (3.1) raised the risk"]:
            fake = mock.Mock(return_value=[{"summary_text": model_text}])
            with mock.patch.object(generator, "load_summarizer", return_value=fake):
                summary = generator.distilbart_summary(PATIENT)
            self.assertEqual(summary, "Lactate (3.1) raised the risk. " + CAVEAT)
        # The model is given the feature sentences only, not the caveat.
        given = fake.call_args.args[0][0]
        self.assertNotIn("model weighed", given)
        self.assertTrue(given.startswith("Lactate (3.1) raised this patient's"))


    def test_batch_returns_one_summary_per_patient_in_order(self):
        other = {"patient_id": "p2", "top_features": [
            {"feature": "Age", "shap_value": -0.2, "value": 71}]}
        fake = mock.Mock(return_value=[{"summary_text": "Lactate raised it."},
                                       {"summary_text": "Age lowered it ."}])
        with mock.patch.object(generator, "load_summarizer", return_value=fake):
            summaries = generator.distilbart_summaries([PATIENT, other])
            self.assertEqual(generator.generate_summaries([PATIENT, other], "distilbart"), summaries)
        self.assertEqual(summaries, ["Lactate raised it. " + CAVEAT, "Age lowered it. " + CAVEAT])
        self.assertEqual(len(fake.call_args.args[0]), 2)
        self.assertEqual(generator.distilbart_summaries([]), [])

    def test_template_backend_still_works_one_at_a_time(self):
        self.assertEqual(generator.generate_summaries([PATIENT], "template"), [generate_summary(PATIENT)])


class FailureTypes(unittest.TestCase):
    def types(self, text):
        return verify_summary(text + " " + CAVEAT, PATIENT)["failure_types"]

    def test_each_failure_gets_its_type(self):
        self.assertEqual(self.types("Lactate and WBC raised the risk."), ["wrong_feature"])
        self.assertEqual(self.types("Lactate lowered the risk."), ["wrong_direction"])
        self.assertEqual(self.types("Lactate (9) raised the risk."), ["wrong_number"])
        self.assertEqual(self.types("Lactate raised the risk due to perfusion."), ["unsupported_wording"])
        self.assertEqual(verify_summary("Lactate raised the risk.", PATIENT)["failure_types"],
                         ["missing_caveat"])

    def test_one_summary_can_fail_in_several_ways(self):
        self.assertEqual(self.types("Lactate (9) lowered the risk and WBC raised it."),
                         ["wrong_direction", "wrong_feature", "wrong_number"])

    def test_a_good_summary_has_no_types(self):
        self.assertEqual(self.types("Lactate (3.1) raised the risk."), [])


class RecordsForTheSavedFile(unittest.TestCase):
    def run_with(self, text):
        """Run summarize_and_verify with a fake backend that writes `text`."""
        with mock.patch.dict(generator.BACKENDS, {"fake": lambda entry: text}):
            return summarize_and_verify(PATIENT, "fake")

    def test_passing_summary_has_no_failure_fields(self):
        record = self.run_with("Lactate (3.1) raised the risk. " + CAVEAT)
        self.assertTrue(record["verified"])
        self.assertNotIn("failure_reason", record)
        self.assertNotIn("fallback_summary", record)

    def test_failing_summary_keeps_its_text_and_gets_a_reason_and_fallback(self):
        bad = "Lactate (5.2) raised the risk. " + CAVEAT
        record = self.run_with(bad)
        self.assertFalse(record["verified"])
        self.assertEqual(record["summary"], bad)
        self.assertEqual(record["failure_types"], ["wrong_number"])
        self.assertIn("says 'Lactate' is 5.2, but the input value is 3.1", record["failure_reason"])
        # The fallback is the template text, and it is safe to show.
        self.assertEqual(record["fallback_summary"], generate_summary(PATIENT))
        self.assertTrue(verify_summary(record["fallback_summary"], PATIENT)["verified"])


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
