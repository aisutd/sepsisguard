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
    check_planted_errors,
    find_mentions,
    plant_wrong_direction,
    plant_wrong_feature,
    plant_wrong_number,
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


def fake_model(rewrite=lambda sentence: sentence):
    """A stand-in for the DistilBART pipeline: one output per input sentence.

    `rewrite` turns each input sentence into the model's "output", so a test
    can imitate the model changing, shortening or mangling a sentence.
    """
    return mock.Mock(side_effect=lambda inputs, **kwargs: [
        {"summary_text": rewrite(sentence)} for sentence in inputs])


class DistilbartBackend(unittest.TestCase):
    def test_caveat_is_added_by_code_not_left_to_the_model(self):
        # Whatever the model returns, the summary ends with the exact caveat,
        # and a missing final full stop does not glue it to the last sentence.
        for strip_stop in [False, True]:
            fake = fake_model(lambda s: s.rstrip(".") if strip_stop else s)
            with mock.patch.object(generator, "load_summarizer", return_value=fake):
                summary = generator.distilbart_summary(PATIENT)
            self.assertEqual(summary, generate_summary(PATIENT))
        # The model is given the feature sentences only, not the caveat.
        for given in fake.call_args.args[0]:
            self.assertNotIn("model weighed", given)

    def test_each_sentence_is_sent_to_the_model_separately(self):
        # The raised group and the lowered group must be two separate inputs,
        # so the model has no full stop between them to drop.
        fake = fake_model()
        with mock.patch.object(generator, "load_summarizer", return_value=fake):
            generator.distilbart_summary(PATIENT)
        self.assertEqual(fake.call_args.args[0], generator.template_sentences(PATIENT))
        self.assertEqual(len(fake.call_args.args[0]), 2)
        self.assertIn("raised", fake.call_args.args[0][0])
        self.assertNotIn("lowered", fake.call_args.args[0][0])
        self.assertIn("lowered", fake.call_args.args[0][1])
        self.assertEqual(fake.call_args.kwargs["min_length"], 1)

    def test_a_dropped_full_stop_no_longer_merges_the_groups(self):
        # The old failure: the model dropped the full stop between the two
        # groups, so "raised" and "lowered" shared a sentence. Each sentence is
        # now rewritten alone, so even with every full stop dropped the code
        # puts one back and the groups stay apart and verifiable.
        fake = fake_model(lambda s: s.rstrip("."))
        with mock.patch.object(generator, "load_summarizer", return_value=fake):
            summary = generator.distilbart_summary(PATIENT)
        result = verify_summary(summary, PATIENT)
        self.assertTrue(result["verified"], result["problems"])
        self.assertNotIn("both raised and lowered", " ".join(result["problems"]))

    def test_batch_returns_one_summary_per_patient_in_order(self):
        other = {"patient_id": "p2", "top_features": [
            {"feature": "Age", "shap_value": -0.2, "value": 71}]}
        # PATIENT has 2 sentences and `other` has 1, so this also checks that
        # the flat list of outputs is regrouped to the right patients.
        fake = mock.Mock(return_value=[{"summary_text": "Lactate raised it."},
                                       {"summary_text": "Age lowered it ."},
                                       {"summary_text": "Age lowered it ."}])
        with mock.patch.object(generator, "load_summarizer", return_value=fake):
            summaries = generator.distilbart_summaries([PATIENT, other])
            self.assertEqual(generator.generate_summaries([PATIENT, other], "distilbart"), summaries)
        self.assertEqual(summaries, ["Lactate raised it. Age lowered it. " + CAVEAT,
                                     "Age lowered it. " + CAVEAT])
        self.assertEqual(len(fake.call_args.args[0]), 3)
        self.assertEqual(generator.distilbart_summaries([]), [])

    def test_wrong_number_of_outputs_raises(self):
        # With one output missing, every later patient would get the wrong
        # sentences, so this must fail loudly instead of regrouping.
        fake = mock.Mock(return_value=[{"summary_text": "Lactate raised it."}])
        with mock.patch.object(generator, "load_summarizer", return_value=fake):
            with self.assertRaises(RuntimeError):
                generator.distilbart_summaries([PATIENT])

    def test_template_backend_still_works_one_at_a_time(self):
        self.assertEqual(generator.generate_summaries([PATIENT], "template"), [generate_summary(PATIENT)])


class PlantedErrors(unittest.TestCase):
    """Plant one known error in a correct summary and check the verifier fails it for that reason."""

    def setUp(self):
        self.correct = generate_summary(PATIENT)
        self.assertTrue(verify_summary(self.correct, PATIENT)["verified"])

    def assertCaught(self, damaged, failure_type):
        self.assertIsNotNone(damaged)
        self.assertNotEqual(damaged, self.correct)
        result = verify_summary(damaged, PATIENT)
        self.assertFalse(result["verified"], damaged)
        self.assertIn(failure_type, result["failure_types"], damaged)

    def test_wrong_feature_is_caught(self):
        damaged = plant_wrong_feature(self.correct, PATIENT)
        # O2Sat is the first column in feature_names.py that is not in this patient's top 5.
        self.assertTrue(damaged.startswith("Oxygen saturation (3.1) raised"), damaged)
        self.assertNotIn("Lactate", damaged)
        self.assertCaught(damaged, "wrong_feature")

    def test_wrong_direction_is_caught(self):
        damaged = plant_wrong_direction(self.correct, PATIENT)
        self.assertIn("lowered this patient's predicted sepsis risk the most", damaged)
        self.assertTrue(damaged.endswith(CAVEAT))  # the caveat is untouched
        self.assertCaught(damaged, "wrong_direction")

    def test_wrong_number_is_caught(self):
        damaged = plant_wrong_number(self.correct, PATIENT)
        self.assertTrue(damaged.startswith("Lactate (13.1) raised"), damaged)  # 3.1 + 10
        self.assertCaught(damaged, "wrong_number")

    def test_nothing_to_plant_returns_none(self):
        # No quoted value means no number to damage, and no "raised"/"lowered" means nothing to swap.
        no_values = {"patient_id": "x", "top_features": [{"feature": "HR", "shap_value": 0.4}]}
        self.assertIsNone(plant_wrong_number(generate_summary(no_values), no_values))
        self.assertIsNone(plant_wrong_direction("Heart rate shifted it.", no_values))

    def test_all_planted_errors_are_caught_on_the_real_patients(self):
        entries = json.loads(Path(VALUES_INPUT).read_text())
        results = check_planted_errors(entries)
        self.assertEqual(set(results), {"wrong_feature", "wrong_direction", "wrong_number"})
        for kind, (caught, planted) in results.items():
            self.assertEqual(planted, len(entries), kind)  # every patient can be damaged each way
            self.assertEqual(caught, planted, kind)


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


class DemoPatients(unittest.TestCase):
    """Round 2: the 8 demo patients (summarize_demo.py and its saved output)."""

    def test_feature_name_key_works_like_feature(self):
        """The demo contract's "feature_name" key gives the same summary as "feature"."""
        renamed = {"patient_id": "test", "top_features": [
            {"feature_name": f["feature"], "shap_value": f["shap_value"], "value": f["value"]}
            for f in PATIENT["top_features"]]}
        self.assertEqual(generate_summary(renamed), generate_summary(PATIENT))
        self.assertTrue(verify_summary(generate_summary(renamed), renamed)["verified"])

    def test_shared_file_used_only_at_the_snapshot_hour(self):
        """An explanation for the right patient at the wrong hour (or no hour) is ignored."""
        import summarize_demo

        patients = [{"patient_id": "p1", "snapshot_icu_hour": 30}]
        rows = [
            {"patient_id": "p1", "top_features": []},                   # no icu_hour
            {"patient_id": "p1", "icu_hour": 29, "top_features": []},   # wrong hour
            {"patient_id": "p9", "icu_hour": 30, "top_features": []},   # not a demo patient
            {"patient_id": "p1", "icu_hour": 30, "top_features": ["ok"]},
        ]
        with mock.patch.object(Path, "exists", return_value=True), \
             mock.patch.object(Path, "read_text", return_value=json.dumps(rows)):
            found = summarize_demo.shared_explanations(patients)
        self.assertEqual(found, {"p1": rows[3]})

    def test_saved_demo_file_is_complete_and_safe(self):
        """Every demo patient has one entry at their snapshot hour, and what the card
        shows (summary if verified, else fallback) always passes the checker."""
        import summarize_demo

        records = json.loads(summarize_demo.OUTPUT_PATH.read_text())
        patients = json.loads(summarize_demo.PATIENTS_PATH.read_text())
        self.assertEqual([(r["patient_id"], r["icu_hour"]) for r in records],
                         [(p["patient_id"], p["snapshot_icu_hour"]) for p in patients])
        for r in records:
            shown = r["summary"] if r["verified"] else r["fallback_summary"]
            self.assertTrue(verify_summary(shown, r)["verified"], r["patient_id"])
            self.assertEqual(r["verified"], r["failure_reason"] is None, r["patient_id"])
        # The frontend copies must not drift from the data files (in Round 1 the
        # 50-patient copy was left on the Week 4 run, so the card showed 29 failures, not 15).
        self.assertEqual(summarize_demo.FRONTEND_COPY.read_text(), summarize_demo.OUTPUT_PATH.read_text())
        sample = summarize_demo.REPO_ROOT / "data/sample/summaries_sample.json"
        sample_copy = summarize_demo.REPO_ROOT / "frontend/public/summaries_sample.json"
        self.assertEqual(sample_copy.read_text(), sample.read_text())


if __name__ == "__main__":
    unittest.main()
