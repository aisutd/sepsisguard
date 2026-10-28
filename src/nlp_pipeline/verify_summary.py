"""Check that a generated summary only says things the SHAP input supports.

For one patient we know the ground truth: their top 5 features, the sign of
each SHAP value, and (when the export has them) the value the model was given
for each feature. A summary passes ("verified": true) only if ALL of these hold:

  1. It names at least one feature (an empty or generic summary is useless).
  2. It names no feature outside the patient's top 5.            wrong_feature
  3. Every sentence that names a feature says exactly one of "raised" or
     "lowered", and that matches the sign of each named feature's SHAP
     value: positive raises the risk, negative lowers it.        wrong_direction
  4. Every number sits right after the feature it describes and matches that
     feature's input value, to the number of decimals written.   wrong_number
  5. Every other word is one our template uses. This is the "fail closed"
     rule: instead of listing bad phrases (a diagnosis, an infection, a
     claim that a value is high, a cause), we accept only wording we can
     account for, so an unfamiliar phrase is rejected rather than waved
     through.                                                    unsupported_wording
  6. It keeps the caveat sentence (CAVEAT in generate_summary.py) word for
     word, so no summary reaches a reader without it.            missing_caveat

The name on the right is the failure type recorded for that problem.

Top-5 features the summary leaves out are reported as a warning but do not
fail it, because naming fewer reasons is incomplete rather than wrong.

Run on a whole file (needs data/sample/shap_top_features_with_values.json):
    python src/nlp_pipeline/verify_summary.py                  # template backend
    python src/nlp_pipeline/verify_summary.py --backend distilbart
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from feature_names import FEATURES, GROUP_TERMS
from generate_summary import (
    CAVEAT,
    format_number,
    generate_summaries,
    generate_summary,
    parse_top_features,
    parse_values,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
VALUES_INPUT = REPO_ROOT / "data/sample/shap_top_features_with_values.json"
MOCK_INPUT = Path(__file__).resolve().parent / "fixtures/mock_shap_top_features.json"
OUTPUT = REPO_ROOT / "data/sample/summaries_sample.json"


# ---------------------------------------------------------------------------
# Step 1: find every feature a summary mentions
# ---------------------------------------------------------------------------

def _build_term_index():
    """List every phrase that refers to a feature, longest phrase first.

    Each item is (compiled regex, set of column names it refers to). One
    phrase usually means one column; group terms like "blood pressure" can
    mean several.
    """
    terms = []
    for column, (name, aliases) in FEATURES.items():
        for phrase in [name, *aliases]:
            terms.append((phrase, {column}))
    for phrase, columns in GROUP_TERMS.items():
        terms.append((phrase, set(columns)))

    # Longest first, so "systolic blood pressure" claims its words before
    # the shorter group term "blood pressure" can match inside it.
    terms.sort(key=lambda t: len(t[0]), reverse=True)

    index = []
    for phrase, columns in terms:
        # Short codes like "HR", "MAP" or "pH" only count when written in
        # their exact case, so the ordinary word "map" is not read as a
        # feature. Multi-word phrases match in any case.
        is_code = " " not in phrase and any(c.isupper() for c in phrase)
        flags = 0 if is_code else re.IGNORECASE
        # Not touching a letter or digit on either side. We avoid \b
        # because some names end in ")" where \b would not match.
        pattern = re.compile(
            rf"(?<![A-Za-z0-9]){re.escape(phrase)}(?![A-Za-z0-9])", flags
        )
        index.append((pattern, columns))
    return index


TERM_INDEX = _build_term_index()


def find_mention_spans(text):
    """Return each feature mention as (matched text, columns, start, end).

    Matched characters are blanked out as we go so a shorter phrase cannot
    match again inside a longer one that was already counted. Blanking keeps
    the length, so start and end are positions in the original text.
    """
    spans = []
    remaining = text
    for pattern, columns in TERM_INDEX:
        for match in pattern.finditer(remaining):
            spans.append((match.group(0), columns, match.start(), match.end()))
        remaining = pattern.sub(lambda m: " " * len(m.group(0)), remaining)
    return sorted(spans, key=lambda span: span[2])


def find_mentions(text):
    """Return each feature mention in text as (matched text, set of columns)."""
    return [(phrase, columns) for phrase, columns, _, _ in find_mention_spans(text)]


# ---------------------------------------------------------------------------
# Step 2: the checks
# ---------------------------------------------------------------------------

# The only direction words we understand. Anything else ("shifted upward",
# "pushed down") is not accepted, because we could not check its direction.
UP_VERBS = ["raise", "raised", "raises", "increase", "increased", "increases"]
DOWN_VERBS = ["lower", "lowered", "lowers", "decrease", "decreased", "decreases",
              "reduce", "reduced", "reduces"]
UP_WORDS = re.compile(r"\b(" + "|".join(UP_VERBS) + r")\b", re.I)
DOWN_WORDS = re.compile(r"\b(" + "|".join(DOWN_VERBS) + r")\b", re.I)

# The fixed wording of the template ("... raised this patient's predicted
# sepsis risk the most, followed by X and Y. Z lowered it."). Feature names
# are not listed here because they are removed before this check.
TEMPLATE_WORDS = {"this", "patient's", "predicted", "estimated", "sepsis", "risk",
                  "the", "most", "followed", "by", "and", "it"}
ALLOWED_WORDS = TEMPLATE_WORDS | set(UP_VERBS) | set(DOWN_VERBS)

WORD = re.compile(r"[A-Za-z]+(?:'[a-z]+)?")
CAVEAT_WORDS = {w.lower() for w in WORD.findall(CAVEAT)}

NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def strip_mentions(text):
    """Blank out every feature phrase, leaving only the words around them."""
    for pattern, _ in TERM_INDEX:
        text = pattern.sub(" ", text)
    return text


def number_matches(written, expected):
    """True if `written` is `expected` rounded to the decimals `written` shows.

    "118" matches 118.4 and "36.2" matches 36.17, but "36.2" does not match 36.9.
    """
    decimals = len(written.split(".")[1]) if "." in written else 0
    return abs(float(written) - expected) <= 0.5 * 10 ** -decimals + 1e-9


def check_numbers(sentence, spans, top, values):
    """Return a problem for each number in one sentence that the input does not back.

    `spans` is find_mention_spans(sentence). A number belongs to the feature
    mentioned most recently before it, which is how the template writes it:
    "heart rate (118)".
    """
    # Blank the feature names so digits inside them ("O2 sat", "HCO3") are
    # not read as numbers.
    blanked = sentence
    for _, _, start, end in spans:
        blanked = blanked[:start] + " " * (end - start) + blanked[end:]

    problems = []
    for match in NUMBER.finditer(blanked):
        written = match.group(0)
        owners = [span for span in spans if span[3] <= match.start()]
        if not owners:
            problems.append(f"gives the number {written} without a feature name before it")
            continue
        phrase, columns, _, _ = owners[-1]
        grounded = columns & top.keys()
        if not grounded:
            continue  # the feature itself is already reported as outside the top 5
        expected = [values[c] for c in grounded if c in values]
        if not expected:
            problems.append(f"gives {written} for '{phrase}', but the input has no value for it")
        elif not any(number_matches(written, e) for e in expected):
            problems.append(f"says '{phrase}' is {written}, but the input value is {format_number(expected[0])}")
    return problems


def verify_summary(summary, entry):
    """Check one summary against one patient entry.

    Returns a dict:
      verified       True only if every hard check passed
      problems       human-readable reasons it failed (empty when verified)
      failure_types  the distinct types of those problems (see the top of this file)
      warnings       soft issues, e.g. top-5 features the summary left out
      mentioned      column names the summary refers to
    """
    top = dict(parse_top_features(entry))  # column -> SHAP value
    values = parse_values(entry)           # column -> value the model was given
    problems, kinds, warnings = [], set(), []

    def fail(kind, message):
        problems.append(message)
        kinds.add(kind)

    # Check 6: the caveat is present word for word. It is then removed so its
    # own wording ("not a diagnosis") is not judged by the checks below.
    if CAVEAT not in summary:
        fail("missing_caveat", "is missing the caveat sentence")
    body = summary.replace(CAVEAT, " ")

    # A mention is grounded if at least one column it could mean is in the
    # top 5 ("blood pressure" is fine if SBP, MAP or DBP is there).
    mentions = find_mentions(body)
    mentioned = set()
    for phrase, columns in mentions:
        grounded = columns & top.keys()
        if grounded:
            mentioned |= grounded
        else:
            fail("wrong_feature", f"mentions '{phrase}', which is not in this patient's top {len(top)}")

    # Check 1: says something.
    if not mentions:
        fail("no_features", "names no features at all")

    # Checks 3 and 4, one sentence at a time.
    for sentence in re.split(r"(?<=[.!?])\s+", body):
        # Numbers are checked in every sentence, so a bare number cannot hide
        # in one that names no feature.
        spans = find_mention_spans(sentence)
        for message in check_numbers(sentence, spans, top, values):
            fail("wrong_number", message)

        named = [(phrase, columns & top.keys()) for phrase, columns, _, _ in spans]
        named = [(p, c) for p, c in named if c]
        if not named:
            continue
        up, down = bool(UP_WORDS.search(sentence)), bool(DOWN_WORDS.search(sentence))
        if up and down:
            fail("wrong_direction", f"says both raised and lowered in one sentence, so no "
                                    f"direction can be checked: '{sentence.strip()}'")
        elif not (up or down):
            phrases = ", ".join(f"'{p}'" for p, _ in named)
            fail("wrong_direction", f"names {phrases} without saying whether it raised or lowered the risk")
        else:
            for phrase, columns in named:
                for column in columns:
                    if up and top[column] < 0:
                        fail("wrong_direction", f"says '{phrase}' raised the risk, but its SHAP value is negative")
                    if down and top[column] > 0:
                        fail("wrong_direction", f"says '{phrase}' lowered the risk, but its SHAP value is positive")

    # Check 5: nothing but template wording around the feature names. A
    # damaged caveat is already reported above, so its words are not listed
    # a second time as unfamiliar.
    allowed = ALLOWED_WORDS if CAVEAT in summary else ALLOWED_WORDS | CAVEAT_WORDS
    unknown = sorted({w.lower() for w in WORD.findall(strip_mentions(body))} - allowed)
    if unknown:
        fail("unsupported_wording", f"uses wording the template never uses: {', '.join(unknown)}")

    # Soft check: which top features went unmentioned.
    omitted = [c for c in top if c not in mentioned]
    if omitted:
        warnings.append(f"leaves out: {', '.join(omitted)}")

    return {
        "verified": not problems,
        "problems": problems,
        "failure_types": sorted(kinds),
        "warnings": warnings,
        "mentioned": sorted(mentioned),
    }


# ---------------------------------------------------------------------------
# Step 3: run over a file of patients
# ---------------------------------------------------------------------------

def summarize_and_verify(entry, backend, summary=None):
    """Check a summary from the chosen backend, generating it if not given.

    Pass `summary` when it was already written, e.g. by generate_summaries.

    The record's "summary" is exactly what the backend wrote and "verified"
    says whether it passed, so a failure stays visible in the saved file with
    its "failure_reason". A failed summary also gets a "fallback_summary":
    the template text, which is safe to show in its place.
    """
    if summary is None:
        summary = generate_summary(entry, backend)
    check = verify_summary(summary, entry)
    record = {
        "patient_id": entry["patient_id"],
        "summary": summary,
        "verified": check["verified"],
        "backend": backend,
    }
    if not check["verified"]:
        fallback = generate_summary(entry, "template")
        fallback_check = verify_summary(fallback, entry)
        if not fallback_check["verified"]:
            # The template is built from the input, so this means a bug here.
            raise RuntimeError(
                f"Template summary for {entry['patient_id']} failed verification: "
                f"{fallback_check['problems']}"
            )
        record["failure_reason"] = "; ".join(check["problems"])
        record["failure_types"] = check["failure_types"]
        record["fallback_summary"] = fallback
    if check["warnings"]:
        record["warnings"] = check["warnings"]
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backend", default="template", choices=["template", "distilbart"])
    parser.add_argument("--input", type=Path, default=VALUES_INPUT,
                        help="SHAP export to summarize (default: the export with feature values)")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    entries = json.loads(args.input.read_text())
    summaries = generate_summaries(entries, args.backend)
    records = [summarize_and_verify(e, args.backend, s) for e, s in zip(entries, summaries)]

    passed = sum(r["verified"] for r in records)
    patients = len({r["patient_id"] for r in records})
    print(f"\n{args.backend}: {passed}/{len(records)} summaries passed ({patients} patients, {args.input.name})")

    failed = [r for r in records if not r["verified"]]
    if failed:
        # One summary can fail in more than one way, so these can add up to more than the failures.
        counts = Counter(t for r in failed for t in r["failure_types"])
        print("Failures by type: " + ", ".join(f"{t} {n}" for t, n in counts.most_common()))
    for r in failed:
        print(f"\n  FAIL {r['patient_id']}: {r['failure_reason']}")
        print(f"       wrote: {r['summary']}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2) + "\n")
    print(f"\nWrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    sys.exit(main())
