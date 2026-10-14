"""Check that a generated summary only says things the SHAP input supports.

For one patient we know the ground truth: their top 5 features and the sign of
each SHAP value. A summary passes ("verified": true) only if ALL of these hold:

  1. It names at least one feature (an empty or generic summary is useless).
  2. It names no feature outside the patient's top 5.  <- main check (Issue #26)
  3. Any sentence that says features "raised" the risk only names features
     with a positive SHAP value, and any "lowered" sentence only names
     features with a negative one.
  4. It makes no diagnostic claim ("has sepsis", "diagnosed with ...").
     SHAP explains a model's estimate; it never confirms a diagnosis.
  5. It brings in no clinical reason the model never saw, such as
     "infection" or "kidney failure". These aren't features, so check 2
     can't catch them, but they read like reasons and would mislead.

Top-5 features the summary leaves out are reported as a warning but do not
fail it, because naming fewer reasons is incomplete rather than wrong.

Run on a whole file:
    python src/nlp_pipeline/verify_summary.py                  # template backend
    python src/nlp_pipeline/verify_summary.py --backend distilbart
"""

import argparse
import json
import re
import sys
from pathlib import Path

from feature_names import FEATURES, GROUP_TERMS
from generate_summary import generate_summary, parse_top_features

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_INPUT = REPO_ROOT / "data/sample/shap_top_features.json"
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


def find_mentions(text):
    """Return each feature mention in text as (matched text, set of columns).

    Matched characters are blanked out as we go so a shorter phrase cannot
    match again inside a longer one that was already counted.
    """
    mentions = []
    remaining = text
    for pattern, columns in TERM_INDEX:
        for match in pattern.finditer(remaining):
            mentions.append((match.group(0), columns))
        remaining = pattern.sub(lambda m: " " * len(m.group(0)), remaining)
    return mentions


# ---------------------------------------------------------------------------
# Step 2: the checks
# ---------------------------------------------------------------------------

UP_WORDS = re.compile(r"\b(raised|raise|raises|increased|increase|increases|pushed up|higher)\b", re.I)
DOWN_WORDS = re.compile(r"\b(lowered|lower|lowers|decreased|decrease|decreases|reduced|reduce|reduces|pushed down)\b", re.I)
DIAGNOSIS_CLAIMS = re.compile(
    r"\b(has sepsis|is septic|diagnosed with|diagnosis of sepsis|"
    r"confirms? sepsis|will develop sepsis|sepsis is present)\b",
    re.I,
)
# Clinical ideas that are NOT model inputs. A summary that offers one of them
# as a reason is inventing it. Extend this list as new failures show up.
OUTSIDE_CLAIMS = re.compile(
    r"\b(infection|infected|pneumonia|bacteria\w*|kidney|renal|liver|hepatic|"
    r"organ (?:failure|dysfunction)|septic shock|shock|dehydrat\w*|"
    r"antibiotics?|medication|history of)\b",
    re.I,
)


def verify_summary(summary, entry):
    """Check one summary against one patient entry.

    Returns a dict:
      verified   True only if every hard check passed
      problems   human-readable reasons it failed (empty when verified)
      warnings   soft issues, e.g. top-5 features the summary left out
      mentioned  column names the summary refers to
    """
    top = dict(parse_top_features(entry))  # column -> SHAP value
    problems, warnings = [], []

    # A mention is grounded if at least one column it could mean is in the
    # top 5 ("blood pressure" is fine if SBP, MAP or DBP is there).
    mentions = find_mentions(summary)
    mentioned = set()
    for phrase, columns in mentions:
        grounded = columns & top.keys()
        if grounded:
            mentioned |= grounded
        else:
            problems.append(f"mentions '{phrase}', which is not in this patient's top {len(top)}")

    # Check 1: says something.
    if not mentions:
        problems.append("names no features at all")

    # Check 3: direction, one sentence at a time. A sentence with both an
    # "up" and a "down" word is ambiguous, so we skip it and warn.
    for sentence in re.split(r"(?<=[.!?])\s+", summary):
        up, down = bool(UP_WORDS.search(sentence)), bool(DOWN_WORDS.search(sentence))
        if not (up or down):
            continue
        named = [(p, c & top.keys()) for p, c in find_mentions(sentence)]
        if up and down:
            if named:
                warnings.append(f"direction not checked (mixed wording): '{sentence}'")
            continue
        for phrase, columns in named:
            for column in columns:
                if up and top[column] < 0:
                    problems.append(f"says '{phrase}' raised the risk, but its SHAP value is negative")
                if down and top[column] > 0:
                    problems.append(f"says '{phrase}' lowered the risk, but its SHAP value is positive")

    # Check 4: no diagnosis.
    claim = DIAGNOSIS_CLAIMS.search(summary)
    if claim:
        problems.append(f"makes a diagnostic claim: '{claim.group(0)}'")

    # Check 5: no reasons from outside the model's inputs.
    for match in OUTSIDE_CLAIMS.finditer(summary):
        problems.append(f"brings in '{match.group(0)}', which is not a model input")

    # Soft check: which top features went unmentioned.
    omitted = [c for c in top if c not in mentioned]
    if omitted:
        warnings.append(f"leaves out: {', '.join(omitted)}")

    return {
        "verified": not problems,
        "problems": problems,
        "warnings": warnings,
        "mentioned": sorted(mentioned),
    }


# ---------------------------------------------------------------------------
# Step 3: run over a file of patients
# ---------------------------------------------------------------------------

def summarize_and_verify(entry, backend):
    """Generate with the chosen backend; fall back to the template if it fails.

    The template is grounded by construction, so a failed model summary is
    never shown. The rejected text is kept in the record for the notes.
    """
    summary = generate_summary(entry, backend)
    check = verify_summary(summary, entry)
    record = {
        "patient_id": entry["patient_id"],
        "summary": summary,
        "verified": check["verified"],
        "backend": backend,
    }
    if not check["verified"] and backend != "template":
        record["rejected_summary"] = summary
        record["rejection_reasons"] = check["problems"]
        record["summary"] = generate_summary(entry, "template")
        check = verify_summary(record["summary"], entry)
        record["verified"] = check["verified"]
        record["backend"] = "template (fallback)"
    if check["warnings"]:
        record["warnings"] = check["warnings"]
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backend", default="template", choices=["template", "distilbart"])
    parser.add_argument("--input", type=Path, default=None,
                        help="SHAP export (default: the real file if present, else the mock)")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    source = args.input or (REAL_INPUT if REAL_INPUT.exists() else MOCK_INPUT)
    if source == MOCK_INPUT:
        print(f"Real SHAP export not found; using mock data: {MOCK_INPUT.name}")
    entries = json.loads(source.read_text())

    records = [summarize_and_verify(e, args.backend) for e in entries]

    first_try_pass = sum("rejected_summary" not in r for r in records)
    print(f"\n{args.backend}: {first_try_pass}/{len(records)} passed on the first try")
    for r in records:
        status = "PASS" if "rejected_summary" not in r else "FAIL -> template"
        print(f"  [{status}] {r['patient_id']}: {r['summary']}")
        for reason in r.get("rejection_reasons", []):
            print(f"      rejected because it {reason}")
            print(f"      rejected text: {r['rejected_summary']}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2) + "\n")
    print(f"\nWrote {len(records)} records to {args.output.relative_to(REPO_ROOT)}")

    # Exit nonzero if anything we are about to show is unverified.
    return 0 if all(r["verified"] for r in records) else 1


if __name__ == "__main__":
    sys.exit(main())
