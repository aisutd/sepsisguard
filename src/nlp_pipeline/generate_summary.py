"""Turn one patient's top SHAP features into a short plain-English summary.

Input format (Immaculate's export, data/sample/shap_top_features.json):
    {"patient_id": "p000123",
     "top_features": [{"feature": "Lactate", "shap_value": 0.42}, ...]}
Each feature may also carry a "value", as in
data/sample/shap_top_features_with_values.json (see export_values.py). When it
is there, the summary quotes it next to the feature name, e.g. "Lactate (3.1)
raised ...". It is the number the model was given, which can be carried forward
or filled in by the cleaning step, so the summary never calls it a measurement.

Two ways to write the summary ("backends"):

  template   (default) Fills a fixed sentence pattern with the feature names.
             Every word comes from the input or from fixed text we wrote, so
             it cannot invent a feature. Fast, needs no model download.

  distilbart Writes the template's feature sentences first, then asks
             DistilBART to rephrase them and adds the caveat itself. Kept so we
             can measure how often a language model drifts from its input. Its
             output must pass verify_summary.py before we use it.

What a SHAP value means here: a positive value pushed this patient's predicted
sepsis risk UP and a negative value pushed it DOWN. That sign is the only thing
the direction words ("raised", "lowered") come from. It is NOT the patient's
measured value (a positive SHAP for Lactate does not mean lactate is high), and
it is NOT a diagnosis. The summary says which way each feature pushed the
estimate and quotes the value the model was given, but never says whether that
value is high or low.
"""

import math
import re
from functools import lru_cache

from feature_names import readable_name

TOP_K = 5
DISTILBART_MODEL = "sshleifer/distilbart-cnn-12-6"

CAVEAT = (
    "These reflect what the model weighed, based on the values it was given, "
    "not a diagnosis."
)

# Yes/no columns. A quoted "(1)" next to "being in a medical ICU" says nothing
# useful, so these are described without their value.
BINARY_FEATURES = {"Gender", "Unit1", "Unit2"}


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------

def check_number(value, label):
    """Raise ValueError unless value is a finite int or float."""
    # bool is a subclass of int in Python, so rule it out explicitly.
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"Invalid {label}: {value!r}")


def parse_top_features(entry):
    """Check one patient entry and return its features, strongest first.

    Returns a list of (column_name, shap_value) pairs, at most TOP_K long,
    sorted by absolute SHAP value. Features with a SHAP value of exactly 0
    had no effect on the prediction, so they are dropped.
    """
    if not isinstance(entry, dict) or "top_features" not in entry:
        raise ValueError("Entry must be a dict with a 'top_features' list.")

    features = []
    for item in entry["top_features"]:
        name = item.get("feature")
        value = item.get("shap_value")

        readable_name(name)  # raises ValueError for an unknown column
        check_number(value, f"SHAP value for {name}")

        if value != 0:
            features.append((name, float(value)))

    if not features:
        raise ValueError(
            f"Patient {entry.get('patient_id')} has no nonzero SHAP features."
        )

    # The export should already be top 5 by |SHAP|, but sort and trim here
    # anyway so the summary never depends on the file being in order.
    features.sort(key=lambda pair: abs(pair[1]), reverse=True)
    return features[:TOP_K]


def parse_values(entry):
    """Return {column: value} for every feature in the entry that has a value."""
    values = {}
    for item in entry["top_features"]:
        if item.get("value") is None:
            continue
        check_number(item["value"], f"value for {item.get('feature')}")
        values[item["feature"]] = float(item["value"])
    return values


def format_number(value):
    """Print a value with at most 2 decimals: 110.0 -> '110', -0.031 -> '-0.03'."""
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


def join_names(names):
    """['a'] -> 'a';  ['a','b'] -> 'a and b';  ['a','b','c'] -> 'a, b, and c'."""
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + f", and {names[-1]}"


# ---------------------------------------------------------------------------
# Backend 1: template
# ---------------------------------------------------------------------------

def template_body(entry):
    """Build the feature sentences (everything except the caveat) from a fixed pattern.

    Features are split by the sign of their SHAP value and each group keeps
    its strongest-first order. The group with the strongest feature comes
    first, e.g.:

        "Lactate (3.1) raised this patient's predicted sepsis risk the most,
        followed by time in the ICU (42) and heart rate (118). Temperature
        (36.2) and age (71) lowered it."

    template_summary() adds the caveat after this.
    """
    features = parse_top_features(entry)
    values = parse_values(entry)

    def label(column):
        """The readable name, followed by the model's input value when we have one."""
        name = readable_name(column)
        if column in values and column not in BINARY_FEATURES:
            return f"{name} ({format_number(values[column])})"
        return name

    raised = [label(f) for f, v in features if v > 0]
    lowered = [label(f) for f, v in features if v < 0]

    # The group holding the single strongest feature goes first, so its
    # first name is the strongest overall and "the most" is true.
    groups = [("raised", raised), ("lowered", lowered)]
    if features[0][1] < 0:
        groups.reverse()
    (lead_verb, lead), (other_verb, other) = groups

    sentences = []
    if lead:
        risk = "this patient's predicted sepsis risk"
        if len(lead) > 1:
            sentences.append(
                f"{lead[0]} {lead_verb} {risk} the most, "
                f"followed by {join_names(lead[1:])}."
            )
        else:
            sentences.append(f"{lead[0]} {lead_verb} {risk}.")
    if other:
        sentences.append(f"{join_names(other)} {other_verb} it.")

    # Capitalize the first letter of each sentence without touching the
    # rest (lowercasing would break names like "AST" or "troponin I").
    return " ".join(s[0].upper() + s[1:] for s in sentences)


def template_summary(entry):
    """The finished template summary: the feature sentences, then the caveat."""
    return f"{template_body(entry)} {CAVEAT}"


# ---------------------------------------------------------------------------
# Backend 2: DistilBART rephrasing (needs torch + transformers + model download)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_summarizer():
    """Load DistilBART once. Imported here so the template path needs no torch."""
    import torch
    from transformers import pipeline

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return pipeline("summarization", model=DISTILBART_MODEL, device=device)


def distilbart_summary(entry):
    """Give DistilBART the template's feature sentences and return its rewrite.

    The caveat is fixed text, so it is added here and never given to the
    model. In a first run on 50 patients, DistilBART dropped it from 39 of 50
    summaries because a summarizer cuts what looks like filler.

    Nothing else is checked here on purpose: verify_summary.py decides
    whether the output is safe to show.
    """
    source_text = template_body(entry)
    result = load_summarizer()(
        source_text, min_length=10, max_length=80, do_sample=False
    )
    text = result[0]["summary_text"].strip()
    # DistilBART sometimes puts a space before punctuation ("rate ." -> "rate.").
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    if text and text[-1] not in ".!?":
        text += "."  # so the caveat starts a new sentence
    return f"{text} {CAVEAT}"


BACKENDS = {
    "template": template_summary,
    "distilbart": distilbart_summary,
}


def generate_summary(entry, backend="template"):
    """Generate a summary for one patient entry with the chosen backend."""
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend {backend!r}; use one of {list(BACKENDS)}")
    return BACKENDS[backend](entry)


if __name__ == "__main__":
    example = {
        "patient_id": "demo",
        "top_features": [
            {"feature": "Lactate", "shap_value": 0.42, "value": 3.1},
            {"feature": "ICULOS", "shap_value": 0.31, "value": 42},
            {"feature": "HR", "shap_value": 0.18, "value": 118},
            {"feature": "Temp", "shap_value": -0.12, "value": 36.2},
            {"feature": "Age", "shap_value": -0.05, "value": 71},
        ],
    }
    print(generate_summary(example))
