"""Turn one patient's top SHAP features into a short plain-English summary.

Input format (Immaculate's export, data/sample/shap_top_features.json):
    {"patient_id": "p000123",
     "top_features": [{"feature": "Lactate", "shap_value": 0.42}, ...]}

Two ways to write the summary ("backends"):

  template   (default) Fills a fixed sentence pattern with the feature names.
             Every word comes from the input or from fixed text we wrote, so
             it cannot invent a feature. Fast, needs no model download.

  distilbart Writes the template text first, then asks DistilBART to rephrase
             it. Kept so we can measure how often a language model drifts from
             its input. Its output must pass verify_summary.py before we use it.

What a SHAP value means here: a positive value pushed this patient's predicted
sepsis risk UP and a negative value pushed it DOWN. It is NOT the patient's
measured value (a positive SHAP for Lactate does not mean lactate is high), and
it is NOT a diagnosis. The summary says which way each feature pushed the
estimate and nothing about the measurements themselves.
"""

import math
import re
from functools import lru_cache

from feature_names import readable_name

TOP_K = 5
DISTILBART_MODEL = "sshleifer/distilbart-cnn-12-6"

CAVEAT = (
    "These reflect what the model weighed, not measured values or a diagnosis."
)


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------

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

        # bool is a subclass of int in Python, so rule it out explicitly.
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"Invalid SHAP value for {name}: {value!r}")

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

def template_summary(entry):
    """Build the summary from a fixed sentence pattern.

    Features are split by the sign of their SHAP value and each group keeps
    its strongest-first order, e.g.:

        "Lactate, time in the ICU, and heart rate raised this patient's
        predicted sepsis risk. Temperature and age lowered it.
        These reflect what the model weighed, not measured values or a
        diagnosis."
    """
    features = parse_top_features(entry)

    raised = [readable_name(f) for f, v in features if v > 0]
    lowered = [readable_name(f) for f, v in features if v < 0]

    sentences = []
    if raised:
        sentences.append(
            f"{join_names(raised)} raised this patient's predicted "
            "sepsis risk."
        )
    if lowered:
        # Say "it" only when the previous sentence already named the risk.
        target = "it" if raised else "this patient's predicted sepsis risk"
        sentences.append(f"{join_names(lowered)} lowered {target}.")
    sentences.append(CAVEAT)

    # Capitalize the first letter of each sentence without touching the
    # rest (lowercasing would break names like "AST" or "troponin I").
    return " ".join(s[0].upper() + s[1:] for s in sentences)


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
    """Give DistilBART the template text and return its rewrite.

    No checking happens here on purpose: verify_summary.py decides whether
    the output is safe to show.
    """
    source_text = template_summary(entry)
    result = load_summarizer()(
        source_text, min_length=10, max_length=80, do_sample=False
    )
    text = result[0]["summary_text"].strip()
    # DistilBART sometimes puts a space before punctuation ("rate ." -> "rate.").
    return re.sub(r"\s+([.,!?;:])", r"\1", text)


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
            {"feature": "Lactate", "shap_value": 0.42},
            {"feature": "ICULOS", "shap_value": 0.31},
            {"feature": "HR", "shap_value": 0.18},
            {"feature": "Temp", "shap_value": -0.12},
            {"feature": "Age", "shap_value": -0.05},
        ],
    }
    print(generate_summary(example))
