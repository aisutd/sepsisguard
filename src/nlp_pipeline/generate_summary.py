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
             DistilBART to rephrase each sentence separately and adds the
             caveat itself. Kept so we can measure how often a language model
             drifts from its input. Its output must pass verify_summary.py
             before we use it.

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
DISTILBART_BATCH_SIZE = 16  # gains level off past about 16

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

def template_sentences(entry):
    """Build the feature sentences (everything except the caveat) from a fixed pattern.

    Returns a list with one sentence per direction group (at most two: one
    "raised", one "lowered"). Features are split by the sign of their SHAP
    value and each group keeps its strongest-first order. The group with the
    strongest feature comes first, e.g.:

        ["Lactate (3.1) raised this patient's predicted sepsis risk the most,
          followed by time in the ICU (42) and heart rate (118).",
         "Temperature (36.2) and age (71) lowered it."]

    The list form matters for DistilBART: each sentence is rewritten on its
    own, so the model can never merge the "raised" and "lowered" groups into
    one sentence (see distilbart_summaries).
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
    return [s[0].upper() + s[1:] for s in sentences]


def template_body(entry):
    """The feature sentences joined into one string, without the caveat."""
    return " ".join(template_sentences(entry))


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


def clean_model_sentence(text):
    """Tidy one sentence of model output so it can be joined with the others.

    Strips it, removes the space DistilBART sometimes puts before punctuation
    ("rate ." -> "rate."), and adds a full stop if there is no ending
    punctuation, so the next sentence never gets glued onto this one.
    """
    text = text.strip()
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    if text and text[-1] not in ".!?":
        text += "."
    return text


def distilbart_summaries(entries, batch_size=DISTILBART_BATCH_SIZE):
    """Rewrite each patient's template sentences with DistilBART, one sentence at a time.

    Why one sentence at a time: when the model was given both template
    sentences at once ("X raised ... the most, followed by A and B. Y lowered
    it."), it dropped the full stop between them in 27 of its 29 failures on
    the 50 patients. "raised" and "lowered" then sat in one sentence, and
    verify_summary.py cannot tie a feature to a direction in a sentence that
    says both. The 2 "true reversals" (p000019, p000060) were the same mistake
    in another form: the model moved the sentence break, so features ended up
    next to the wrong verb. Rewriting each sentence on its own means the model
    has no break to drop or move: each output is one sentence and we join them.

    All sentences from all patients are flattened into one list and sent in
    batches, which was about 3x faster than one call per patient (27.6 s to
    10.7 s at batch_size 16) with identical text, because decoding is greedy.
    We then regroup the outputs by patient, in order.

    min_length is 1 because each input is a single short sentence; the old
    value of 10 forced the model to pad a short sentence such as "Temperature
    lowered it." with extra words. max_length stays at 80.

    The caveat is fixed text, so it is added here and never given to the
    model (a summarizer cut it from 39 of 50 summaries as filler).

    Nothing else is checked here on purpose: verify_summary.py decides
    whether the output is safe to show.
    """
    if not entries:
        return []

    per_patient = [template_sentences(entry) for entry in entries]
    inputs = [sentence for sentences in per_patient for sentence in sentences]

    results = load_summarizer()(
        inputs, min_length=1, max_length=80, do_sample=False,
        batch_size=batch_size,
    )
    # Regrouping below relies on one output per input, in order. If the
    # counts differ, patients would silently get each other's sentences.
    if len(results) != len(inputs):
        raise RuntimeError(
            f"DistilBART returned {len(results)} outputs for {len(inputs)} sentences."
        )

    summaries = []
    position = 0
    for sentences in per_patient:
        outputs = results[position:position + len(sentences)]
        position += len(sentences)
        cleaned = (clean_model_sentence(r["summary_text"]) for r in outputs)
        # Skip empty outputs so they do not leave a double space; the
        # verifier then fails a summary whose sentences all came back empty.
        text = " ".join(s for s in cleaned if s)
        summaries.append(f"{text} {CAVEAT}")
    return summaries


def distilbart_summary(entry):
    """One patient's DistilBART summary (see distilbart_summaries)."""
    return distilbart_summaries([entry])[0]


BACKENDS = {
    "template": template_summary,
    "distilbart": distilbart_summary,
}

# Backends that are faster given many patients at once.
BATCH_BACKENDS = {
    "distilbart": distilbart_summaries,
}


def generate_summary(entry, backend="template"):
    """Generate a summary for one patient entry with the chosen backend."""
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend {backend!r}; use one of {list(BACKENDS)}")
    return BACKENDS[backend](entry)


def generate_summaries(entries, backend="template"):
    """Generate a summary for each entry, batching when the backend supports it."""
    if backend in BATCH_BACKENDS:
        return BATCH_BACKENDS[backend](entries)
    return [generate_summary(entry, backend) for entry in entries]


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
