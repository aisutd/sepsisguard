import math
import re
from functools import lru_cache

import torch
from transformers import pipeline


MODEL_NAME = "sshleifer/distilbart-cnn-12-6"

# Expand abbreviations so the generated text is easier to read.
FEATURE_NAMES = {
    "lactate": "lactate",
    "resp_rate": "respiratory rate",
    "heart_rate": "heart rate",
    "sbp": "systolic blood pressure",
    "temperature": "temperature",
    "wbc": "white blood cell count",
    "spo2": "oxygen saturation",
}


@lru_cache(maxsize=1)
def load_summarizer():
    """Load the model once and reuse it across calls."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    return pipeline(
        "summarization",
        model=MODEL_NAME,
        device=device,
    )


def get_top_features(feature_importance):
    """Validate scores and return the three most influential features."""
    if not isinstance(feature_importance, dict) or not feature_importance:
        raise ValueError("Provide a nonempty dictionary of feature scores.")

    for feature, score in feature_importance.items():
        if feature not in FEATURE_NAMES:
            raise ValueError(f"Unknown feature: {feature}")

        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
        ):
            raise ValueError(f"Invalid score for {feature}: {score}")

    # Rank by magnitude only. This does not interpret contribution direction
    # or imply that a patient's measurement is high or low.
    ranked = sorted(
        feature_importance.items(),
        key=lambda item: abs(item[1]),
        reverse=True,
    )

    top_features = [
        FEATURE_NAMES[feature]
        for feature, score in ranked
        if score != 0
    ][:3]

    if not top_features:
        raise ValueError("At least one feature must have a nonzero score.")

    return top_features


def build_input(feature_importance):
    """Convert feature scores into factual source text for the model."""
    names = ", ".join(get_top_features(feature_importance))

    # These scores describe model influence, not patient measurements.
    # Supply source text for summarization rather than chat instructions.
    return (
        f"The prediction model identified these leading contributors "
        f"to its sepsis risk prediction: {names}. "
        "They were selected by the magnitude of their feature scores. "
        "These scores describe model influence, not measured patient "
        "values or a confirmed diagnosis."
    )


def generate_summary(feature_importance):
    """Generate text and return up to two complete sentences."""
    source_text = build_input(feature_importance)
    summarizer = load_summarizer()

    # Allow enough space for complete output. Sentence selection below
    # controls the final length without cutting words off mid-sentence.
    result = summarizer(
        source_text,
        min_length=10,
        max_length=100,
        do_sample=False,
    )

    raw_summary = result[0]["summary_text"].strip()

    # Remove the extra spaces DistilBART sometimes adds before punctuation.
    raw_summary = re.sub(r"\s+([.,!?;:])", r"\1", raw_summary)

    # Print the full generation so model behavior remains visible.
    print(f"Raw generation: {raw_summary}")

    # This simple splitter is sufficient for the controlled source text.
    # Discard unfinished trailing text and retain at most two sentences.
    sentences = re.findall(r"[^.!?]+[.!?]", raw_summary)
    summary = " ".join(sentence.strip() for sentence in sentences[:2])

    if not summary:
        raise ValueError("The model did not generate a complete sentence.")

    # Catch missing feature names. This is an omission check, not a full
    # factual verification system.
    expected = get_top_features(feature_importance)
    missing = [
        name for name in expected
        if name.lower() not in summary.lower()
    ]

    if missing:
        raise ValueError(f"Summary omitted features: {', '.join(missing)}")

    return summary


def main():
    # Mock inputs exercise different features and mixed signed scores.
    # Signed scores are ranked by magnitude; their direction is not reported.
    mock_inputs = [
        {"lactate": 0.42, "resp_rate": 0.31, "heart_rate": 0.18},
        {"sbp": 0.45, "heart_rate": 0.28, "temperature": 0.12},
        {"wbc": 0.38, "temperature": 0.29, "lactate": 0.20},
        {"spo2": -0.35, "resp_rate": 0.27, "sbp": -0.16},
    ]

    for number, features in enumerate(mock_inputs, start=1):
        print(f"\nExample {number}")
        print(f"Input: {features}")
        print(f"Source: {build_input(features)}")
        print(f"Summary: {generate_summary(features)}")


if __name__ == "__main__":
    main()