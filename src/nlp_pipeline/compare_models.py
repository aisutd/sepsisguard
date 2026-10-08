import time

from transformers import pipeline


# Two Hugging Face models we want to compare.
# Both are designed for text summarization.
MODEL_NAMES = [
    "facebook/bart-large-cnn",
    "sshleifer/distilbart-cnn-12-6",
]


# These are simple examples of the kind of information SepsisGuard
# may eventually need to turn into short, readable explanations.
SAMPLE_TEXTS = [
    (
        "The patient's heart rate increased over the last six hours. "
        "Systolic blood pressure decreased during the same period. "
        "The model identified heart rate and blood pressure as two of "
        "the strongest contributors to the patient's increased risk score."
    ),
    (
        "Lactate levels were elevated and white blood cell count was above "
        "the normal range. Temperature also increased. These measurements "
        "were identified as important contributors to the model's prediction."
    ),
    (
        "The patient's oxygen saturation remained stable, but respiratory "
        "rate increased and blood pressure decreased. The model assigned "
        "greater importance to respiratory rate and blood pressure than "
        "oxygen saturation."
    ),
]


def test_model(model_name):
    """Load one model and run all sample inputs through it."""

    print("\n" + "=" * 70)
    print(f"MODEL: {model_name}")
    print("=" * 70)

    # pipeline() loads both the model and its tokenizer.
    summarizer = pipeline(
        "summarization",
        model=model_name,
    )

    start_time = time.time()

    # Give every model the exact same inputs so the comparison is fair.
    for number, text in enumerate(SAMPLE_TEXTS, start=1):
        # NOTE: bart-large-cnn ships with a default min_length of 56, which is
        # larger than our 50-token cap and makes generation impossible.
        # We override the minimum explicitly so both models use the same limits.
        result = summarizer(
            text,
            min_new_tokens=10,   # shortest summary we accept
            max_new_tokens=50,   # longest summary we accept
            do_sample=False,     # deterministic output -> fair comparison
        )

        summary = result[0]["summary_text"]

        print(f"\nSample {number}")
        print(f"Input:   {text}")
        print(f"Output:  {summary}")

    elapsed_time = time.time() - start_time

    print(f"\nTotal inference time: {elapsed_time:.2f} seconds")


def main():
    # Run the same test for each candidate model.
    for model_name in MODEL_NAMES:
        test_model(model_name)


if __name__ == "__main__":
    main()