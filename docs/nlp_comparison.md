# NLP model comparison

Compared facebook/bart-large-cnn and sshleifer/distilbart-cnn-12-6 on three
sample sentences about vitals and risk-score contributors.

## Results
- BART-large: output varied between samples and dropped facts
  (e.g. "Temperature also increased"). 3.39s total.
- DistilBART: nearly copied the input every time, keeping all facts
  with stray spaces before periods. 1.14s total.

## Which felt more consistent, and why
DistilBART was more consistent, but mostly because it echoes the input
rather than summarizing. BART-large summarizes more, but silently lost
clinically relevant details, which is a risk for SepsisGuard.

## Caveats
Only 3 short samples. Both models are trained on news text, not clinical
text. Timings are rough (Apple mps device, includes warm-up).