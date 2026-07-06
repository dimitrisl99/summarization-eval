# Summarization Eval Pipeline

A small MLOps-style project that benchmarks an LLM's summarization quality against
a fixed "golden set" of texts, using a second LLM as an automated judge. Built as a
learning project to improve my knowledge and skills on LLM & AI.

## What this does

1. Takes a fixed AI generated (for copywright reasons) set of 10 English texts (`golden_set.json`) covering varied topics
   (technology, health, economy, science, history, environment).
2. Sends each text to a "model under test" and generates a summary.
3. Sends the original text + summary to a separate "judge" model, which scores the
   summary from 1-5 based on coverage, accuracy, and conciseness.
4. Aggregates the scores into an average, which can be tracked over time as a
   baseline — a future run scoring meaningfully lower would indicate a regression.

## Models used

- **Generator (model under test):** `llama-3.1-8b-instant` (Groq)
- **Judge:** `llama-3.3-70b-versatile` (Groq)

Both are free-tier models accessed via Groq's API.

### Why the same provider for both generator and judge?

Using a stronger/larger model to judge a smaller model's output is a common and
reasonable pattern (it mirrors how a cheaper model might be used in production
while a more capable model evaluates its outputs). However, since both models
come from the same underlying model family (Llama), there's a known risk of
**self-preference bias** — a judge model may rate outputs more favorably if they
share stylistic similarities with its own family, compared to how an independent
model (e.g. a different provider) might score the same output.

This project uses same-family free models specifically because it was built for
learning purposes and to avoid incurring API costs. In a production setting, a
more rigorous approach would use an independent, ideally more capable, judge model
from a different provider to reduce this bias.

## Setup

1. Clone the repo and create a virtual environment:
```bash
   python -m venv .venv
```
2. Activate it and install dependencies:
```bash
   pip install -r requirements.txt
```
3. Create a `.env` file in the project root with your Groq API key:
```
GROQ_API_KEY=your_key_here
```
4. Run the eval:
```bash
   python run_eval.py
```

## Project structure
```commandline
eval_project/
├── .env                  # API keys (not committed)
├── .gitignore
├── golden_set.json        # Fixed set of texts used for evaluation
├── run_eval.py            # Main eval script
├── baseline.json          # Saved results from the most recent baseline run
└── requirements.txt
```
## Notes / learnings while building this
- Before trusting the judge model's scores, I ran a sanity check: fed it a
  deliberately unrelated/wrong summary to confirm it would actually assign a low
  score, rather than defaulting to high scores regardless of quality. It correctly
  scored the bad summary 1/5, confirming the judge is discriminative rather than
  just generous.
- On the initial 10-text golden set, `llama-3.1-8b-instant` scored 5/5 on every
  single item. This isn't a flaw in the eval — it reflects that the texts used are
  relatively straightforward for the model. A more challenging or adversarial
  golden set would likely surface more variation in scores.
- When simulating a regression by increasing the generator's temperature to an
  extreme value (1.8-2.0), a **single evaluation run was not a reliable signal** —
  repeated runs at the same temperature produced meaningfully different average
  scores (e.g. 4.60 vs 4.80) purely due to the generator's own randomness. This
  was fixed by running the regression scenario multiple times (3 runs) and
  averaging across runs, which produced a much more stable signal.
- At temperature=1.8, the average score dropped by ~0.2-0.3 (out of 5) versus
  baseline, which was borderline against the initial threshold. At temperature=2.0
  (max), the drop increased to ~0.43, clearly exceeding the threshold and
  correctly triggering the "regression detected" failure path (exit code 1). This
  confirms the pipeline can distinguish a real quality drop from normal run-to-run
  noise, and that `llama-3.1-8b-instant` is fairly robust to sampling randomness
  on this summarization task even under high temperature.
- Threshold was set to 0.25 (out of a 1-5 scale) based on empirical observation
  of the noise floor across repeated baseline-temperature runs, not chosen
  arbitrarily.

## Status

Core eval pipeline complete and validated: baseline tracking, LLM-as-judge
scoring, and regression detection (including multi-run averaging to reduce
noise) all confirmed working, with both a "pass" and a "fail" scenario observed.

Next step: wire this into a GitHub Actions workflow so the eval runs
automatically and blocks a deploy if a regression is detected.
