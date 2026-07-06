import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

with open("golden_set.json", "r", encoding="utf-8") as f:
    golden_set = json.load(f)

with open("baseline.json", "r", encoding="utf-8") as f:
    baseline_data = json.load(f)


def summarize_text(text, temperature):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": f"Summarize the following text in 2-3 sentences:\n\n{text}"
            }
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content


def judge_summary(original_text, summary):
    judge_prompt = f"""You are an expert evaluator of text summaries. Rate the following summary on a scale of 1-5 based on these criteria:
- Coverage: Does it capture the key points of the original text?
- Accuracy: Does it avoid introducing false information not in the original?
- Conciseness: Is it appropriately brief without losing important meaning?

Original text:
{original_text}

Summary to evaluate:
{summary}

Respond ONLY with valid JSON in this exact format, with no other text before or after:
{{"score": <integer 1-5>, "reasoning": "<one sentence explanation>"}}"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": judge_prompt}],
        temperature=0,
    )
    raw_output = response.choices[0].message.content

    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        print(f"WARNING: Could not parse judge output: {raw_output}")
        return {"score": None, "reasoning": "PARSE_ERROR"}


# Simulate a "regression" using a much higher temperature
REGRESSION_TEMPERATURE = 2.0

NUM_RUNS = 3

run_averages = []

for run_number in range(1, NUM_RUNS + 1):
    print(f"\n=== RUN {run_number}/{NUM_RUNS} ===")
    results = []

    for item in golden_set:
        summary = summarize_text(item["text"], temperature=REGRESSION_TEMPERATURE)
        judge_result = judge_summary(item["text"], summary)
        results.append({
            "id": item["id"],
            "score": judge_result["score"],
        })

    run_average = sum(r["score"] for r in results) / len(results)
    print(f"Run {run_number} average: {run_average:.2f}")
    run_averages.append(run_average)

# Average across all runs — this is our final, more stable signal
current_average = sum(run_averages) / len(run_averages)
baseline_average = baseline_data["average_score"]

print("\n--- COMPARISON ---")
print(f"Individual run averages: {[round(a, 2) for a in run_averages]}")
print(f"Baseline average:        {baseline_average:.2f}")
print(f"Current average (mean of {NUM_RUNS} runs): {current_average:.2f}")

THRESHOLD = 0.25
drop = baseline_average - current_average

if drop > THRESHOLD:
    print(f"\n❌ REGRESSION DETECTED: score dropped by {drop:.2f} (threshold: {THRESHOLD})")
    exit(1)
else:
    print(f"\n✅ OK: score drop ({drop:.2f}) within acceptable threshold ({THRESHOLD})")
    exit(0)