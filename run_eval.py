import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY) # create client --> the way that we will comunicate with API (our connection with GROQ)

# Load the golden set
with open("golden_set.json", "r", encoding="utf-8") as f:
    golden_set = json.load(f)

print(f"Loaded {len(golden_set)} texts from golden set.")
print(f"GROQ key loaded: {GROQ_API_KEY is not None}")

# Create Summarizing function
def summarize_text(text):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant", #the model that we use
        messages=[
            {
                "role": "user",
                "content": f"Summarize the following text in 2-3 sentences:\n\n{text}"
            }
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content #we want only the answer text

# Create the "Judge"
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
        model="llama-3.3-70b-versatile", #for Judge we use a better model
        messages=[
            {"role": "user", "content": judge_prompt}
        ],
        temperature=0,
    )
    raw_output = response.choices[0].message.content

    try:
        parsed = json.loads(raw_output)
        return parsed
    except json.JSONDecodeError:
        print(f"WARNING: Could not parse judge output as JSON: {raw_output}")
        return {"score": None, "reasoning": "PARSE_ERROR"}

# Test with just the first text in our golden set
results = []

for item in golden_set:
    print(f"Processing item {item['id']}...")

    summary = summarize_text(item["text"])
    judge_result = judge_summary(item["text"], summary)

    results.append({
        "id": item["id"],
        "topic": item["topic"],
        "summary": summary,
        "score": judge_result["score"],
        "reasoning": judge_result["reasoning"],
    })

# Print a simple report
print("\n--- RESULTS ---")
for r in results:
    print(f"ID {r['id']} ({r['topic']}): score={r['score']} - {r['reasoning']}")