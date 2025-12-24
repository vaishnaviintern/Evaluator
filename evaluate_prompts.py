import pandas as pd
import numpy as np
import requests
import json
import re
from sentence_transformers import SentenceTransformer
from time import time

# CONFIG
CSV_PATH = "clean_prompt_dataset.csv"
OUTPUT_PATH = "evaluation_results2.csv"
OUTPUT_PATH_WITH_PROMPTS = "evaluation_results_with_prompts2.csv"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1:8b"

CLARITY_THRESHOLD = 0.7
LOG_EVERY_N = 10          # progress log frequency
CHECKPOINT_EVERY_N = 25   # save partial CSV

# LOAD DATA

df = pd.read_csv(CSV_PATH)
eval_df = df[df["enhanced_prompt"].notna()].copy()

total_rows = len(eval_df)

print(f"Total rows: {len(df)}")
print(f"Evaluating: {total_rows} enhanced prompts")

# EMBEDDINGS (cheap signals)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def intent_similarity(user_prompt, enhanced_prompt):
    return cosine(
        embedder.encode(user_prompt),
        embedder.encode(enhanced_prompt)
    )

def token_ratio(user_prompt, enhanced_prompt):
    return len(enhanced_prompt.split()) / max(len(user_prompt.split()), 1)

# JUDGE PROMPT

JUDGE_SYSTEM_PROMPT = """
You are an evaluator for a PROMPT ENHANCEMENT SYSTEM.

You will receive:
1. The original user prompt
2. The enhanced prompt

Your job is to EVALUATE the enhancement.

Rules:
- Do NOT rewrite the prompt
- Do NOT suggest improvements
- Output ONLY valid JSON
- No text before or after JSON

Evaluate:
1. intent_preserved
2. added_value
3. over_enhanced
4. hallucination
5. factual_correct (null if not applicable)
6. clarity_score (0.0-1.0)

Decision rules:
ACCEPT only if:
- intent_preserved = true
- hallucination = false
- factual_correct != false
- added_value = true
- clarity_score >= 0.7

Otherwise REJECT.

Return ONLY this JSON schema:
{
  "intent_preserved": boolean,
  "added_value": boolean,
  "over_enhanced": boolean,
  "hallucination": boolean,
  "factual_correct": boolean | null,
  "clarity_score": number,
  "action": "accept" | "reject",
  "reason": "short explanation"
}
"""

# SAFE JSON PARSER

def safe_json_parse(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found")
    return json.loads(match.group())

# OLLAMA JUDGE (with retry)

def ollama_judge(user_prompt, enhanced_prompt):
    payload = {
        "model": MODEL_NAME,
        "system": JUDGE_SYSTEM_PROMPT,
        "prompt": f"""
USER PROMPT:
{user_prompt}

ENHANCED PROMPT:
{enhanced_prompt}
""",
        "stream": False
    }

    # First attempt
    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    text = response.json().get("response", "")

    try:
        return safe_json_parse(text)
    except Exception:
        # Retry once
        retry_prompt = (
            "Your previous response was invalid JSON.\n"
            "Return ONLY valid JSON matching the schema.\n\n"
            + payload["prompt"]
        )

        payload["prompt"] = retry_prompt
        retry_response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        retry_text = retry_response.json().get("response", "")

        try:
            return safe_json_parse(retry_text)
        except Exception:
            # Final conservative fallback
            return {
                "intent_preserved": False,
                "added_value": False,
                "over_enhanced": True,
                "hallucination": True,
                "factual_correct": False,
                "clarity_score": 0.0,
                "action": "reject",
                "reason": "Invalid JSON from evaluator"
            }

# RUN EVALUATION

results = []
start_time = time()

for i, (idx, row) in enumerate(eval_df.iterrows(), start=1):
    sim = intent_similarity(row["user_prompt"], row["enhanced_prompt"])
    ratio = token_ratio(row["user_prompt"], row["enhanced_prompt"])
    verdict = ollama_judge(row["user_prompt"], row["enhanced_prompt"])

    results.append({
        "row_id": idx,
        "user_prompt": row["user_prompt"],
        "enhanced_prompt": row["enhanced_prompt"],
        "intent_similarity": round(sim, 4),
        "token_ratio": round(ratio, 2),
        **verdict
    })

    # Progress log
    if i % LOG_EVERY_N == 0 or i == total_rows:
        percent = (i / total_rows) * 100
        elapsed = time() - start_time
        print(f"[{i}/{total_rows}] ({percent:.1f}%) completed | elapsed {elapsed/60:.1f} min")

    # Checkpoint save
    if i % CHECKPOINT_EVERY_N == 0:
        pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)

# FINAL SAVE

out_df = pd.DataFrame(results)

# CSV 1: evaluation-only
eval_only_cols = [
    "row_id",
    "intent_similarity",
    "token_ratio",
    "intent_preserved",
    "added_value",
    "over_enhanced",
    "hallucination",
    "factual_correct",
    "clarity_score",
    "action",
    "reason"
]
out_df[eval_only_cols].to_csv(OUTPUT_PATH, index=False)

# CSV 2: prompts + evaluation
out_df.to_csv(OUTPUT_PATH_WITH_PROMPTS, index=False)

print("✅ Evaluation complete")
print(f"📄 Results saved to {OUTPUT_PATH}")
print(f"📄 Results with prompts saved to {OUTPUT_PATH_WITH_PROMPTS}")

