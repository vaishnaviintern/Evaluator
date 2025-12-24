import pandas as pd
import numpy as np
import requests
import json
import re
from sentence_transformers import SentenceTransformer
from time import time

# =========================
# CONFIG
# =========================
CSV_PATH = "clean_prompt_dataset.csv"

OUTPUT_PATH = "evaluation_results_media_aware_v2.csv"
OUTPUT_PATH_WITH_PROMPTS = "evaluation_results_media_aware_v2_with_prompts.csv"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1:8b"

LOG_EVERY_N = 10
CHECKPOINT_EVERY_N = 25

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(CSV_PATH)
eval_df = df[df["enhanced_prompt"].notna()].copy()
total_rows = len(eval_df)

print(f"Total rows: {len(df)}")
print(f"Evaluating: {total_rows} enhanced prompts")

# =========================
# EMBEDDINGS (ONCE)
# =========================
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# =========================
# MEDIA DETECTION (IMPROVED)
# =========================
MEDIA_KEYWORDS = [
    "image", "photo", "picture", "visual","img", "pic",
    "video", "motion", "animation", "cinematic", "vid"
]

def is_media_prompt(row):
    text = row["user_prompt"].lower()

    # domain / intent based (if present)
    domain = str(row.get("domain", "")).lower()
    intent = str(row.get("intent", "")).lower()

    if any(k in domain for k in ["image", "video", "media"]):
        return True
    if any(k in intent for k in ["image", "video", "visual", "motion"]):
        return True
    if any(k in text for k in MEDIA_KEYWORDS):
        return True

    return False

# =========================
# LIGHT HALLUCINATION CHECK (MEDIA)
# =========================
def media_hallucination_check(user_prompt, enhanced_prompt):
    """
    Heuristic checks for obvious creative drift
    """
    up = user_prompt.lower()
    ep = enhanced_prompt.lower()

    forbidden_additions = [
        ("audio", up),
        ("voiceover", up),
        ("dialogue", up),
        ("story", up),
        ("character", up),
    ]

    for word, original in forbidden_additions:
        if word in ep and word not in original:
            return True

    return False

# =========================
# MEDIA EVALUATOR
# =========================
def media_evaluator(user_prompt, enhanced_prompt, sim, ratio):
    intent_preserved = sim >= 0.5
    added_value = len(enhanced_prompt.split()) > len(user_prompt.split())
    hallucination = media_hallucination_check(user_prompt, enhanced_prompt)

    # extreme expansion guardrail
    if ratio > 60:
        return {
            "intent_preserved": False,
            "added_value": False,
            "over_enhanced": True,
            "hallucination": True,
            "factual_correct": None,
            "clarity_score": 0.4,
            "action": "reject",
            "reason": "Extreme expansion likely indicates creative hallucination"
        }

    accept = intent_preserved and added_value and not hallucination

    return {
        "intent_preserved": intent_preserved,
        "added_value": added_value,
        "over_enhanced": False,  # not applicable for media
        "hallucination": hallucination,
        "factual_correct": None,
        "clarity_score": 0.9 if accept else 0.6,
        "action": "accept" if accept else "reject",
        "reason": "Media-aware heuristic evaluation"
    }

# =========================
# TEXT JUDGE PROMPT
# =========================
JUDGE_SYSTEM_PROMPT = """
You are an evaluator for a PROMPT ENHANCEMENT SYSTEM.

Rules:
- Output ONLY valid JSON
- No text before or after JSON

Return ONLY this JSON:
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

def safe_json_parse(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON found")
    return json.loads(match.group())

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

    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    text = response.json().get("response", "")

    try:
        return safe_json_parse(text)
    except Exception:
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

# =========================
# RUN EVALUATION
# =========================
results = []
start_time = time()

for i, (idx, row) in enumerate(eval_df.iterrows(), start=1):
    sim = cosine(
        embedder.encode(row["user_prompt"]),
        embedder.encode(row["enhanced_prompt"])
    )
    ratio = len(row["enhanced_prompt"].split()) / max(len(row["user_prompt"].split()), 1)

    if is_media_prompt(row):
        verdict = media_evaluator(
            row["user_prompt"],
            row["enhanced_prompt"],
            sim,
            ratio
        )
    else:
        verdict = ollama_judge(
            row["user_prompt"],
            row["enhanced_prompt"]
        )

    results.append({
        "row_id": idx,
        "user_prompt": row["user_prompt"],
        "enhanced_prompt": row["enhanced_prompt"],
        "intent_similarity": round(sim, 4),
        "token_ratio": round(ratio, 2),
        **verdict
    })

    if i % LOG_EVERY_N == 0 or i == total_rows:
        elapsed = (time() - start_time) / 60
        print(f"[{i}/{total_rows}] completed | {elapsed:.1f} min")

    if i % CHECKPOINT_EVERY_N == 0:
        pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)

# =========================
# FINAL SAVE
# =========================
out_df = pd.DataFrame(results)

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
out_df.to_csv(OUTPUT_PATH_WITH_PROMPTS, index=False)

print("✅ Media-aware evaluation v2 complete")
print(f"📄 Saved: {OUTPUT_PATH}")
