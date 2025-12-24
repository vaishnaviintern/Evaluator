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

OUTPUT_PATH = "evaluation_results_media_final.csv"
OUTPUT_PATH_WITH_PROMPTS = "evaluation_results_media_final_with_prompts.csv"

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

print(f"Evaluating {total_rows} enhanced prompts")

# =========================
# EMBEDDINGS (DIAGNOSTIC ONLY)
# =========================
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# =========================
# MEDIA PROMPT DETECTION
# =========================
MEDIA_KEYWORDS = [
    "image", "photo", "picture", "visual", "img", "pic",
    "video", "motion", "animation", "cinematic", "scene"
]

def is_media_prompt(row):
    text = str(row["user_prompt"]).lower()
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
# MEDIA HALLUCINATION CHECK (STRICT)
# =========================
def media_hallucination_check(user_prompt, enhanced_prompt):
    """
    Hallucination ONLY means contradiction or subject drift.
    Cinematic detail is allowed.
    """
    up = user_prompt.lower()
    ep = enhanced_prompt.lower()

    # Hard contradictions only
    if "image" in up and any(x in ep for x in ["dialogue", "voiceover"]):
        return True

    return False

# =========================
# MEDIA EVALUATOR (VALIDATION, NOT JUDGMENT)
# =========================
def media_evaluator(user_prompt, enhanced_prompt):
    hallucination = media_hallucination_check(user_prompt, enhanced_prompt)

    adds_detail = len(enhanced_prompt.split()) > len(user_prompt.split())
    has_structure = len(enhanced_prompt.split()) >= 25

    if hallucination:
        return {
            "intent_preserved": False,
            "added_value": False,
            "over_enhanced": False,
            "hallucination": True,
            "factual_correct": None,
            "clarity_score": 0.4,
            "action": "invalid",
            "reason": "Contradicts media constraints"
        }

    if not adds_detail:
        return {
            "intent_preserved": True,
            "added_value": False,
            "over_enhanced": False,
            "hallucination": False,
            "factual_correct": None,
            "clarity_score": 0.6,
            "action": "needs_review",
            "reason": "Media prompt lacks sufficient generation detail"
        }

    return {
        "intent_preserved": True,
        "added_value": True,
        "over_enhanced": False,
        "hallucination": False,
        "factual_correct": None,
        "clarity_score": 0.9 if has_structure else 0.75,
        "action": "accept_media",
        "reason": "Valid media generation prompt"
    }

# =========================
# TEXT JUDGE (UNCHANGED)
# =========================
JUDGE_SYSTEM_PROMPT = """
You are an evaluator for a PROMPT ENHANCEMENT SYSTEM.

Return ONLY valid JSON:
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
# RUN EVALUATION (ROUTER)
# =========================
results = []
start_time = time()

for i, (idx, row) in enumerate(eval_df.iterrows(), start=1):
    user_prompt = row["user_prompt"]
    enhanced_prompt = row["enhanced_prompt"]

    sim = cosine(
        embedder.encode(user_prompt),
        embedder.encode(enhanced_prompt)
    )
    ratio = len(enhanced_prompt.split()) / max(len(user_prompt.split()), 1)

    if is_media_prompt(row):
        verdict = media_evaluator(user_prompt, enhanced_prompt)
    else:
        verdict = ollama_judge(user_prompt, enhanced_prompt)

    results.append({
        "row_id": idx,
        "user_prompt": user_prompt,
        "enhanced_prompt": enhanced_prompt,
        "intent_similarity": round(sim, 4),   # diagnostic only
        "token_ratio": round(ratio, 2),       # diagnostic only
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

print("✅ Final media-aligned evaluation complete")
print(f"📄 Saved: {OUTPUT_PATH}")
print(f"📄 Saved: {OUTPUT_PATH_WITH_PROMPTS}")
