# ============================================================
# Evaluator Version 4
# Purpose: Semantic correctness validation for prompt enhancement
# ============================================================

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

OUTPUT_PATH = "evaluation_results_v4.csv"
OUTPUT_PATH_WITH_PROMPTS = "evaluation_results_v4_with_prompts.csv"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1:8b"

LOG_EVERY_N = 10
CHECKPOINT_EVERY_N = 25

# Similarity guardrails
SIMILARITY_LOW_MEDIA = 0.35
SIMILARITY_LOW_TEXT = 0.30

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(CSV_PATH)
eval_df = df[df["enhanced_prompt"].notna()].copy()
total_rows = len(eval_df)

print(f"Evaluator V4 | Evaluating {total_rows} enhanced prompts")

# =========================
# EMBEDDINGS (GUARDRAIL)
# =========================
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

# =========================
# MEDIA MODALITY DETECTION
# =========================
def is_media_prompt(user_prompt, enhanced_prompt):
    """
    Media = prompt intended to be executed directly
    by an image or video generation model.
    """
    up = user_prompt.lower()
    ep = enhanced_prompt.lower()

    # Explicit generation intent
    generation_phrases = [
        "generate an image", "create an image", "make an image",
        "generate a video", "create a video", "make a video",
        "render", "turn this image into a video","create an img","make a vid of"
    ]

    if any(p in up for p in generation_phrases):
        return True

    # Platform-aligned media structure (secondary signal)
    media_sections = [
        "## camera", "## visuals", "## scene",
        "## lighting", "## audio", "## music"
    ]

    if any(s in ep for s in media_sections):
        return True

    return False

# =========================
# LLM JUDGE (SIGNALS ONLY)
# =========================
JUDGE_SYSTEM_PROMPT = """
You are an evaluator for a PROMPT ENHANCEMENT SYSTEM.

Your role:
- Analyze semantic correctness ONLY
- Do NOT decide accept/reject
- Do NOT judge style or verbosity

Return ONLY valid JSON with these fields:
{
  "intent_preserved": boolean,
  "hallucination": boolean,
  "factual_correct": boolean | null,
  "added_value": boolean,
  "over_enhanced": boolean,
  "clarity_score": number,
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
        # Conservative fallback
        return {
            "intent_preserved": False,
            "hallucination": True,
            "factual_correct": False,
            "added_value": False,
            "over_enhanced": True,
            "clarity_score": 0.0,
            "reason": "Invalid JSON from LLM judge"
        }

# =========================
# DECISION LOGIC
# =========================
def decide_action(prompt_type, signals, sim):
    """
    Deterministic decision logic.
    """
    intent_ok = signals["intent_preserved"]
    hallucination = signals["hallucination"]
    factual = signals["factual_correct"]

    if prompt_type == "media":
        if sim < SIMILARITY_LOW_MEDIA:
            return "MEDIA_REJECT", "Low semantic similarity to original media intent"

        if not intent_ok or hallucination:
            return "MEDIA_REJECT", signals["reason"]

        return "MEDIA_ACCEPT", "Valid media prompt with preserved intent"

    else:  # text
        if sim < SIMILARITY_LOW_TEXT:
            return "REJECT", "Low semantic similarity to user intent"

        if not intent_ok or hallucination or factual is False:
            return "REJECT", signals["reason"]

        return "ACCEPT", "Valid enhanced prompt with preserved intent"

# =========================
# RUN EVALUATION
# =========================
results = []
start_time = time()

for i, (idx, row) in enumerate(eval_df.iterrows(), start=1):
    user_prompt = row["user_prompt"]
    enhanced_prompt = row["enhanced_prompt"]

    sim = intent_similarity(user_prompt, enhanced_prompt)
    ratio = token_ratio(user_prompt, enhanced_prompt)

    prompt_type = "media" if is_media_prompt(user_prompt, enhanced_prompt) else "text"

    signals = ollama_judge(user_prompt, enhanced_prompt)

    action, reason = decide_action(prompt_type, signals, sim)

    results.append({
        "row_id": idx,
        "prompt_type": prompt_type,
        "user_prompt": user_prompt,
        "enhanced_prompt": enhanced_prompt,
        "intent_similarity": round(sim, 4),
        "token_ratio": round(ratio, 2),
        **signals,
        "action": action,
        "final_reason": reason
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
    "prompt_type",
    "intent_similarity",
    "token_ratio",
    "intent_preserved",
    "hallucination",
    "factual_correct",
    "added_value",
    "over_enhanced",
    "clarity_score",
    "action",
    "final_reason"
]

out_df[eval_only_cols].to_csv(OUTPUT_PATH, index=False)
out_df.to_csv(OUTPUT_PATH_WITH_PROMPTS, index=False)

print("Evaluator V4 completed successfully")
print(f"Saved: {OUTPUT_PATH}")
print(f"Saved: {OUTPUT_PATH_WITH_PROMPTS}")
