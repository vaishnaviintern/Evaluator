# Prompt Enhancement Evaluator System

A comprehensive evaluation framework for assessing AI-powered prompt enhancement quality across text and media generation tasks.

---

## 📋 Overview

This repository contains **three progressive evaluator versions** designed to validate and score prompt enhancements using a combination of LLM-based evaluation and heuristic checks.

### Evaluator Versions

| Version | Script | Status | Description |
|---------|--------|--------|-------------|
| **v1** | `evaluate_prompts.py` | ✅ Stable | Basic LLM-based evaluation for text prompts |
| **v2** | `evaluate_prompts_media_aware_v2.py` | ✅ Stable | Adds media-aware heuristics for image/video prompts |
| **v3** | `evaluate_prompts_media_aligned_final.py` | 🎯 **Current** | Enhanced media validation with strict hallucination checks |

---

## 🔍 Evaluator Details

### 1. **v1: Basic Evaluator** (`evaluate_prompts.py`)

The foundational evaluator using LLM-as-a-judge methodology.

**Features:**
- Uses Ollama (Llama 3.1 8B) as evaluation judge
- Calculates intent similarity with sentence embeddings
- Measures token expansion ratio
- Retry logic for invalid JSON responses
- Checkpoint saving every 25 evaluations

**Evaluation Criteria:**
- Intent preservation
- Added value
- Over-enhancement detection
- Hallucination detection
- Factual correctness
- Clarity score (0.0-1.0)

**Output:**
- `evaluation_results2.csv` - Evaluation metrics only
- `evaluation_results_with_prompts2.csv` - Full dataset with prompts

---

### 2. **v2: Media-Aware Evaluator** (`evaluate_prompts_media_aware_v2.py`)

Adds intelligent routing for media generation prompts (images/videos).

**Improvements over v1:**
- ✅ **Media Detection**: Keyword + domain/intent-based routing
- ✅ **Dual Evaluation Path**: LLM for text, heuristics for media
- ✅ **Hallucination Guards**: Detects inappropriate audio/dialogue additions
- ✅ **Extreme Expansion Filter**: Rejects >60x token expansion

**Media Detection Keywords:**
```python
["image", "photo", "picture", "visual", "img", "pic",
 "video", "motion", "animation", "cinematic", "vid"]
```

**Output:**
- `evaluation_results_media_aware_v2.csv` - Evaluation metrics
- `evaluation_results_media_aware_v2_with_prompts.csv` - With prompts

---

### 3. **v3: Media-Aligned Final** (`evaluate_prompts_media_aligned_final.py`) 🎯

The **production-ready evaluator** with refined media validation logic.

**Key Enhancements:**
- 🔥 **Strict Media Validation**: Only flags true contradictions (not creative detail)
- 🔥 **Action Categories**: `accept`, `reject`, `invalid`, `needs_review`, `accept_media`
- 🔥 **Structured Media Checks**: Minimum 25-word threshold for generation prompts
- 🔥 **Clean Separation**: Media prompts use validation, not judgment

**Philosophy Change:**
```
v2: "Does this have hallucinations?"
v3: "Does this contradict the original request?"
```

**Example:**
```
User: "Create an image of a sunset"
Enhanced: "Create a vibrant sunset image with orange and purple hues, 
          reflections on water, golden hour lighting, 4K resolution"

v2: ❌ Reject (hallucination - added colors/details)
v3: ✅ Accept (valid creative expansion, no contradiction)
```

**Output:**
- `evaluation_results_media_final.csv` - Evaluation metrics
- `evaluation_results_media_final_with_prompts.csv` - With prompts

---

## 🚀 Getting Started

### Prerequisites

```bash
# Python 3.8+
pip install pandas numpy requests sentence-transformers
```

**Required Services:**
- [Ollama](https://ollama.ai/) running locally
- Model: `llama3.1:8b` pulled

```bash
# Pull the model
ollama pull llama3.1:8b

# Start Ollama server (if not running)
ollama serve
```

### Input Data Format

All evaluators expect a CSV file (`clean_prompt_dataset.csv`) with:

| Column | Required | Description |
|--------|----------|-------------|
| `user_prompt` | ✅ | Original user input |
| `enhanced_prompt` | ✅ | Enhanced version to evaluate |
| `domain` | ⚪ | Prompt domain (optional, helps media detection) |
| `intent` | ⚪ | User intent (optional, helps media detection) |

---

## 📖 Usage

### Running v1 (Basic Evaluator)

```bash
python evaluate_prompts.py
```

**Configuration:**
```python
CSV_PATH = "clean_prompt_dataset.csv"
OUTPUT_PATH = "evaluation_results2.csv"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1:8b"
CLARITY_THRESHOLD = 0.7
```

---

### Running v2 (Media-Aware)

```bash
python evaluate_prompts_media_aware_v2.py
```

**Configuration:**
```python
CSV_PATH = "clean_prompt_dataset.csv"
OUTPUT_PATH = "evaluation_results_media_aware_v2.csv"
LOG_EVERY_N = 10          # Progress logging
CHECKPOINT_EVERY_N = 25   # Intermediate saves
```

**Special Features:**
- Auto-detects media prompts via keywords
- Applies lighter validation for image/video tasks
- Flags >60x expansion as likely hallucination

---

### Running v3 (Media-Aligned Final) 🎯 **RECOMMENDED**

```bash
python evaluate_prompts_media_aligned_final.py
```

**Configuration:**
```python
CSV_PATH = "clean_prompt_dataset.csv"
OUTPUT_PATH = "evaluation_results_media_final.csv"
OUTPUT_PATH_WITH_PROMPTS = "evaluation_results_media_final_with_prompts.csv"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1:8b"

LOG_EVERY_N = 10
CHECKPOINT_EVERY_N = 25
```

**Why Use v3?**
- ✅ Most accurate for media generation prompts
- ✅ Reduces false positives (creative detail ≠ hallucination)
- ✅ Production-tested logic
- ✅ Clear action categories

---

## 📊 Output Format

All evaluators produce CSV files with these columns:

| Column | Type | Description |
|--------|------|-------------|
| `row_id` | int | Original dataset index |
| `intent_similarity` | float | Cosine similarity (0-1) - diagnostic |
| `token_ratio` | float | Enhanced/original length ratio - diagnostic |
| `intent_preserved` | bool | Original intent maintained |
| `added_value` | bool | Enhancement adds meaningful content |
| `over_enhanced` | bool | Excessive elaboration detected |
| `hallucination` | bool | False information introduced |
| `factual_correct` | bool/null | Factual accuracy (when applicable) |
| `clarity_score` | float | Quality rating (0.0-1.0 or 0-10) |
| `action` | string | Decision: `accept`, `reject`, `needs_review`, `accept_media`, `invalid` |
| `reason` | string | Evaluation explanation |

---

## 🔧 Advanced Configuration

### Adjusting Evaluation Thresholds

**In v3:**
```python
# Modify media hallucination strictness
def media_hallucination_check(user_prompt, enhanced_prompt):
    up = user_prompt.lower()
    ep = enhanced_prompt.lower()
    
    # Add custom contradiction rules
    if "image" in up and "audio" in ep:
        return True
    
    return False
```

### Changing Model

```python
# Use different Ollama model
MODEL_NAME = "llama3:70b"  # More accurate, slower
MODEL_NAME = "mistral:7b"  # Faster, less accurate
```

### Batch Processing

```python
# Process subset of data
eval_df = df[df["enhanced_prompt"].notna()].head(100)  # First 100 rows
```

---

## 📈 Performance Metrics

Based on analysis of 1,169 evaluations:

| Metric | v2 Results | Interpretation |
|--------|-----------|----------------|
| **Acceptance Rate** | 69.3% | Good enhancement quality |
| **Intent Preservation** | 89.8% | Strong alignment |
| **Hallucination Rate** | 12.0% | Needs monitoring |
| **Over-Enhancement** | 7.3% | Well controlled |
| **Value Addition** | 71.9% | Effective improvements |

**Speed:**
- ~5-10 seconds per evaluation (with Ollama)
- ~1,000 prompts ≈ 2-3 hours
- Checkpoint saves prevent data loss

---

## 🐛 Troubleshooting

### Common Issues

**1. `Connection Error: Ollama not running`**
```bash
# Start Ollama
ollama serve

# Verify it's running
curl http://localhost:11434/api/generate
```

**2. `Invalid JSON from evaluator`**
- Model occasionally returns malformed JSON
- v1 has retry logic
- v3 has fallback verdict

**3. `KeyError: 'enhanced_prompt'`**
- Check CSV has required columns
- Ensure no null values in `enhanced_prompt`

**4. Slow Evaluation**
```python
# Reduce checkpoint frequency
CHECKPOINT_EVERY_N = 50  # Default: 25

# Use smaller model
MODEL_NAME = "llama3.1:8b"  # vs llama3:70b
```

---

## 📂 Project Structure

```
Evaluator/
├── evaluate_prompts.py                           # v1: Basic evaluator
├── evaluate_prompts_media_aware_v2.py           # v2: Media-aware
├── evaluate_prompts_media_aligned_final.py      # v3: Production (MAIN)
├── clean_prompt_dataset.csv                      # Input data
├── evaluation_results2.csv                       # v1 output
├── evaluation_results_media_aware_v2.csv        # v2 output
├── evaluation_results_media_final.csv           # v3 output (MAIN)
├── analyze_patterns.py                           # Pattern analysis tool
├── analyze_accepted.py                           # Accepted prompts analyzer
├── generate_report.py                            # Stats generator
├── get_examples.py                               # Example extractor
└── README.md                                     # This file
```

---

## 🔬 Analysis Tools

### Pattern Analysis
```bash
python analyze_patterns.py
```
Identifies common rejection reasons and quality patterns.

### Accepted Prompts Analysis
```bash
python analyze_accepted.py
```
Extracts characteristics of successful enhancements.

### Report Generation
```bash
python generate_report.py
```
Creates statistical summary of evaluation results.

---

## 📝 Evaluation Criteria Explained

### Intent Preservation
**Definition:** Original user goal is maintained after enhancement.

**Example:**
```
Original: "Write a poem about nature"
Enhanced: "Create a 16-line nature poem in iambic pentameter"
✅ Intent preserved (still about writing a nature poem)
```

### Added Value
**Definition:** Enhancement provides meaningful improvements.

**Example:**
```
Original: "Explain AI"
Enhanced: "Explain artificial intelligence, covering machine learning, 
          neural networks, and real-world applications in healthcare and finance"
✅ Added value (more specific, structured)
```

### Over-Enhancement
**Definition:** Excessive detail beyond reasonable interpretation.

**Example:**
```
Original: "Design a logo"
Enhanced: "Design a minimalist logo using Helvetica Neue font, 
          #2C3E50 color, golden ratio proportions, vector format, 
          300 DPI, CMYK color space, trademarked symbol..."
❌ Over-enhanced (too many unasked constraints)
```

### Hallucination
**Definition:** Information not grounded in original request.

**v2 (stricter):** Any creative addition
**v3 (balanced):** Direct contradictions only

```
Original: "Create an image of a cat"
Enhanced: "Create a photorealistic image of a fluffy Persian cat"

v2: ❌ Hallucination (added "Persian", "fluffy")
v3: ✅ No hallucination (creative detail, not contradiction)
```

---

## 🎯 Best Practices

### For Production Use

1. **Use v3** (`evaluate_prompts_media_aligned_final.py`) for latest logic
2. **Enable checkpointing** to prevent data loss on long runs
3. **Monitor hallucination rate** - target <5%
4. **Review edge cases** manually (extreme ratios, low similarity)

### For Development

```python
# Test on small sample first
eval_df = df.head(50)

# Increase logging
LOG_EVERY_N = 1  # Log every evaluation
```

### For Custom Domains

**Add domain-specific keywords:**
```python
CODE_KEYWORDS = ["code", "function", "script", "program"]
DESIGN_KEYWORDS = ["design", "ui", "ux", "mockup"]

def is_code_prompt(row):
    return any(k in row["user_prompt"].lower() for k in CODE_KEYWORDS)
```

---

## 📊 Metrics Dashboard

After running evaluations, calculate key metrics:

```python
import pandas as pd

df = pd.read_csv("evaluation_results_media_final.csv")

print(f"Acceptance Rate: {(df['action'] == 'accept').mean():.1%}")
print(f"Intent Preservation: {df['intent_preserved'].mean():.1%}")
print(f"Hallucination Rate: {df['hallucination'].mean():.1%}")
print(f"Avg Clarity Score: {df['clarity_score'].mean():.2f}")
print(f"Median Token Ratio: {df['token_ratio'].median():.1f}x")
```

---

## 🤝 Contributing

### Reporting Issues
- Include evaluator version
- Provide sample prompts that fail
- Share configuration settings

### Suggesting Improvements
- Propose new heuristics for media validation
- Share domain-specific evaluation criteria
- Contribute analysis scripts

---

## 📄 License

This evaluation system is part of the ThinkVelocity prompt enhancement project.

---

## 🙏 Acknowledgments

- **Ollama** for local LLM inference
- **sentence-transformers** for semantic similarity
- **LLM-as-a-judge** methodology from OpenAI/Anthropic research

---

## 📞 Support

For questions or issues:
1. Check **Troubleshooting** section
2. Review **Common Issues** in output CSV
3. Open an issue with sample data

---

**Version:** 3.0 (Media-Aligned Final)  
**Last Updated:** December 24, 2025  
**Status:** ✅ Production Ready
#   E v a l u a t o r  
 