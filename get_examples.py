import pandas as pd

# Load data
df = pd.read_csv('evaluation_results_with_prompts2.csv')
rejected = df[df['action']=='reject'].copy()

# Get examples for each category
def categorize_prompt(text):
    text_lower = str(text).lower()
    if any(word in text_lower for word in ['image', 'picture', 'photo', 'pic', 'visual', 'draw', 'generate an image', 'create an image']):
        return 'Image Generation'
    elif any(word in text_lower for word in ['video', 'motion', 'animation', 'animate']):
        return 'Video/Motion'
    elif any(word in text_lower for word in ['ppt', 'presentation', 'slides', 'powerpoint']):
        return 'Presentation'
    elif any(word in text_lower for word in ['website', 'web app', 'dashboard', 'logo', 'ui', 'ux']):
        return 'Web/App/Design'
    elif any(word in text_lower for word in ['write', 'document', 'report', 'letter', 'content', 'text']):
        return 'Text/Document'
    elif any(word in text_lower for word in ['data', 'analysis', 'sql', 'excel', 'tableau', 'analyst']):
        return 'Data/Analytics'
    elif any(word in text_lower for word in ['code', 'program', 'function', 'algorithm', 'python', 'java']):
        return 'Programming'
    elif any(word in text_lower for word in ['find', 'search', 'information', 'about', 'details']):
        return 'Information Request'
    else:
        return 'Other'

rejected['category'] = rejected['user_prompt'].apply(categorize_prompt)
rejected['prompt_length'] = rejected['user_prompt'].str.split().str.len()

# Get examples for the report
print("="*80)
print("SAMPLE REJECTED PROMPTS BY CATEGORY")
print("="*80)

categories = rejected['category'].value_counts().index[:5]

for cat in categories:
    cat_data = rejected[rejected['category']==cat].head(3)
    print(f"\n\n{'='*80}")
    print(f"CATEGORY: {cat} ({len(rejected[rejected['category']==cat])} rejections)")
    print(f"{'='*80}")
    
    for idx, row in cat_data.iterrows():
        print(f"\n--- Example {idx} ---")
        print(f"User Prompt: {row['user_prompt'][:200]}...")
        print(f"Enhanced Prompt: {row['enhanced_prompt'][:300]}...")
        print(f"\nMetrics:")
        print(f"  - Intent Similarity: {row['intent_similarity']:.3f}")
        print(f"  - Token Ratio: {row['token_ratio']:.1f}x")
        print(f"  - Clarity Score: {row['clarity_score']:.2f}")
        print(f"  - Intent Preserved: {row['intent_preserved']}")
        print(f"  - Over Enhanced: {row['over_enhanced']}")
        print(f"  - Hallucination: {row['hallucination']}")
        print(f"Reason: {row['reason']}")

# Get simple prompt examples
print(f"\n\n{'='*80}")
print("SIMPLE PROMPTS (<10 words) - High Rejection Rate Examples")
print(f"{'='*80}")

simple_rejected = rejected[rejected['prompt_length'] < 10].sort_values('token_ratio', ascending=False).head(5)

for idx, row in simple_rejected.iterrows():
    print(f"\n--- Example {idx} ---")
    print(f"User Prompt ({row['prompt_length']} words): {row['user_prompt']}")
    print(f"Enhanced Length: {len(row['enhanced_prompt'].split())} words (Ratio: {row['token_ratio']:.1f}x)")
    print(f"Intent Similarity: {row['intent_similarity']:.3f}")
    print(f"Over Enhanced: {row['over_enhanced']}")
    print(f"Reason: {row['reason'][:150]}...")
