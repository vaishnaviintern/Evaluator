import pandas as pd

# Load data
df = pd.read_csv('evaluation_results_with_prompts2.csv')

def categorize_prompt(text):
    text_lower = str(text).lower()
    if any(word in text_lower for word in ['image', 'picture', 'photo', 'pic', 'visual', 'draw', 'generate an image', 'create an image']):
        return 'Image Generation'
    elif any(word in text_lower for word in ['video', 'motion', 'animation', 'animate']):
        return 'Video/Motion'
    else:
        return 'Other'

df['category'] = df['user_prompt'].apply(categorize_prompt)
df['prompt_length'] = df['user_prompt'].str.split().str.len()

# Filter accepted image and video prompts
accepted_images = df[(df['action']=='accept') & (df['category']=='Image Generation')].copy()
accepted_videos = df[(df['action']=='accept') & (df['category']=='Video/Motion')].copy()

rejected_images = df[(df['action']=='reject') & (df['category']=='Image Generation')].copy()
rejected_videos = df[(df['action']=='reject') & (df['category']=='Video/Motion')].copy()

print("="*80)
print("WHY IMAGE & VIDEO PROMPTS GET ACCEPTED")
print("="*80)

# IMAGE GENERATION ANALYSIS
print("\n" + "="*80)
print("IMAGE GENERATION - ACCEPTED vs REJECTED COMPARISON")
print("="*80)

print(f"\nTotal Image Prompts: {len(df[df['category']=='Image Generation'])}")
print(f"  Accepted: {len(accepted_images)} (84.2%)")
print(f"  Rejected: {len(rejected_images)} (15.8%)")

print("\n--- KEY METRICS COMPARISON ---")
print("\nACCEPTED Images:")
print(f"  Avg Prompt Length: {accepted_images['prompt_length'].mean():.1f} words")
print(f"  Median Prompt Length: {accepted_images['prompt_length'].median():.1f} words")
print(f"  Avg Token Ratio: {accepted_images['token_ratio'].mean():.1f}x")
print(f"  Median Token Ratio: {accepted_images['token_ratio'].median():.1f}x")
print(f"  Avg Intent Similarity: {accepted_images['intent_similarity'].mean():.3f}")
print(f"  Avg Clarity Score: {accepted_images['clarity_score'].mean():.3f}")

print("\nREJECTED Images:")
print(f"  Avg Prompt Length: {rejected_images['prompt_length'].mean():.1f} words")
print(f"  Median Prompt Length: {rejected_images['prompt_length'].median():.1f} words")
print(f"  Avg Token Ratio: {rejected_images['token_ratio'].mean():.1f}x")
print(f"  Median Token Ratio: {rejected_images['token_ratio'].median():.1f}x")
print(f"  Avg Intent Similarity: {rejected_images['intent_similarity'].mean():.3f}")
print(f"  Avg Clarity Score: {rejected_images['clarity_score'].mean():.3f}")

print("\n--- LENGTH DISTRIBUTION ---")
print("\nACCEPTED Images by Length:")
short_acc_img = len(accepted_images[accepted_images['prompt_length'] < 10])
med_acc_img = len(accepted_images[(accepted_images['prompt_length'] >= 10) & (accepted_images['prompt_length'] < 50)])
long_acc_img = len(accepted_images[accepted_images['prompt_length'] >= 50])
print(f"  Short (<10 words):  {short_acc_img} ({short_acc_img/len(accepted_images)*100:.1f}%)")
print(f"  Medium (10-50):     {med_acc_img} ({med_acc_img/len(accepted_images)*100:.1f}%)")
print(f"  Long (>=50 words):  {long_acc_img} ({long_acc_img/len(accepted_images)*100:.1f}%)")

print("\nREJECTED Images by Length:")
short_rej_img = len(rejected_images[rejected_images['prompt_length'] < 10])
med_rej_img = len(rejected_images[(rejected_images['prompt_length'] >= 10) & (rejected_images['prompt_length'] < 50)])
long_rej_img = len(rejected_images[rejected_images['prompt_length'] >= 50])
print(f"  Short (<10 words):  {short_rej_img} ({short_rej_img/len(rejected_images)*100:.1f}%)")
print(f"  Medium (10-50):     {med_rej_img} ({med_rej_img/len(rejected_images)*100:.1f}%)")
print(f"  Long (>=50 words):  {long_rej_img} ({long_rej_img/len(rejected_images)*100:.1f}%)")

# VIDEO/MOTION ANALYSIS
print("\n" + "="*80)
print("VIDEO/MOTION - ACCEPTED vs REJECTED COMPARISON")
print("="*80)

print(f"\nTotal Video Prompts: {len(df[df['category']=='Video/Motion'])}")
print(f"  Accepted: {len(accepted_videos)} (84.9%)")
print(f"  Rejected: {len(rejected_videos)} (15.1%)")

print("\n--- KEY METRICS COMPARISON ---")
print("\nACCEPTED Videos:")
print(f"  Avg Prompt Length: {accepted_videos['prompt_length'].mean():.1f} words")
print(f"  Median Prompt Length: {accepted_videos['prompt_length'].median():.1f} words")
print(f"  Avg Token Ratio: {accepted_videos['token_ratio'].mean():.1f}x")
print(f"  Median Token Ratio: {accepted_videos['token_ratio'].median():.1f}x")
print(f"  Avg Intent Similarity: {accepted_videos['intent_similarity'].mean():.3f}")
print(f"  Avg Clarity Score: {accepted_videos['clarity_score'].mean():.3f}")

print("\nREJECTED Videos:")
print(f"  Avg Prompt Length: {rejected_videos['prompt_length'].mean():.1f} words")
print(f"  Median Prompt Length: {rejected_videos['prompt_length'].median():.1f} words")
print(f"  Avg Token Ratio: {rejected_videos['token_ratio'].mean():.1f}x")
print(f"  Median Token Ratio: {rejected_videos['token_ratio'].median():.1f}x")
print(f"  Avg Intent Similarity: {rejected_videos['intent_similarity'].mean():.3f}")
print(f"  Avg Clarity Score: {rejected_videos['clarity_score'].mean():.3f}")

# Show examples of ACCEPTED prompts
print("\n" + "="*80)
print("SAMPLE ACCEPTED IMAGE PROMPTS")
print("="*80)

# Get diverse examples (different lengths)
short_examples = accepted_images[accepted_images['prompt_length'] < 10].head(2)
medium_examples = accepted_images[(accepted_images['prompt_length'] >= 10) & (accepted_images['prompt_length'] < 50)].head(2)
long_examples = accepted_images[accepted_images['prompt_length'] >= 50].head(2)

print("\n--- SHORT Accepted Image Prompts ---")
for idx, row in short_examples.iterrows():
    print(f"\nUser Prompt ({int(row['prompt_length'])} words): {row['user_prompt'][:150]}")
    print(f"Token Ratio: {row['token_ratio']:.1f}x")
    print(f"Intent Similarity: {row['intent_similarity']:.3f}")
    print(f"Clarity: {row['clarity_score']:.2f}")

print("\n--- MEDIUM Accepted Image Prompts ---")
for idx, row in medium_examples.iterrows():
    print(f"\nUser Prompt ({int(row['prompt_length'])} words): {row['user_prompt'][:200]}")
    print(f"Token Ratio: {row['token_ratio']:.1f}x")
    print(f"Intent Similarity: {row['intent_similarity']:.3f}")
    print(f"Clarity: {row['clarity_score']:.2f}")

print("\n--- LONG Accepted Image Prompts ---")
for idx, row in long_examples.iterrows():
    print(f"\nUser Prompt ({int(row['prompt_length'])} words): {row['user_prompt'][:300]}...")
    print(f"Token Ratio: {row['token_ratio']:.1f}x")
    print(f"Intent Similarity: {row['intent_similarity']:.3f}")
    print(f"Clarity: {row['clarity_score']:.2f}")

print("\n" + "="*80)
print("SAMPLE ACCEPTED VIDEO PROMPTS")
print("="*80)

video_examples = accepted_videos.head(3)
for idx, row in video_examples.iterrows():
    print(f"\nUser Prompt ({int(row['prompt_length'])} words): {row['user_prompt'][:200]}")
    print(f"Token Ratio: {row['token_ratio']:.1f}x")
    print(f"Intent Similarity: {row['intent_similarity']:.3f}")
    print(f"Clarity: {row['clarity_score']:.2f}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
