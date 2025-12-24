import pandas as pd
import re
from collections import Counter

# Load data
df = pd.read_csv('evaluation_results_with_prompts2.csv')
rejected = df[df['action']=='reject'].copy()
accepted = df[df['action']=='accept'].copy()

print("="*80)
print("REJECTION PATTERN ANALYSIS REPORT")
print("="*80)

# 1. PROMPT LENGTH ANALYSIS
print("\n" + "="*80)
print("1. PROMPT LENGTH ANALYSIS")
print("="*80)

rejected['prompt_length'] = rejected['user_prompt'].str.split().str.len()
accepted['prompt_length'] = accepted['user_prompt'].str.split().str.len()

print("\nRejected prompts - Length statistics:")
print(rejected['prompt_length'].describe())

print("\n\nPrompt length distribution (REJECTED):")
short_rej = len(rejected[rejected['prompt_length'] < 10])
medium_rej = len(rejected[(rejected['prompt_length'] >= 10) & (rejected['prompt_length'] < 50)])
long_rej = len(rejected[rejected['prompt_length'] >= 50])
print(f"  Short (<10 words):   {short_rej} ({short_rej/len(rejected)*100:.1f}%)")
print(f"  Medium (10-50):      {medium_rej} ({medium_rej/len(rejected)*100:.1f}%)")
print(f"  Long (>=50 words):   {long_rej} ({long_rej/len(rejected)*100:.1f}%)")

print("\n\nPrompt length distribution (ACCEPTED):")
short_acc = len(accepted[accepted['prompt_length'] < 10])
medium_acc = len(accepted[(accepted['prompt_length'] >= 10) & (accepted['prompt_length'] < 50)])
long_acc = len(accepted[accepted['prompt_length'] >= 50])
print(f"  Short (<10 words):   {short_acc} ({short_acc/len(accepted)*100:.1f}%)")
print(f"  Medium (10-50):      {medium_acc} ({medium_acc/len(accepted)*100:.1f}%)")
print(f"  Long (>=50 words):   {long_acc} ({long_acc/len(accepted)*100:.1f}%)")

# 2. PROMPT TYPE CATEGORIZATION
print("\n" + "="*80)
print("2. PROMPT TYPE CATEGORIZATION")
print("="*80)

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
accepted['category'] = accepted['user_prompt'].apply(categorize_prompt)

print("\nRejection by Category:")
print(rejected['category'].value_counts())

print("\n\nAcceptance by Category:")
print(accepted['category'].value_counts())

# 3. OVER-ENHANCEMENT PATTERNS
print("\n" + "="*80)
print("3. OVER-ENHANCEMENT PATTERNS")
print("="*80)

over_enh = rejected[rejected['over_enhanced']==True].copy()
print(f"\nTotal over-enhanced rejections: {len(over_enh)} / {len(rejected)} ({len(over_enh)/len(rejected)*100:.1f}%)")

print("\nOver-enhancement by category:")
print(over_enh['category'].value_counts())

print("\n\nToken ratio for over-enhanced prompts by category:")
over_enh['token_ratio_group'] = pd.cut(over_enh['token_ratio'], bins=[0, 10, 20, 30, 50, 200], labels=['<10x', '10-20x', '20-30x', '30-50x', '>50x'])
print(over_enh.groupby('category')['token_ratio'].agg(['mean', 'median', 'max']))

# 4. INTENT PRESERVATION FAILURES
print("\n" + "="*80)
print("4. INTENT PRESERVATION FAILURES")
print("="*80)

intent_lost = rejected[rejected['intent_preserved']==False].copy()
print(f"\nTotal intent preservation failures: {len(intent_lost)} / {len(rejected)} ({len(intent_lost)/len(rejected)*100:.1f}%)")

print("\nIntent lost by category:")
print(intent_lost['category'].value_counts())

print("\n\nIntent similarity for failed prompts by category:")
print(intent_lost.groupby('category')['intent_similarity'].agg(['mean', 'median', 'min']))

# 5. SPECIFIC PATTERNS
print("\n" + "="*80)
print("5. SPECIFIC REJECTION PATTERNS")
print("="*80)

# Pattern: Simple vs Complex
df['prompt_length'] = df['user_prompt'].str.split().str.len()
simple_prompts = rejected[rejected['prompt_length'] < 10]
print(f"\nSimple prompts (<10 words) - Rejection rate:")
total_simple = len(df[df['prompt_length'] < 10])
print(f"  Rejected: {len(simple_prompts)} / {total_simple} ({len(simple_prompts)/total_simple*100:.1f}%)")
print(f"  Avg token ratio: {simple_prompts['token_ratio'].mean():.1f}x")
print(f"  Over-enhanced: {simple_prompts['over_enhanced'].sum()} ({simple_prompts['over_enhanced'].sum()/len(simple_prompts)*100:.1f}%)")


# Pattern: By clarity score
print("\n\nClarity score distribution (rejected):")
print(rejected['clarity_score'].describe())

print("\n\nRejected despite high clarity (>=0.8):")
high_clarity_rej = rejected[rejected['clarity_score'] >= 0.8]
print(f"  Count: {len(high_clarity_rej)} ({len(high_clarity_rej)/len(rejected)*100:.1f}%)")
print(f"  Main reasons:")
print(high_clarity_rej.groupby(['over_enhanced', 'intent_preserved']).size())

print("\n\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
