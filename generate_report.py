import pandas as pd
import numpy as np

# CONFIG
CSV_PATH = "clean_prompt_dataset.csv"
EVALUATION_PATH = "evaluation_results.csv"
OUTPUT_PATH = "comprehensive_evaluation_report.csv"

# LOAD DATA
df = pd.read_csv(CSV_PATH)
eval_df = pd.read_csv(EVALUATION_PATH)

print(f"Total prompts in dataset: {len(df)}")
print(f"Total evaluations: {len(eval_df)}")

# MERGE THE DATA
# The eval_df has row_id which corresponds to the index in the original dataset
merged_df = pd.merge(
    eval_df,
    df,
    left_on='row_id',
    right_index=True,
    how='left'
)

# SELECT AND REORDER COLUMNS FOR THE REPORT
report_columns = [
    'row_id',
    'user_prompt',
    'enhanced_prompt',
    'intent_similarity',
    'token_ratio',
    'intent_preserved',
    'added_value',
    'over_enhanced',
    'hallucination',
    'factual_correct',
    'clarity_score',
    'action',
    'reason',
    'domain',
    'intent',
    'mode',
    'llm_used'
]

# CREATE THE FINAL REPORT
report_df = merged_df[report_columns].copy()

# SAVE THE COMPREHENSIVE REPORT
report_df.to_csv(OUTPUT_PATH, index=False)

# GENERATE SUMMARY STATISTICS
print("\n" + "="*60)
print("EVALUATION SUMMARY")
print("="*60)

# Overall Stats
total = len(report_df)
accepted = (report_df['action'] == 'accept').sum()
rejected = (report_df['action'] == 'reject').sum()

print(f"\n[*] Overall Performance:")
print(f"  Total Evaluated: {total}")
print(f"  [+] Accepted: {accepted} ({accepted/total*100:.1f}%)")
print(f"  [-] Rejected: {rejected} ({rejected/total*100:.1f}%)")

# Quality Metrics
print(f"\n[#] Average Quality Scores:")
print(f"  Intent Similarity: {report_df['intent_similarity'].mean():.3f}")
print(f"  Clarity Score: {report_df['clarity_score'].mean():.3f}")
print(f"  Token Ratio: {report_df['token_ratio'].mean():.2f}x")

# Boolean Metrics
print(f"\n[>] Quality Indicators:")
print(f"  Intent Preserved: {(report_df['intent_preserved']==True).sum()}/{total} ({(report_df['intent_preserved']==True).sum()/total*100:.1f}%)")
print(f"  Added Value: {(report_df['added_value']==True).sum()}/{total} ({(report_df['added_value']==True).sum()/total*100:.1f}%)")
print(f"  Over-Enhanced: {(report_df['over_enhanced']==True).sum()}/{total} ({(report_df['over_enhanced']==True).sum()/total*100:.1f}%)")
print(f"  Hallucination: {(report_df['hallucination']==True).sum()}/{total} ({(report_df['hallucination']==True).sum()/total*100:.1f}%)")

# BREAK DOWN BY DOMAIN
print(f"\n[D] Performance by Domain:")
domain_stats = report_df.groupby('domain').agg({
    'action': lambda x: (x == 'accept').sum(),
    'row_id': 'count'
}).rename(columns={'action': 'accepted', 'row_id': 'total'})
domain_stats['acceptance_rate'] = (domain_stats['accepted'] / domain_stats['total'] * 100).round(1)
for domain, row in domain_stats.iterrows():
    print(f"  {domain}: {row['accepted']}/{row['total']} ({row['acceptance_rate']}%)")

# FAILURE ANALYSIS
print(f"\n[!] Common Failure Reasons:")
rejected_reasons = report_df[report_df['action'] == 'reject']['reason'].value_counts()
for reason, count in rejected_reasons.head(5).items():
    print(f"  * {reason}: {count} times")

print(f"\n[OK] Report saved to: {OUTPUT_PATH}")
print("="*60)
