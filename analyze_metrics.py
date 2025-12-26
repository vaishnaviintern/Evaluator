import csv
import statistics

def analyze_csv(filename):
    actions = {}
    intent_similarity = []
    clarity_scores = []
    intent_preserved_count = 0
    added_value_count = 0
    over_enhanced_count = 0
    hallucination_count = 0
    total_rows = 0
    
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            
            # Action distribution
            action = row.get('action', 'unknown')
            actions[action] = actions.get(action, 0) + 1
            
            # Numeric metrics
            try:
                sim = float(row.get('intent_similarity', 0))
                intent_similarity.append(sim)
            except ValueError:
                pass
                
            try:
                clarity = float(row.get('clarity_score', 0))
                if clarity > 0: # filter out empty or 0 if relevant, or keep
                    clarity_scores.append(clarity)
            except ValueError:
                pass

            # Boolean metrics
            if row.get('intent_preserved') == 'True':
                intent_preserved_count += 1
            if row.get('added_value') == 'True':
                added_value_count += 1
            if row.get('over_enhanced') == 'True':
                over_enhanced_count += 1
            if row.get('hallucination') == 'True':
                hallucination_count += 1

    print(f"Total Rows: {total_rows}")
    print("\nAction Distribution:")
    for action, count in actions.items():
        percentage = (count / total_rows) * 100
        print(f"  {action}: {count} ({percentage:.2f}%)")

    if intent_similarity:
        print(f"\nAverage Intent Similarity: {statistics.mean(intent_similarity):.4f}")
    if clarity_scores:
        print(f"Average Clarity Score: {statistics.mean(clarity_scores):.4f}")

    print(f"\nIntent Preserved: {intent_preserved_count} ({intent_preserved_count/total_rows*100:.2f}%)")
    print(f"Added Value: {added_value_count} ({added_value_count/total_rows*100:.2f}%)")
    print(f"Over Enhanced: {over_enhanced_count} ({over_enhanced_count/total_rows*100:.2f}%)")
    print(f"Hallucination: {hallucination_count} ({hallucination_count/total_rows*100:.2f}%)")

if __name__ == "__main__":
    analyze_csv(r'c:\Users\PC 3\Documents\VP2\ThinkVelocity\Evaluator\evaluation_results_media_final.csv')
