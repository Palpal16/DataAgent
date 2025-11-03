import json
import os
import subprocess
import csv
from io import StringIO
from typing import Dict, List, Tuple

def text_to_csv(text: str) -> List[List[str]]:
    """Convert text table to CSV rows.
    
    Handles both space-separated and pipe-separated formats.
    """
    lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
    if not lines:
        return []
    
    rows = []
    for line in lines:
        # Try splitting by multiple spaces first
        if '  ' in line:
            parts = [p.strip() for p in line.split() if p.strip()]
        # Try pipe separator
        elif '|' in line:
            parts = [p.strip() for p in line.split('|') if p.strip()]
        # Fallback to comma
        else:
            parts = [p.strip() for p in line.split(',') if p.strip()]
        
        if parts:
            rows.append(parts)
    
    return rows

def save_csv(rows: List[List[str]], filepath: str):
    """Save rows to CSV file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

def run_cpp_comparator(actual_csv: str, expected_csv: str, keys: List[str] = None) -> Dict:
    """Run the C++ comparator and return JSON result."""
    cmd = [
        './cpp_evaluator/build/resultcmp',
        '--actual', actual_csv,
        '--expected', expected_csv,
    ]
    
    if keys:
        cmd.extend(['--key', ','.join(keys)])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return json.loads(result.stdout)
    except subprocess.TimeoutExpired:
        return {"error": "Comparator timeout"}
    except json.JSONDecodeError:
        return {"error": f"Invalid JSON output: {result.stdout}"}
    except FileNotFoundError:
        return {"error": "Comparator binary not found. Run: cmake --build cpp_evaluator/build"}

def evaluate_dataset(dataset_path: str, output_dir: str = 'evaluation/csv_queries'):
    """Evaluate all queries in the dataset."""
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    os.makedirs(output_dir, exist_ok=True)
    results = []
    
    for i, item in enumerate(dataset):
        print(f"\nEvaluating query {i+1}/{len(dataset)}...")
        print(f"Prompt: {item.get('prompt', 'N/A')}")
        
        # Extract data
        generated = item.get('generated_data', '')
        ground_truth = item.get('ground_truth', '')
        
        if not generated or not ground_truth:
            print(f"  ⚠️  Skipping: missing data")
            results.append({
                'query_id': i,
                'prompt': item.get('prompt'),
                'error': 'Missing generated_data or ground_truth'
            })
            continue
        
        # Convert to CSV
        try:
            gen_rows = text_to_csv(generated)
            gt_rows = text_to_csv(ground_truth)
            
            if not gen_rows or not gt_rows:
                print(f"  ⚠️  Skipping: empty data after parsing")
                results.append({
                    'query_id': i,
                    'prompt': item.get('prompt'),
                    'error': 'Empty data after parsing'
                })
                continue
            
            # Save to CSV
            gen_csv = os.path.join(output_dir, f'query_{i}_generated.csv')
            gt_csv = os.path.join(output_dir, f'query_{i}_ground_truth.csv')
            save_csv(gen_rows, gen_csv)
            save_csv(gt_rows, gt_csv)
            
            # Run comparator
            eval_result = run_cpp_comparator(gen_csv, gt_csv)
            eval_result['query_id'] = i
            eval_result['prompt'] = item.get('prompt')
            eval_result['sql_query'] = item.get('generated_sql', 'N/A')
            
            results.append(eval_result)
            
            # Print result
            if eval_result.get('equal'):
                print(f"  ✅ PASS")
            else:
                print(f"  ❌ FAIL")
                print(f"     Mismatched rows: {eval_result.get('mismatched_rows', 0)}")
                print(f"     Mismatched columns: {eval_result.get('mismatched_columns', [])}")
        
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results.append({
                'query_id': i,
                'prompt': item.get('prompt'),
                'error': str(e)
            })
    
    # Save evaluation results
    results_file = os.path.join(output_dir, 'evaluation_summary.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    total = len(results)
    passed = sum(1 for r in results if r.get('equal', False))
    failed = total - passed
    
    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total queries: {total}")
    print(f"Passed: {passed} ({100*passed/total if total else 0:.1f}%)")
    print(f"Failed: {failed} ({100*failed/total if total else 0:.1f}%)")
    print(f"\nDetailed results saved to: {results_file}")
    
    return results

if __name__ == '__main__':
    # Ensure comparator is built
    if not os.path.exists('./cpp_evaluator/build/resultcmp'):
        print("Building C++ comparator...")
        subprocess.run(['cmake', '-S', 'cpp_evaluator', '-B', 'cpp_evaluator/build', '-G', 'Ninja'], check=True)
        subprocess.run(['cmake', '--build', 'cpp_evaluator/build', '--config', 'Release'], check=True)
    
    evaluate_dataset('evaluation/query_dataset.json')