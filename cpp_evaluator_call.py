import json
import os
import subprocess
import csv
from io import StringIO
from typing import Dict, List, Tuple


def run_cpp_comparator(actual_csv: str, expected_csv: str, keys: List[str] = None) -> Dict:
    """Run the C++ comparator and return JSON result."""
    cmd = [
        './cpp_evaluator/build/resultcmp',
        '--actual', actual_csv,
        '--expected', expected_csv,
    ]
    
    if keys:
        cmd.extend(['--key', ','.join(keys)])
    

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    return result.stdout

if __name__ == '__main__':
    # Ensure comparator is built
    if not os.path.exists('./cpp_evaluator/build/resultcmp'):
        print("Building C++ comparator...")
        subprocess.run(['cmake', '-S', 'cpp_evaluator', '-B', 'cpp_evaluator/build', '-G', 'Ninja'], check=True)
        subprocess.run(['cmake', '--build', 'cpp_evaluator/build', '--config', 'Release'], check=True)

    PREFIX = 'gpt' # options: 'my', 'claude', 'gpt'
    index = 2
    path_gt = f"evaluation/csv_queries/{PREFIX}_{index}_gt.csv"
    path_gen = f"evaluation/csv_queries/{PREFIX}_{index}_gen.csv"
    res = run_cpp_comparator(path_gt, path_gen)
    print(res)
