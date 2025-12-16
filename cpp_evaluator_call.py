import json
import os
import subprocess
import csv
from io import StringIO
from typing import Dict, List, Tuple

COMPARATOR = r"C:\Users\Recup\OneDrive\Documentos\Books\PACS\DataAgent\build\Debug\resultcmp.exe"


def run_cpp_comparator(actual_csv: str, expected_csv: str, keys: List[str] = None) -> Dict:
    """Run the C++ comparator and return JSON result."""
    cmd = ['./cpp_evaluator/build/resultcmp', '--actual', actual_csv, '--expected', expected_csv]
    
    if keys:
        cmd.extend(['--key', ','.join(keys)])
    

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    print(result)

def _ensure_dir(path: str) -> None:
    """Create parent directory for a file path if missing."""
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def save_result_to_csv(gt_path: str, gen_path: str, json_str: str, out_csv: str) -> None:
    """Append a row to out_csv with columns [GT, Gen, Json]. Creates file with header if missing."""
    _ensure_dir(out_csv)
    write_header = not os.path.exists(out_csv) or os.path.getsize(out_csv) == 0
    with open(out_csv, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['GT', 'Gen', 'Json'])
        if write_header:
            writer.writeheader()
        writer.writerow({'GT': gt_path, 'Gen': gen_path, 'Json': json_str})


def run_cpp_comparator_fixed(actual_csv: str, expected_csv: str, keys=None, exe_path: str = None, out_csv: str = None) -> dict:
    exe = exe_path or COMPARATOR
    cmd = [exe, '--actual', actual_csv, '--expected', expected_csv]
    if keys:
        cmd += ['--key', ','.join(keys)]
    # Stream C++ debug (stderr) to terminal, capture stdout for JSON
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=None, text=True, timeout=30)
    json_text = (result.stdout or '').strip()
    if json_text:
        print(json_text)
        try:
            data = json.loads(json_text)
        except json.JSONDecodeError:
            data = {}
        if out_csv:
            save_result_to_csv(actual_csv, expected_csv, json_text, out_csv)
        return data
    return {}

if __name__ == '__main__':

    PREFIX = "gpt" # options: 'my', 'claude', 'gpt'
    index = 0
    path_gt = f"evaluation/csv_queries/{PREFIX}_{index}_gt.csv"
    path_gen = f"evaluation/csv_queries/{PREFIX}_{index}_gen.csv"
    print(f"Comparing {path_gt} and {path_gen} with keys ['week', 'store_id']")

    # Resolve comparator path (prefer fixed; build if missing; fallback to local Ninja build)
    fixed_path = COMPARATOR
    local_build_base = './cpp_evaluator/build/resultcmp'
    local_exe = local_build_base + ('.exe' if os.name == 'nt' else '')

    exe_to_use = None
    if fixed_path and os.path.exists(fixed_path):
        exe_to_use = fixed_path
        print(f"Using fixed comparator: {exe_to_use}")
    else:
        if not os.path.exists(local_exe):
            print("Fixed comparator not found. Building C++ comparator with CMake (Ninja)...")
            subprocess.run(['cmake', '-S', 'cpp_evaluator', '-B', 'cpp_evaluator/build', '-G', 'Ninja'], check=True)
            subprocess.run(['cmake', '--build', 'cpp_evaluator/build', '--config', 'Release'], check=True)
        exe_to_use = local_exe
        print(f"Using locally built comparator: {exe_to_use}")

    # Save/append results to evaluation/cplusplus/results.csv with columns [GT, Gen, Json]
    run_cpp_comparator_fixed(
        path_gt,
        path_gen,
        keys=["week", "store_id"],
        exe_path=exe_to_use,
        out_csv=f"evaluation/cplusplus/results_{PREFIX}_{index}.csv",
    )



