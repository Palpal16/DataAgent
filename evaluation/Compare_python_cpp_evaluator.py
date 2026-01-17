import argparse
import json
import os
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Agent.utils import compare_csv, bleu_score, run_cpp_comparator, prepare_gt_from_dataset


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def win_to_wsl(path: Path) -> str:
    raw = str(path.resolve())
    if len(raw) >= 3 and raw[1:3] == ":\\":
        drive = raw[0].lower()
        rest = raw[3:].replace("\\", "/")
        return f"/mnt/{drive}/{rest}"
    return raw


def run_cpp_comparator_wsl(actual_csv: Path, expected_csv: Path, evaluator_exe: Path, keys):
    eval_keys = ",".join(keys) if keys else ""
    actual_wsl = win_to_wsl(actual_csv)
    expected_wsl = win_to_wsl(expected_csv)
    evaluator_wsl = win_to_wsl(evaluator_exe)

    cmd = (
        f"{shlex.quote(evaluator_wsl)} "
        f"--actual {shlex.quote(actual_wsl)} "
        f"--expected {shlex.quote(expected_wsl)}"
    )
    if eval_keys:
        cmd += f" --key {shlex.quote(eval_keys)}"

    proc = os.popen(f"wsl bash -lc {shlex.quote(cmd)}")
    stdout = proc.read().strip()
    exit_code = proc.close()
    exit_code = 0 if exit_code is None else int(exit_code)

    try:
        report = json.loads(stdout) if stdout else {}
    except json.JSONDecodeError:
        report = {"equal": False, "error": "Invalid JSON from comparator", "raw": stdout}
    report["exit_code"] = exit_code
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate output/test_* using evaluation/our_dataset.json."
    )
    parser.add_argument(
        "--output-dir",
        default="output/test_1",
        help="Path to output/test_* folder",
    )
    parser.add_argument(
        "--dataset",
        default="evaluation/our_dataset.json",
        help="Path to dataset JSON with gt_data/gt_analysis",
    )
    parser.add_argument(
        "--run-csv",
        default="run_data.csv",
        help="CSV filename inside output dir",
    )
    parser.add_argument(
        "--write-gt-csv",
        default="gt_data.csv",
        help="Filename to write gt_data CSV inside output dir",
    )
    parser.add_argument(
        "--iou-type",
        default="rows",
        choices=["rows", "columns", "table"],
        help="IoU type to highlight",
    )
    parser.add_argument(
        "--cpp-evaluator",
        default=None,
        help="Path to C++ comparator executable (optional)",
    )
    parser.add_argument(
        "--cpp-evaluator-wsl",
        action="store_true",
        help="Use WSL to run the C++ comparator (Linux binary)",
    )
    parser.add_argument(
        "--eval-keys",
        default=None,
        help="Comma-separated key columns for C++ comparator (optional)",
    )
    args = parser.parse_args()

    root = Path.cwd()
    output_dir = (root / args.output_dir).resolve()
    dataset_path = (root / args.dataset).resolve()
    run_csv_path = output_dir / args.run_csv
    gt_csv_path = output_dir / args.write_gt_csv

    results_path = output_dir / "all_results.json"
    if not results_path.exists():
        raise SystemExit(f"Missing {results_path}")
    if not dataset_path.exists():
        raise SystemExit(f"Missing {dataset_path}")
    if not run_csv_path.exists():
        raise SystemExit(f"Missing {run_csv_path}")

    results = load_json(results_path)
    if not results:
        raise SystemExit("all_results.json is empty")
    result = results[0]
    prompt = result.get("prompt", "")
    answer_text = " ".join(result.get("answer", []))

    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        prep = prepare_gt_from_dataset(
            prompt=prompt,
            dataset_path=str(dataset_path),
            output_dir=str(output_dir),
            gt_csv_filename=args.write_gt_csv,
            gt_results_filename="gt_results.json",
            gt_text_filename="gt_analysis.txt",
        )
    except ValueError as exc:
        raise SystemExit(str(exc))
    case = prep["case"]
    gt_csv_path = Path(prep["gt_csv_path"])

    cols_iou, rows_iou, data_iou = compare_csv(str(run_csv_path), str(gt_csv_path))
    summary = {
        "csv_iou": {"columns": cols_iou, "rows": rows_iou, "table": data_iou},
    }

    gt_analysis = case.get("gt_analysis", "") if case else ""
    if gt_analysis:
        summary["bleu"] = bleu_score(answer_text, gt_analysis)
    else:
        summary["bleu"] = None

    if args.cpp_evaluator:
        keys = [k.strip() for k in (args.eval_keys or "").split(",") if k.strip()] or None
        if args.cpp_evaluator_wsl:
            cpp_report = run_cpp_comparator_wsl(
                actual_csv=run_csv_path,
                expected_csv=gt_csv_path,
                evaluator_exe=Path(args.cpp_evaluator),
                keys=keys,
            )
        else:
            cpp_report = run_cpp_comparator(
                actual_csv=str(run_csv_path),
                expected_csv=str(gt_csv_path),
                evaluator_exe=args.cpp_evaluator,
                keys=keys,
            )
        summary["cpp_report"] = cpp_report

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
