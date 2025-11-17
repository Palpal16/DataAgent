import json
import os
import shlex
import subprocess
import tempfile
from typing import Dict, List, Optional

import pandas as pd


def _table_str_to_dataframe(table_text: str) -> pd.DataFrame:
    """Convert pandas.to_string() fixed-width table back to a DataFrame.

    Falls back to a single-column frame if parsing fails.
    """
    from io import StringIO

    buf = StringIO(table_text or "")
    try:
        df = pd.read_fwf(buf)
        if df is None or df.empty:
            return pd.DataFrame()
        return df
    except Exception:
        return pd.DataFrame({"data": [table_text]})


def write_actual_csv_from_table_str(table_text: str, out_path: str) -> None:
    df = _table_str_to_dataframe(table_text)
    # Normalize column names to strings
    df.columns = [str(c) for c in df.columns]
    df.to_csv(out_path, index=False)


def run_cpp_comparator(
    *,
    evaluator_exe: str,
    actual_csv: str,
    expected_csv: str,
    keys: Optional[List[str]] = None,
    float_abs: float = 1e-8,
    float_rel: float = 1e-6,
    case_insensitive: bool = False,
    stream_debug: bool = False,
) -> Dict:
    args = [evaluator_exe, "--actual", actual_csv, "--expected", expected_csv,
            "--float-abs", str(float_abs), "--float-rel", str(float_rel)]
    if keys:
        args += ["--key", ",".join(keys)]
    if case_insensitive:
        args += ["--case-insensitive"]

    # If stream_debug is True, inherit stderr so C++ debug (sent to stderr) prints to terminal.
    # Keep stdout captured to parse JSON report.
    if stream_debug:
        proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=None, text=True)
    else:
        proc = subprocess.run(args, capture_output=True, text=True)
    stdout = proc.stdout.strip()
    try:
        report = json.loads(stdout) if stdout else {}
    except json.JSONDecodeError:
        report = {"equal": False, "error": "Invalid JSON from comparator", "raw": stdout}
    report["exit_code"] = proc.returncode
    if proc.returncode not in (0, 1):
        # Non-comparison error, include stderr
        report.setdefault("error", proc.stderr.strip())
    return report


def evaluate_table_str_against_expected(
    *,
    table_text: str,
    expected_csv_path: str,
    evaluator_exe: Optional[str] = None,
    keys: Optional[List[str]] = None,
    float_abs: float = 1e-8,
    float_rel: float = 1e-6,
    case_insensitive: bool = False,
    stream_debug: bool = False,
) -> Dict:
    evaluator = evaluator_exe or os.path.join("cpp_evaluator", "build", "resultcmp.exe")
    with tempfile.TemporaryDirectory() as td:
        actual_csv = os.path.join(td, "actual.csv")
        write_actual_csv_from_table_str(table_text, actual_csv)
        return run_cpp_comparator(
            evaluator_exe=evaluator,
            actual_csv=actual_csv,
            expected_csv=expected_csv_path,
            keys=keys,
            float_abs=float_abs,
            float_rel=float_rel,
            case_insensitive=case_insensitive,
            stream_debug=stream_debug,
        )


