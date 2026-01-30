"""
Post-process `output_google_colab` runs to extract per-stage timing and emissions.

Inputs (expected in `output_google_colab/`):
- `**/stage_metrics.json`: per-test metrics (stage durations + estimated emissions)
- `codecarbon_batch.csv` (optional): CodeCarbon runs, emissions + energy + duration
- `batch_times.csv` (optional): total wall-clock duration per test + exit code

Outputs (default written under `evaluation/`):
- `stage_metrics_long.csv`: one row per (run, test_case, stage)
- `stage_metrics_runs.csv`: one row per (run, test_case) with totals + joined CodeCarbon

Run:
  python -m evaluation.post_processing_output
or:
  python evaluation/post_processing_output.py --output-root output_google_colab
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


# Experiment folder naming has evolved over time. We support both:
# - rep1_tsfalse_bon2_t0_tmax_0
# - rep1_tsfalse_bon2_t0_tmax0p15   (p encodes decimal point)
_EXP_RE = re.compile(
    r"^rep(?P<rep>\d+)_ts(?P<two_stage_cot>true|false)_bon(?P<best_of_n>\d+)"
    r"_t(?P<t0>[\dp]+)_tmax_?(?P<tmax>[\dp]+)_?$"
)
_TEST_RE = re.compile(r"^test_(?P<test_case>\d+)$")


@dataclass(frozen=True)
class RunInfo:
    experiment: str
    test_dir: str
    test_case: Optional[int]
    rep: Optional[int]
    two_stage_cot: Optional[bool]
    best_of_n: Optional[int]
    t0: Optional[float]
    tmax: Optional[float]


def _safe_int(x: Any) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def _safe_float(x: Any) -> Optional[float]:
    """Parse float, accepting strings like '0p15' (used in directory naming)."""
    try:
        if isinstance(x, str):
            x = x.replace("p", ".")
        return float(x)
    except Exception:
        return None


def _parse_run_info(stage_metrics_path: Path) -> RunInfo:
    # .../output_google_colab/<experiment>/<test_dir>/stage_metrics.json
    test_dir = stage_metrics_path.parent.name
    experiment = stage_metrics_path.parent.parent.name if stage_metrics_path.parent.parent else ""

    test_case: Optional[int] = None
    m_test = _TEST_RE.match(test_dir)
    if m_test:
        test_case = _safe_int(m_test.group("test_case"))

    rep = two_stage_cot = best_of_n = None
    t0: Optional[float] = None
    tmax: Optional[float] = None
    m_exp = _EXP_RE.match(experiment)
    if m_exp:
        rep = _safe_int(m_exp.group("rep"))
        two_stage_cot = m_exp.group("two_stage_cot") == "true"
        best_of_n = _safe_int(m_exp.group("best_of_n"))
        t0 = _safe_float(m_exp.group("t0"))
        tmax = _safe_float(m_exp.group("tmax"))

    return RunInfo(
        experiment=experiment,
        test_dir=test_dir,
        test_case=test_case,
        rep=rep,
        two_stage_cot=two_stage_cot,
        best_of_n=best_of_n,
        t0=t0,
        tmax=tmax,
    )


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _index_by_stage(items: Iterable[Dict[str, Any]], *, stage_key: str = "stage") -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for it in items:
        stage = it.get(stage_key)
        if isinstance(stage, str) and stage:
            out[stage] = dict(it)
    return out


def _closest_by_duration_ms(df: pd.DataFrame, *, target_duration_ms: Optional[float]) -> Optional[pd.Series]:
    if df.empty:
        return None
    if target_duration_ms is None or pd.isna(target_duration_ms):
        # fallback: latest by timestamp if present, else first
        if "timestamp" in df.columns:
            d2 = df.copy()
            d2["timestamp"] = pd.to_datetime(d2["timestamp"], errors="coerce")
            d2 = d2.sort_values("timestamp", ascending=False, na_position="last")
            return d2.iloc[0]
        return df.iloc[0]

    if "duration" in df.columns:
        # codecarbon duration is in seconds
        dur_ms = pd.to_numeric(df["duration"], errors="coerce") * 1000.0
    elif "duration_ms" in df.columns:
        dur_ms = pd.to_numeric(df["duration_ms"], errors="coerce")
    else:
        return df.iloc[0]

    diff = (dur_ms - float(target_duration_ms)).abs()
    idx = diff.idxmin()
    try:
        return df.loc[idx]
    except Exception:
        return df.iloc[0]


def build_reports(output_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    stage_files = sorted(output_root.glob("**/stage_metrics.json"))

    # Optional global CSVs
    codecarbon_path = output_root / "codecarbon_batch.csv"
    batch_times_path = output_root / "batch_times.csv"

    codecarbon_df = pd.read_csv(codecarbon_path) if codecarbon_path.exists() else pd.DataFrame()
    batch_times_df = pd.read_csv(batch_times_path) if batch_times_path.exists() else pd.DataFrame()

    # Normalize key types if present
    if not codecarbon_df.empty and "test_case" in codecarbon_df.columns:
        codecarbon_df["test_case"] = pd.to_numeric(codecarbon_df["test_case"], errors="coerce").astype("Int64")
    if not batch_times_df.empty and "test_case" in batch_times_df.columns:
        batch_times_df["test_case"] = pd.to_numeric(batch_times_df["test_case"], errors="coerce").astype("Int64")

    long_rows: List[Dict[str, Any]] = []
    run_rows: List[Dict[str, Any]] = []

    for p in stage_files:
        info = _parse_run_info(p)
        payload = _read_json(p)

        prompt = payload.get("prompt")
        prompt = prompt if isinstance(prompt, str) else None

        stage_metrics = payload.get("stage_metrics", [])
        stage_metrics = stage_metrics if isinstance(stage_metrics, list) else []
        stage_dur = _index_by_stage(stage_metrics)

        emissions_obj = payload.get("emissions", {})
        emissions_obj = emissions_obj if isinstance(emissions_obj, dict) else {}
        total_emissions_kg = emissions_obj.get("total_emissions_kg")
        stage_emissions = emissions_obj.get("stage_emissions_est", [])
        stage_emissions = stage_emissions if isinstance(stage_emissions, list) else []
        stage_emi = _index_by_stage(stage_emissions)

        # Compute totals from stage_metrics
        durations_ms = [
            float(it.get("duration_ms")) for it in stage_dur.values() if it.get("duration_ms") is not None
        ]
        total_duration_ms = sum(durations_ms) if durations_ms else None

        # Build long rows by stage (union of stage names in both lists)
        stages = sorted(set(stage_dur.keys()) | set(stage_emi.keys()))
        for stage in stages:
            dur_ms = stage_dur.get(stage, {}).get("duration_ms")
            emi_kg_est = stage_emi.get(stage, {}).get("emissions_kg_est")
            long_rows.append(
                {
                    "experiment": info.experiment,
                    "test_dir": info.test_dir,
                    "test_case": info.test_case,
                    "rep": info.rep,
                    "two_stage_cot": info.two_stage_cot,
                    "best_of_n": info.best_of_n,
                    "t0": info.t0,
                    "tmax": info.tmax,
                    "prompt": prompt,
                    "stage": stage,
                    "duration_ms": dur_ms,
                    "emissions_kg_est": emi_kg_est,
                    "stage_metrics_path": str(p),
                }
            )

        # Join (best-effort) CodeCarbon + batch_times by (test_case, prompt) and closest duration
        cc_row: Optional[pd.Series] = None
        if not codecarbon_df.empty:
            cc_sub = codecarbon_df
            if info.test_case is not None and "test_case" in cc_sub.columns:
                cc_sub = cc_sub[cc_sub["test_case"] == info.test_case]
            if prompt is not None and "prompt" in cc_sub.columns:
                cc_sub = cc_sub[cc_sub["prompt"] == prompt]
            cc_row = _closest_by_duration_ms(cc_sub, target_duration_ms=total_duration_ms)

        bt_row: Optional[pd.Series] = None
        if not batch_times_df.empty:
            bt_sub = batch_times_df
            if info.test_case is not None and "test_case" in bt_sub.columns:
                bt_sub = bt_sub[bt_sub["test_case"] == info.test_case]
            if prompt is not None and "prompt" in bt_sub.columns:
                bt_sub = bt_sub[bt_sub["prompt"] == prompt]
            bt_row = _closest_by_duration_ms(bt_sub, target_duration_ms=total_duration_ms)

        run_row: Dict[str, Any] = {
            "experiment": info.experiment,
            "test_dir": info.test_dir,
            "test_case": info.test_case,
            "rep": info.rep,
            "two_stage_cot": info.two_stage_cot,
            "best_of_n": info.best_of_n,
            "t0": info.t0,
            "tmax": info.tmax,
            "prompt": prompt,
            "total_duration_ms_from_stages": total_duration_ms,
            "total_emissions_kg_from_stage_metrics": total_emissions_kg,
            "stage_metrics_path": str(p),
        }

        if bt_row is not None:
            run_row["batch_times_duration_ms"] = bt_row.get("duration_ms")
            run_row["batch_times_exit_code"] = bt_row.get("exit_code")

        if cc_row is not None:
            # Keep a few high-signal columns; keep others if you want later.
            for col in [
                "timestamp",
                "project_name",
                "run_id",
                "experiment_id",
                "duration",
                "emissions",
                "energy_consumed",
                "cpu_energy",
                "gpu_energy",
                "ram_energy",
            ]:
                if col in cc_row.index:
                    run_row[f"codecarbon_{col}"] = cc_row.get(col)

        run_rows.append(run_row)

    long_df = pd.DataFrame(long_rows)
    runs_df = pd.DataFrame(run_rows)

    # Consistent ordering
    if not long_df.empty:
        long_df = long_df.sort_values(["experiment", "test_case", "stage"], na_position="last")
    if not runs_df.empty:
        runs_df = runs_df.sort_values(["experiment", "test_case"], na_position="last")

    return long_df, runs_df


def build_score_reports(output_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build BLEU/SPICE (and related) score reports from `**/best_result.json`.

    Returns:
      - scores_runs_df: one row per (experiment, test_case)
      - scores_summary_df: aggregated means/stds per experiment configuration
    """
    best_files = sorted(output_root.glob("**/best_result.json"))

    rows: List[Dict[str, Any]] = []
    for p in best_files:
        info = _parse_run_info(p)
        payload = _read_json(p)

        prompt = payload.get("prompt")
        prompt = prompt if isinstance(prompt, str) else None

        text_scores = payload.get("text_scores", {})
        text_scores = text_scores if isinstance(text_scores, dict) else {}

        rows.append(
            {
                "experiment": info.experiment,
                "test_dir": info.test_dir,
                "test_case": info.test_case,
                "rep": info.rep,
                "two_stage_cot": info.two_stage_cot,
                "best_of_n": info.best_of_n,
                "t0": info.t0,
                "tmax": info.tmax,
                "prompt": prompt,
                "csv_score": payload.get("csv_score"),
                "text_score": payload.get("text_score"),
                "bleu": text_scores.get("bleu"),
                "spice": text_scores.get("spice"),
                "best_score": payload.get("best_score"),
                "best_idx": payload.get("best_idx"),
                "best_result_path": str(p),
            }
        )

    scores_runs_df = pd.DataFrame(rows)
    if not scores_runs_df.empty:
        scores_runs_df = scores_runs_df.sort_values(["experiment", "test_case"], na_position="last")

    # Summary for comparisons: aggregate across test cases within the same config
    group_cols = ["rep", "two_stage_cot", "best_of_n", "t0", "tmax", "experiment"]
    if scores_runs_df.empty:
        scores_summary_df = pd.DataFrame()
    else:
        num_cols = ["csv_score", "text_score", "bleu", "spice", "best_score"]
        agg: Dict[str, List[str]] = {c: ["mean", "std", "min", "max"] for c in num_cols if c in scores_runs_df.columns}
        scores_summary_df = (
            scores_runs_df.groupby(group_cols, dropna=False)
            .agg(agg)
            .reset_index()
        )
        # Flatten MultiIndex columns
        scores_summary_df.columns = [
            f"{a}_{b}" if b else a for a, b in (c if isinstance(c, tuple) else (c, "") for c in scores_summary_df.columns)
        ]
        # Add counts
        scores_summary_df["n_tests"] = scores_runs_df.groupby(group_cols, dropna=False).size().values

    return scores_runs_df, scores_summary_df


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Post-process output_google_colab stage metrics + CodeCarbon.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "output_google_colab",
        help="Path to output_google_colab directory.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory to write the output CSVs (default: evaluation/).",
    )
    args = parser.parse_args(argv)

    out_root = args.output_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    long_df, runs_df = build_reports(out_root)
    scores_runs_df, scores_summary_df = build_score_reports(out_root)

    long_path = out_dir / "stage_metrics_long.csv"
    runs_path = out_dir / "stage_metrics_runs.csv"
    scores_runs_path = out_dir / "scores_runs.csv"
    scores_summary_path = out_dir / "scores_summary.csv"

    long_df.to_csv(long_path, index=False)
    runs_df.to_csv(runs_path, index=False)
    scores_runs_df.to_csv(scores_runs_path, index=False)
    scores_summary_df.to_csv(scores_summary_path, index=False)

    # Also store next to the raw Colab outputs for convenience
    out_root.mkdir(parents=True, exist_ok=True)
    long_path_root = out_root / "stage_metrics_long.csv"
    runs_path_root = out_root / "stage_metrics_runs.csv"
    scores_runs_path_root = out_root / "scores_runs.csv"
    scores_summary_path_root = out_root / "scores_summary.csv"
    long_df.to_csv(long_path_root, index=False)
    runs_df.to_csv(runs_path_root, index=False)
    scores_runs_df.to_csv(scores_runs_path_root, index=False)
    scores_summary_df.to_csv(scores_summary_path_root, index=False)

    print(f"Wrote {len(long_df)} rows -> {long_path}")
    print(f"Wrote {len(runs_df)} rows -> {runs_path}")
    print(f"Wrote {len(scores_runs_df)} rows -> {scores_runs_path}")
    print(f"Wrote {len(scores_summary_df)} rows -> {scores_summary_path}")
    print(f"Wrote {len(long_df)} rows -> {long_path_root}")
    print(f"Wrote {len(runs_df)} rows -> {runs_path_root}")
    print(f"Wrote {len(scores_runs_df)} rows -> {scores_runs_path_root}")
    print(f"Wrote {len(scores_summary_df)} rows -> {scores_summary_path_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

