from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from evaluation.difficulty_labels import Difficulty3, load_prompt_difficulty_map


def _coerce_bool(x) -> Optional[bool]:
    if isinstance(x, bool):
        return x
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return True
    if s in {"false", "0", "no", "n"}:
        return False
    return None


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Compute best config per difficulty from evaluation/scores_runs.csv")
    ap.add_argument("--scores-csv", default="evaluation/scores_runs.csv", help="Path to scores_runs.csv")
    ap.add_argument("--dataset-json", default="evaluation/miguel.json", help="Dataset JSON with prompt->difficulty")
    ap.add_argument("--out", default="evaluation/best_config_by_difficulty.json", help="Output JSON path")
    args = ap.parse_args(argv)

    prompt2diff = load_prompt_difficulty_map(args.dataset_json)

    scores_path = Path(args.scores_csv)
    df = pd.read_csv(scores_path)
    if "prompt" not in df.columns:
        raise SystemExit(f"Missing 'prompt' column in {scores_path}")
    if "csv_score" not in df.columns:
        raise SystemExit(f"Missing 'csv_score' column in {scores_path}")

    # Map prompt -> difficulty (3 classes) and keep only labeled prompts
    df = df.copy()
    df["difficulty"] = df["prompt"].astype(str).map(lambda p: prompt2diff.get(p.strip(), None))
    df = df[df["difficulty"].notna()].copy()
    if df.empty:
        raise SystemExit("No rows matched between scores CSV and dataset prompt->difficulty mapping.")

    # Normalize config knobs (we pick best *config*, not best repetition)
    df["two_stage_cot"] = df["two_stage_cot"].map(_coerce_bool).fillna(False).astype(bool)
    df["best_of_n"] = pd.to_numeric(df["best_of_n"], errors="coerce").fillna(1).astype(int)
    df["t0"] = pd.to_numeric(df["t0"], errors="coerce").fillna(0.0).astype(float)
    df["tmax"] = pd.to_numeric(df["tmax"], errors="coerce").fillna(0.0).astype(float)
    df["csv_score"] = pd.to_numeric(df["csv_score"], errors="coerce").fillna(0.0).astype(float)

    # For each difficulty: pick config with highest mean csv_score
    g = (
        df.groupby(["difficulty", "two_stage_cot", "best_of_n", "t0", "tmax"], dropna=False)
        .agg(n=("csv_score", "count"), mean_csv_score=("csv_score", "mean"))
        .reset_index()
        .sort_values(["difficulty", "mean_csv_score", "n"], ascending=[True, False, False])
    )

    best_by_diff: Dict[str, Dict[str, object]] = {}
    for diff in ["easy", "medium", "hard"]:
        sub = g[g["difficulty"] == diff]
        if sub.empty:
            continue
        row = sub.iloc[0].to_dict()
        best_by_diff[diff] = {
            "two_stage_cot": bool(row["two_stage_cot"]),
            "best_of_n": int(row["best_of_n"]),
            "temperature": float(row["t0"]),
            "temperature_max": float(row["tmax"]),
            "mean_csv_score": float(row["mean_csv_score"]),
            "n": int(row["n"]),
        }

    # Global fallback: best overall by mean csv_score
    g_all = (
        df.groupby(["two_stage_cot", "best_of_n", "t0", "tmax"], dropna=False)
        .agg(n=("csv_score", "count"), mean_csv_score=("csv_score", "mean"))
        .reset_index()
        .sort_values(["mean_csv_score", "n"], ascending=[False, False])
        .reset_index(drop=True)
    )
    fallback = g_all.iloc[0].to_dict() if len(g_all) else {}

    out = {
        "schema": 1,
        "scores_csv": str(scores_path.as_posix()),
        "dataset_json": str(Path(args.dataset_json).as_posix()),
        "difficulty_labels": ["easy", "medium", "hard"],
        "best_config_by_difficulty": best_by_diff,
        "fallback_best_config": {
            "two_stage_cot": bool(fallback.get("two_stage_cot", False)),
            "best_of_n": int(fallback.get("best_of_n", 1)),
            "temperature": float(fallback.get("t0", 0.0)),
            "temperature_max": float(fallback.get("tmax", 0.0)),
            "mean_csv_score": float(fallback.get("mean_csv_score", 0.0) or 0.0),
            "n": int(fallback.get("n", 0) or 0),
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

