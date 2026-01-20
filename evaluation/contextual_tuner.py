import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd

try:
    # Reuse parsing + enrichment utilities
    from evaluation.config_tuner import compute_reward, enrich_from_best_result_json, load_sweep_csv  # type: ignore
except Exception:
    # If running as a standalone script without module path, fall back to local import style.
    from config_tuner import compute_reward, enrich_from_best_result_json, load_sweep_csv  # type: ignore


def _load_dataset_difficulty(dataset_json: Optional[str]) -> Dict[str, str]:
    """
    Optional mapping: prompt -> difficulty (if the dataset contains it).
    If not present, returns empty mapping.
    """
    if not dataset_json:
        return {}
    try:
        with open(dataset_json, "r", encoding="utf-8") as f:
            cases = json.load(f)
        out: Dict[str, str] = {}
        for c in cases:
            p = (c or {}).get("prompt")
            d = (c or {}).get("difficulty")
            if isinstance(p, str) and p and isinstance(d, str) and d:
                out[p] = d
        return out
    except Exception:
        return {}


def _infer_task_type(prompt: str) -> str:
    p = (prompt or "").lower()
    viz_kw = ["plot", "chart", "graph", "visual", "bar", "line", "scatter", "hist", "histogram"]
    if any(k in p for k in viz_kw):
        return "viz"
    # “SQL-like” user requests: group-by / join / filter language
    sql_kw = ["select", "group by", "join", "where", "order by", "limit", "sql", "query"]
    if any(k in p for k in sql_kw):
        return "sql"
    return "analysis"


def _length_bucket(prompt: str) -> str:
    n = len((prompt or "").strip())
    if n < 40:
        return "short"
    if n < 120:
        return "medium"
    return "long"


def make_context_bucket(prompt: str, *, difficulty: Optional[str] = None) -> str:
    """
    Discrete context bucket for a prompt.
    This is the “context” part of a contextual bandit.
    """
    task = _infer_task_type(prompt)
    lb = _length_bucket(prompt)
    diff = (difficulty or "unknown").lower()
    return f"task={task}|len={lb}|difficulty={diff}"


def _extract_actions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Ensure two_stage_cot exists (enrichment usually fills it). Missing -> False.
    if "two_stage_cot" not in out.columns:
        out["two_stage_cot"] = False
    out["two_stage_cot"] = out["two_stage_cot"].fillna(False).astype(bool)
    out["best_of_n"] = pd.to_numeric(out["best_of_n"], errors="coerce").fillna(1).astype(int)
    return out


def train_policy_table(
    df: pd.DataFrame,
    *,
    dataset_prompt_to_difficulty: Dict[str, str],
    min_bucket_n: int = 3,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - per_bucket_action: best action per bucket (by mean reward)
      - global_action: best action overall (fallback)
    """
    df = _extract_actions(df)

    # Compute context bucket per row
    diffs = df["prompt"].astype(str).map(lambda p: dataset_prompt_to_difficulty.get(p, "unknown"))
    df = df.copy()
    df["difficulty"] = diffs
    df["context_bucket"] = [
        make_context_bucket(p, difficulty=d) for p, d in zip(df["prompt"].astype(str).tolist(), diffs.tolist())
    ]

    # Aggregate reward by (bucket, action)
    g = (
        df.groupby(["context_bucket", "best_of_n", "two_stage_cot"], dropna=False)
        .agg(n=("reward", "count"), mean_reward=("reward", "mean"), mean_best=("best_score", "mean"), mean_time_ms=("duration_ms", "mean"))
        .reset_index()
    )

    # For each bucket: pick best action, but only if we have enough samples (min_bucket_n)
    g_valid = g[g["n"] >= int(min_bucket_n)].copy()
    idx = g_valid.groupby(["context_bucket"])["mean_reward"].idxmax()
    per_bucket = g_valid.loc[idx].sort_values("mean_reward", ascending=False).reset_index(drop=True)

    # Global fallback (ignore context)
    g2 = (
        df.groupby(["best_of_n", "two_stage_cot"], dropna=False)
        .agg(n=("reward", "count"), mean_reward=("reward", "mean"), mean_best=("best_score", "mean"), mean_time_ms=("duration_ms", "mean"))
        .reset_index()
        .sort_values(["mean_reward", "mean_best"], ascending=[False, False])
        .reset_index(drop=True)
    )
    return per_bucket, g2


def recommend_for_prompt(
    prompt: str,
    *,
    per_bucket_action: pd.DataFrame,
    global_action: pd.DataFrame,
    difficulty: str = "unknown",
) -> Dict[str, object]:
    bucket = make_context_bucket(prompt, difficulty=difficulty)
    match = per_bucket_action[per_bucket_action["context_bucket"] == bucket]
    if len(match):
        row = match.iloc[0].to_dict()
        source = "bucket"
    else:
        row = (global_action.iloc[0].to_dict() if len(global_action) else {"best_of_n": 1, "two_stage_cot": False})
        source = "global"

    return {
        "prompt": prompt,
        "difficulty": difficulty,
        "context_bucket": bucket,
        "source": source,
        "best_of_n": int(row.get("best_of_n", 1)),
        "two_stage_cot": bool(row.get("two_stage_cot", False)),
        "stats": {
            "mean_reward": float(row.get("mean_reward", 0.0) or 0.0),
            "mean_best": float(row.get("mean_best", 0.0) or 0.0),
            "mean_time_ms": float(row.get("mean_time_ms", 0.0) or 0.0),
            "n": int(row.get("n", 0) or 0),
        },
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Contextual (prompt-based) config recommender from sweep logs")
    ap.add_argument("csv", nargs="+", help="One or more sweep_results.csv files")
    ap.add_argument("--dataset-json", default=None, help="Optional dataset JSON containing prompt->difficulty")
    ap.add_argument("--min-bucket-n", type=int, default=3, help="Min samples per bucket to trust bucket policy")
    ap.add_argument("--no-enrich", action="store_true", help="Do not read <test_dir>/best_result.json")
    ap.add_argument("--w-best", type=float, default=1.0, help="Reward weight for best_score")
    ap.add_argument("--w-viz", type=float, default=0.0, help="Reward weight for viz_text_score")
    ap.add_argument("--w-time-s", type=float, default=0.0, help="Reward penalty per second")
    ap.add_argument("--w-co2-kg", type=float, default=0.0, help="Reward penalty per kgCO2")
    ap.add_argument("--prompt", default=None, help="If provided, print recommendation for this prompt")
    ap.add_argument("--difficulty", default="unknown", help="Optional difficulty for --prompt (easy/medium/hard/...)")
    ap.add_argument("--export-policy", default=None, help="Write trained policy JSON to this path")
    args = ap.parse_args(argv)

    dfs = [load_sweep_csv(p) for p in args.csv]
    df = pd.concat(dfs, ignore_index=True)
    if not args.no_enrich:
        df = enrich_from_best_result_json(df)

    df["reward"] = compute_reward(
        df,
        w_best=args.w_best,
        w_viz=args.w_viz,
        w_time_s=args.w_time_s,
        w_co2_kg=args.w_co2_kg,
    )

    prompt_to_diff = _load_dataset_difficulty(args.dataset_json)
    per_bucket, global_action = train_policy_table(
        df,
        dataset_prompt_to_difficulty=prompt_to_diff,
        min_bucket_n=args.min_bucket_n,
    )

    if args.export_policy:
        pol = {
            "min_bucket_n": int(args.min_bucket_n),
            "reward_weights": {
                "w_best": args.w_best,
                "w_viz": args.w_viz,
                "w_time_s": args.w_time_s,
                "w_co2_kg": args.w_co2_kg,
            },
            "per_bucket": per_bucket.to_dict(orient="records"),
            "global": global_action.head(10).to_dict(orient="records"),
        }
        with open(args.export_policy, "w", encoding="utf-8") as f:
            json.dump(pol, f, indent=2)

    if args.prompt:
        diff = args.difficulty
        rec = recommend_for_prompt(
            args.prompt,
            per_bucket_action=per_bucket,
            global_action=global_action,
            difficulty=diff,
        )
        print(json.dumps(rec, indent=2))
        print("\n# YAML overrides")
        print(f"two_stage_cot: {str(rec['two_stage_cot']).lower()}")
        print(f"best_of_n: {rec['best_of_n']}")
        return 0

    # Otherwise: print a short global summary
    print("\n=== Global best actions (top 10) ===")
    print(global_action.head(10).to_string(index=False))
    print("\n=== Buckets learned (top 20 by mean_reward) ===")
    print(per_bucket.head(20).to_string(index=False))
    print("\nTip: pass --prompt \"...\" to get a recommendation for a new prompt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

