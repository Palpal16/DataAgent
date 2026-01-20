import argparse
import json
import os
import re
from typing import Dict, List, Optional

import pandas as pd

def _repair_concatenated_sweep_csv_text(text: str) -> str:
    """
    Some sweep_results.csv files can be corrupted by missing newlines, resulting in multiple
    records concatenated on one line:
      ... ,./output/.../test_1<next_sweep_id>,...

    We repair this by inserting a newline between the test_dir path and the next numeric sweep_id.
    This is safe because the last column (test_dir) never contains commas.
    """
    # Normalize line endings first
    t = text.replace("\r\n", "\n").replace("\r", "\n")

    # Insert newline when a test_dir path is immediately followed by a sweep_id token like "12,1,1,".
    # This is safe because the last column (test_dir) never contains commas.
    t = re.sub(r"(\./output/[^,\n]+)(?=(\d+,\d+,\d+,))", r"\1\n", t)

    return t


def load_sweep_csv(path: str) -> pd.DataFrame:
    raw = open(path, "r", encoding="utf-8", errors="replace").read()
    raw = _repair_concatenated_sweep_csv_text(raw)

    # Drop duplicated headers if present mid-file.
    lines = raw.split("\n")
    if not lines:
        raise ValueError(f"Empty CSV: {path}")
    header = lines[0].strip()
    kept = [lines[0]]
    for ln in lines[1:]:
        if not ln.strip():
            continue
        if ln.strip() == header:
            continue
        kept.append(ln)
    cleaned = "\n".join(kept) + "\n"

    from io import StringIO

    df = pd.read_csv(StringIO(cleaned))
    df["source_csv"] = os.path.abspath(path)
    return df


def _safe_load_json(path: str) -> Dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def enrich_from_best_result_json(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich each row by reading <test_dir>/best_result.json if present.
    This allows us to recover fields not present in older sweep CSVs, such as:
      - two_stage_cot
      - viz_text_score / viz_bleu / viz_spice
    """
    if "test_dir" not in df.columns:
        return df

    two_stage: List[Optional[bool]] = []
    viz_text_score: List[Optional[float]] = []
    viz_bleu: List[Optional[float]] = []
    viz_spice: List[Optional[float]] = []

    for td in df["test_dir"].astype(str).tolist():
        best_path = os.path.join(str(td), "best_result.json")
        best = _safe_load_json(best_path)
        ts = best.get("two_stage_cot", None)
        two_stage.append(bool(ts) if isinstance(ts, bool) else None)
        viz_text_score.append(float(best.get("viz_text_score", 0.0) or 0.0) if best else None)
        viz_bleu.append(float(best.get("viz_bleu", 0.0) or 0.0) if best else None)
        viz_spice.append(float(best.get("viz_spice", 0.0) or 0.0) if best else None)

    out = df.copy()
    if "two_stage_cot" not in out.columns:
        out["two_stage_cot"] = two_stage
    if "viz_text_score" not in out.columns:
        out["viz_text_score"] = viz_text_score
    if "viz_bleu" not in out.columns:
        out["viz_bleu"] = viz_bleu
    if "viz_spice" not in out.columns:
        out["viz_spice"] = viz_spice
    return out


def compute_reward(
    df: pd.DataFrame,
    *,
    w_best: float,
    w_time_s: float,
    w_co2_kg: float,
    w_viz: float,
) -> pd.Series:
    best = pd.to_numeric(df.get("best_score", 0.0), errors="coerce").fillna(0.0)
    dur_ms = pd.to_numeric(df.get("duration_ms", 0.0), errors="coerce").fillna(0.0)
    co2 = pd.to_numeric(df.get("total_emissions_kg", 0.0), errors="coerce").fillna(0.0)
    viz = pd.to_numeric(df.get("viz_text_score", 0.0), errors="coerce").fillna(0.0)

    dur_s = dur_ms / 1000.0
    # Reward: maximize accuracy, penalize time and emissions; optionally add visualization quality weight
    return (w_best * best) + (w_viz * viz) - (w_time_s * dur_s) - (w_co2_kg * co2)


def summarize_by_config(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["best_of_n", "temperature", "temperature_max"]
    if "two_stage_cot" in df.columns:
        keys.append("two_stage_cot")

    agg = df.groupby(keys, dropna=False).agg(
        n=("best_score", "count"),
        mean_best=("best_score", "mean"),
        std_best=("best_score", "std"),
        mean_time_ms=("duration_ms", "mean"),
        mean_co2_kg=("total_emissions_kg", "mean"),
        mean_text=("text_score", "mean"),
        mean_csv=("csv_score", "mean"),
        mean_viz=("viz_text_score", "mean"),
        mean_reward=("reward", "mean"),
    )
    return agg.reset_index().sort_values(["mean_reward", "mean_best"], ascending=[False, False])


def pareto_front(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return non-dominated configs for (maximize mean_best, minimize mean_time_ms, minimize mean_co2_kg).
    """
    cols = ["mean_best", "mean_time_ms", "mean_co2_kg"]
    x = df[cols].copy()
    x["idx"] = range(len(df))

    keep = []
    for i, row in x.iterrows():
        dominated = False
        for j, other in x.iterrows():
            if i == j:
                continue
            # other dominates row if >= best and <= time and <= co2, and at least one strict
            if (
                other["mean_best"] >= row["mean_best"]
                and other["mean_time_ms"] <= row["mean_time_ms"]
                and other["mean_co2_kg"] <= row["mean_co2_kg"]
                and (
                    other["mean_best"] > row["mean_best"]
                    or other["mean_time_ms"] < row["mean_time_ms"]
                    or other["mean_co2_kg"] < row["mean_co2_kg"]
                )
            ):
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return df.iloc[keep].sort_values(["mean_best", "mean_time_ms"], ascending=[False, True])


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Offline bandit-style tuner for sweep_results.csv")
    ap.add_argument("csv", nargs="+", help="One or more sweep_results.csv paths")
    ap.add_argument("--no-enrich", action="store_true", help="Do not read <test_dir>/best_result.json to enrich rows")
    ap.add_argument("--w-best", type=float, default=1.0, help="Weight for best_score (accuracy)")
    ap.add_argument("--w-viz", type=float, default=0.0, help="Extra weight for viz_text_score (optional)")
    ap.add_argument("--w-time-s", type=float, default=0.0, help="Penalty per second of runtime (duration_ms/1000)")
    ap.add_argument("--w-co2-kg", type=float, default=0.0, help="Penalty per kgCO2")
    ap.add_argument("--topk", type=int, default=10, help="How many configs to print")
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

    summary = summarize_by_config(df)
    print("\n=== Top configs by mean reward ===")
    print(summary.head(args.topk).to_string(index=False))

    print("\n=== Pareto front (mean_best vs mean_time_ms vs mean_co2_kg) ===")
    pf = pareto_front(summary)
    print(pf.head(args.topk).to_string(index=False))

    best = summary.iloc[0].to_dict() if len(summary) else {}
    if best:
        print("\n=== Recommended YAML overrides ===")
        print(f"two_stage_cot: {str(best.get('two_stage_cot', 'false')).lower()}")
        print(f"best_of_n: {int(best.get('best_of_n', 1))}")
        print(f"temperature: {float(best.get('temperature', 0.1))}")
        tmax = best.get("temperature_max", "")
        if isinstance(tmax, str) and tmax.strip():
            print(f"temperature_max: \"{tmax}\"")
        else:
            print("temperature_max: null")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

