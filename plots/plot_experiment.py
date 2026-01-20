from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    agg: str = "mean"  # "mean" or "sum"


DEFAULT_METRICS: list[MetricSpec] = [
    MetricSpec("csv_score", "CSV score (mean)", "mean"),
    MetricSpec("bleu", "BLEU (mean)", "mean"),
    MetricSpec("spice", "SPICE (mean)", "mean"),
    MetricSpec("duration_ms", "Duration (ms, mean)", "mean"),
    MetricSpec("total_emissions_kg", "Emissions (kg, mean)", "mean"),
]


def _required_columns_present(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns in sweep_results.csv: "
            + ", ".join(missing)
            + f"\nAvailable columns: {list(df.columns)}"
        )


def load_sweep_results(experiment_dir: Path) -> pd.DataFrame:
    sweep_csv = experiment_dir / "sweep_results.csv"
    if not sweep_csv.exists():
        raise FileNotFoundError(f"Could not find {sweep_csv}")

    df = pd.read_csv(sweep_csv)
    _required_columns_present(
        df,
        [
            "best_of_n",
            "temperature",
            "csv_score",
            "bleu",
            "spice",
            "duration_ms",
            "total_emissions_kg",
        ],
    )

    # Normalize types
    df["best_of_n"] = pd.to_numeric(df["best_of_n"], errors="coerce").astype("Int64")
    df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
    for c in ["csv_score", "bleu", "spice", "duration_ms", "total_emissions_kg"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["best_of_n", "temperature"])
    return df


def aggregate_metric(
    df: pd.DataFrame, metric: MetricSpec
) -> pd.DataFrame:
    grouped = df.groupby(["best_of_n", "temperature"], dropna=False)[metric.key]
    if metric.agg == "sum":
        agg_df = grouped.sum().reset_index(name="value")
    elif metric.agg == "mean":
        agg_df = grouped.mean().reset_index(name="value")
    else:
        raise ValueError(f"Unsupported aggregation '{metric.agg}' for {metric.key}")

    # Ensure stable ordering
    agg_df = agg_df.sort_values(["best_of_n", "temperature"])
    return agg_df


def _pivot_for_heatmap(agg_df: pd.DataFrame) -> pd.DataFrame:
    pivot = agg_df.pivot(index="best_of_n", columns="temperature", values="value")
    pivot = pivot.sort_index(axis=0).sort_index(axis=1)
    return pivot


def plot_heatmap(pivot: pd.DataFrame, title: str, out_path: Path) -> None:
    fig_w = max(6.0, 1.2 * max(1, pivot.shape[1]))
    fig_h = max(4.0, 0.8 * max(1, pivot.shape[0]))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)

    data = pivot.to_numpy()
    im = ax.imshow(data, aspect="auto")
    fig.colorbar(im, ax=ax, shrink=0.9)

    ax.set_title(title)
    ax.set_xlabel("temperature")
    ax.set_ylabel("best_of_n")

    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels([str(c) for c in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels([str(i) for i in pivot.index])

    # Annotate cells
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = data[i, j]
            if pd.isna(v):
                continue
            ax.text(j, i, f"{v:.3g}", ha="center", va="center", fontsize=9, color="black")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_lines(agg_df: pd.DataFrame, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)

    for t, sub in agg_df.groupby("temperature"):
        sub = sub.sort_values("best_of_n")
        ax.plot(
            sub["best_of_n"].astype(int),
            sub["value"],
            marker="o",
            linewidth=2,
            label=f"T={t:g}",
        )

    ax.set_title(title)
    ax.set_xlabel("best_of_n")
    ax.set_ylabel("value")
    ax.grid(True, alpha=0.25)
    ax.legend(title="temperature", loc="best")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _safe_stem(s: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in s).strip("_")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Plot metrics by best_of_n and temperature for a given experiment folder "
            "(expects sweep_results.csv)."
        )
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path("output/exp1_bestof_sweep"),
        help="Path to experiment directory (e.g. output/exp1_bestof_sweep).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for plots. Default: <experiment-dir>/plots",
    )
    parser.add_argument(
        "--format",
        choices=["png", "pdf"],
        default="png",
        help="Output file format.",
    )

    args = parser.parse_args()
    experiment_dir: Path = args.experiment_dir
    out_dir: Path = args.out_dir or (experiment_dir / "plots")

    df = load_sweep_results(experiment_dir)

    # If there are multiple test_cases/prompts, we still aggregate across them.
    # Users can filter upstream by subsetting sweep_results.csv if needed.
    for metric in DEFAULT_METRICS:
        agg_df = aggregate_metric(df, metric)
        pivot = _pivot_for_heatmap(agg_df)

        stem = _safe_stem(f"{metric.key}_{metric.agg}")
        plot_heatmap(
            pivot,
            title=f"{metric.label} by best_of_n × temperature",
            out_path=out_dir / f"heatmap_{stem}.{args.format}",
        )
        plot_lines(
            agg_df,
            title=f"{metric.label} vs best_of_n (colored by temperature)",
            out_path=out_dir / f"lines_{stem}.{args.format}",
        )

    # Also generate "total" variants for duration/emissions for convenience
    for metric in [
        MetricSpec("duration_ms", "Duration (ms, sum)", "sum"),
        MetricSpec("total_emissions_kg", "Emissions (kg, sum)", "sum"),
    ]:
        agg_df = aggregate_metric(df, metric)
        pivot = _pivot_for_heatmap(agg_df)
        stem = _safe_stem(f"{metric.key}_{metric.agg}")
        plot_heatmap(
            pivot,
            title=f"{metric.label} by best_of_n × temperature",
            out_path=out_dir / f"heatmap_{stem}.{args.format}",
        )
        plot_lines(
            agg_df,
            title=f"{metric.label} vs best_of_n (colored by temperature)",
            out_path=out_dir / f"lines_{stem}.{args.format}",
        )

    print(f"Wrote plots to: {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

