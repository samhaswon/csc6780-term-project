#!/usr/bin/env python3

"""
Plot manager timings.
Run with: python3 plot_manager_timings.py --csv manager_timings.csv --outdir plots
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # ensure non-interactive backend for headless environments
# pylint: disable=wrong-import-position
import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    """Parse CLI arguments or display the help message."""
    p = argparse.ArgumentParser(description="Plot average manager timings per model vs nodes.")
    p.add_argument(
        "--csv",
        type=Path,
        default=Path("manager_timings.csv"),
        help="Path to manager_timings.csv",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("plots"),
        help="Directory to write plots into",
    )
    return p.parse_args()


# pylint: disable=too-many-locals
def main():
    """main"""
    args = parse_args()

    if not args.csv.exists():
        raise SystemExit(f"CSV not found: {args.csv}")

    args.outdir.mkdir(parents=True, exist_ok=True)

    # Load CSV
    df = pd.read_csv(args.csv)

    # Validate expected columns
    required_cols = {"base_session", "nodes", "base_time", "patch_time", "total_time"}
    missing = required_cols - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns in CSV: {sorted(missing)}")

    # Ensure types are correct
    df = df.copy()
    df["nodes"] = pd.to_numeric(df["nodes"], errors="coerce").astype("Int64")
    for c in ("base_time", "patch_time", "total_time"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(
        subset=["base_session", "nodes", "base_time", "patch_time", "total_time"]
    )  # keep valid rows

    # Aggregate: mean per (base_session, nodes)
    agg = (
        df.groupby(
            ["base_session", "nodes"], as_index=False
        )[["base_time", "patch_time", "total_time"]]
        .mean()
    )

    # Mapping for display names (graph step only)
    pretty_names = {
        "birefnet": "BiRefNet",
        "u2net": "U²-Net",
        "u2netp": "U²-NetP",
    }

    models = sorted(agg["base_session"].unique())

    # Style
    colors = {
        "base_time": "#1f77b4",   # blue
        "patch_time": "#ff7f0e",  # orange
        "total_time": "#2ca02c",  # green
    }
    labels = {
        "base_time": "Base",
        "patch_time": "Patch",
        "total_time": "Total",
    }

    for model in models:
        sub = agg[agg["base_session"] == model].sort_values("nodes")

        fig, ax = plt.subplots(figsize=(8, 5))

        for col in ("base_time", "patch_time", "total_time"):
            ax.plot(
                sub["nodes"],
                sub[col],
                marker="o",
                linewidth=2,
                markersize=4,
                color=colors[col],
                label=labels[col],
            )

        ax.set_xlabel("Nodes")
        ax.set_ylabel("Average Time (s)")
        title = f"{pretty_names.get(model, model)}: Average Times vs Nodes"
        ax.set_title(title)
        ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)
        ax.legend(frameon=False)

        out_path = args.outdir / f"manager_timings_{model}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    # Overall average patch_time across all models per node
    overall = (
        agg.groupby(["nodes"], as_index=False)[["patch_time"]]
        .mean()
        .sort_values("nodes")
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        overall["nodes"],
        overall["patch_time"],
        marker="o",
        linewidth=2,
        markersize=4,
        color=colors["patch_time"],
        label="Patch (avg across models)",
    )
    ax.set_xlabel("Nodes")
    ax.set_ylabel("Average Patch Time (s)")
    ax.set_title("Average Patch Time vs Nodes (All Models)")
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)
    ax.legend(frameon=False)

    out_path_overall = args.outdir / "manager_timings_patch_overall.png"
    fig.tight_layout()
    fig.savefig(out_path_overall, dpi=150)
    plt.close(fig)

    # Comparison plot: total_time per model vs nodes on one figure
    fig, ax = plt.subplots(figsize=(9, 5))
    model_palette = list(plt.cm.tab10.colors)
    for i, model in enumerate(models):
        sub = agg[agg["base_session"] == model].sort_values("nodes")
        ax.plot(
            sub["nodes"],
            sub["total_time"],
            marker="o",
            linewidth=2,
            markersize=4,
            color=model_palette[i % len(model_palette)],
            label=pretty_names.get(model, model),
        )
    ax.set_xlabel("Nodes")
    ax.set_ylabel("Average Total Time (s)")
    ax.set_title("Total Time vs Nodes (By Model)")
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)
    ax.legend(title="Model", frameon=False, ncol=1)
    out_total_by_model = args.outdir / "manager_timings_total_by_model.png"
    fig.tight_layout()
    fig.savefig(out_total_by_model, dpi=150)
    plt.close(fig)

    fig_log, ax_log = plt.subplots(figsize=(9, 5))
    for i, model in enumerate(models):
        sub = agg[agg["base_session"] == model].sort_values("nodes")
        ax_log.plot(
            sub["nodes"],
            sub["total_time"],
            marker="o",
            linewidth=2,
            markersize=4,
            color=model_palette[i % len(model_palette)],
            label=pretty_names.get(model, model),
        )
    ax_log.set_xlabel("Nodes")
    ax_log.set_ylabel("Average Total Time (s)")
    ax_log.set_yscale("log")
    ax_log.set_title("Total Time vs Nodes (By Model, log-scale)")
    ax_log.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)
    ax_log.legend(title="Model", frameon=False, ncol=1)
    out_total_log = args.outdir / "manager_timings_total_by_model_log.png"
    fig_log.tight_layout()
    fig_log.savefig(out_total_log, dpi=150)
    plt.close(fig_log)

    # Bar chart: average base_time per model (averaged over node means)
    base_by_model = (
        agg.groupby(["base_session"], as_index=False)[["base_time"]]
        .mean()
        .sort_values("base_time")
    )
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x_labels = [pretty_names.get(m, m) for m in base_by_model["base_session"]]
    ax.bar(x_labels, base_by_model["base_time"], color=colors["base_time"], alpha=0.9)
    ax.set_ylabel("Average Base Time (s)")
    ax.set_title("Average Base Time by Model")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.8, alpha=0.7)
    for idx, v in enumerate(base_by_model["base_time"]):
        ax.text(idx, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    out_base_bar = args.outdir / "manager_base_time_by_model.png"
    fig.savefig(out_base_bar, dpi=150)
    plt.close(fig)

    print(
        f"Wrote {len(models)} model plot(s), total-time comparison plots (linear/log), base-time bar chart, and overall patch plot to: {args.outdir}"
    )


if __name__ == "__main__":
    main()
