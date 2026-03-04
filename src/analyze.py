"""Analysis and Visualisation — load benchmark results and generate figures.

Loads all JSON benchmark reports, produces a combined DataFrame, generates
publication-quality figures, and compares against prior work data.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
PRIOR_WORK_DIR = PROJECT_ROOT / "prior_work"
CARBON_DATA = PROJECT_ROOT / "data" / "carbon_cache" / "roi_sample_7day.csv"
FIGURES_DIR = PROJECT_ROOT / "paper" / "figures"
TABLES_DIR = PROJECT_ROOT / "paper" / "tables"

# Model parameter counts (billions)
MODEL_PARAMS = {
    "meta-llama/Llama-3.2-1B-Instruct": 1.0,
    "Qwen/Qwen2.5-1.5B-Instruct": 1.5,
    "google/gemma-2-2b-it": 2.0,
    "meta-llama/Llama-3.2-3B-Instruct": 3.0,
    "microsoft/Phi-3-mini-4k-instruct": 3.8,
    "mistralai/Mistral-7B-Instruct-v0.3": 7.0,
    "Qwen/Qwen2.5-7B-Instruct": 7.0,
    "meta-llama/Llama-3.1-8B-Instruct": 8.0,
    "google/gemma-2-9b-it": 9.0,
    "microsoft/Phi-3-medium-4k-instruct": 14.0,
    "Qwen/Qwen2.5-32B-Instruct": 32.0,
    "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4": 32.0,
    "mistralai/Mixtral-8x7B-Instruct-v0.1": 46.7,
    "TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ": 46.7,
    "meta-llama/Llama-3.3-70B-Instruct": 70.0,
    "shuyuej/Llama-3.3-70B-Instruct-GPTQ": 70.0,
}

MODEL_FAMILY = {
    "meta-llama/Llama-3.2-1B-Instruct": "Llama",
    "meta-llama/Llama-3.2-3B-Instruct": "Llama",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama",
    "meta-llama/Llama-3.3-70B-Instruct": "Llama",
    "shuyuej/Llama-3.3-70B-Instruct-GPTQ": "Llama",
    "mistralai/Mistral-7B-Instruct-v0.3": "Mistral",
    "mistralai/Mixtral-8x7B-Instruct-v0.1": "Mixtral",
    "TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ": "Mixtral",
    "Qwen/Qwen2.5-1.5B-Instruct": "Qwen",
    "Qwen/Qwen2.5-7B-Instruct": "Qwen",
    "Qwen/Qwen2.5-32B-Instruct": "Qwen",
    "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4": "Qwen",
    "microsoft/Phi-3-mini-4k-instruct": "Phi",
    "microsoft/Phi-3-medium-4k-instruct": "Phi",
    "google/gemma-2-2b-it": "Gemma",
    "google/gemma-2-9b-it": "Gemma",
}

# Colourblind-friendly palette
FAMILY_COLOURS = {
    "Llama": "#0077BB",
    "Mistral": "#EE7733",
    "Mixtral": "#CC3311",
    "Qwen": "#009988",
    "Phi": "#EE3377",
    "Gemma": "#33BBEE",
}


def load_all_reports() -> pd.DataFrame:
    """Load all JSON benchmark reports into a single DataFrame."""
    records = []
    for json_file in RESULTS_DIR.rglob("benchmark_*.json"):
        with open(json_file) as f:
            report = json.load(f)

        model_name = report["model"]["name"]
        precision = report["model"]["precision"]
        baseline_watts = report["summary"]["idle_baseline_watts"]
        params_b = MODEL_PARAMS.get(model_name, 0)
        family = MODEL_FAMILY.get(model_name, "Unknown")

        for r in report["results"]:
            records.append({
                "model": model_name,
                "model_short": model_name.split("/")[-1],
                "params_b": params_b,
                "family": family,
                "precision": precision,
                "prompt_id": r["prompt_id"],
                "task_type": r["task_type"],
                "batch_size": r["batch_size"],
                "n_runs": r["runs"],
                "j_per_tok_mean": r["metrics"]["joules_per_token"]["mean"],
                "j_per_tok_std": r["metrics"]["joules_per_token"]["std"],
                "j_per_tok_ci_lo": r["metrics"]["joules_per_token"]["ci_95_lower"],
                "j_per_tok_ci_hi": r["metrics"]["joules_per_token"]["ci_95_upper"],
                "tok_per_s_mean": r["metrics"]["tokens_per_second"]["mean"],
                "tok_per_s_std": r["metrics"]["tokens_per_second"]["std"],
                "mean_watts": r["metrics"]["mean_watts"]["mean"],
                "mean_watts_std": r["metrics"]["mean_watts"]["std"],
                "output_tokens_mean": r["metrics"]["output_tokens"]["mean"],
                "input_tokens_mean": r["metrics"]["input_tokens"]["mean"],
                "baseline_watts": baseline_watts,
                "report_file": str(json_file),
            })

    if not records:
        logger.warning("No benchmark reports found in %s", RESULTS_DIR)
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df = df.sort_values(["params_b", "precision", "batch_size", "prompt_id"])
    return df


def export_combined_csv(df: pd.DataFrame) -> Path:
    """Export combined results to CSV."""
    out = RESULTS_DIR / "combined_results.csv"
    df.to_csv(out, index=False)
    logger.info("Combined CSV saved: %s", out)
    return out


def _power_law(x, a, b):
    """Power law: y = a * x^b."""
    return a * np.power(x, b)


def figure_1_scaling_law(df: pd.DataFrame) -> None:
    """Figure 1: J/tok vs Model Size (log-log scaling law)."""
    # Use batch_size=1. Prefer fp16, but include gptq for models only available as GPTQ.
    bs1 = df[df["batch_size"] == 1]
    if bs1.empty:
        logger.warning("No bs=1 data for scaling law plot")
        return

    # For each model, prefer fp16 data; fall back to gptq/awq if fp16 unavailable
    best_rows = []
    for model in bs1["model"].unique():
        model_data = bs1[bs1["model"] == model]
        fp16_data = model_data[model_data["precision"] == "fp16"]
        if not fp16_data.empty:
            best_rows.append(fp16_data)
        else:
            # Use whatever precision is available (gptq, awq, int8, int4)
            best_rows.append(model_data)
    fp16 = pd.concat(best_rows) if best_rows else bs1

    # Average across prompts per model
    grouped = fp16.groupby(["model", "params_b", "family", "precision"]).agg(
        j_per_tok=("j_per_tok_mean", "mean"),
        j_per_tok_err=("j_per_tok_std", "mean"),
    ).reset_index()

    # Deduplicate: keep only one entry per param count for fitting
    # (prefer fp16 over gptq for same model size)
    grouped = grouped.sort_values("precision").drop_duplicates(subset=["params_b"], keep="first")

    fig, ax = plt.subplots(figsize=(7, 5))

    for family, colour in FAMILY_COLOURS.items():
        subset = grouped[grouped["family"] == family]
        if subset.empty:
            continue
        # Mark GPTQ points with a different marker
        fp16_pts = subset[subset["precision"] == "fp16"]
        gptq_pts = subset[subset["precision"].isin(["gptq", "awq"])]
        if not fp16_pts.empty:
            ax.scatter(fp16_pts["params_b"], fp16_pts["j_per_tok"],
                       color=colour, label=family, s=80, zorder=5,
                       edgecolors="black", linewidth=0.5)
        if not gptq_pts.empty:
            ax.scatter(gptq_pts["params_b"], gptq_pts["j_per_tok"],
                       color=colour, label=f"{family} (GPTQ)", s=80, zorder=5,
                       marker="D", edgecolors="black", linewidth=0.5)

    # Fit power law across all data
    x = grouped["params_b"].values
    y = grouped["j_per_tok"].values
    if len(x) >= 3:
        try:
            popt, _ = curve_fit(_power_law, x, y, p0=[0.05, 0.5], maxfev=5000)
            x_fit = np.logspace(np.log10(x.min() * 0.8), np.log10(x.max() * 1.2), 100)
            y_fit = _power_law(x_fit, *popt)
            ss_res = np.sum((y - _power_law(x, *popt)) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            ax.plot(x_fit, y_fit, "k--", alpha=0.6,
                    label=f"Fit: J/tok = {popt[0]:.3f} N^{popt[1]:.2f} (R²={r2:.2f})")
        except RuntimeError:
            logger.warning("Power law fit failed")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Model Parameters (Billions)", fontsize=12)
    ax.set_ylabel("J/tok (mean, bs=1)", fontsize=12)
    ax.set_title("Energy per Token vs Model Size", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig1_scaling_law.pdf", dpi=300)
    fig.savefig(FIGURES_DIR / "fig1_scaling_law.png", dpi=300)
    plt.close(fig)
    logger.info("Figure 1 saved")


def _best_precision_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return data preferring fp16 per model, falling back to gptq/awq."""
    rows = []
    for model in df["model"].unique():
        m = df[df["model"] == model]
        fp16 = m[m["precision"] == "fp16"]
        rows.append(fp16 if not fp16.empty else m)
    return pd.concat(rows) if rows else df


def figure_2_efficiency_frontier(df: pd.DataFrame) -> None:
    """Figure 2: J/tok vs Throughput (efficiency frontier)."""
    fp16 = _best_precision_data(df)

    # Average across prompts per (model, batch_size)
    grouped = fp16.groupby(["model_short", "params_b", "family", "batch_size"]).agg(
        j_per_tok=("j_per_tok_mean", "mean"),
        tok_per_s=("tok_per_s_mean", "mean"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(7, 5))

    for family, colour in FAMILY_COLOURS.items():
        subset = grouped[grouped["family"] == family]
        if subset.empty:
            continue
        ax.scatter(subset["tok_per_s"], subset["j_per_tok"],
                   color=colour, label=family, s=60, alpha=0.7, edgecolors="black", linewidth=0.3)

    # Pareto frontier
    pts = grouped[["tok_per_s", "j_per_tok"]].values
    if len(pts) > 0:
        # Sort by throughput ascending
        pts = pts[pts[:, 0].argsort()]
        pareto = [pts[0]]
        for p in pts[1:]:
            if p[1] <= pareto[-1][1]:
                pareto.append(p)
        pareto = np.array(pareto)
        ax.plot(pareto[:, 0], pareto[:, 1], "r-", linewidth=2, alpha=0.5, label="Pareto frontier")

    ax.set_xlabel("Throughput (tok/s)", fontsize=12)
    ax.set_ylabel("J/tok", fontsize=12)
    ax.set_title("Energy Efficiency Frontier", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2_efficiency_frontier.pdf", dpi=300)
    fig.savefig(FIGURES_DIR / "fig2_efficiency_frontier.png", dpi=300)
    plt.close(fig)
    logger.info("Figure 2 saved")


def figure_3_batch_size(df: pd.DataFrame) -> None:
    """Figure 3: Batch Size Effect on J/tok and throughput."""
    fp16 = _best_precision_data(df)

    grouped = fp16.groupby(["model_short", "family", "batch_size"]).agg(
        j_per_tok=("j_per_tok_mean", "mean"),
        tok_per_s=("tok_per_s_mean", "mean"),
    ).reset_index()

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax2 = ax1.twinx()

    models = grouped["model_short"].unique()
    for model in models:
        subset = grouped[grouped["model_short"] == model].sort_values("batch_size")
        family = subset["family"].iloc[0]
        colour = FAMILY_COLOURS.get(family, "#999999")
        ax1.plot(subset["batch_size"], subset["j_per_tok"], "o-",
                 color=colour, label=model, markersize=5)
        ax2.plot(subset["batch_size"], subset["tok_per_s"], "s--",
                 color=colour, alpha=0.4, markersize=4)

    ax1.set_xlabel("Batch Size", fontsize=12)
    ax1.set_ylabel("J/tok (solid lines)", fontsize=12)
    ax2.set_ylabel("tok/s (dashed lines)", fontsize=12, alpha=0.6)
    ax1.set_title("Batch Size Effect on Energy and Throughput", fontsize=13)
    ax1.legend(fontsize=7, loc="upper right", ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale("log", base=2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig3_batch_size.pdf", dpi=300)
    fig.savefig(FIGURES_DIR / "fig3_batch_size.png", dpi=300)
    plt.close(fig)
    logger.info("Figure 3 saved")


def figure_4_quantisation(df: pd.DataFrame) -> None:
    """Figure 4: Quantisation Impact (Llama-3.1-8B fp16 vs int8 vs int4)."""
    quant = df[df["model"].str.contains("Llama-3.1-8B")]
    if quant.empty:
        logger.warning("No Llama-3.1-8B data for quantisation plot")
        return

    grouped = quant.groupby(["precision", "batch_size"]).agg(
        j_per_tok=("j_per_tok_mean", "mean"),
        tok_per_s=("tok_per_s_mean", "mean"),
        watts=("mean_watts", "mean"),
    ).reset_index()

    # Use batch_size=1 for bar chart
    bs1 = grouped[grouped["batch_size"] == 1]
    if bs1.empty:
        bs1 = grouped.groupby("precision").first().reset_index()

    precisions = ["fp16", "int8", "int4"]
    available = [p for p in precisions if p in bs1["precision"].values]
    if not available:
        return

    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    metrics = [("j_per_tok", "J/tok"), ("tok_per_s", "tok/s"), ("watts", "Mean Watts")]
    colours = {"fp16": "#0077BB", "int8": "#EE7733", "int4": "#CC3311"}

    for ax, (col, label) in zip(axes, metrics):
        vals = [bs1[bs1["precision"] == p][col].values[0] if p in available else 0
                for p in available]
        bars = ax.bar(available, vals, color=[colours.get(p, "#999") for p in available])
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label, fontsize=12)
        # Add value labels
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.3f}" if val < 10 else f"{val:.1f}",
                    ha="center", va="bottom", fontsize=9)

    fig.suptitle("Quantisation Impact: Llama-3.1-8B (bs=1)", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig4_quantisation.pdf", dpi=300)
    fig.savefig(FIGURES_DIR / "fig4_quantisation.png", dpi=300)
    plt.close(fig)
    logger.info("Figure 4 saved")


def figure_5_prompt_type(df: pd.DataFrame) -> None:
    """Figure 5: Prompt Type Effect (heatmap)."""
    bs1 = df[df["batch_size"] == 1]
    fp16 = _best_precision_data(bs1)
    fp16 = fp16[fp16["batch_size"] == 1]
    if fp16.empty:
        return

    pivot = fp16.pivot_table(
        values="j_per_tok_mean",
        index="model_short",
        columns="task_type",
        aggfunc="mean",
    )

    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(8, max(4, len(pivot) * 0.5)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=10, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, label="J/tok")
    ax.set_title("J/tok by Prompt Type and Model (bs=1, fp16)", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig5_prompt_type.pdf", dpi=300)
    fig.savefig(FIGURES_DIR / "fig5_prompt_type.png", dpi=300)
    plt.close(fig)
    logger.info("Figure 5 saved")


def figure_6_architecture_comparison(df: pd.DataFrame) -> None:
    """Figure 6: Architecture Comparison at similar parameter counts (7-9B)."""
    models_7_9b = df[
        (df["params_b"] >= 7) & (df["params_b"] <= 9) &
        (df["precision"] == "fp16") & (df["batch_size"] == 1)
    ]
    if models_7_9b.empty:
        logger.warning("No 7-9B fp16 bs=1 data for architecture comparison")
        return

    grouped = models_7_9b.groupby(["model_short", "family"]).agg(
        j_per_tok=("j_per_tok_mean", "mean"),
        j_per_tok_std=("j_per_tok_std", "mean"),
        tok_per_s=("tok_per_s_mean", "mean"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(7, 4))
    colours = [FAMILY_COLOURS.get(f, "#999") for f in grouped["family"]]
    bars = ax.bar(grouped["model_short"], grouped["j_per_tok"],
                  yerr=grouped["j_per_tok_std"], capsize=4,
                  color=colours, edgecolor="black", linewidth=0.5)

    ax.set_ylabel("J/tok (mean, bs=1)", fontsize=12)
    ax.set_title("Architecture Comparison: 7-9B Models (fp16, bs=1)", fontsize=13)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig6_architecture.pdf", dpi=300)
    fig.savefig(FIGURES_DIR / "fig6_architecture.png", dpi=300)
    plt.close(fig)
    logger.info("Figure 6 saved")


def figure_7_carbon_variation(df: pd.DataFrame) -> None:
    """Figure 7: Carbon Variation using Irish grid data."""
    if not CARBON_DATA.exists():
        logger.warning("Carbon data not found: %s", CARBON_DATA)
        return

    carbon = pd.read_csv(CARBON_DATA)
    c_min = carbon["gco2_per_kwh"].min()
    c_mean = carbon["gco2_per_kwh"].mean()
    c_max = carbon["gco2_per_kwh"].max()

    # Use bs=1, prefer fp16 but include gptq for models without fp16
    bs1 = df[df["batch_size"] == 1]
    fp16_bs1 = _best_precision_data(bs1)
    fp16_bs1 = fp16_bs1[fp16_bs1["batch_size"] == 1]
    if fp16_bs1.empty:
        return

    grouped = fp16_bs1.groupby(["model_short", "params_b"]).agg(
        j_per_tok=("j_per_tok_mean", "mean"),
    ).reset_index().sort_values("params_b")

    def gco2_per_tok(j_per_tok, carbon_intensity):
        return (j_per_tok / 3_600_000) * carbon_intensity

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(grouped))
    width = 0.25

    for i, (c_val, c_label, colour) in enumerate([
        (c_min, f"Min ({c_min:.0f})", "#33BBEE"),
        (c_mean, f"Mean ({c_mean:.0f})", "#EE7733"),
        (c_max, f"Max ({c_max:.0f})", "#CC3311"),
    ]):
        vals = [gco2_per_tok(j, c_val) * 1e6 for j in grouped["j_per_tok"]]
        ax.bar([xi + i * width for xi in x], vals, width,
               label=f"{c_label} gCO2/kWh", color=colour, alpha=0.8)

    ax.set_xticks([xi + width for xi in x])
    ax.set_xticklabels(grouped["model_short"], fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("ugCO2eq/tok", fontsize=12)
    ax.set_title("Carbon Impact per Token at Irish Grid Intensity Levels", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig7_carbon_variation.pdf", dpi=300)
    fig.savefig(FIGURES_DIR / "fig7_carbon_variation.png", dpi=300)
    plt.close(fig)
    logger.info("Figure 7 saved")


def prior_work_scaling_overlay(df: pd.DataFrame) -> None:
    """Overlay new results with energy-bench Pythia scaling law."""
    pythia_csv = PRIOR_WORK_DIR / "energy_bench_pythia_combined.csv"
    if not pythia_csv.exists():
        logger.warning("Pythia CSV not found")
        return

    pythia = pd.read_csv(pythia_csv)
    # Average J/tok per model at bs=1 (or bs=2 if bs=1 unavailable)
    pythia_agg = pythia.groupby(["model_id", "model_params_est"]).agg(
        j_per_tok=("joules_per_token", "mean"),
    ).reset_index()
    pythia_agg["params_b"] = pythia_agg["model_params_est"] / 1e9

    # New framework data — prefer fp16, include gptq for large models
    new_bs1 = _best_precision_data(df[df["batch_size"] == 1])
    new_bs1 = new_bs1[new_bs1["batch_size"] == 1]
    new_agg = new_bs1.groupby(["model_short", "params_b", "family", "precision"]).agg(
        j_per_tok=("j_per_tok_mean", "mean"),
    ).reset_index()
    # Deduplicate by params_b (prefer fp16)
    new_agg = new_agg.sort_values("precision").drop_duplicates(
        subset=["params_b"], keep="first"
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    # Pythia data
    ax.scatter(pythia_agg["params_b"], pythia_agg["j_per_tok"],
               marker="x", s=60, color="#999999", label="energy-bench Pythia", zorder=3)

    # energy-bench scaling law: E = 0.048 * (N/70M)^0.8
    x_law = np.logspace(-1.2, 2.0, 200)
    y_law = 0.048 * (x_law / 0.07) ** 0.8
    ax.plot(x_law, y_law, "k--", alpha=0.4, label="Pythia fit (alpha=0.8)")

    # New framework data
    for family, colour in FAMILY_COLOURS.items():
        subset = new_agg[new_agg["family"] == family]
        if subset.empty:
            continue
        ax.scatter(subset["params_b"], subset["j_per_tok"],
                   color=colour, label=f"New: {family}", s=80, zorder=5,
                   edgecolors="black", linewidth=0.5)

    # Fit new power law if enough points
    if len(new_agg) >= 3:
        try:
            popt, _ = curve_fit(_power_law, new_agg["params_b"].values,
                                new_agg["j_per_tok"].values, p0=[0.05, 0.5])
            x_fit = np.logspace(np.log10(new_agg["params_b"].min() * 0.8),
                                np.log10(new_agg["params_b"].max() * 1.2), 100)
            ax.plot(x_fit, _power_law(x_fit, *popt), "r-", alpha=0.6,
                    label=f"New fit (alpha={popt[1]:.2f})")
        except RuntimeError:
            pass

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Model Parameters (Billions)", fontsize=12)
    ax.set_ylabel("J/tok", fontsize=12)
    ax.set_title("Scaling Law: New Framework vs energy-bench Pythia", fontsize=13)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "prior_scaling_overlay.pdf", dpi=300)
    fig.savefig(FIGURES_DIR / "prior_scaling_overlay.png", dpi=300)
    plt.close(fig)
    logger.info("Prior work scaling overlay saved")


def prior_work_quantisation_comparison(df: pd.DataFrame) -> None:
    """Compare quantisation multipliers with energy-bench Mistral-7B data."""
    quant_files = {
        "fp16": PRIOR_WORK_DIR / "energy_bench_quant_fp16.csv",
        "int8": PRIOR_WORK_DIR / "energy_bench_quant_int8.csv",
        "nf4": PRIOR_WORK_DIR / "energy_bench_quant_nf4.csv",
    }

    prior = {}
    for prec, path in quant_files.items():
        if path.exists():
            data = pd.read_csv(path)
            prior[prec] = data["joules_per_token"].mean()

    if not prior:
        logger.warning("No prior quantisation data found")
        return

    # New framework Llama-3.1-8B data
    new_quant = df[df["model"].str.contains("Llama-3.1-8B")]
    if new_quant.empty:
        logger.warning("No Llama-3.1-8B data for quantisation comparison")
        return

    new_prec = {}
    for prec in ["fp16", "int8", "int4"]:
        subset = new_quant[(new_quant["precision"] == prec) & (new_quant["batch_size"] == 1)]
        if not subset.empty:
            new_prec[prec] = subset["j_per_tok_mean"].mean()

    fig, ax = plt.subplots(figsize=(7, 4))

    labels = []
    prior_vals = []
    new_vals = []

    mapping = {"fp16": "fp16", "int8": "int8", "nf4": "int4"}
    for prior_key, new_key in mapping.items():
        if prior_key in prior and new_key in new_prec:
            labels.append(new_key.upper())
            prior_vals.append(prior[prior_key])
            new_vals.append(new_prec[new_key])

    if not labels:
        return

    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, prior_vals, width, label="energy-bench (Mistral-7B)", color="#999999")
    ax.bar(x + width / 2, new_vals, width, label="New framework (Llama-3.1-8B)", color="#0077BB")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("J/tok", fontsize=12)
    ax.set_title("Quantisation Impact: energy-bench vs New Framework", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "prior_quantisation_comparison.pdf", dpi=300)
    fig.savefig(FIGURES_DIR / "prior_quantisation_comparison.png", dpi=300)
    plt.close(fig)
    logger.info("Prior work quantisation comparison saved")


def prior_work_batch_saturation_overlay(df: pd.DataFrame) -> None:
    """Overlay batch size curves with energy-bench Pythia-6.9B data."""
    batch_csv = PRIOR_WORK_DIR / "energy_bench_batch_saturation.csv"
    if not batch_csv.exists():
        logger.warning("Batch saturation CSV not found")
        return

    prior = pd.read_csv(batch_csv)
    prior_agg = prior.groupby("batch_size").agg(
        j_per_tok=("joules_per_token", "mean"),
    ).reset_index()

    # New framework — prefer fp16, include gptq for large models
    best = _best_precision_data(df)

    fig, ax = plt.subplots(figsize=(7, 5))

    # Prior data
    ax.plot(prior_agg["batch_size"], prior_agg["j_per_tok"], "x--",
            color="#999999", label="energy-bench Pythia-6.9B", markersize=8)

    # New framework per model
    models = best.groupby("model_short")["params_b"].first().sort_values()
    for model in models.index:
        subset = best[best["model_short"] == model]
        batch_avg = subset.groupby("batch_size")["j_per_tok_mean"].mean().reset_index()
        batch_avg = batch_avg.sort_values("batch_size")
        family = subset["family"].iloc[0]
        colour = FAMILY_COLOURS.get(family, "#999")
        ax.plot(batch_avg["batch_size"], batch_avg["j_per_tok_mean"], "o-",
                color=colour, label=model, markersize=5)

    ax.set_xlabel("Batch Size", fontsize=12)
    ax.set_ylabel("J/tok", fontsize=12)
    ax.set_title("Batch Size Saturation: New Framework vs energy-bench", fontsize=13)
    ax.set_xscale("log", base=2)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "prior_batch_saturation.pdf", dpi=300)
    fig.savefig(FIGURES_DIR / "prior_batch_saturation.png", dpi=300)
    plt.close(fig)
    logger.info("Prior work batch saturation overlay saved")


def generate_summary_table(df: pd.DataFrame) -> None:
    """Generate summary table for the paper."""
    if df.empty:
        return

    # Best J/tok per model (any batch size, fp16 preferred)
    summary_rows = []
    for model in df["model"].unique():
        model_data = df[df["model"] == model]
        for prec in model_data["precision"].unique():
            prec_data = model_data[model_data["precision"] == prec]
            avg = prec_data.groupby("batch_size").agg(
                j_per_tok=("j_per_tok_mean", "mean"),
                tok_per_s=("tok_per_s_mean", "mean"),
                watts=("mean_watts", "mean"),
            ).reset_index()

            best_idx = avg["j_per_tok"].idxmin()
            best = avg.loc[best_idx]
            summary_rows.append({
                "Model": model.split("/")[-1],
                "Params (B)": MODEL_PARAMS.get(model, 0),
                "Precision": prec,
                "Best J/tok": f"{best['j_per_tok']:.4f}",
                "Best tok/s": f"{best['tok_per_s']:.1f}",
                "Best Batch": int(best["batch_size"]),
                "Mean Watts": f"{best['watts']:.1f}",
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values("Params (B)")

    # Save as CSV
    summary_df.to_csv(TABLES_DIR / "summary_table.csv", index=False)

    # Save as LaTeX
    latex = summary_df.to_latex(index=False, column_format="lrllrrl")
    with open(TABLES_DIR / "summary_table.tex", "w") as f:
        f.write(latex)

    logger.info("Summary table saved")


def generate_cross_study_table(df: pd.DataFrame) -> None:
    """Generate cross-study comparison table with energy-bench."""
    rows = []

    # energy-bench key metrics
    rows.append({
        "Metric": "Scaling exponent (alpha)",
        "energy-bench": "0.80 (Pythia, R²=0.99)",
        "New Framework": "TBD from fit",
    })

    # Load Pythia data for reference
    pythia_csv = PRIOR_WORK_DIR / "energy_bench_pythia_combined.csv"
    if pythia_csv.exists():
        pythia = pd.read_csv(pythia_csv)
        p7b = pythia[pythia["model_params_est"].between(6e9, 8e9)]
        if not p7b.empty:
            rows.append({
                "Metric": "J/tok ~7B model (bs=1)",
                "energy-bench": f"{p7b['joules_per_token'].mean():.3f} (Pythia-6.9B)",
                "New Framework": "TBD",
            })

    # Quantisation reference
    rows.append({
        "Metric": "INT8 energy multiplier",
        "energy-bench": "2.88x (Mistral-7B)",
        "New Framework": "TBD (Llama-3.1-8B)",
    })
    rows.append({
        "Metric": "NF4/INT4 energy multiplier",
        "energy-bench": "3.72x (Mistral-7B)",
        "New Framework": "TBD (Llama-3.1-8B)",
    })

    # Fill in new framework values if available
    if not df.empty:
        # Scaling fit — include GPTQ models for wider range
        best_bs1 = _best_precision_data(df[df["batch_size"] == 1])
        fp16_bs1 = best_bs1[best_bs1["batch_size"] == 1]
        if not fp16_bs1.empty:
            new_agg = fp16_bs1.groupby("params_b")["j_per_tok_mean"].mean().reset_index()
            if len(new_agg) >= 3:
                try:
                    popt, _ = curve_fit(_power_law, new_agg["params_b"].values,
                                        new_agg["j_per_tok_mean"].values, p0=[0.05, 0.5])
                    ss_res = np.sum((new_agg["j_per_tok_mean"].values -
                                     _power_law(new_agg["params_b"].values, *popt)) ** 2)
                    ss_tot = np.sum((new_agg["j_per_tok_mean"].values -
                                     new_agg["j_per_tok_mean"].mean()) ** 2)
                    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                    rows[0]["New Framework"] = f"{popt[1]:.2f} (multi-arch, R²={r2:.2f})"
                except RuntimeError:
                    pass

        # Fill in 7B model J/tok
        models_7b = fp16_bs1[fp16_bs1["params_b"].between(6.5, 8.5)]
        if not models_7b.empty:
            avg_7b = models_7b.groupby("model")["j_per_tok_mean"].mean()
            # Find the row for 7B metric and update it
            for i, row in enumerate(rows):
                if "J/tok ~7B" in row["Metric"]:
                    models_str = ", ".join(
                        m.split("/")[-1] for m in avg_7b.index
                    )
                    rows[i]["New Framework"] = (
                        f"{avg_7b.mean():.3f} (avg: {models_str})"
                    )
                    break

        # Quantisation multipliers
        llama8b = df[df["model"].str.contains("Llama-3.1-8B")]
        if not llama8b.empty:
            fp16_val = llama8b[(llama8b["precision"] == "fp16") & (llama8b["batch_size"] == 1)]
            if not fp16_val.empty:
                fp16_jtok = fp16_val["j_per_tok_mean"].mean()
                for prec, idx in [("int8", -2), ("int4", -1)]:
                    prec_data = llama8b[(llama8b["precision"] == prec) & (llama8b["batch_size"] == 1)]
                    if not prec_data.empty:
                        mult = prec_data["j_per_tok_mean"].mean() / fp16_jtok
                        rows[idx]["New Framework"] = f"{mult:.2f}x (Llama-3.1-8B)"

    table = pd.DataFrame(rows)
    table.to_csv(TABLES_DIR / "cross_study_comparison.csv", index=False)
    latex = table.to_latex(index=False)
    with open(TABLES_DIR / "cross_study_comparison.tex", "w") as f:
        f.write(latex)

    logger.info("Cross-study comparison table saved")


def run_analysis():
    """Run the full analysis pipeline."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading all benchmark reports...")
    df = load_all_reports()

    if df.empty:
        logger.error("No data to analyse. Run benchmarks first.")
        return

    logger.info("Loaded %d rows from %d reports", len(df), df["report_file"].nunique())

    # Export combined CSV
    export_combined_csv(df)

    # Generate all 7 primary figures
    logger.info("Generating figures...")
    figure_1_scaling_law(df)
    figure_2_efficiency_frontier(df)
    figure_3_batch_size(df)
    figure_4_quantisation(df)
    figure_5_prompt_type(df)
    figure_6_architecture_comparison(df)
    figure_7_carbon_variation(df)

    # Prior work comparisons
    logger.info("Generating prior work comparisons...")
    prior_work_scaling_overlay(df)
    prior_work_quantisation_comparison(df)
    prior_work_batch_saturation_overlay(df)

    # Tables
    logger.info("Generating tables...")
    generate_summary_table(df)
    generate_cross_study_table(df)

    logger.info("Analysis complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_analysis()
