"""
Step 11 — Multi-Biomarker Panel Analysis
========================================
Combines AFP, GPC3, DKK1, and MDK into multi-marker detection circuits and
compares panel performance against every single-biomarker baseline already
computed in steps 8-10.

This implementation is intentionally aligned with the current project plan:
  - Step 11 stays focused on multi-marker fusion, activation probability,
    mutual information, and SR gain.
  - Label-based ROC/AUC framing is deferred to the planned Step 13.

Fusion strategies evaluated here
--------------------------------
  1. Logical-OR            — activate if any biomarker Hill output crosses 0.5
  2. Majority-Vote (K=2)   — activate if at least 2 biomarkers activate
  3. Majority-Vote (K=3)   — activate if at least 3 biomarkers activate
  4. Weighted-Sum          — MI(sigma*)-weighted linear combination of Hill outputs

For each strategy x noise type the script computes:
  • Activation probability curve vs noise intensity sigma
  • Mutual information curve   vs noise intensity sigma
  • Peak sigma* for MI
  • SR gain delta MI and delta activation relative to the best single-marker
    baseline under the same noise type

Outputs
-------
  output/step11_panel_activation_curves.png
  output/step11_panel_mi_curves.png
  output/step11_panel_mi_heatmap.png
  output/step11_panel_sr_gain_bar.png
  output/step11_panel_summary.csv
  output/step11_panel_summary.tex

Inputs
------
  output/tcga_lihc_afp_statistics.json
  output/gene_activation_threshold_K.json
  output/sr_full_results.csv
  output/step10_sr_peak_summary.csv
"""

from __future__ import annotations

import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import mutual_info_score

from noise_utils import (
    DEFAULT_LEVY_ALPHA,
    DEFAULT_LEVY_BETA,
    DEFAULT_OU_STEP,
    DEFAULT_OU_TAU,
    generate_noise,
)
from step10_sr_peak_characterization import find_sigma_star, smooth_curve, standardise_sr_dataframe

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BIOMARKERS = ["AFP", "GPC3", "DKK1", "MDK"]
NOISE_TYPES = ["Gaussian", "OU", "Levy"]
NOISE_TO_INTERNAL = {"Gaussian": "gaussian", "OU": "ou", "Levy": "levy"}
SIGMA_GRID = np.linspace(0.0, 6.0, 40)
NUM_TRIALS = 120
HILL_N = 4.0
DECISION_THRESHOLD = 0.5
MI_BINS = 12

FIG_BG = "white"
AX_BG = "white"
GRID_C = "#cccccc"
TICK_C = "#333333"
SPINE_C = "#999999"

BM_COLORS = {
    "AFP": "#1f77b4",
    "GPC3": "#d62728",
    "DKK1": "#2ca02c",
    "MDK": "#ff7f0e",
}
PANEL_COLORS = {
    "Logical-OR": "#9467bd",
    "Majority-Vote (K=2)": "#e377c2",
    "Majority-Vote (K=3)": "#8c564b",
    "Weighted-Sum": "#17becf",
}


def load_biomarker_matrix() -> np.ndarray:
    """Load the matched early-stage biomarker matrix from step 3 output."""
    path = os.path.join(OUTPUT_DIR, "tcga_lihc_afp_statistics.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required biomarker statistics file not found: {path}")

    with open(path) as handle:
        payload = json.load(handle)

    if "biomarkers" not in payload:
        raise ValueError("Expected multi-biomarker structure under 'biomarkers' in step 3 output.")

    columns = []
    lengths = []
    for biomarker in BIOMARKERS:
        if biomarker not in payload["biomarkers"]:
            raise ValueError(f"Biomarker {biomarker} missing from step 3 output.")
        values = np.array(payload["biomarkers"][biomarker]["values"], dtype=float)
        columns.append(values)
        lengths.append(len(values))

    if len(set(lengths)) != 1:
        raise ValueError(
            "Biomarker value arrays do not have the same length. "
            "Rerun step 3 so the panel analysis uses aligned patient vectors."
        )

    matrix = np.column_stack(columns)
    print(f"[load] Biomarker matrix loaded from {path} with shape {matrix.shape}")
    return matrix


def load_thresholds() -> np.ndarray:
    """Load per-biomarker activation thresholds from step 4."""
    path = os.path.join(OUTPUT_DIR, "gene_activation_threshold_K.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required threshold file not found: {path}")

    with open(path) as handle:
        payload = json.load(handle)

    if "biomarker_thresholds" not in payload:
        raise ValueError("Expected 'biomarker_thresholds' in step 4 output.")

    thresholds = np.array(
        [payload["biomarker_thresholds"][bm]["activation_threshold_K"] for bm in BIOMARKERS],
        dtype=float,
    )
    print(f"[load] Thresholds loaded from {path}")
    return thresholds


def load_single_curves() -> pd.DataFrame:
    """Load the single-marker activation/MI curves from step 8/9."""
    path = os.path.join(OUTPUT_DIR, "sr_full_results.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required single-marker curve file not found: {path}")

    df = standardise_sr_dataframe(pd.read_csv(path)).copy()
    df["noise_type"] = df["noise_type"].replace({"Gaussian": "Gaussian", "OU": "OU", "Levy": "Levy"})
    required = {"biomarker", "noise_type", "sigma", "activation_prob", "mutual_information"}
    if not required.issubset(df.columns):
        raise ValueError(f"Single-marker SR file missing required columns: {required - set(df.columns)}")

    print(f"[load] Single-marker curves loaded from {path}")
    return df


def load_step10_weights() -> dict[str, np.ndarray]:
    """
    Use step 10 MI(sigma*) values as the weighting scheme for the weighted-sum panel.
    """
    path = os.path.join(OUTPUT_DIR, "step10_sr_peak_summary.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required step 10 summary file not found: {path}")

    df = pd.read_csv(path)
    weights_by_noise: dict[str, np.ndarray] = {}
    for noise_type in NOISE_TYPES:
        rows = df[df["Noise Type"] == noise_type].set_index("Biomarker")
        values = np.array([rows.loc[bm, "MI(sigma*)"] for bm in BIOMARKERS], dtype=float)
        values = np.clip(values, 1e-6, None)
        weights_by_noise[noise_type] = values / values.sum()
    print(f"[load] Weighted-sum weights loaded from {path}")
    return weights_by_noise


def hill_matrix(x: np.ndarray, thresholds: np.ndarray, n: float = HILL_N) -> np.ndarray:
    """Hill activation applied column-wise with per-biomarker thresholds."""
    clipped = np.clip(x, 0.0, None)
    xn = clipped**n
    kn = thresholds**n
    return xn / (kn + xn)


def discretized_mi(score: np.ndarray, decision: np.ndarray, bins: int = MI_BINS) -> float:
    """Estimate MI(score, binary decision) in bits via discretisation."""
    if np.allclose(score, score[0]):
        return 0.0
    _, edges = np.histogram(score, bins=bins)
    score_disc = np.digitize(score, edges[1:-1], right=False)
    return mutual_info_score(score_disc, decision.astype(int)) / np.log(2.0)


def add_noise_matrix(signal_matrix: np.ndarray, noise_type: str, sigma: float, trial: int) -> np.ndarray:
    """Generate one noisy cohort matrix using the shared noise generators."""
    internal = NOISE_TO_INTERNAL[noise_type]
    noisy = np.empty_like(signal_matrix, dtype=float)
    for col_idx in range(signal_matrix.shape[1]):
        seed = trial * 100 + col_idx
        eta = generate_noise(
            internal,
            signal_matrix.shape[0],
            sigma,
            tau=DEFAULT_OU_TAU,
            h=DEFAULT_OU_STEP,
            levy_alpha=DEFAULT_LEVY_ALPHA,
            levy_beta=DEFAULT_LEVY_BETA,
            seed=seed,
        )
        noisy[:, col_idx] = signal_matrix[:, col_idx] + eta
    return noisy


def strategy_outputs(hill_out: np.ndarray, weights: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Return continuous score and binary decision for each panel strategy.
    """
    activated = (hill_out >= DECISION_THRESHOLD).astype(int)
    active_count = activated.sum(axis=1)

    or_score = np.max(hill_out, axis=1)
    or_decision = (active_count >= 1).astype(int)

    mv2_score = active_count / hill_out.shape[1]
    mv2_decision = (active_count >= 2).astype(int)

    mv3_score = active_count / hill_out.shape[1]
    mv3_decision = (active_count >= 3).astype(int)

    weighted_score = hill_out @ weights
    weighted_decision = (weighted_score >= DECISION_THRESHOLD).astype(int)

    return {
        "Logical-OR": (or_score, or_decision),
        "Majority-Vote (K=2)": (mv2_score, mv2_decision),
        "Majority-Vote (K=3)": (mv3_score, mv3_decision),
        "Weighted-Sum": (weighted_score, weighted_decision),
    }


def compute_panel_curves(signal_matrix: np.ndarray, thresholds: np.ndarray, weights: np.ndarray, noise_type: str) -> dict:
    """Run the Monte Carlo sweep for one noise family across all panel strategies."""
    panel = {
        strategy: {"act_mean": [], "act_std": [], "mi_mean": [], "mi_std": []}
        for strategy in PANEL_COLORS
    }

    for sigma in SIGMA_GRID:
        trial_metrics = {strategy: {"act": [], "mi": []} for strategy in PANEL_COLORS}

        for trial in range(NUM_TRIALS):
            noisy = add_noise_matrix(signal_matrix, noise_type, float(sigma), trial)
            hill_out = hill_matrix(noisy, thresholds)
            outputs = strategy_outputs(hill_out, weights)

            for strategy, (score, decision) in outputs.items():
                trial_metrics[strategy]["act"].append(float(decision.mean()))
                trial_metrics[strategy]["mi"].append(float(discretized_mi(score, decision)))

        for strategy in PANEL_COLORS:
            act_arr = np.array(trial_metrics[strategy]["act"], dtype=float)
            mi_arr = np.array(trial_metrics[strategy]["mi"], dtype=float)
            panel[strategy]["act_mean"].append(float(act_arr.mean()))
            panel[strategy]["act_std"].append(float(act_arr.std()))
            panel[strategy]["mi_mean"].append(float(mi_arr.mean()))
            panel[strategy]["mi_std"].append(float(mi_arr.std()))

    for strategy in PANEL_COLORS:
        for key in ("act_mean", "mi_mean"):
            arr = np.array(panel[strategy][key], dtype=float)
            smoothed = smooth_curve(SIGMA_GRID, arr)
            if key == "act_mean":
                smoothed = np.clip(smoothed, 0.0, 1.0)
            else:
                smoothed = np.clip(smoothed, 0.0, None)
            panel[strategy][key] = smoothed
        for key in ("act_std", "mi_std"):
            panel[strategy][key] = np.array(panel[strategy][key], dtype=float)

    return panel


def build_single_results(single_df: pd.DataFrame) -> dict[str, dict]:
    """Organise the already-computed single-marker curves by noise type and biomarker."""
    results: dict[str, dict] = {noise_type: {} for noise_type in NOISE_TYPES}
    for noise_type in NOISE_TYPES:
        sub_noise = single_df[single_df["noise_type"] == noise_type]
        for biomarker in BIOMARKERS:
            sub = sub_noise[sub_noise["biomarker"] == biomarker].sort_values("sigma")
            if sub.empty:
                raise ValueError(f"Missing single-marker curve for {biomarker} / {noise_type}")
            results[noise_type][biomarker] = {
                "sigma": sub["sigma"].to_numpy(dtype=float),
                "act_mean": smooth_curve(sub["sigma"].to_numpy(dtype=float), sub["activation_prob"].to_numpy(dtype=float)),
                "mi_mean": smooth_curve(sub["sigma"].to_numpy(dtype=float), sub["mutual_information"].to_numpy(dtype=float)),
            }
    return results


def peak_metrics(sigmas: np.ndarray, act_curve: np.ndarray, mi_curve: np.ndarray) -> dict[str, float]:
    """Compute sigma*, MI(sigma*), and Act(sigma*) from smoothed curves."""
    sigma_star = find_sigma_star(sigmas, mi_curve)
    return {
        "sigma_star": float(sigma_star),
        "mi_at_star": float(np.interp(sigma_star, sigmas, mi_curve)),
        "act_at_star": float(np.interp(sigma_star, sigmas, act_curve)),
    }


def run_panel_analysis(signal_matrix: np.ndarray, thresholds: np.ndarray, single_results: dict, weights_by_noise: dict) -> tuple[dict, pd.DataFrame]:
    """Main step-11 analysis loop."""
    results: dict[str, dict] = {}
    rows = []

    for noise_type in NOISE_TYPES:
        print(f"\n  -- Noise type: {noise_type} --")
        panel_curves = compute_panel_curves(signal_matrix, thresholds, weights_by_noise[noise_type], noise_type)

        single_peaks = {}
        for biomarker in BIOMARKERS:
            metrics = peak_metrics(
                single_results[noise_type][biomarker]["sigma"],
                single_results[noise_type][biomarker]["act_mean"],
                single_results[noise_type][biomarker]["mi_mean"],
            )
            single_peaks[biomarker] = metrics
            rows.append(
                {
                    "Noise Type": noise_type,
                    "Method": f"Single — {biomarker}",
                    "Type": "Single",
                    "sigma* (MI max)": round(metrics["sigma_star"], 4),
                    "MI(sigma*)": round(metrics["mi_at_star"], 5),
                    "Act. Prob(sigma*)": round(metrics["act_at_star"], 4),
                    "delta MI vs best single": 0.0,
                    "delta Act vs best single": 0.0,
                }
            )
            print(
                f"    Single {biomarker:4s}  sigma* = {metrics['sigma_star']:.3f}  "
                f"MI* = {metrics['mi_at_star']:.4f}"
            )

        best_single_peak = max(single_peaks.values(), key=lambda item: item["mi_at_star"])
        best_single_mi = best_single_peak["mi_at_star"]
        best_single_act = best_single_peak["act_at_star"]

        panel_peaks = {}
        for strategy in PANEL_COLORS:
            metrics = peak_metrics(SIGMA_GRID, panel_curves[strategy]["act_mean"], panel_curves[strategy]["mi_mean"])
            panel_peaks[strategy] = metrics
            rows.append(
                {
                    "Noise Type": noise_type,
                    "Method": strategy,
                    "Type": "Panel",
                    "sigma* (MI max)": round(metrics["sigma_star"], 4),
                    "MI(sigma*)": round(metrics["mi_at_star"], 5),
                    "Act. Prob(sigma*)": round(metrics["act_at_star"], 4),
                    "delta MI vs best single": round(metrics["mi_at_star"] - best_single_mi, 5),
                    "delta Act vs best single": round(metrics["act_at_star"] - best_single_act, 5),
                }
            )
            print(
                f"    {strategy:20s}  sigma* = {metrics['sigma_star']:.3f}  "
                f"MI* = {metrics['mi_at_star']:.4f}  "
                f"delta MI vs best single = {metrics['mi_at_star'] - best_single_mi:+.4f}"
            )

        results[noise_type] = {
            "single": single_results[noise_type],
            "panel": panel_curves,
            "weights": weights_by_noise[noise_type],
        }

    summary = pd.DataFrame(rows)
    return results, summary


def save_tables(summary: pd.DataFrame) -> None:
    csv_path = os.path.join(OUTPUT_DIR, "step11_panel_summary.csv")
    summary.to_csv(csv_path, index=False)
    print(f"\n[save] {csv_path}")

    tex_path = os.path.join(OUTPUT_DIR, "step11_panel_summary.tex")
    with open(tex_path, "w") as handle:
        handle.write("% Step 11 — Multi-Biomarker Panel Summary\n\n")
        handle.write("\\begin{table}[htbp]\\centering\n")
        handle.write(
            "\\caption{Panel vs single-biomarker stochastic-resonance performance. "
            "Step 11 is restricted to label-free fusion metrics; ROC/AUC framing is "
            "deferred to the planned Step 13.}\n"
        )
        handle.write("\\label{tab:panel}\n")
        handle.write("\\resizebox{\\textwidth}{!}{%\n")
        handle.write("\\begin{tabular}{llccccc}\\toprule\n")
        handle.write(
            "Noise & Method & $\\sigma^*$ & MI($\\sigma^*$) & "
            "Act.~Prob($\\sigma^*$) & $\\Delta$MI vs best single & "
            "$\\Delta P$ vs best single \\\\\n\\midrule\n"
        )
        previous_noise = None
        for _, row in summary.iterrows():
            noise_label = row["Noise Type"] if row["Noise Type"] != previous_noise else ""
            previous_noise = row["Noise Type"]
            bold = "\\textbf" if row["Type"] == "Panel" else ""
            handle.write(
                f"{noise_label} & {bold}{{{row['Method']}}} & "
                f"{row['sigma* (MI max)']:.3f} & "
                f"{row['MI(sigma*)']:.4f} & "
                f"{row['Act. Prob(sigma*)']:.3f} & "
                f"${row['delta MI vs best single']:+.4f}$ & "
                f"${row['delta Act vs best single']:+.4f}$ \\\\\n"
            )
        handle.write("\\bottomrule\n\\end{tabular}}\\end{table}\n")
    print(f"[save] {tex_path}")


def style_axes(ax, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    ax.set_facecolor(AX_BG)
    if title:
        ax.set_title(title, color="black", fontsize=10, fontweight="bold", pad=6)
    if xlabel:
        ax.set_xlabel(xlabel, color=TICK_C, fontsize=8.5)
    if ylabel:
        ax.set_ylabel(ylabel, color=TICK_C, fontsize=8.5)
    for spine in ax.spines.values():
        spine.set_color(SPINE_C)
    ax.tick_params(colors=TICK_C, labelsize=8)
    ax.grid(True, color=GRID_C, lw=0.55, linestyle="--", alpha=0.7)


def plot_activation(results: dict) -> None:
    fig, axes = plt.subplots(len(NOISE_TYPES), 1, figsize=(12, 4.5 * len(NOISE_TYPES)))
    fig.patch.set_facecolor(FIG_BG)
    fig.suptitle(
        "Activation Probability vs Noise Intensity\nPanel Strategies vs Individual Biomarkers",
        color="black",
        fontsize=13,
        fontweight="bold",
    )

    for idx, noise_type in enumerate(NOISE_TYPES):
        ax = axes[idx]
        noise_results = results[noise_type]

        for biomarker in BIOMARKERS:
            ax.plot(
                noise_results["single"][biomarker]["sigma"],
                noise_results["single"][biomarker]["act_mean"],
                color=BM_COLORS[biomarker],
                lw=1.1,
                linestyle="--",
                alpha=0.7,
                label=f"Single {biomarker}",
            )

        for strategy, color in PANEL_COLORS.items():
            ax.plot(SIGMA_GRID, noise_results["panel"][strategy]["act_mean"], color=color, lw=2.1, label=strategy)

        ax.axhline(0.5, color="#444c56", lw=0.7, linestyle=":")
        ax.legend(fontsize=8, facecolor=AX_BG, labelcolor="black", edgecolor=SPINE_C, ncol=2, loc="upper left")
        style_axes(ax, title=f"{noise_type} Noise", xlabel="Noise Intensity sigma", ylabel="Activation Probability")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(OUTPUT_DIR, "step11_panel_activation_curves.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=FIG_BG)
    plt.close()
    print(f"[fig]  {out_path}")


def plot_mi(results: dict) -> None:
    fig, axes = plt.subplots(len(NOISE_TYPES), 1, figsize=(12, 4.5 * len(NOISE_TYPES)))
    fig.patch.set_facecolor(FIG_BG)
    fig.suptitle(
        "Mutual Information vs Noise Intensity\nPanel Strategies vs Individual Biomarkers",
        color="black",
        fontsize=13,
        fontweight="bold",
    )

    for idx, noise_type in enumerate(NOISE_TYPES):
        ax = axes[idx]
        noise_results = results[noise_type]

        for biomarker in BIOMARKERS:
            ax.plot(
                noise_results["single"][biomarker]["sigma"],
                noise_results["single"][biomarker]["mi_mean"],
                color=BM_COLORS[biomarker],
                lw=1.1,
                linestyle="--",
                alpha=0.7,
                label=f"Single {biomarker}",
            )

        for strategy, color in PANEL_COLORS.items():
            mi_curve = noise_results["panel"][strategy]["mi_mean"]
            ax.plot(SIGMA_GRID, mi_curve, color=color, lw=2.1, label=strategy)
            sigma_star = find_sigma_star(SIGMA_GRID, mi_curve)
            mi_star = float(np.interp(sigma_star, SIGMA_GRID, mi_curve))
            ax.scatter([sigma_star], [mi_star], color=color, s=55, zorder=5, edgecolors="black", linewidths=0.7)

        ax.legend(fontsize=8, facecolor=AX_BG, labelcolor="black", edgecolor=SPINE_C, ncol=2, loc="upper right")
        style_axes(ax, title=f"{noise_type} Noise", xlabel="Noise Intensity sigma", ylabel="Mutual Information (bits)")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(OUTPUT_DIR, "step11_panel_mi_curves.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=FIG_BG)
    plt.close()
    print(f"[fig]  {out_path}")


def plot_mi_heatmap(summary: pd.DataFrame) -> None:
    method_order = [f"Single — {bm}" for bm in BIOMARKERS] + list(PANEL_COLORS.keys())
    pivot = summary.pivot_table(values="MI(sigma*)", index="Method", columns="Noise Type").reindex(index=method_order, columns=NOISE_TYPES)
    data = pivot.to_numpy(dtype=float)

    cmap = LinearSegmentedColormap.from_list(
        "panel_mi",
        ["#0d1117", "#1f3a5f", "#2e6da4", "#58a6ff", "#ffffff"],
    )

    fig, ax = plt.subplots(figsize=(7, 9))
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(AX_BG)
    image = ax.imshow(data, cmap=cmap, aspect="auto", vmin=float(data.min()), vmax=float(data.max()))

    for i in range(data.shape[0]):
        is_panel = method_order[i] in PANEL_COLORS
        for j in range(data.shape[1]):
            value = data[i, j]
            text_color = "white" if value > data.min() + 0.6 * (data.max() - data.min()) else "black"
            weight = "bold" if is_panel else "normal"
            ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=9.5, fontweight=weight, color=text_color)

    ax.axhline(len(BIOMARKERS) - 0.5, color="#b38600", lw=1.5, linestyle="--")
    ax.text(len(NOISE_TYPES) - 0.45, len(BIOMARKERS) - 0.55, "▼ Panels", color="#b38600", fontsize=8, va="bottom", ha="right")

    ax.set_xticks(range(len(NOISE_TYPES)))
    ax.set_xticklabels(NOISE_TYPES, color=TICK_C, fontsize=10)
    ax.set_yticks(range(len(method_order)))
    ax.set_yticklabels(method_order, color=TICK_C, fontsize=9)
    for spine in ax.spines.values():
        spine.set_color(SPINE_C)
    ax.tick_params(colors=TICK_C)

    cbar = fig.colorbar(image, ax=ax, pad=0.02, fraction=0.025)
    cbar.set_label("MI(sigma*)", color=TICK_C, fontsize=9)
    cbar.ax.yaxis.set_tick_params(color=TICK_C, labelcolor=TICK_C)
    cbar.outline.set_edgecolor(SPINE_C)

    ax.set_title("Peak MI — Panel vs Single Biomarker\nacross Noise Types", color="black", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Noise Type", color=TICK_C, fontsize=10)
    ax.set_ylabel("Method", color=TICK_C, fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "step11_panel_mi_heatmap.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=FIG_BG)
    plt.close()
    print(f"[fig]  {out_path}")


def plot_sr_gain_bar(summary: pd.DataFrame) -> None:
    panel_summary = summary[summary["Type"] == "Panel"].copy()
    fig, axes = plt.subplots(1, len(NOISE_TYPES), figsize=(15, 5.5), sharey=True)
    fig.patch.set_facecolor(FIG_BG)
    fig.suptitle(
        "SR Gain Relative to Best Single Biomarker\n(delta MI at each strategy's sigma*)",
        color="black",
        fontsize=13,
        fontweight="bold",
    )

    for idx, noise_type in enumerate(NOISE_TYPES):
        ax = axes[idx]
        sub = panel_summary[panel_summary["Noise Type"] == noise_type]
        methods = list(PANEL_COLORS.keys())
        values = [float(sub[sub["Method"] == method]["delta MI vs best single"].iloc[0]) for method in methods]
        colors = [PANEL_COLORS[method] for method in methods]
        x = np.arange(len(methods))
        bars = ax.bar(x, values, color=colors, alpha=0.88, edgecolor=SPINE_C, linewidth=0.6, width=0.65)
        ax.axhline(0.0, color="#b38600", lw=1.2, linestyle="--")

        for bar, value in zip(bars, values):
            y = bar.get_height()
            offset = 0.002 if y >= 0 else -0.006
            ax.text(bar.get_x() + bar.get_width() / 2.0, y + offset, f"{value:+.4f}", ha="center", va="bottom" if y >= 0 else "top", fontsize=7.5, color="black", rotation=0)

        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=8, color=TICK_C)
        style_axes(ax, title=f"{noise_type} Noise", ylabel="delta MI vs best single" if idx == 0 else "")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = os.path.join(OUTPUT_DIR, "step11_panel_sr_gain_bar.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=FIG_BG)
    plt.close()
    print(f"[fig]  {out_path}")


def print_summary(summary: pd.DataFrame) -> None:
    sep = "-" * 108
    print(f"\n{sep}")
    print("  STEP 11 — MULTI-BIOMARKER PANEL SUMMARY")
    print(sep)
    header = (
        f"{'Noise':>9}  {'Method':<22}  {'sigma*':>8}  "
        f"{'MI(sigma*)':>10}  {'Act P*':>8}  {'delta MI':>10}  {'delta P':>9}"
    )
    print(header)
    print(sep)
    for noise_type in NOISE_TYPES:
        sub = summary[summary["Noise Type"] == noise_type]
        for _, row in sub.iterrows():
            flag = "< PANEL" if row["Type"] == "Panel" else ""
            print(
                f"{noise_type:>9}  {row['Method']:<22}  {row['sigma* (MI max)']:>8.3f}  "
                f"{row['MI(sigma*)']:>10.4f}  {row['Act. Prob(sigma*)']:>8.3f}  "
                f"{row['delta MI vs best single']:>+10.4f}  {row['delta Act vs best single']:>+9.4f}  {flag}"
            )
    print(sep)

    print("\n  Best panel strategy per noise type (highest delta MI)")
    print("  " + "-" * 60)
    for noise_type in NOISE_TYPES:
        sub = summary[(summary["Noise Type"] == noise_type) & (summary["Type"] == "Panel")]
        best = sub.loc[sub["delta MI vs best single"].idxmax()]
        print(
            f"  {noise_type:>9}  ->  {best['Method']:<20}  "
            f"delta MI = {best['delta MI vs best single']:+.4f}"
        )


def run_step11_multi_biomarker_panel() -> pd.DataFrame:
    """Run the complete step-11 workflow and return the summary table."""
    print("=" * 62)
    print("  STEP 11 — Multi-Biomarker Panel Analysis")
    print("=" * 62)

    print("\n[step] Loading inputs ...")
    signal_matrix = load_biomarker_matrix()
    thresholds = load_thresholds()
    single_df = load_single_curves()
    weights_by_noise = load_step10_weights()

    print("\n[step] Running panel analysis ...")
    single_results = build_single_results(single_df)
    results, summary = run_panel_analysis(signal_matrix, thresholds, single_results, weights_by_noise)

    print("\n[step] Saving tables ...")
    save_tables(summary)

    print("\n[step] Generating figures ...")
    plot_activation(results)
    plot_mi(results)
    plot_mi_heatmap(summary)
    plot_sr_gain_bar(summary)

    print_summary(summary)

    print("=" * 62)
    print(f"  Step 11 complete. Outputs in {OUTPUT_DIR}/")
    print("=" * 62)
    return summary


if __name__ == "__main__":
    run_step11_multi_biomarker_panel()
