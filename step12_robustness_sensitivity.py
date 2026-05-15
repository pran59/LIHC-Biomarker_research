"""
Step 12 — Robustness / Sensitivity Analysis
===========================================
Tests whether the SR-peak conclusions from steps 10-11 remain qualitatively
stable when three modelling parameters are varied around the current pipeline
reference configuration:

  P1 — Hill coefficient          n  in {1, 2, 3, 4}
  P2 — Activation threshold      K  at {50th, 75th, 90th} percentile
  P3 — OU correlation time       tau in {0.1, 0.5, 1.0, 2.0, 5.0}

This implementation is intentionally aligned with the existing project state:
  - It uses the real matched early-stage biomarker vectors from step 3.
  - It keeps step 12 label-free and does not introduce ROC/AUC early.
  - It reuses the shared Gaussian / OU / Levy generators from noise_utils.py.
  - It keeps the same MI / activation definitions used in the SR pipeline.

For every parameter x level x (biomarker, noise type) combination the script
records:
  - sigma*          : noise level that maximises mutual information
  - MI(sigma*)      : mutual information at the peak
  - delta MI        : MI(sigma*) - MI(sigma=0)
  - Act(sigma*)     : activation probability at the peak
  - delta Act       : Act(sigma*) - Act(sigma=0)

Outputs
-------
  output/step12_sensitivity_full.csv
  output/step12_sensitivity_summary.csv
  output/step12_sensitivity_summary.tex
  output/step12_tornado.png
  output/step12_sigma_star_sensitivity.png
  output/step12_delta_mi_sensitivity.png
  output/step12_delta_act_sensitivity.png
  output/step12_sr_hill_sweep.png
  output/step12_sr_percentile_sweep.png
  output/step12_sr_tau_sweep.png

Inputs
------
  output/tcga_lihc_afp_statistics.json
  output/gene_activation_threshold_K.json
"""

from __future__ import annotations

import hashlib
import json
import os
import warnings

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "output", ".cache")
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(CACHE_DIR, "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", CACHE_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from sklearn.metrics import mutual_info_score

from noise_utils import (
    DEFAULT_LEVY_ALPHA,
    DEFAULT_LEVY_BETA,
    DEFAULT_OU_STEP,
    DEFAULT_OU_TAU,
    generate_noise,
)
from step10_sr_peak_characterization import find_sigma_star, smooth_curve

warnings.filterwarnings("ignore")

OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BIOMARKERS = ["AFP", "GPC3", "DKK1", "MDK"]
NOISE_TYPES = ["Gaussian", "OU", "Levy"]
NOISE_TO_INTERNAL = {"Gaussian": "gaussian", "OU": "ou", "Levy": "levy"}

HILL_LEVELS = [1, 2, 3, 4]
PERCENTILE_LEVELS = [50, 75, 90]
OU_TAU_LEVELS = [0.1, 0.5, 1.0, 2.0, 5.0]

SIGMA_GRID = np.linspace(0.0, 6.0, 40)
NUM_TRIALS = 80
MI_BINS = 12
DECISION_THRESHOLD = 0.5

REF_HILL_N = 4.0
REF_PERCENTILE = 75
REF_OU_TAU = float(DEFAULT_OU_TAU)

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
NT_COLORS = {
    "Gaussian": "#1f77b4",
    "OU": "#d62728",
    "Levy": "#9467bd",
}
HILL_COLORS = {1: "#d62728", 2: "#1f77b4", 3: "#2ca02c", 4: "#ff7f0e"}
PERCENTILE_COLORS = {50: "#ff7f0e", 75: "#1f77b4", 90: "#9467bd"}
TAU_COLORS = {0.1: "#d62728", 0.5: "#e377c2", 1.0: "#1f77b4", 2.0: "#2ca02c", 5.0: "#9467bd"}

CMAP_SIGMA = LinearSegmentedColormap.from_list(
    "step12_sigma", ["#0d2137", "#1a4f6e", "#2980b9", "#00d4ff", "#a8ff78"]
)
CMAP_GAIN = LinearSegmentedColormap.from_list(
    "step12_gain", ["#5c0a0a", "#c0392b", "#e8c56a", "#a8ff78"]
)
CMAP_ACT = LinearSegmentedColormap.from_list(
    "step12_act", ["#1a0533", "#4a1273", "#2980b9", "#27ae60", "#a8ff78"]
)


def stable_seed(*parts: object) -> int:
    """Deterministic seed so sweep levels reuse the same underlying noise draws."""
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


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
            "Rerun step 3 so the sensitivity sweep uses aligned patient vectors."
        )

    matrix = np.column_stack(columns)
    print(f"[load] Biomarker matrix loaded from {path} with shape {matrix.shape}")
    return matrix


def load_reference_thresholds() -> np.ndarray:
    """Load the existing step 4 thresholds so the P75 sweep can be validated."""
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
    print(f"[load] Reference thresholds loaded from {path}")
    return thresholds


def compute_percentile_thresholds(signal_matrix: np.ndarray) -> dict[int, np.ndarray]:
    """Recompute threshold vectors directly from the real matched biomarker values."""
    thresholds = {
        percentile: np.percentile(signal_matrix, percentile, axis=0).astype(float)
        for percentile in PERCENTILE_LEVELS
    }
    for percentile, values in thresholds.items():
        joined = ", ".join(f"{bm}={val:.4f}" for bm, val in zip(BIOMARKERS, values))
        print(f"[prep] Percentile P{percentile}: {joined}")
    return thresholds


def validate_reference_thresholds(reference: np.ndarray, computed_p75: np.ndarray) -> None:
    """Sanity-check that the live step 4 thresholds match the P75 values from step 3."""
    diff = np.abs(reference - computed_p75)
    if np.any(diff > 1e-3):
        print(
            "[warn] Step 4 thresholds differ from recomputed P75 values by more than 1e-3. "
            "Proceeding with the live P75 values computed from step 3 for the sweep."
        )
    else:
        print("[check] Step 4 thresholds match recomputed P75 values.")


def hill_output(x_in: np.ndarray, threshold_k: float, hill_n: float) -> np.ndarray:
    """Hill activation with the same absolute-value guard used in the SR scripts."""
    x_abs = np.abs(x_in)
    xn = x_abs**hill_n
    kn = threshold_k**hill_n
    return xn / (kn + xn)


def discretize(values: np.ndarray, bins: int = MI_BINS) -> np.ndarray:
    """Match the histogram-based discretisation used in the SR pipeline."""
    _, edges = np.histogram(values, bins=bins)
    return np.digitize(values, edges[:-1])


def mutual_information_bits(x_noisy: np.ndarray, decisions: np.ndarray) -> float:
    """MI(noisy input, binary circuit output) in bits."""
    if np.all(decisions == decisions[0]):
        return 0.0
    x_disc = discretize(x_noisy)
    return mutual_info_score(x_disc, decisions.astype(int)) / np.log(2.0)


def compute_curves(
    signal: np.ndarray,
    threshold_k: float,
    noise_type: str,
    hill_n: float,
    ou_tau: float,
    biomarker: str,
) -> dict[str, np.ndarray]:
    """Monte Carlo SR sweep for one biomarker under one configuration."""
    act_mean = []
    act_std = []
    mi_mean = []
    mi_std = []

    internal_noise = NOISE_TO_INTERNAL[noise_type]

    for sigma_idx, sigma in enumerate(SIGMA_GRID):
        act_trials = []
        mi_trials = []
        for trial in range(NUM_TRIALS):
            seed = stable_seed("step12", biomarker, noise_type, sigma_idx, trial)
            eta = generate_noise(
                internal_noise,
                len(signal),
                float(sigma),
                tau=ou_tau,
                h=DEFAULT_OU_STEP,
                levy_alpha=DEFAULT_LEVY_ALPHA,
                levy_beta=DEFAULT_LEVY_BETA,
                seed=seed,
            )
            x_noisy = signal + eta
            y = hill_output(x_noisy, threshold_k, hill_n)
            decisions = (y > DECISION_THRESHOLD).astype(int)

            act_trials.append(float(decisions.mean()))
            mi_trials.append(mutual_information_bits(x_noisy, decisions))

        act_arr = np.array(act_trials, dtype=float)
        mi_arr = np.array(mi_trials, dtype=float)
        act_mean.append(float(act_arr.mean()))
        act_std.append(float(act_arr.std()))
        mi_mean.append(float(mi_arr.mean()))
        mi_std.append(float(mi_arr.std()))

    act_curve = np.clip(smooth_curve(SIGMA_GRID, np.array(act_mean, dtype=float)), 0.0, 1.0)
    mi_curve = np.clip(smooth_curve(SIGMA_GRID, np.array(mi_mean, dtype=float)), 0.0, 1.0)

    return {
        "act_mean": act_curve,
        "act_std": np.array(act_std, dtype=float),
        "mi_mean": mi_curve,
        "mi_std": np.array(mi_std, dtype=float),
    }


def peak_metrics(sigmas: np.ndarray, act_curve: np.ndarray, mi_curve: np.ndarray) -> dict[str, float]:
    """Extract sigma* and derived peak metrics from one SR curve."""
    sigma_star = find_sigma_star(sigmas, mi_curve)
    mi_at_star = float(np.interp(sigma_star, sigmas, mi_curve))
    act_at_star = float(np.interp(sigma_star, sigmas, act_curve))
    mi_baseline = float(mi_curve[0])
    act_baseline = float(act_curve[0])
    return {
        "sigma_star": float(sigma_star),
        "mi_peak": mi_at_star,
        "delta_mi": mi_at_star - mi_baseline,
        "act_at_star": act_at_star,
        "delta_act": act_at_star - act_baseline,
    }


def sweep_hill(signal_matrix: np.ndarray, thresholds_by_percentile: dict[int, np.ndarray]) -> pd.DataFrame:
    """Vary the Hill coefficient n while fixing P75 and tau_ref."""
    rows = []
    total = len(HILL_LEVELS) * len(NOISE_TYPES) * len(BIOMARKERS)
    done = 0
    thresholds = thresholds_by_percentile[REF_PERCENTILE]

    for hill_n in HILL_LEVELS:
        for noise_type in NOISE_TYPES:
            for biomarker_idx, biomarker in enumerate(BIOMARKERS):
                curves = compute_curves(
                    signal_matrix[:, biomarker_idx],
                    thresholds[biomarker_idx],
                    noise_type,
                    float(hill_n),
                    REF_OU_TAU,
                    biomarker,
                )
                metrics = peak_metrics(SIGMA_GRID, curves["act_mean"], curves["mi_mean"])
                rows.append(
                    {
                        "parameter": "Hill-n",
                        "level_label": f"n={hill_n}",
                        "level_value": float(hill_n),
                        "noise_type": noise_type,
                        "biomarker": biomarker,
                        **metrics,
                        "_act_curve": curves["act_mean"],
                        "_mi_curve": curves["mi_mean"],
                    }
                )
                done += 1
                print(
                    f"  [Hill-n] {done:>3}/{total}  n={hill_n}  {noise_type:8s}  "
                    f"{biomarker:4s}  sigma*={metrics['sigma_star']:.3f}  "
                    f"delta MI={metrics['delta_mi']:+.4f}",
                    end="\r",
                )
    print()
    return pd.DataFrame(rows)


def sweep_percentile(signal_matrix: np.ndarray, thresholds_by_percentile: dict[int, np.ndarray]) -> pd.DataFrame:
    """Vary the percentile-defined threshold while fixing n_ref and tau_ref."""
    rows = []
    total = len(PERCENTILE_LEVELS) * len(NOISE_TYPES) * len(BIOMARKERS)
    done = 0

    for percentile in PERCENTILE_LEVELS:
        thresholds = thresholds_by_percentile[percentile]
        for noise_type in NOISE_TYPES:
            for biomarker_idx, biomarker in enumerate(BIOMARKERS):
                curves = compute_curves(
                    signal_matrix[:, biomarker_idx],
                    thresholds[biomarker_idx],
                    noise_type,
                    REF_HILL_N,
                    REF_OU_TAU,
                    biomarker,
                )
                metrics = peak_metrics(SIGMA_GRID, curves["act_mean"], curves["mi_mean"])
                rows.append(
                    {
                        "parameter": "Percentile",
                        "level_label": f"P{percentile}",
                        "level_value": float(percentile),
                        "noise_type": noise_type,
                        "biomarker": biomarker,
                        **metrics,
                        "_act_curve": curves["act_mean"],
                        "_mi_curve": curves["mi_mean"],
                    }
                )
                done += 1
                print(
                    f"  [Percentile] {done:>3}/{total}  P{percentile}  {noise_type:8s}  "
                    f"{biomarker:4s}  sigma*={metrics['sigma_star']:.3f}  "
                    f"delta MI={metrics['delta_mi']:+.4f}",
                    end="\r",
                )
    print()
    return pd.DataFrame(rows)


def sweep_tau(signal_matrix: np.ndarray, thresholds_by_percentile: dict[int, np.ndarray]) -> pd.DataFrame:
    """Vary OU correlation time only for OU noise while fixing n_ref and P75."""
    rows = []
    total = len(OU_TAU_LEVELS) * len(BIOMARKERS)
    done = 0
    thresholds = thresholds_by_percentile[REF_PERCENTILE]

    for tau in OU_TAU_LEVELS:
        for biomarker_idx, biomarker in enumerate(BIOMARKERS):
            curves = compute_curves(
                signal_matrix[:, biomarker_idx],
                thresholds[biomarker_idx],
                "OU",
                REF_HILL_N,
                float(tau),
                biomarker,
            )
            metrics = peak_metrics(SIGMA_GRID, curves["act_mean"], curves["mi_mean"])
            rows.append(
                {
                    "parameter": "OU-tau",
                    "level_label": f"tau={tau:g}",
                    "level_value": float(tau),
                    "noise_type": "OU",
                    "biomarker": biomarker,
                    **metrics,
                    "_act_curve": curves["act_mean"],
                    "_mi_curve": curves["mi_mean"],
                }
            )
            done += 1
            print(
                f"  [OU-tau] {done:>3}/{total}  tau={tau:<3g}  {biomarker:4s}  "
                f"sigma*={metrics['sigma_star']:.3f}  delta MI={metrics['delta_mi']:+.4f}",
                end="\r",
            )
    print()
    return pd.DataFrame(rows)


def build_sensitivity_summary(*frames: pd.DataFrame) -> pd.DataFrame:
    """Summarise min / max / range across sweep levels for each metric."""
    reference_levels = {
        "Hill-n": REF_HILL_N,
        "Percentile": float(REF_PERCENTILE),
        "OU-tau": REF_OU_TAU,
    }
    metrics = [
        ("sigma_star", "sigma*"),
        ("mi_peak", "MI(sigma*)"),
        ("delta_mi", "delta MI"),
        ("act_at_star", "Act. Prob(sigma*)"),
        ("delta_act", "delta Act"),
    ]

    rows = []
    for frame in frames:
        parameter = str(frame["parameter"].iloc[0])
        for noise_type in sorted(frame["noise_type"].unique()):
            for biomarker in BIOMARKERS:
                sub = frame[(frame["noise_type"] == noise_type) & (frame["biomarker"] == biomarker)]
                if sub.empty:
                    continue
                for column, label in metrics:
                    values = sub[column].to_numpy(dtype=float)
                    ref_rows = sub[np.isclose(sub["level_value"], reference_levels[parameter])]
                    ref_value = float(ref_rows[column].iloc[0]) if not ref_rows.empty else np.nan
                    rows.append(
                        {
                            "parameter": parameter,
                            "noise_type": noise_type,
                            "biomarker": biomarker,
                            "metric": label,
                            "min_value": float(values.min()),
                            "max_value": float(values.max()),
                            "range_value": float(values.max() - values.min()),
                            "reference_value": ref_value,
                        }
                    )
    return pd.DataFrame(rows)


def save_tables(full_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    """Persist the full sensitivity sweep and a condensed LaTeX summary."""
    full_cols = [col for col in full_df.columns if not col.startswith("_")]
    full_path = os.path.join(OUTPUT_DIR, "step12_sensitivity_full.csv")
    summary_path = os.path.join(OUTPUT_DIR, "step12_sensitivity_summary.csv")

    full_df[full_cols].to_csv(full_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    tex_path = os.path.join(OUTPUT_DIR, "step12_sensitivity_summary.tex")
    tex_metrics = {"sigma*", "delta MI", "delta Act"}
    tex_df = summary_df[summary_df["metric"].isin(tex_metrics)].copy()

    with open(tex_path, "w") as handle:
        handle.write("% Step 12 — Sensitivity Analysis Summary Table\n\n")
        handle.write("\\begin{table}[htbp]\\centering\n")
        handle.write(
            "\\caption{Sensitivity ranges for the key robustness metrics under "
            "Hill-coefficient, threshold-percentile, and OU-correlation sweeps. "
            "Range = max - min across levels. Reference values correspond to the "
            "current pipeline configuration $n=4$, $P75$, and $\\tau=5.0$.}\n"
        )
        handle.write("\\label{tab:step12_sensitivity}\n")
        handle.write("\\resizebox{\\textwidth}{!}{%\n")
        handle.write("\\begin{tabular}{lllcccccc}\\toprule\n")
        handle.write(
            "Parameter & Noise & Biomarker & Metric & Min & Max & Range & Ref \\\\\n\\midrule\n"
        )

        previous = ("", "", "")
        for _, row in tex_df.iterrows():
            param_str = row["parameter"] if row["parameter"] != previous[0] else ""
            noise_str = row["noise_type"] if (row["parameter"], row["noise_type"]) != previous[:2] else ""
            biomarker_str = (
                row["biomarker"]
                if (row["parameter"], row["noise_type"], row["biomarker"]) != previous
                else ""
            )
            previous = (row["parameter"], row["noise_type"], row["biomarker"])
            handle.write(
                f"{param_str} & {noise_str} & {biomarker_str} & {row['metric']} & "
                f"{row['min_value']:.4f} & {row['max_value']:.4f} & "
                f"\\textbf{{{row['range_value']:.4f}}} & {row['reference_value']:.4f} \\\\\n"
            )

        handle.write("\\bottomrule\n\\end{tabular}}\\end{table}\n")

    print(f"[save] {full_path}")
    print(f"[save] {summary_path}")
    print(f"[save] {tex_path}")


def style_axes(ax, title: str = "", xlabel: str = "", ylabel: str = "", fontsize: int = 9) -> None:
    """Shared dark-theme styling."""
    ax.set_facecolor(AX_BG)
    if title:
        ax.set_title(title, color="black", fontsize=fontsize, fontweight="bold", pad=5)
    if xlabel:
        ax.set_xlabel(xlabel, color=TICK_C, fontsize=max(fontsize - 1, 7))
    if ylabel:
        ax.set_ylabel(ylabel, color=TICK_C, fontsize=max(fontsize - 1, 7))
    for spine in ax.spines.values():
        spine.set_color(SPINE_C)
    ax.tick_params(colors=TICK_C, labelsize=max(fontsize - 2, 7))
    ax.grid(True, color=GRID_C, lw=0.5, linestyle="--", alpha=0.6)


def plot_tornado(summary_df: pd.DataFrame) -> None:
    """Show the total sensitivity range for sigma*, delta MI, and delta Act."""
    metrics = [("sigma*", "sigma* range"), ("delta MI", "delta MI range"), ("delta Act", "delta Act range")]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor(FIG_BG)
    fig.suptitle("Step 12 — Total Sensitivity Range by Parameter Family", color="black", fontsize=13, fontweight="bold")

    for ax, (metric, xlabel) in zip(axes, metrics):
        sub = summary_df[summary_df["metric"] == metric].copy()
        grouped = (
            sub.groupby(["parameter", "biomarker"])["range_value"]
            .max()
            .unstack(fill_value=0.0)
            .reindex(columns=BIOMARKERS)
        )
        grouped["total"] = grouped.sum(axis=1)
        grouped = grouped.sort_values("total", ascending=True)

        y_positions = np.arange(len(grouped))
        left = np.zeros(len(grouped), dtype=float)
        for biomarker in BIOMARKERS:
            values = grouped[biomarker].to_numpy(dtype=float)
            ax.barh(
                y_positions,
                values,
                left=left,
                height=0.6,
                color=BM_COLORS[biomarker],
                alpha=0.85,
                edgecolor=SPINE_C,
                linewidth=0.5,
            )
            left += values

        totals = grouped["total"].to_numpy(dtype=float)
        for idx, total in enumerate(totals):
            ax.text(total + max(totals.max() * 0.02, 0.002), idx, f"{total:.3f}", color="black", fontsize=8, va="center")

        ax.set_yticks(y_positions)
        ax.set_yticklabels(grouped.index.tolist(), color=TICK_C, fontsize=9, fontweight="bold")
        style_axes(ax, title=metric, xlabel=xlabel)
        if ax is axes[0]:
            handles = [plt.Rectangle((0, 0), 1, 1, color=BM_COLORS[bm], alpha=0.85) for bm in BIOMARKERS]
            ax.legend(
                handles,
                BIOMARKERS,
                fontsize=7.5,
                facecolor=AX_BG,
                labelcolor="black",
                edgecolor=SPINE_C,
                loc="lower right",
            )

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = os.path.join(OUTPUT_DIR, "step12_tornado.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=FIG_BG)
    plt.close()
    print(f"[fig]  {out_path}")


def plot_metric_grid(
    hill_df: pd.DataFrame,
    percentile_df: pd.DataFrame,
    tau_df: pd.DataFrame,
    *,
    metric_col: str,
    title: str,
    cbar_label: str,
    fmt: str,
    cmap,
    out_path: str,
) -> None:
    """Composite grid figure with rows for parameter families and columns for noise types."""
    datasets = [
        ("Hill coefficient n", hill_df),
        ("Threshold percentile", percentile_df),
        ("OU correlation tau", tau_df),
    ]
    all_values = []
    for _, frame in datasets:
        all_values.extend(frame[metric_col].to_numpy(dtype=float))
    all_values_arr = np.array(all_values, dtype=float)
    vmin = float(all_values_arr.min())
    vmax = float(all_values_arr.max())
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6

    fig, axes = plt.subplots(3, 3, figsize=(15.5, 11.5))
    fig.patch.set_facecolor("white")
    fig.suptitle(title, color="black", fontsize=13, fontweight="bold")

    for row_idx, (row_label, frame) in enumerate(datasets):
        for col_idx, noise_type in enumerate(NOISE_TYPES):
            ax = axes[row_idx][col_idx]
            ax.set_facecolor("white")
            for spine in ax.spines.values():
                spine.set_color(SPINE_C)

            sub = frame[frame["noise_type"] == noise_type]
            if sub.empty:
                ax.text(0.5, 0.5, "not applicable", color=TICK_C, fontsize=9, ha="center", va="center")
                ax.set_xticks([])
                ax.set_yticks([])
                if row_idx == 0:
                    ax.set_title(f"{noise_type} noise", color="black", fontsize=10, fontweight="bold", pad=6)
                continue

            level_order = sub.sort_values("level_value")["level_label"].drop_duplicates().tolist()
            pivot = (
                sub.pivot_table(values=metric_col, index="level_label", columns="biomarker", aggfunc="mean")
                .reindex(index=level_order, columns=BIOMARKERS)
            )
            data = pivot.to_numpy(dtype=float)
            ax.imshow(data, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    norm_value = (data[i, j] - vmin) / max(vmax - vmin, 1e-9)
                    text_color = "white" if norm_value > 0.55 else "black"
                    ax.text(
                        j,
                        i,
                        format(data[i, j], fmt),
                        ha="center",
                        va="center",
                        fontsize=8.5,
                        color=text_color,
                        fontweight="bold",
                    )

            ax.set_xticks(range(len(BIOMARKERS)))
            ax.set_xticklabels(BIOMARKERS, color=TICK_C, fontsize=9)
            ax.set_yticks(range(len(level_order)))
            ax.set_yticklabels(level_order, color=TICK_C, fontsize=8.5)
            ax.tick_params(colors=TICK_C)
            if row_idx == 0:
                ax.set_title(f"{noise_type} noise", color="black", fontsize=10, fontweight="bold", pad=6)
            if col_idx == 0:
                ax.set_ylabel(row_label, color=TICK_C, fontsize=9)
            ax.set_xlabel("Biomarker", color=TICK_C, fontsize=8.5)

    cbar_ax = fig.add_axes([0.92, 0.12, 0.015, 0.76])
    sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(cbar_label, color=TICK_C, fontsize=9)
    cbar.ax.yaxis.set_tick_params(color=TICK_C, labelcolor=TICK_C)
    cbar.outline.set_edgecolor(SPINE_C)

    plt.tight_layout(rect=[0, 0, 0.91, 0.95])
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=FIG_BG)
    plt.close()
    print(f"[fig]  {out_path}")


def sr_overlay(
    frame: pd.DataFrame,
    *,
    color_map: dict[float, str],
    reference_level: float,
    title_prefix: str,
    label_prefix: str,
    out_path: str,
    noise_types: list[str] | None = None,
) -> None:
    """Overlay MI curves for one parameter sweep."""
    if noise_types is None:
        noise_types = [noise for noise in NOISE_TYPES if noise in frame["noise_type"].unique()]
    levels = frame.sort_values("level_value")["level_value"].drop_duplicates().tolist()

    n_rows = len(noise_types)
    n_cols = len(BIOMARKERS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.6 * n_rows), sharex=True)
    if n_rows == 1:
        axes = np.array([axes])
    fig.patch.set_facecolor(FIG_BG)
    fig.suptitle(f"Step 12 — {title_prefix} Sweep (MI vs sigma)", color="black", fontsize=13, fontweight="bold")

    for row_idx, noise_type in enumerate(noise_types):
        for col_idx, biomarker in enumerate(BIOMARKERS):
            ax = axes[row_idx][col_idx]
            for level in levels:
                sub = frame[
                    (frame["noise_type"] == noise_type)
                    & (frame["biomarker"] == biomarker)
                    & np.isclose(frame["level_value"], level)
                ]
                if sub.empty:
                    continue
                row = sub.iloc[0]
                mi_curve = np.array(row["_mi_curve"], dtype=float)
                sigma_star = float(row["sigma_star"])
                is_reference = np.isclose(level, reference_level)
                color = color_map.get(level, "#888888")
                ax.plot(
                    SIGMA_GRID,
                    mi_curve,
                    color=color,
                    lw=2.4 if is_reference else 1.2,
                    linestyle="-" if is_reference else "--",
                    alpha=0.95,
                    label=f"{label_prefix}={level:g}",
                )
                ax.axvline(sigma_star, color=color, lw=0.7, linestyle=":", alpha=0.55)

            if row_idx == 0:
                ax.set_title(biomarker, color="black", fontsize=10, fontweight="bold", pad=5)
            style_axes(
                ax,
                xlabel="Noise intensity sigma" if row_idx == n_rows - 1 else "",
                ylabel=f"{noise_type}\nMI (bits)" if col_idx == 0 else "",
            )

    handles = [
        plt.Line2D(
            [0],
            [0],
            color=color_map.get(level, "#888888"),
            lw=2.4 if np.isclose(level, reference_level) else 1.2,
            linestyle="-" if np.isclose(level, reference_level) else "--",
            label=f"{label_prefix}={level:g}",
        )
        for level in levels
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(len(handles), 5),
        fontsize=8.5,
        facecolor=AX_BG,
        labelcolor="black",
        edgecolor=SPINE_C,
        bbox_to_anchor=(0.5, -0.01),
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=FIG_BG)
    plt.close()
    print(f"[fig]  {out_path}")


def plot_sr_overlays(hill_df: pd.DataFrame, percentile_df: pd.DataFrame, tau_df: pd.DataFrame) -> None:
    """Render the three curve-overlay figures described in the step plan."""
    sr_overlay(
        hill_df,
        color_map=HILL_COLORS,
        reference_level=REF_HILL_N,
        title_prefix="Hill coefficient n",
        label_prefix="n",
        out_path=os.path.join(OUTPUT_DIR, "step12_sr_hill_sweep.png"),
    )
    sr_overlay(
        percentile_df,
        color_map=PERCENTILE_COLORS,
        reference_level=float(REF_PERCENTILE),
        title_prefix="Threshold percentile",
        label_prefix="P",
        out_path=os.path.join(OUTPUT_DIR, "step12_sr_percentile_sweep.png"),
    )
    sr_overlay(
        tau_df,
        color_map=TAU_COLORS,
        reference_level=REF_OU_TAU,
        title_prefix="OU correlation time tau",
        label_prefix="tau",
        out_path=os.path.join(OUTPUT_DIR, "step12_sr_tau_sweep.png"),
        noise_types=["OU"],
    )


def print_full_summary(*frames: pd.DataFrame) -> None:
    """Console summary of the full sweep results."""
    separator = "─" * 98
    for frame in frames:
        parameter = str(frame["parameter"].iloc[0])
        print(f"\n{separator}")
        print(f"  {parameter} sweep")
        print(separator)
        print(
            f"  {'Level':>8}  {'Noise':>9}  {'Biomarker':>8}  "
            f"{'sigma*':>7}  {'MI*':>8}  {'delta MI':>9}  {'delta Act':>10}"
        )
        print(separator)
        for _, row in frame.sort_values(["level_value", "noise_type", "biomarker"]).iterrows():
            print(
                f"  {row['level_label']:>8}  {row['noise_type']:>9}  {row['biomarker']:>8}  "
                f"{row['sigma_star']:>7.3f}  {row['mi_peak']:>8.4f}  "
                f"{row['delta_mi']:>+9.4f}  {row['delta_act']:>+10.4f}"
            )


def stability_verdict(summary_df: pd.DataFrame) -> None:
    """Qualitative readout of whether SR gains stay positive under each sweep."""
    eps = 1e-6

    print("\n" + "═" * 80)
    print("  QUALITATIVE STABILITY VERDICT")
    print("═" * 80)
    print(
        f"  {'Parameter':>12}  {'Metric':>10}  {'Max range':>10}  "
        f"{'Stable?':>10}  Interpretation"
    )
    print("─" * 80)

    verdict_metrics = [("sigma*", "sigma*"), ("delta MI", "delta MI"), ("delta Act", "delta Act")]
    for parameter in ["Hill-n", "Percentile", "OU-tau"]:
        parameter_df = summary_df[summary_df["parameter"] == parameter]
        for metric_key, metric_label in verdict_metrics:
            metric_df = parameter_df[parameter_df["metric"] == metric_key]
            if metric_df.empty:
                continue
            max_range = float(metric_df["range_value"].max())
            if max_range <= eps:
                stable = True
                if parameter == "Hill-n":
                    interpretation = "Binary SR metrics are invariant under the current 0.5 decision rule"
                else:
                    interpretation = "Metric is effectively unchanged across sweep levels"
            elif metric_key == "sigma*":
                stable = max_range < 2.0
                interpretation = "Peak location shifts modestly across levels" if stable else "Peak location is parameter-sensitive"
            else:
                stable = bool((metric_df["min_value"] >= -eps).all())
                interpretation = "SR gain remains positive across levels" if stable else "Some levels weaken the SR gain"
            print(
                f"  {parameter:>12}  {metric_label:>10}  {max_range:>10.4f}  "
                f"{('yes' if stable else 'watch'):>10}  {interpretation}"
            )
    print("═" * 80 + "\n")


def run_step12_robustness_sensitivity() -> pd.DataFrame:
    """Main entry point used by the runner and by direct execution."""
    print("=" * 66)
    print("  STEP 12 — Robustness / Sensitivity Analysis")
    print("=" * 66)

    print("\n[step] Loading matched biomarker vectors and reference thresholds ...")
    signal_matrix = load_biomarker_matrix()
    reference_thresholds = load_reference_thresholds()
    thresholds_by_percentile = compute_percentile_thresholds(signal_matrix)
    validate_reference_thresholds(reference_thresholds, thresholds_by_percentile[REF_PERCENTILE])

    print("\n[step] Sweep 1/3 — Hill coefficient n ...")
    hill_df = sweep_hill(signal_matrix, thresholds_by_percentile)

    print("\n[step] Sweep 2/3 — Threshold percentile ...")
    percentile_df = sweep_percentile(signal_matrix, thresholds_by_percentile)

    print("\n[step] Sweep 3/3 — OU correlation time tau ...")
    tau_df = sweep_tau(signal_matrix, thresholds_by_percentile)

    print("\n[step] Building summary tables ...")
    full_df = pd.concat([hill_df, percentile_df, tau_df], ignore_index=True)
    summary_df = build_sensitivity_summary(hill_df, percentile_df, tau_df)
    save_tables(full_df, summary_df)

    print("\n[step] Plotting tornado and sensitivity grids ...")
    plot_tornado(summary_df)
    plot_metric_grid(
        hill_df,
        percentile_df,
        tau_df,
        metric_col="sigma_star",
        title="Step 12 — sigma* sensitivity across parameter sweeps",
        cbar_label="sigma*",
        fmt=".3f",
        cmap=CMAP_SIGMA,
        out_path=os.path.join(OUTPUT_DIR, "step12_sigma_star_sensitivity.png"),
    )
    plot_metric_grid(
        hill_df,
        percentile_df,
        tau_df,
        metric_col="delta_mi",
        title="Step 12 — delta MI sensitivity across parameter sweeps",
        cbar_label="delta MI [bits]",
        fmt="+.4f",
        cmap=CMAP_GAIN,
        out_path=os.path.join(OUTPUT_DIR, "step12_delta_mi_sensitivity.png"),
    )
    plot_metric_grid(
        hill_df,
        percentile_df,
        tau_df,
        metric_col="delta_act",
        title="Step 12 — delta activation sensitivity across parameter sweeps",
        cbar_label="delta activation",
        fmt="+.4f",
        cmap=CMAP_ACT,
        out_path=os.path.join(OUTPUT_DIR, "step12_delta_act_sensitivity.png"),
    )

    print("\n[step] Plotting SR curve overlays ...")
    plot_sr_overlays(hill_df, percentile_df, tau_df)

    print_full_summary(hill_df, percentile_df, tau_df)
    stability_verdict(summary_df)

    print("=" * 66)
    print(f"  Step 12 complete. Outputs in {OUTPUT_DIR}/")
    print("=" * 66)
    return full_df


if __name__ == "__main__":
    run_step12_robustness_sensitivity()
