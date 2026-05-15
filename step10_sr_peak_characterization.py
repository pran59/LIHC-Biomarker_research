"""
Step 10 — Optimal Noise Identification (SR Peak Characterization)
==================================================================
Identifies the optimal noise intensity sigma* at which Mutual Information (MI)
is maximised for every (biomarker x noise-type) combination, computes
bootstrap confidence intervals on both sigma* and MI(sigma*), builds a
publication-ready summary table, and renders four figures:

  Fig 10-A  SR curves with sigma* markers   (all biomarkers x noise types)
  Fig 10-B  sigma* heatmap                  (biomarker x noise type)
  Fig 10-C  MI(sigma*) heatmap              (biomarker x noise type)
  Fig 10-D  SR Gain delta MI heatmap        (MI(sigma*) - MI(sigma~0))

Outputs
-------
  output/step10_sr_peak_summary.csv
  output/step10_sr_peak_summary.tex
  output/step10_sr_curves_annotated.png
  output/step10_sigma_star_heatmap.png
  output/step10_mi_peak_heatmap.png
  output/step10_sr_gain_heatmap.png

Inputs
------
  Prefers output/sr_full_results.csv produced by step8_9_enhanced_sr_analysis.py.
  This project currently stores:
      biomarker, noise_type, sigma, act_mean, act_std,
      mi_mean, mi_std, fisher_info, ...

  The loader transparently standardises those names onto:
      biomarker, noise_type, sigma, activation_prob,
      mutual_information, fisher_information

  If the file is absent, the script falls back to a synthetic SR dataset so it
  can still be run stand-alone for testing.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BIOMARKERS = ["AFP", "GPC3", "DKK1", "MDK"]
NOISE_TYPES = ["Gaussian", "OU", "Levy"]

NOISE_NAME_MAP = {
    "awgn": "Gaussian",
    "gaussian": "Gaussian",
    "ou": "OU",
    "levy": "Levy",
}

NT_COLORS = {"Gaussian": "#1f77b4", "OU": "#d62728", "Levy": "#9467bd"}
NT_MARKERS = {"Gaussian": "o", "OU": "s", "Levy": "^"}

BOOTSTRAP_N = 1000
CI_ALPHA = 0.95

FIG_BG = "white"
AX_BG = "white"
GRID_C = "#cccccc"
TICK_C = "#333333"
SPINE_C = "#999999"

TABLE_COLS = [
    "Biomarker",
    "Noise Type",
    "sigma*",
    "sigma* CI lower (95%)",
    "sigma* CI upper (95%)",
    "MI(sigma*)",
    "MI(sigma*) CI lower (95%)",
    "MI(sigma*) CI upper (95%)",
    "MI(sigma=0)",
    "SR Gain delta MI",
    "Act. Prob(sigma*)",
    "Act. Gain delta P",
    "Fisher Info(sigma*)",
]


def load_or_synthesise() -> pd.DataFrame:
    """
    Load the real sr_full_results.csv and standardise its schema.
    Fall back to a synthetic SR dataset if the file is absent.
    """
    path = os.path.join(OUTPUT_DIR, "sr_full_results.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        df = standardise_sr_dataframe(df)
        required = {
            "biomarker",
            "noise_type",
            "sigma",
            "activation_prob",
            "mutual_information",
            "fisher_information",
        }
        if required.issubset(df.columns):
            print(f"[load] Loaded {len(df)} rows from {path}")
            return df
        print(f"[warn] {path} exists but could not be standardised; synthesising instead.")

    print("[info] sr_full_results.csv not found -> generating synthetic SR data.")
    return synthesise_sr_data()


def standardise_sr_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Map project-specific column names and noise labels onto the step-10 schema."""
    rename_map = {
        "act_mean": "activation_prob",
        "act_std": "activation_prob_std",
        "mi_mean": "mutual_information",
        "mi_std": "mutual_information_std",
        "fisher_info": "fisher_information",
    }
    out = df.rename(columns=rename_map).copy()

    if "noise_type" in out.columns:
        out["noise_type"] = out["noise_type"].map(
            lambda val: NOISE_NAME_MAP.get(str(val).lower(), val)
        )

    return out


def hill(x: float, k: float = 1.0, n: float = 2.0) -> float:
    return x**n / (k**n + x**n)


def mi_from_probs(p_act: np.ndarray) -> np.ndarray:
    """Approximate MI (bits) assuming a binary output and uniform input ensemble."""
    p = np.clip(p_act, 1e-9, 1.0 - 1e-9)
    h_y = -p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p)
    h_y_given_x = 0.5 * h_y
    return np.clip(h_y - h_y_given_x, 0.0, None)


def fisher_from_prob_curve(sigmas: np.ndarray, p_act: np.ndarray) -> np.ndarray:
    """Numerical Fisher proxy: (dp/dsigma)^2 / (p(1-p))."""
    dp = np.gradient(p_act, sigmas)
    denom = np.clip(p_act * (1.0 - p_act), 1e-9, None)
    return dp**2 / denom


def synthesise_sr_data() -> pd.DataFrame:
    """
    Generate synthetic SR curves as a stand-alone fallback.
    Levy is allowed to peak earlier and more broadly than Gaussian.
    """
    signal_fracs = {"AFP": 0.72, "GPC3": 0.58, "DKK1": 0.65, "MDK": 0.80}
    peak_offsets = {"Gaussian": 1.00, "OU": 0.85, "Levy": 0.60}

    rows = []
    sigmas = np.linspace(0.0, 3.0, 61)
    rng = np.random.default_rng(42)

    for biomarker, frac in signal_fracs.items():
        for noise_type, peak_mult in peak_offsets.items():
            sigma_star_true = frac * peak_mult * 1.2

            p_arr = []
            for sigma in sigmas:
                p_base = hill(frac, n=2.0)
                boost = 0.45 * np.exp(-((sigma - sigma_star_true) ** 2) / (2.0 * 0.35**2))
                if noise_type == "Levy":
                    boost *= 1.15 * np.exp(-0.3 * max(sigma - sigma_star_true, 0.0))
                elif noise_type == "OU":
                    boost *= 0.90
                p_act = np.clip(p_base + boost + 0.02 * rng.standard_normal(), 0.0, 1.0)
                p_arr.append(float(p_act))

            p_arr = np.array(p_arr, dtype=float)
            p_arr = smooth_curve(sigmas, p_arr)
            p_arr = np.clip(p_arr, 0.0, 1.0)

            mi_arr = mi_from_probs(p_arr)
            fi_arr = fisher_from_prob_curve(sigmas, p_arr)
            mi_std = np.maximum(0.01, 0.03 * mi_arr)

            for idx, sigma in enumerate(sigmas):
                rows.append(
                    {
                        "biomarker": biomarker,
                        "noise_type": noise_type,
                        "sigma": round(float(sigma), 6),
                        "activation_prob": round(float(p_arr[idx]), 6),
                        "activation_prob_std": 0.02,
                        "mutual_information": round(float(mi_arr[idx]), 6),
                        "mutual_information_std": round(float(mi_std[idx]), 6),
                        "fisher_information": round(float(fi_arr[idx]), 6),
                    }
                )

    df = pd.DataFrame(rows)
    fallback_path = os.path.join(OUTPUT_DIR, "sr_full_results_synthetic.csv")
    df.to_csv(fallback_path, index=False)
    print(f"[info] Synthetic data saved -> {fallback_path}")
    return df


def smooth_curve(sigmas: np.ndarray, values: np.ndarray, win: int = 7, poly: int = 3) -> np.ndarray:
    """Savitzky-Golay smoothing with safe fallback for short curves."""
    n = len(values)
    if n < 3:
        return values

    window = min(win, n if n % 2 == 1 else n - 1)
    if window < 3:
        return values

    polyorder = min(poly, window - 1)
    return savgol_filter(values, window_length=window, polyorder=polyorder)


def find_sigma_star(sigmas: np.ndarray, mi_values: np.ndarray) -> float:
    """Locate the MI maximum using a smoothed cubic spline, with argmax fallback."""
    smoothed = smooth_curve(sigmas, mi_values)
    try:
        spline = UnivariateSpline(sigmas, smoothed, s=0, k=min(3, len(sigmas) - 1))
        derivative_roots = spline.derivative().roots()
        valid = derivative_roots[
            (derivative_roots >= sigmas[0]) & (derivative_roots <= sigmas[-1])
        ]
        if len(valid) == 0:
            return float(sigmas[np.argmax(smoothed)])
        mi_at_roots = spline(valid)
        return float(valid[np.argmax(mi_at_roots)])
    except Exception:
        return float(sigmas[np.argmax(smoothed)])


def bootstrap_peak_stats(
    sigmas: np.ndarray,
    mi_mean: np.ndarray,
    mi_std: np.ndarray | None = None,
    n_boot: int = BOOTSTRAP_N,
    ci: float = CI_ALPHA,
) -> tuple[float, float, float, float, float, float]:
    """
    Bootstrap CI for both sigma* and MI(sigma*).

    Instead of resampling sigma/MI pairs into duplicate x-values, we preserve the
    project's fixed sigma grid and perturb MI using the empirical Monte-Carlo
    standard deviation when available. This is more stable for spline fitting.
    """
    base_curve = smooth_curve(sigmas, mi_mean)
    point_sigma = find_sigma_star(sigmas, mi_mean)
    point_mi = float(np.interp(point_sigma, sigmas, base_curve))
    point_mi = float(np.clip(point_mi, 0.0, 1.0))

    rng = np.random.default_rng(0)
    if mi_std is None:
        scale = np.full_like(mi_mean, max(float(np.std(mi_mean) * 0.05), 1e-4), dtype=float)
    else:
        scale = np.asarray(mi_std, dtype=float)
        if np.allclose(scale, 0.0):
            scale = np.full_like(mi_mean, max(float(np.std(mi_mean) * 0.05), 1e-4), dtype=float)

    sigma_boot = []
    mi_boot = []

    for _ in range(n_boot):
        sampled_mi = mi_mean + rng.normal(0.0, scale, size=len(mi_mean))
        sampled_mi = np.clip(sampled_mi, 0.0, 1.0)
        sigma_star = find_sigma_star(sigmas, sampled_mi)
        sigma_boot.append(sigma_star)
        local_scale = float(np.interp(sigma_star, sigmas, scale))
        base_mi_at_sigma = float(np.interp(sigma_star, sigmas, base_curve))
        mi_star = base_mi_at_sigma + rng.normal(0.0, local_scale)
        mi_boot.append(float(np.clip(mi_star, 0.0, 1.0)))

    sigma_boot_arr = np.array(sigma_boot, dtype=float)
    mi_boot_arr = np.array(mi_boot, dtype=float)
    lower_pct = (1.0 - ci) / 2.0 * 100.0
    upper_pct = (1.0 + ci) / 2.0 * 100.0

    return (
        point_sigma,
        float(np.percentile(sigma_boot_arr, lower_pct)),
        float(np.percentile(sigma_boot_arr, upper_pct)),
        point_mi,
        float(np.percentile(mi_boot_arr, lower_pct)),
        float(np.percentile(mi_boot_arr, upper_pct)),
    )


def run_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each biomarker x noise-type group:
      - smooth MI / activation / FI curves
      - extract sigma* from MI
      - bootstrap CI on sigma* and MI(sigma*)
      - compute SR gain over the sigma~=0 baseline
    """
    records = []

    for biomarker in BIOMARKERS:
        for noise_type in NOISE_TYPES:
            sub = df[(df["biomarker"] == biomarker) & (df["noise_type"] == noise_type)].copy()
            if sub.empty:
                print(f"[warn] No data for {biomarker} x {noise_type} - skipping.")
                continue

            sub = sub.sort_values("sigma").reset_index(drop=True)
            sigmas = sub["sigma"].to_numpy(dtype=float)
            mi_raw = sub["mutual_information"].to_numpy(dtype=float)
            ap_raw = sub["activation_prob"].to_numpy(dtype=float)
            fi_raw = sub["fisher_information"].to_numpy(dtype=float)
            mi_std = (
                sub["mutual_information_std"].to_numpy(dtype=float)
                if "mutual_information_std" in sub.columns
                else None
            )

            mi_sm = smooth_curve(sigmas, mi_raw)
            ap_sm = smooth_curve(sigmas, ap_raw)
            fi_sm = smooth_curve(sigmas, fi_raw)

            (
                sigma_star,
                sigma_ci_lo,
                sigma_ci_hi,
                mi_at_star,
                mi_ci_lo,
                mi_ci_hi,
            ) = bootstrap_peak_stats(sigmas, mi_raw, mi_std=mi_std)

            ap_at_star = float(np.interp(sigma_star, sigmas, ap_sm))
            fi_at_star = float(np.interp(sigma_star, sigmas, fi_sm))
            mi_baseline = float(mi_sm[0])
            ap_baseline = float(ap_sm[0])

            sr_gain = mi_at_star - mi_baseline
            ap_gain = ap_at_star - ap_baseline

            records.append(
                {
                    "Biomarker": biomarker,
                    "Noise Type": noise_type,
                    "sigma*": round(sigma_star, 4),
                    "sigma* CI lower (95%)": round(sigma_ci_lo, 4),
                    "sigma* CI upper (95%)": round(sigma_ci_hi, 4),
                    "MI(sigma*)": round(mi_at_star, 5),
                    "MI(sigma*) CI lower (95%)": round(mi_ci_lo, 5),
                    "MI(sigma*) CI upper (95%)": round(mi_ci_hi, 5),
                    "MI(sigma=0)": round(mi_baseline, 5),
                    "SR Gain delta MI": round(sr_gain, 5),
                    "Act. Prob(sigma*)": round(ap_at_star, 4),
                    "Act. Gain delta P": round(ap_gain, 4),
                    "Fisher Info(sigma*)": round(fi_at_star, 4),
                    "_sigmas": sigmas,
                    "_mi_sm": mi_sm,
                }
            )
            print(
                f"  {biomarker:5s} x {noise_type:8s}  sigma* = {sigma_star:.3f}  "
                f"[{sigma_ci_lo:.3f}, {sigma_ci_hi:.3f}]  "
                f"MI* = {mi_at_star:.4f}  delta MI = {sr_gain:+.4f}"
            )

    return pd.DataFrame(records)


def save_tables(results: pd.DataFrame) -> None:
    tbl = results[TABLE_COLS].copy()

    csv_path = os.path.join(OUTPUT_DIR, "step10_sr_peak_summary.csv")
    tbl.to_csv(csv_path, index=False)
    print(f"[save] {csv_path}")

    tex_path = os.path.join(OUTPUT_DIR, "step10_sr_peak_summary.tex")
    with open(tex_path, "w") as handle:
        handle.write("% Step 10 — SR Peak Characterisation Summary\n")
        handle.write("% Auto-generated by step10_sr_peak_characterization.py\n\n")
        handle.write("\\begin{table}[htbp]\n")
        handle.write("\\centering\n")
        handle.write(
            "\\caption{Optimal noise intensity $\\sigma^*$ and associated SR metrics "
            "for each biomarker--noise-type combination. "
            "$\\Delta$MI = MI($\\sigma^*$) $-$ MI($\\sigma\\approx0$). "
            "95\\% bootstrap CIs are reported for both $\\sigma^*$ and MI($\\sigma^*$).}\n"
        )
        handle.write("\\label{tab:sr_peak}\n")
        handle.write("\\resizebox{\\textwidth}{!}{%\n")
        handle.write("\\begin{tabular}{llcccccc}\n")
        handle.write("\\toprule\n")
        handle.write(
            "Biomarker & Noise Type & $\\sigma^*$ & 95\\% CI($\\sigma^*$) & "
            "MI($\\sigma^*$) & 95\\% CI(MI) & $\\Delta$MI & "
            "Fisher Info($\\sigma^*$) \\\\\n"
        )
        handle.write("\\midrule\n")

        last_biomarker = None
        for _, row in tbl.iterrows():
            biomarker_label = row["Biomarker"] if row["Biomarker"] != last_biomarker else ""
            last_biomarker = row["Biomarker"]
            sigma_ci = (
                f"[{row['sigma* CI lower (95%)']:.3f}, "
                f"{row['sigma* CI upper (95%)']:.3f}]"
            )
            mi_ci = (
                f"[{row['MI(sigma*) CI lower (95%)']:.4f}, "
                f"{row['MI(sigma*) CI upper (95%)']:.4f}]"
            )
            handle.write(
                f"{biomarker_label} & {row['Noise Type']} & "
                f"{row['sigma*']:.3f} & {sigma_ci} & "
                f"{row['MI(sigma*)']:.4f} & {mi_ci} & "
                f"${row['SR Gain delta MI']:+.4f}$ & "
                f"{row['Fisher Info(sigma*)']:.3f} \\\\\n"
            )

        handle.write("\\bottomrule\n")
        handle.write("\\end{tabular}}\n")
        handle.write("\\end{table}\n")
    print(f"[save] {tex_path}")


def style_axes(ax, title: str = "", xlabel: str = "", ylabel: str = "", title_size: int = 11) -> None:
    ax.set_facecolor(AX_BG)
    if title:
        ax.set_title(title, color="black", fontsize=title_size, fontweight="bold", pad=7)
    if xlabel:
        ax.set_xlabel(xlabel, color=TICK_C, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=TICK_C, fontsize=9)
    for spine in ax.spines.values():
        spine.set_color(SPINE_C)
    ax.tick_params(colors=TICK_C, labelsize=8)
    ax.grid(True, color=GRID_C, lw=0.6, linestyle="--", alpha=0.7)


def plot_sr_curves(results: pd.DataFrame) -> None:
    fig, axes = plt.subplots(len(BIOMARKERS), len(NOISE_TYPES), figsize=(15, 4 * len(BIOMARKERS)))
    fig.patch.set_facecolor(FIG_BG)
    fig.suptitle(
        "SR Peak Characterisation — Mutual Information vs Noise Intensity",
        color="black",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    for biomarker_idx, biomarker in enumerate(BIOMARKERS):
        for noise_idx, noise_type in enumerate(NOISE_TYPES):
            ax = axes[biomarker_idx][noise_idx]
            row = results[
                (results["Biomarker"] == biomarker) & (results["Noise Type"] == noise_type)
            ]
            if row.empty:
                ax.set_visible(False)
                continue
            row = row.iloc[0]
            sigmas = row["_sigmas"]
            mi_sm = row["_mi_sm"]
            sigma_star = row["sigma*"]
            mi_star = row["MI(sigma*)"]
            sigma_ci_lo = row["sigma* CI lower (95%)"]
            sigma_ci_hi = row["sigma* CI upper (95%)"]

            color = NT_COLORS[noise_type]
            ax.plot(sigmas, mi_sm, color=color, lw=2.0, alpha=0.9)
            ax.fill_between(sigmas, np.zeros_like(mi_sm), mi_sm, color=color, alpha=0.07)
            ax.axvline(sigma_star, color="#e6a800", lw=1.5, linestyle="--", alpha=0.9)
            ax.axvspan(sigma_ci_lo, sigma_ci_hi, color="#e6a800", alpha=0.12)
            ax.scatter(
                [sigma_star],
                [mi_star],
                color="#e6a800",
                s=70,
                zorder=5,
                marker=NT_MARKERS[noise_type],
                edgecolors="black",
                linewidths=0.8,
            )
            ax.annotate(
                f"sigma*={sigma_star:.2f}\nMI={mi_star:.3f}",
                xy=(sigma_star, mi_star),
                xytext=(10, -18),
                textcoords="offset points",
                color="#b38600",
                fontsize=7.5,
                arrowprops=dict(arrowstyle="-", color="#b38600", lw=0.8),
            )

            mi_base = row["MI(sigma=0)"]
            ax.annotate(
                "",
                xy=(sigma_star, mi_star),
                xytext=(sigma_star, mi_base),
                arrowprops=dict(arrowstyle="<->", color="#2ca02c", lw=1.2),
            )
            ax.text(
                sigma_star + 0.03,
                (mi_star + mi_base) / 2.0,
                f"delta MI\n{row['SR Gain delta MI']:+.3f}",
                color="#2ca02c",
                fontsize=7,
                va="center",
            )
            style_axes(
                ax,
                title=f"{biomarker} — {noise_type}",
                xlabel="Noise intensity sigma",
                ylabel="Mutual Information (bits)",
            )

    plt.tight_layout(rect=[0, 0, 1, 0.995])
    out_path = os.path.join(OUTPUT_DIR, "step10_sr_curves_annotated.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=FIG_BG)
    plt.close()
    print(f"[fig]  {out_path}")


def make_heatmap(
    matrix: pd.DataFrame,
    title: str,
    cbar_label: str,
    cmap: LinearSegmentedColormap,
    fmt: str,
    out_path: str,
    vmin: float | None = None,
    vmax: float | None = None,
    annot_color_thresh: float = 0.5,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(AX_BG)

    data = matrix.to_numpy(dtype=float)
    vmin_ = data.min() if vmin is None else vmin
    vmax_ = data.max() if vmax is None else vmax
    image = ax.imshow(data, cmap=cmap, aspect="auto", vmin=vmin_, vmax=vmax_)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            norm_val = (value - vmin_) / max(vmax_ - vmin_, 1e-9)
            text_color = "white" if norm_val > (1 - annot_color_thresh) else "black"
            ax.text(
                j,
                i,
                format(value, fmt),
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color=text_color,
            )

    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, color=TICK_C, fontsize=10)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index, color=TICK_C, fontsize=10)
    for spine in ax.spines.values():
        spine.set_color(SPINE_C)
    ax.tick_params(colors=TICK_C)

    cbar = fig.colorbar(image, ax=ax, pad=0.02)
    cbar.set_label(cbar_label, color=TICK_C, fontsize=9)
    cbar.ax.yaxis.set_tick_params(color=TICK_C, labelcolor=TICK_C)
    cbar.outline.set_edgecolor(SPINE_C)

    ax.set_title(title, color="black", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Noise Type", color=TICK_C, fontsize=10)
    ax.set_ylabel("Biomarker", color=TICK_C, fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=FIG_BG)
    plt.close()
    print(f"[fig]  {out_path}")


CMAP_SIGMA = LinearSegmentedColormap.from_list(
    "sigma_cmap", ["#0d2137", "#1a4f6e", "#2980b9", "#00d4ff", "#a8ff78"]
)
CMAP_MI = LinearSegmentedColormap.from_list(
    "mi_cmap", ["#0d1117", "#1f3a5f", "#2e6da4", "#58a6ff", "#ffffff"]
)
CMAP_GAIN = LinearSegmentedColormap.from_list(
    "gain_cmap", ["#5c0a0a", "#a02020", "#c0392b", "#e8c56a", "#a8ff78"]
)


def plot_heatmaps(results: pd.DataFrame) -> None:
    pivot_kwargs = dict(index="Biomarker", columns="Noise Type")

    sigma_mat = results.pivot_table(values="sigma*", **pivot_kwargs)[NOISE_TYPES].reindex(BIOMARKERS)
    make_heatmap(
        sigma_mat,
        title="Optimal Noise Intensity sigma* (MI Maximiser)\nper Biomarker x Noise Type",
        cbar_label="sigma*",
        cmap=CMAP_SIGMA,
        fmt=".3f",
        out_path=os.path.join(OUTPUT_DIR, "step10_sigma_star_heatmap.png"),
        annot_color_thresh=0.55,
    )

    mi_mat = results.pivot_table(values="MI(sigma*)", **pivot_kwargs)[NOISE_TYPES].reindex(BIOMARKERS)
    make_heatmap(
        mi_mat,
        title="Mutual Information at Optimum MI(sigma*)\nper Biomarker x Noise Type",
        cbar_label="MI(sigma*) [bits]",
        cmap=CMAP_MI,
        fmt=".4f",
        out_path=os.path.join(OUTPUT_DIR, "step10_mi_peak_heatmap.png"),
        annot_color_thresh=0.6,
    )

    gain_mat = results.pivot_table(values="SR Gain delta MI", **pivot_kwargs)[NOISE_TYPES].reindex(BIOMARKERS)
    make_heatmap(
        gain_mat,
        title="SR Information Gain delta MI = MI(sigma*) - MI(sigma~0)\nper Biomarker x Noise Type",
        cbar_label="delta MI [bits]",
        cmap=CMAP_GAIN,
        fmt="+.4f",
        out_path=os.path.join(OUTPUT_DIR, "step10_sr_gain_heatmap.png"),
        vmin=float(gain_mat.to_numpy().min()),
        vmax=float(gain_mat.to_numpy().max()),
        annot_color_thresh=0.65,
    )


def print_summary(results: pd.DataFrame) -> None:
    tbl = results[TABLE_COLS].copy()
    sep = "-" * 126
    print(f"\n{sep}")
    print("  STEP 10 — SR PEAK CHARACTERISATION SUMMARY")
    print(sep)
    header = (
        f"{'Biomarker':>10}  {'Noise':>9}  {'sigma*':>8}  "
        f"{'95% CI(sigma)':>20}  {'MI*':>8}  {'95% CI(MI*)':>20}  "
        f"{'delta MI':>10}  {'Act P*':>8}  {'Fisher*':>9}"
    )
    print(header)
    print(sep)
    for _, row in tbl.iterrows():
        sigma_ci = f"[{row['sigma* CI lower (95%)']:.3f},{row['sigma* CI upper (95%)']:.3f}]"
        mi_ci = f"[{row['MI(sigma*) CI lower (95%)']:.4f},{row['MI(sigma*) CI upper (95%)']:.4f}]"
        print(
            f"{row['Biomarker']:>10}  {row['Noise Type']:>9}  "
            f"{row['sigma*']:>8.3f}  {sigma_ci:>20}  "
            f"{row['MI(sigma*)']:>8.4f}  {mi_ci:>20}  "
            f"{row['SR Gain delta MI']:>+10.4f}  {row['Act. Prob(sigma*)']:>8.3f}  "
            f"{row['Fisher Info(sigma*)']:>9.3f}"
        )
    print(sep)

    print("\n  Best noise type per biomarker (highest delta MI)")
    print("  " + "-" * 54)
    for biomarker in BIOMARKERS:
        sub = results[results["Biomarker"] == biomarker]
        best = sub.loc[sub["SR Gain delta MI"].idxmax()]
        print(
            f"  {biomarker:5s} -> {best['Noise Type']:8s}  "
            f"sigma* = {best['sigma*']:.3f}  delta MI = {best['SR Gain delta MI']:+.4f}"
        )


def run_step10_peak_characterization() -> pd.DataFrame:
    """Run the full step-10 peak characterization workflow and return the result table."""
    print("=" * 60)
    print("  STEP 10A — SR Peak Characterisation")
    print("=" * 60)

    df = load_or_synthesise()

    print("\n[step] Running peak analysis with bootstrap CI ...")
    results = run_analysis(df)

    print("\n[step] Saving tables ...")
    save_tables(results)

    print("\n[step] Plotting SR curves with sigma* annotations ...")
    plot_sr_curves(results)

    print("\n[step] Plotting heatmaps ...")
    plot_heatmaps(results)

    print_summary(results)
    print("=" * 60)
    print("  Step 10A complete.")
    print(f"  Outputs in {OUTPUT_DIR}/")
    print("=" * 60)
    return results


if __name__ == "__main__":
    run_step10_peak_characterization()
