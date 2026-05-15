"""
Step 13 — ROC / AUC Clinical Framing
=====================================
Converts SR activation probability into a clinically interpretable detection
metric.  For each (biomarker × noise type × σ) the Hill-function output is
used as a continuous detection score with binary ground-truth labels:
    1 = early-stage tumour   (N ≈ 189)
    0 = matched normal tissue (N = 50)

ROC curves and AUC are computed at every noise level.  Bootstrap CIs (n=1000)
are reported for AUC at the optimal σ*.

Outputs
-------
  output/step13_auc_vs_sigma.png
  output/step13_roc_at_sigma_star.png
  output/step13_auc_heatmap.png
  output/step13_auc_summary.csv
  output/step13_auc_summary.tex
"""

from __future__ import annotations

import json
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_curve, roc_auc_score

from noise_utils import (
    DEFAULT_LEVY_ALPHA,
    DEFAULT_LEVY_BETA,
    DEFAULT_OU_STEP,
    DEFAULT_OU_TAU,
    generate_noise,
)
from step10_sr_peak_characterization import find_sigma_star, smooth_curve

warnings.filterwarnings("ignore")

# ==================================================
# PATHS
# ==================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLINICAL_FILE = os.path.join(BASE_DIR, "data", "clinical.project-tcga-lihc.json")
EXPRESSION_FILE = os.path.join(BASE_DIR, "data", "TCGA.LIHC.sampleMap:HiSeqV2.tsv")
THRESHOLD_FILE = os.path.join(OUTPUT_DIR, "gene_activation_threshold_K.json")

# ==================================================
# CONSTANTS
# ==================================================

BIOMARKERS = ["AFP", "GPC3", "DKK1", "MDK"]
NOISE_TYPES = ["Gaussian", "OU", "Levy"]
NOISE_TO_INTERNAL = {"Gaussian": "gaussian", "OU": "ou", "Levy": "levy"}

HILL_N = 4.0
SIGMA_GRID = np.linspace(0.0, 6.0, 40)
NUM_TRIALS = 200
BOOTSTRAP_N = 1000
CI_ALPHA = 0.95

# Dark theme
FIG_BG = "white"
AX_BG = "white"
GRID_C = "#cccccc"
TICK_C = "#333333"
SPINE_C = "#999999"

NT_COLORS = {"Gaussian": "#1f77b4", "OU": "#d62728", "Levy": "#9467bd"}
NT_MARKERS = {"Gaussian": "o", "OU": "s", "Levy": "^"}

CMAP_AUC = LinearSegmentedColormap.from_list(
    "auc_cmap", ["#0d1117", "#1f3a5f", "#2e6da4", "#58a6ff", "#a8ff78", "#ffffff"]
)

# ==================================================
# DATA LOADING
# ==================================================


def load_expression_matrix():
    """Load the full HiSeqV2 expression matrix (genes × samples)."""
    df = pd.read_csv(EXPRESSION_FILE, sep="\t")
    df.set_index(df.columns[0], inplace=True)
    return df


def load_clinical():
    """Load clinical JSON and build patient-id → stage/vital lookup."""
    with open(CLINICAL_FILE) as f:
        data = json.load(f)
    records = []
    for pt in data:
        pid = pt.get("submitter_id")
        diags = pt.get("diagnoses", [])
        stage = diags[0].get("ajcc_pathologic_stage", "") if diags else ""
        records.append({"patient_id": pid, "stage": stage})
    return records


def extract_tumor_normal_vectors(expr_df, clinical_records):
    """
    Extract expression vectors for:
      - Early-stage tumour samples (barcode xx-01 = primary tumour)
      - Normal tissue samples     (barcode xx-11 = solid tissue normal)
    Returns dict[biomarker] → (tumor_values, normal_values)
    """
    # Identify early-stage patient IDs
    early_ids = set()
    for rec in clinical_records:
        if rec["stage"] in ("Stage I", "Stage II"):
            early_ids.add(rec["patient_id"])

    # Classify columns by sample type
    tumor_cols = []
    normal_cols = []
    for col in expr_df.columns:
        if len(col) < 15:
            continue
        sample_type = col[13:15]
        patient_id = col[:12]
        if sample_type == "01" and patient_id in early_ids:
            tumor_cols.append(col)
        elif sample_type == "11":
            normal_cols.append(col)

    print(f"[load] Early-stage tumour samples: {len(tumor_cols)}")
    print(f"[load] Normal tissue samples: {len(normal_cols)}")

    results = {}
    for gene in BIOMARKERS:
        if gene not in expr_df.index:
            raise ValueError(f"Gene {gene} not found in expression matrix")
        tumor_vals = expr_df.loc[gene, tumor_cols].values.astype(float)
        normal_vals = expr_df.loc[gene, normal_cols].values.astype(float)
        # Average duplicates per patient for tumor
        patient_map = {}
        for col, val in zip(tumor_cols, tumor_vals):
            pid = col[:12]
            patient_map.setdefault(pid, []).append(val)
        tumor_vals_dedup = np.array([np.mean(v) for v in patient_map.values()])
        # Normal: average per patient too
        normal_map = {}
        for col, val in zip(normal_cols, normal_vals):
            pid = col[:12]
            normal_map.setdefault(pid, []).append(val)
        normal_vals_dedup = np.array([np.mean(v) for v in normal_map.values()])

        results[gene] = (tumor_vals_dedup, normal_vals_dedup)
        print(f"  {gene}: tumor N={len(tumor_vals_dedup)}, normal N={len(normal_vals_dedup)}")

    return results


def load_thresholds():
    """Load per-biomarker activation thresholds from step 4."""
    with open(THRESHOLD_FILE) as f:
        data = json.load(f)
    return {
        bm: float(data["biomarker_thresholds"][bm]["activation_threshold_K"])
        for bm in BIOMARKERS
    }


# ==================================================
# MODEL
# ==================================================


def hill(x, K, n=HILL_N):
    """Hill activation with absolute-value guard."""
    xn = np.abs(x) ** n
    return xn / (K ** n + xn)


def compute_auc_at_sigma(
    tumor_signal, normal_signal, threshold_k, noise_type, sigma, num_trials=NUM_TRIALS
):
    """
    Compute AUC across multiple noise trials at a given sigma.
    Returns array of per-trial AUC values.
    """
    internal = NOISE_TO_INTERNAL[noise_type]
    n_tumor = len(tumor_signal)
    n_normal = len(normal_signal)
    combined = np.concatenate([tumor_signal, normal_signal])
    labels = np.concatenate([np.ones(n_tumor), np.zeros(n_normal)])

    trial_aucs = []
    for trial in range(num_trials):
        eta = generate_noise(
            internal,
            len(combined),
            float(sigma),
            tau=DEFAULT_OU_TAU,
            h=DEFAULT_OU_STEP,
            levy_alpha=DEFAULT_LEVY_ALPHA,
            levy_beta=DEFAULT_LEVY_BETA,
            seed=trial,
        )
        noisy = combined + eta
        scores = hill(noisy, threshold_k)
        try:
            auc = roc_auc_score(labels, scores)
        except ValueError:
            auc = 0.5
        trial_aucs.append(auc)

    return np.array(trial_aucs, dtype=float)


def bootstrap_auc_ci(auc_trials, n_boot=BOOTSTRAP_N, ci=CI_ALPHA):
    """Bootstrap CI on the mean AUC across trials."""
    rng = np.random.default_rng(42)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(auc_trials, size=len(auc_trials), replace=True)
        means.append(sample.mean())
    means = np.array(means)
    lo = np.percentile(means, (1 - ci) / 2 * 100)
    hi = np.percentile(means, (1 + ci) / 2 * 100)
    return float(np.mean(auc_trials)), float(lo), float(hi)


# ==================================================
# MAIN SWEEP
# ==================================================


def run_auc_sweep(bio_vectors, thresholds):
    """Run AUC vs sigma sweep for all biomarkers × noise types."""
    records = []

    for gene in BIOMARKERS:
        tumor, normal = bio_vectors[gene]
        K = thresholds[gene]

        for nt in NOISE_TYPES:
            print(f"  {gene} × {nt} ...", end="", flush=True)
            auc_means = []
            auc_stds = []

            for sigma_idx, sigma in enumerate(SIGMA_GRID):
                trial_aucs = compute_auc_at_sigma(tumor, normal, K, nt, sigma)
                auc_means.append(float(trial_aucs.mean()))
                auc_stds.append(float(trial_aucs.std()))

            auc_curve = np.array(auc_means)
            auc_std_arr = np.array(auc_stds)
            auc_smooth = smooth_curve(SIGMA_GRID, auc_curve)
            sigma_star = find_sigma_star(SIGMA_GRID, auc_smooth)

            # Bootstrap CI at sigma*
            sigma_star_idx = np.argmin(np.abs(SIGMA_GRID - sigma_star))
            star_trials = compute_auc_at_sigma(
                tumor, normal, K, nt, SIGMA_GRID[sigma_star_idx]
            )
            auc_star, ci_lo, ci_hi = bootstrap_auc_ci(star_trials)

            # Baseline AUC at sigma=0
            baseline_trials = compute_auc_at_sigma(tumor, normal, K, nt, 0.0)
            auc_baseline = float(baseline_trials.mean())

            records.append({
                "Biomarker": gene,
                "Noise Type": nt,
                "sigma*": round(sigma_star, 4),
                "AUC(sigma*)": round(auc_star, 5),
                "AUC CI lower": round(ci_lo, 5),
                "AUC CI upper": round(ci_hi, 5),
                "AUC(sigma=0)": round(auc_baseline, 5),
                "delta AUC": round(auc_star - auc_baseline, 5),
                "_auc_curve": auc_smooth,
                "_auc_std": auc_std_arr,
                "_star_trials": star_trials,
            })
            print(f" σ*={sigma_star:.2f}  AUC*={auc_star:.4f}  ΔAUC={auc_star - auc_baseline:+.4f}")

    return pd.DataFrame(records)


# ==================================================
# FIGURES
# ==================================================


def style_axes(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(AX_BG)
    if title:
        ax.set_title(title, color="black", fontsize=10, fontweight="bold", pad=6)
    if xlabel:
        ax.set_xlabel(xlabel, color=TICK_C, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=TICK_C, fontsize=9)
    for spine in ax.spines.values():
        spine.set_color(SPINE_C)
    ax.tick_params(colors=TICK_C, labelsize=8)
    ax.grid(True, color=GRID_C, lw=0.5, linestyle="--", alpha=0.6)


def plot_auc_vs_sigma(results):
    """AUC vs noise intensity for all biomarkers × noise types."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    fig.patch.set_facecolor(FIG_BG)
    fig.suptitle(
        "ROC-AUC vs Noise Intensity — Early-Stage Tumour vs Normal Tissue\n"
        "Stochastic Resonance in Diagnostic Detection",
        color="black", fontsize=13, fontweight="bold",
    )

    for idx, gene in enumerate(BIOMARKERS):
        ax = axes.flat[idx]
        for _, row in results[results["Biomarker"] == gene].iterrows():
            nt = row["Noise Type"]
            auc_curve = row["_auc_curve"]
            auc_std = row["_auc_std"]
            sigma_star = row["sigma*"]
            auc_star = row["AUC(sigma*)"]

            ax.plot(SIGMA_GRID, auc_curve, color=NT_COLORS[nt], lw=2, label=nt)
            ax.fill_between(
                SIGMA_GRID,
                auc_curve - auc_std,
                auc_curve + auc_std,
                alpha=0.12,
                color=NT_COLORS[nt],
            )
            ax.scatter(
                [sigma_star], [auc_star],
                color=NT_COLORS[nt], s=70, zorder=5,
                marker=NT_MARKERS[nt], edgecolors="white", linewidths=0.8,
            )

        ax.axhline(0.5, color="#b38600", lw=0.8, linestyle=":", alpha=0.7,
                    label="Random (AUC=0.5)")
        ax.legend(fontsize=8, facecolor=AX_BG, labelcolor="black", edgecolor=SPINE_C)
        style_axes(ax, title=gene, xlabel="Noise Intensity σ", ylabel="AUC")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(OUTPUT_DIR, "step13_auc_vs_sigma.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=FIG_BG)
    plt.close()
    print(f"[fig]  {path}")


def plot_roc_at_sigma_star(results, bio_vectors, thresholds):
    """ROC curves at the optimal σ* for each biomarker (best noise type)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor(FIG_BG)
    fig.suptitle(
        "ROC Curves at Optimal Noise Intensity σ*\nEarly-Stage HCC vs Normal Tissue",
        color="black", fontsize=13, fontweight="bold",
    )

    for idx, gene in enumerate(BIOMARKERS):
        ax = axes.flat[idx]
        gene_results = results[results["Biomarker"] == gene]
        tumor, normal = bio_vectors[gene]
        K = thresholds[gene]
        combined = np.concatenate([tumor, normal])
        labels = np.concatenate([np.ones(len(tumor)), np.zeros(len(normal))])

        for _, row in gene_results.iterrows():
            nt = row["Noise Type"]
            sigma_star = row["sigma*"]
            internal = NOISE_TO_INTERNAL[nt]

            # Average ROC across trials at sigma*
            mean_fpr = np.linspace(0, 1, 100)
            tpr_list = []
            for trial in range(min(50, NUM_TRIALS)):
                eta = generate_noise(
                    internal, len(combined), float(sigma_star),
                    tau=DEFAULT_OU_TAU, h=DEFAULT_OU_STEP,
                    levy_alpha=DEFAULT_LEVY_ALPHA, levy_beta=DEFAULT_LEVY_BETA,
                    seed=trial,
                )
                scores = hill(combined + eta, K)
                fpr, tpr, _ = roc_curve(labels, scores)
                tpr_list.append(np.interp(mean_fpr, fpr, tpr))

            mean_tpr = np.mean(tpr_list, axis=0)
            auc_val = row["AUC(sigma*)"]
            ax.plot(
                mean_fpr, mean_tpr,
                color=NT_COLORS[nt], lw=2,
                label=f"{nt} (AUC={auc_val:.3f}, σ*={sigma_star:.2f})",
            )

        ax.plot([0, 1], [0, 1], color="#ffd966", lw=0.8, linestyle=":")
        ax.legend(fontsize=7.5, facecolor=AX_BG, labelcolor="black",
                  edgecolor=SPINE_C, loc="lower right")
        style_axes(ax, title=gene, xlabel="False Positive Rate", ylabel="True Positive Rate")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(OUTPUT_DIR, "step13_roc_at_sigma_star.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=FIG_BG)
    plt.close()
    print(f"[fig]  {path}")


def plot_auc_heatmap(results):
    """AUC(σ*) heatmap — biomarker × noise type."""
    pivot = results.pivot_table(
        values="AUC(sigma*)", index="Biomarker", columns="Noise Type"
    )[NOISE_TYPES].reindex(BIOMARKERS)
    data = pivot.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(AX_BG)
    image = ax.imshow(data, cmap=CMAP_AUC, aspect="auto",
                      vmin=max(0.4, data.min() - 0.05), vmax=min(1.0, data.max() + 0.05))

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            norm_val = (val - data.min()) / max(data.max() - data.min(), 1e-9)
            color = "white" if norm_val > 0.55 else "black"
            ax.text(j, i, f"{val:.4f}", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=color)

    ax.set_xticks(range(len(NOISE_TYPES)))
    ax.set_xticklabels(NOISE_TYPES, color=TICK_C, fontsize=10)
    ax.set_yticks(range(len(BIOMARKERS)))
    ax.set_yticklabels(BIOMARKERS, color=TICK_C, fontsize=10)
    for spine in ax.spines.values():
        spine.set_color(SPINE_C)
    ax.tick_params(colors=TICK_C)

    cbar = fig.colorbar(image, ax=ax, pad=0.02)
    cbar.set_label("AUC at σ*", color=TICK_C, fontsize=9)
    cbar.ax.yaxis.set_tick_params(color=TICK_C, labelcolor=TICK_C)
    cbar.outline.set_edgecolor(SPINE_C)

    ax.set_title("AUC at Optimal Noise σ*\nEarly-Stage Tumour vs Normal Tissue",
                  color="black", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Noise Type", color=TICK_C, fontsize=10)
    ax.set_ylabel("Biomarker", color=TICK_C, fontsize=10)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "step13_auc_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=FIG_BG)
    plt.close()
    print(f"[fig]  {path}")


# ==================================================
# TABLES
# ==================================================


def save_tables(results):
    """Save CSV and LaTeX summary tables."""
    cols = [c for c in results.columns if not c.startswith("_")]
    tbl = results[cols].copy()

    csv_path = os.path.join(OUTPUT_DIR, "step13_auc_summary.csv")
    tbl.to_csv(csv_path, index=False)
    print(f"[save] {csv_path}")

    tex_path = os.path.join(OUTPUT_DIR, "step13_auc_summary.tex")
    with open(tex_path, "w") as f:
        f.write("% Step 13 — ROC/AUC Summary Table\n\n")
        f.write("\\begin{table}[htbp]\\centering\n")
        f.write(
            "\\caption{AUC at optimal noise intensity $\\sigma^*$ for early-stage "
            "HCC detection (tumour vs.\\ matched normal tissue). "
            "Bootstrap 95\\% CIs from 1000 resamples.}\n"
        )
        f.write("\\label{tab:auc}\n")
        f.write("\\resizebox{\\textwidth}{!}{%\n")
        f.write("\\begin{tabular}{llccccc}\\toprule\n")
        f.write(
            "Biomarker & Noise Type & $\\sigma^*$ & AUC($\\sigma^*$) & "
            "95\\% CI & AUC($\\sigma{=}0$) & $\\Delta$AUC \\\\\n\\midrule\n"
        )
        prev_bm = None
        for _, row in tbl.iterrows():
            bm_label = row["Biomarker"] if row["Biomarker"] != prev_bm else ""
            prev_bm = row["Biomarker"]
            ci = f"[{row['AUC CI lower']:.4f}, {row['AUC CI upper']:.4f}]"
            f.write(
                f"{bm_label} & {row['Noise Type']} & {row['sigma*']:.3f} & "
                f"{row['AUC(sigma*)']:.4f} & {ci} & "
                f"{row['AUC(sigma=0)']:.4f} & ${row['delta AUC']:+.4f}$ \\\\\n"
            )
        f.write("\\bottomrule\n\\end{tabular}}\\end{table}\n")
    print(f"[save] {tex_path}")


# ==================================================
# ENTRY POINT
# ==================================================


def run_step13():
    """Run the complete Step 13 workflow."""
    print("=" * 62)
    print("  STEP 13 — ROC / AUC Clinical Framing")
    print("=" * 62)

    print("\n[step] Loading expression matrix ...")
    expr_df = load_expression_matrix()

    print("[step] Loading clinical data ...")
    clinical = load_clinical()

    print("[step] Extracting tumour vs normal vectors ...")
    bio_vectors = extract_tumor_normal_vectors(expr_df, clinical)

    print("[step] Loading thresholds ...")
    thresholds = load_thresholds()

    print("\n[step] Running AUC vs σ sweep ...")
    results = run_auc_sweep(bio_vectors, thresholds)

    print("\n[step] Saving tables ...")
    save_tables(results)

    print("\n[step] Generating figures ...")
    plot_auc_vs_sigma(results)
    plot_roc_at_sigma_star(results, bio_vectors, thresholds)
    plot_auc_heatmap(results)

    # Print summary
    print("\n" + "=" * 80)
    print("  STEP 13 — AUC SUMMARY")
    print("=" * 80)
    for gene in BIOMARKERS:
        sub = results[results["Biomarker"] == gene]
        best = sub.loc[sub["AUC(sigma*)"].idxmax()]
        print(
            f"  {gene:5s} → best: {best['Noise Type']:8s}  σ*={best['sigma*']:.3f}  "
            f"AUC={best['AUC(sigma*)']:.4f} [{best['AUC CI lower']:.4f}, {best['AUC CI upper']:.4f}]  "
            f"ΔAUC={best['delta AUC']:+.4f}"
        )

    print(f"\n  Outputs in {OUTPUT_DIR}/")
    print("=" * 62)
    return results


if __name__ == "__main__":
    run_step13()
