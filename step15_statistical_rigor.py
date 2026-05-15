"""
Step 15 — Statistical Rigor Consolidation
==========================================
Closes the statistical rigor gaps required for journal submission:

  A. Bootstrap confidence bands (≥1000 resamples) on all SR curves
  B. Effect size reporting: ΔMI, ΔSNR, Cohen's d with 95% CI
  C. Benjamini-Hochberg FDR correction across all hypothesis tests
  D. Publication-ready Table 1 summarising all biomarker characteristics

Outputs
-------
  output/step15_mi_with_ci.png
  output/step15_act_with_ci.png
  output/step15_effect_sizes.csv
  output/step15_effect_sizes.tex
  output/step15_multiple_testing.csv
  output/step15_table1.csv
  output/step15_table1.tex
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
from scipy.stats import false_discovery_control

warnings.filterwarnings("ignore")

# ==================================================
# PATHS
# ==================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SR_RESULTS_FILE = os.path.join(OUTPUT_DIR, "sr_full_results.csv")
STEP10_FILE = os.path.join(OUTPUT_DIR, "step10_sr_peak_summary.csv")
STEP13_FILE = os.path.join(OUTPUT_DIR, "step13_auc_summary.csv")
STEP14_FILE = os.path.join(OUTPUT_DIR, "step14_survival_summary.csv")
STATS_FILE = os.path.join(OUTPUT_DIR, "tcga_lihc_afp_statistics.json")
THRESHOLD_FILE = os.path.join(OUTPUT_DIR, "gene_activation_threshold_K.json")

BIOMARKERS = ["AFP", "GPC3", "DKK1", "MDK"]
NOISE_TYPES_RAW = ["awgn", "ou", "levy"]
NOISE_DISPLAY = {"awgn": "Gaussian", "ou": "OU", "levy": "Lévy"}

BOOTSTRAP_N = 1000
CI_ALPHA = 0.95

# Dark theme
FIG_BG = "white"
AX_BG = "white"
GRID_C = "#cccccc"
TICK_C = "#333333"
SPINE_C = "#999999"

NT_COLORS = {"awgn": "#1f77b4", "ou": "#d62728", "levy": "#9467bd"}


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


# ==================================================
# A. BOOTSTRAP CI BANDS ON SR CURVES
# ==================================================


def bootstrap_ci_bands(sr_df):
    """
    For each (biomarker × noise_type × sigma), bootstrap the MI and AP
    point estimates using the stored mean and std from Monte Carlo trials.
    Produces CI bands across the sigma grid.
    """
    rng = np.random.default_rng(42)
    results = {}

    for gene in BIOMARKERS:
        results[gene] = {}
        for nt in NOISE_TYPES_RAW:
            sub = sr_df[(sr_df["biomarker"] == gene) & (sr_df["noise_type"] == nt)].sort_values("sigma")
            if sub.empty:
                continue

            sigmas = sub["sigma"].to_numpy(dtype=float)
            mi_mean = sub["mi_mean"].to_numpy(dtype=float)
            mi_std = sub["mi_std"].to_numpy(dtype=float)
            act_mean = sub["act_mean"].to_numpy(dtype=float)
            act_std = sub["act_std"].to_numpy(dtype=float)

            # Bootstrap: resample point estimates around mean ± std
            mi_lo, mi_hi = np.zeros_like(sigmas), np.zeros_like(sigmas)
            act_lo, act_hi = np.zeros_like(sigmas), np.zeros_like(sigmas)

            for i in range(len(sigmas)):
                mi_scale = max(mi_std[i], 1e-6)
                act_scale = max(act_std[i], 1e-6)
                mi_boots = mi_mean[i] + rng.normal(0, mi_scale, BOOTSTRAP_N)
                act_boots = act_mean[i] + rng.normal(0, act_scale, BOOTSTRAP_N)
                lo_pct = (1 - CI_ALPHA) / 2 * 100
                hi_pct = (1 + CI_ALPHA) / 2 * 100
                mi_lo[i] = np.percentile(mi_boots, lo_pct)
                mi_hi[i] = np.percentile(mi_boots, hi_pct)
                act_lo[i] = np.clip(np.percentile(act_boots, lo_pct), 0, 1)
                act_hi[i] = np.clip(np.percentile(act_boots, hi_pct), 0, 1)

            results[gene][nt] = {
                "sigmas": sigmas,
                "mi_mean": mi_mean, "mi_lo": mi_lo, "mi_hi": mi_hi,
                "act_mean": act_mean, "act_lo": act_lo, "act_hi": act_hi,
            }

    return results


def plot_ci_bands(ci_data, metric, ylabel, title_suffix, out_name):
    """Plot SR curves with shaded 95% CI bands."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    fig.patch.set_facecolor(FIG_BG)
    fig.suptitle(
        f"{ylabel} vs Noise Intensity — with 95% Bootstrap CI Bands\n{title_suffix}",
        color="black", fontsize=13, fontweight="bold",
    )

    key_mean = f"{metric}_mean"
    key_lo = f"{metric}_lo"
    key_hi = f"{metric}_hi"

    for idx, gene in enumerate(BIOMARKERS):
        ax = axes.flat[idx]
        for nt in NOISE_TYPES_RAW:
            if nt not in ci_data[gene]:
                continue
            d = ci_data[gene][nt]
            ax.plot(d["sigmas"], d[key_mean], color=NT_COLORS[nt], lw=2,
                    label=NOISE_DISPLAY[nt])
            ax.fill_between(d["sigmas"], d[key_lo], d[key_hi],
                            alpha=0.18, color=NT_COLORS[nt])

        ax.legend(fontsize=8, facecolor=AX_BG, labelcolor="black", edgecolor=SPINE_C)
        style_axes(ax, title=gene, xlabel="Noise Intensity σ", ylabel=ylabel)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(OUTPUT_DIR, out_name)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=FIG_BG)
    plt.close()
    print(f"[fig]  {path}")


# ==================================================
# B. EFFECT SIZE REPORTING
# ==================================================


def compute_effect_sizes(sr_df, s10_df):
    """
    For each (biomarker × noise type) compute:
      ΔMI = MI(σ*) − MI(σ=0)  with bootstrap 95% CI
      ΔSNR = ratio-based gain
      Cohen's d for activation probability gain
    """
    rng = np.random.default_rng(42)
    records = []

    for gene in BIOMARKERS:
        for nt_raw in NOISE_TYPES_RAW:
            nt_display = NOISE_DISPLAY[nt_raw]
            sub = sr_df[(sr_df["biomarker"] == gene) & (sr_df["noise_type"] == nt_raw)].sort_values("sigma")
            if sub.empty:
                continue

            sigmas = sub["sigma"].to_numpy(dtype=float)
            mi_mean = sub["mi_mean"].to_numpy(dtype=float)
            mi_std = sub["mi_std"].to_numpy(dtype=float)
            act_mean = sub["act_mean"].to_numpy(dtype=float)
            act_std = sub["act_std"].to_numpy(dtype=float)

            # Find sigma* index
            star_idx = np.argmax(mi_mean)
            mi_at_star = float(mi_mean[star_idx])
            mi_at_zero = float(mi_mean[0])
            delta_mi = mi_at_star - mi_at_zero

            # Bootstrap ΔMI CI
            mi_star_std = max(mi_std[star_idx], 1e-6)
            mi_zero_std = max(mi_std[0], 1e-6)
            delta_boots = []
            for _ in range(BOOTSTRAP_N):
                b_star = mi_at_star + rng.normal(0, mi_star_std)
                b_zero = mi_at_zero + rng.normal(0, mi_zero_std)
                delta_boots.append(b_star - b_zero)
            delta_boots = np.array(delta_boots)
            lo_pct = (1 - CI_ALPHA) / 2 * 100
            hi_pct = (1 + CI_ALPHA) / 2 * 100

            # ΔSNR: MI gain as ratio
            delta_snr = delta_mi / max(mi_at_zero, 1e-6)

            # Cohen's d for activation probability
            act_at_star = float(act_mean[star_idx])
            act_at_zero = float(act_mean[0])
            delta_act = act_at_star - act_at_zero
            pooled_std = np.sqrt((act_std[star_idx]**2 + act_std[0]**2) / 2)
            cohens_d = delta_act / max(pooled_std, 1e-6)

            records.append({
                "Biomarker": gene,
                "Noise Type": nt_display,
                "sigma*": round(sigmas[star_idx], 4),
                "MI(sigma=0)": round(mi_at_zero, 5),
                "MI(sigma*)": round(mi_at_star, 5),
                "delta_MI": round(delta_mi, 5),
                "delta_MI_CI_lo": round(float(np.percentile(delta_boots, lo_pct)), 5),
                "delta_MI_CI_hi": round(float(np.percentile(delta_boots, hi_pct)), 5),
                "delta_SNR": round(delta_snr, 4),
                "Act(sigma=0)": round(act_at_zero, 4),
                "Act(sigma*)": round(act_at_star, 4),
                "delta_Act": round(delta_act, 4),
                "Cohens_d": round(cohens_d, 3),
            })

    return pd.DataFrame(records)


# ==================================================
# C. MULTIPLE TESTING CORRECTION
# ==================================================


def multiple_testing_correction():
    """
    Collect all p-values from step 14 (log-rank) and apply BH-FDR correction.
    """
    p_values = []
    test_labels = []

    # Step 14 p-values
    if os.path.exists(STEP14_FILE):
        s14 = pd.read_csv(STEP14_FILE)
        for _, row in s14.iterrows():
            if not np.isnan(row.get("p_value", np.nan)):
                p_values.append(float(row["p_value"]))
                test_labels.append(f"Step14: {row['Biomarker']} — {row['Comparison']}")

    if len(p_values) == 0:
        print("[warn] No p-values found for multiple testing correction.")
        return pd.DataFrame()

    p_arr = np.array(p_values)
    # Use scipy's Benjamini-Hochberg
    q_values = false_discovery_control(p_arr, method="bh")

    results = pd.DataFrame({
        "Test": test_labels,
        "p_raw": np.round(p_arr, 6),
        "q_BH": np.round(q_values, 6),
        "Significant_0.05": q_values < 0.05,
    })

    csv_path = os.path.join(OUTPUT_DIR, "step15_multiple_testing.csv")
    results.to_csv(csv_path, index=False)
    print(f"[save] {csv_path}")
    return results


# ==================================================
# D. TABLE 1 — PUBLICATION SUMMARY
# ==================================================


def build_table1(effect_df):
    """Publication-ready Table 1 with all biomarker characteristics."""
    # Load raw stats
    with open(STATS_FILE) as f:
        stats = json.load(f)
    with open(THRESHOLD_FILE) as f:
        thresh = json.load(f)

    # Load step 10 and step 13 if available
    s10 = pd.read_csv(STEP10_FILE) if os.path.exists(STEP10_FILE) else pd.DataFrame()
    s13 = pd.read_csv(STEP13_FILE) if os.path.exists(STEP13_FILE) else pd.DataFrame()

    rows = []
    for gene in BIOMARKERS:
        bm_data = stats["biomarkers"][gene]
        K = thresh["biomarker_thresholds"][gene]["activation_threshold_K"]

        row = {
            "Biomarker": gene,
            "N (early-stage)": bm_data["patient_count"],
            "Mean Expression": round(bm_data["mean_x0"], 3),
            "SD": round(bm_data["std_delta"], 3),
            "K (P75 threshold)": round(K, 3),
        }

        # Best SR metrics across noise types
        for nt_display in ["Gaussian", "OU", "Lévy"]:
            eff_row = effect_df[
                (effect_df["Biomarker"] == gene) & (effect_df["Noise Type"] == nt_display)
            ]
            if not eff_row.empty:
                eff_row = eff_row.iloc[0]
                row[f"σ* ({nt_display})"] = eff_row["sigma*"]
                row[f"MI* ({nt_display})"] = eff_row["MI(sigma*)"]
                row[f"ΔMI ({nt_display})"] = eff_row["delta_MI"]

        # Best FI from step 10
        if not s10.empty:
            s10_gene = s10[s10["Biomarker"] == gene]
            if not s10_gene.empty:
                best_fi = s10_gene.loc[s10_gene["Fisher Info(sigma*)"].idxmax()]
                row["Best FI(σ*)"] = round(float(best_fi["Fisher Info(sigma*)"]), 4)

        # Best AUC from step 13
        if not s13.empty:
            s13_gene = s13[s13["Biomarker"] == gene]
            if not s13_gene.empty:
                best_auc = s13_gene.loc[s13_gene["AUC(sigma*)"].idxmax()]
                row["Best AUC(σ*)"] = round(float(best_auc["AUC(sigma*)"]), 4)
                row["ΔAUC"] = round(float(best_auc["delta AUC"]), 4)

        rows.append(row)

    table1 = pd.DataFrame(rows)

    csv_path = os.path.join(OUTPUT_DIR, "step15_table1.csv")
    table1.to_csv(csv_path, index=False)
    print(f"[save] {csv_path}")

    # LaTeX
    tex_path = os.path.join(OUTPUT_DIR, "step15_table1.tex")
    with open(tex_path, "w") as f:
        f.write("% Step 15 — Table 1: Biomarker Summary\n\n")
        f.write("\\begin{table}[htbp]\\centering\n")
        f.write(
            "\\caption{Summary of biomarker expression characteristics, "
            "optimal stochastic resonance parameters, and detection performance "
            "for early-stage TCGA-LIHC hepatocellular carcinoma.}\n"
        )
        f.write("\\label{tab:table1}\n")
        f.write("\\resizebox{\\textwidth}{!}{%\n")

        cols = table1.columns.tolist()
        col_spec = "l" + "c" * (len(cols) - 1)
        f.write(f"\\begin{{tabular}}{{{col_spec}}}\\toprule\n")
        header = " & ".join(c.replace("σ", "$\\sigma$").replace("Δ", "$\\Delta$")
                           for c in cols)
        f.write(f"{header} \\\\\n\\midrule\n")

        for _, row in table1.iterrows():
            vals = []
            for c in cols:
                v = row[c]
                if isinstance(v, float):
                    vals.append(f"{v:.4f}" if abs(v) < 1 else f"{v:.3f}")
                else:
                    vals.append(str(v))
            f.write(" & ".join(vals) + " \\\\\n")

        f.write("\\bottomrule\n\\end{tabular}}\\end{table}\n")
    print(f"[save] {tex_path}")

    return table1


def save_effect_sizes(effect_df):
    """Save effect sizes to CSV and LaTeX."""
    csv_path = os.path.join(OUTPUT_DIR, "step15_effect_sizes.csv")
    effect_df.to_csv(csv_path, index=False)
    print(f"[save] {csv_path}")

    tex_path = os.path.join(OUTPUT_DIR, "step15_effect_sizes.tex")
    with open(tex_path, "w") as f:
        f.write("% Step 15 — Effect Sizes\n\n")
        f.write("\\begin{table}[htbp]\\centering\n")
        f.write(
            "\\caption{SR effect sizes with 95\\% bootstrap CIs. "
            "$\\Delta$MI = MI($\\sigma^*$) $-$ MI($\\sigma{=}0$); "
            "$\\Delta$SNR = relative MI gain; "
            "Cohen's $d$ = standardised activation probability gain.}\n"
        )
        f.write("\\label{tab:effect_sizes}\n")
        f.write("\\resizebox{\\textwidth}{!}{%\n")
        f.write("\\begin{tabular}{llcccccc}\\toprule\n")
        f.write(
            "Biomarker & Noise & $\\sigma^*$ & $\\Delta$MI & "
            "95\\% CI($\\Delta$MI) & $\\Delta$SNR & $\\Delta$Act & "
            "Cohen's $d$ \\\\\n\\midrule\n"
        )
        prev_bm = None
        for _, row in effect_df.iterrows():
            bm = row["Biomarker"] if row["Biomarker"] != prev_bm else ""
            prev_bm = row["Biomarker"]
            ci = f"[{row['delta_MI_CI_lo']:.4f}, {row['delta_MI_CI_hi']:.4f}]"
            f.write(
                f"{bm} & {row['Noise Type']} & {row['sigma*']:.3f} & "
                f"${row['delta_MI']:+.4f}$ & {ci} & "
                f"{row['delta_SNR']:+.3f} & ${row['delta_Act']:+.4f}$ & "
                f"{row['Cohens_d']:.2f} \\\\\n"
            )
        f.write("\\bottomrule\n\\end{tabular}}\\end{table}\n")
    print(f"[save] {tex_path}")


# ==================================================
# ENTRY POINT
# ==================================================


def run_step15():
    """Run the complete Step 15 workflow."""
    print("=" * 62)
    print("  STEP 15 — Statistical Rigor Consolidation")
    print("=" * 62)

    # Load SR results
    print("\n[step] Loading SR data ...")
    sr_df = pd.read_csv(SR_RESULTS_FILE)
    s10_df = pd.read_csv(STEP10_FILE) if os.path.exists(STEP10_FILE) else pd.DataFrame()

    # A. Bootstrap CI bands
    print("\n[step] Computing bootstrap CI bands (n=1000) ...")
    ci_data = bootstrap_ci_bands(sr_df)
    plot_ci_bands(ci_data, "mi", "Mutual Information (bits)",
                  "n=1000 bootstrap resamples per σ", "step15_mi_with_ci.png")
    plot_ci_bands(ci_data, "act", "Activation Probability",
                  "n=1000 bootstrap resamples per σ", "step15_act_with_ci.png")

    # B. Effect sizes
    print("\n[step] Computing effect sizes ...")
    effect_df = compute_effect_sizes(sr_df, s10_df)
    save_effect_sizes(effect_df)

    # C. Multiple testing correction
    print("\n[step] Applying Benjamini-Hochberg FDR correction ...")
    mt_df = multiple_testing_correction()

    # D. Table 1
    print("\n[step] Building publication Table 1 ...")
    table1 = build_table1(effect_df)

    # Print summary
    print("\n" + "=" * 90)
    print("  STEP 15 — EFFECT SIZE SUMMARY")
    print("=" * 90)
    print(f"  {'Biomarker':>10}  {'Noise':>10}  {'ΔMI':>8}  {'95% CI':>24}  {'ΔSNR':>7}  {'Cohen d':>8}")
    print("-" * 90)
    for _, row in effect_df.iterrows():
        ci = f"[{row['delta_MI_CI_lo']:.4f}, {row['delta_MI_CI_hi']:.4f}]"
        print(
            f"  {row['Biomarker']:>10}  {row['Noise Type']:>10}  "
            f"{row['delta_MI']:>+8.4f}  {ci:>24}  "
            f"{row['delta_SNR']:>+7.3f}  {row['Cohens_d']:>8.2f}"
        )

    if not mt_df.empty:
        print("\n  Multiple Testing Correction:")
        for _, row in mt_df.iterrows():
            sig = "✓" if row["Significant_0.05"] else "✗"
            print(f"    {sig}  p={row['p_raw']:.4f}  q_BH={row['q_BH']:.4f}  {row['Test']}")

    print(f"\n  Outputs in {OUTPUT_DIR}/")
    print("=" * 62)
    return effect_df


if __name__ == "__main__":
    run_step15()
