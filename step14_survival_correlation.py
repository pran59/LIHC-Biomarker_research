"""
Step 14 — Survival / Clinical Outcome Correlation (TCGA)
=========================================================
Uses TCGA-LIHC clinical endpoints (OS) to test whether patients whose
biomarker expression sits in the SR-benefiting sub-threshold regime have
distinct outcomes.

Analysis scope: ALL stages (I–IV) for maximum statistical power.

For each biomarker the cohort is split into:
  A) Sub-threshold  (expression < K)  — the SR-benefiting group
  B) Supra-threshold (expression ≥ K) — the already-detectable group

Additionally, within the sub-threshold group a finer split tests:
  A1) SR-zone  (K − σ* ≤ expression < K)  — near-threshold, most SR-amplified
  A2) Deep sub-threshold (expression < K − σ*)

Kaplan-Meier curves and log-rank tests are computed for each split.

Outputs
-------
  output/step14_km_curves.png
  output/step14_km_sr_zone.png
  output/step14_survival_summary.csv
  output/step14_survival_summary.tex
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

try:
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
except ImportError:
    raise ImportError("lifelines is required for step 14. Install: pip install lifelines")

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
STEP10_FILE = os.path.join(OUTPUT_DIR, "step10_sr_peak_summary.csv")

BIOMARKERS = ["AFP", "GPC3", "DKK1", "MDK"]

# Dark theme
FIG_BG = "white"
AX_BG = "white"
GRID_C = "#cccccc"
TICK_C = "#333333"
SPINE_C = "#999999"

BM_COLORS = {"AFP": "#1f77b4", "GPC3": "#d62728", "DKK1": "#2ca02c", "MDK": "#ff7f0e"}
GROUP_COLORS = {"Sub-threshold": "#1f77b4", "Supra-threshold": "#d62728"}
ZONE_COLORS = {"SR zone": "#2ca02c", "Deep sub-threshold": "#9467bd", "Supra-threshold": "#d62728"}


# ==================================================
# DATA LOADING
# ==================================================


def build_survival_dataset():
    """
    Merge clinical (survival) and expression data for ALL patients with
    both endpoints available.  Returns a DataFrame with columns:
      patient_id, vital_status, time_days, event, stage, + one col per biomarker
    """
    # --- Clinical ---
    with open(CLINICAL_FILE) as f:
        clinical = json.load(f)

    clin_records = []
    for pt in clinical:
        pid = pt.get("submitter_id")
        dem = pt.get("demographic", {})
        vs = dem.get("vital_status", "")
        dtd = dem.get("days_to_death")
        diags = pt.get("diagnoses", [])
        dtlf = diags[0].get("days_to_last_follow_up") if diags else None
        stage = diags[0].get("ajcc_pathologic_stage", "") if diags else ""

        if vs not in ("Alive", "Dead"):
            continue

        event = 1 if vs == "Dead" else 0
        time_days = dtd if dtd is not None else dtlf
        if time_days is None or float(time_days) <= 0:
            continue

        clin_records.append({
            "patient_id": pid,
            "vital_status": vs,
            "time_days": float(time_days),
            "event": event,
            "stage": stage,
        })

    clin_df = pd.DataFrame(clin_records).drop_duplicates(subset="patient_id")
    print(f"[load] Clinical: {len(clin_df)} patients with OS data "
          f"(events={clin_df['event'].sum()})")

    # --- Expression ---
    expr_df = pd.read_csv(EXPRESSION_FILE, sep="\t")
    expr_df.set_index(expr_df.columns[0], inplace=True)

    # Map columns to patient IDs (tumour samples only)
    tumor_cols = [c for c in expr_df.columns if len(c) >= 15 and c[13:15] == "01"]
    col_to_pid = {c: c[:12] for c in tumor_cols}

    # Extract biomarker values per patient (average if multiple samples)
    for gene in BIOMARKERS:
        if gene not in expr_df.index:
            raise ValueError(f"Gene {gene} not in expression matrix")
        vals = expr_df.loc[gene, tumor_cols]
        gene_df = pd.DataFrame({"col": tumor_cols, gene: vals.values.astype(float)})
        gene_df["patient_id"] = gene_df["col"].map(col_to_pid)
        gene_df = gene_df.groupby("patient_id")[gene].mean().reset_index()
        clin_df = clin_df.merge(gene_df, on="patient_id", how="left")

    # Drop patients without expression
    clin_df = clin_df.dropna(subset=BIOMARKERS)
    print(f"[load] After expression merge: {len(clin_df)} patients "
          f"(events={clin_df['event'].sum()})")

    return clin_df


def load_thresholds_and_sigma_star():
    """Load K thresholds (step 4) and σ* values (step 10)."""
    with open(THRESHOLD_FILE) as f:
        k_data = json.load(f)
    thresholds = {
        bm: float(k_data["biomarker_thresholds"][bm]["activation_threshold_K"])
        for bm in BIOMARKERS
    }

    # σ* from step 10 (use best noise type = highest MI gain)
    sigma_stars = {}
    if os.path.exists(STEP10_FILE):
        s10 = pd.read_csv(STEP10_FILE)
        for bm in BIOMARKERS:
            sub = s10[s10["Biomarker"] == bm]
            if not sub.empty:
                best = sub.loc[sub["SR Gain delta MI"].idxmax()]
                sigma_stars[bm] = float(best["sigma*"])
            else:
                sigma_stars[bm] = 1.0
    else:
        sigma_stars = {bm: 1.0 for bm in BIOMARKERS}

    return thresholds, sigma_stars


# ==================================================
# ANALYSIS
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
    ax.grid(True, color=GRID_C, lw=0.5, linestyle="--", alpha=0.4)


def km_analysis(dataset, thresholds, sigma_stars):
    """
    Run Kaplan-Meier + log-rank for:
      1. Sub-threshold vs. supra-threshold (per biomarker)
      2. SR-zone vs. deep sub-threshold vs. supra-threshold (per biomarker)
    """
    records = []

    # --- Figure 1: Sub vs Supra threshold ---
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
    fig1.patch.set_facecolor(FIG_BG)
    fig1.suptitle(
        "Kaplan-Meier Survival: Sub-threshold vs Supra-threshold Expressors\n"
        "All TCGA-LIHC stages — split at activation threshold K",
        color="black", fontsize=13, fontweight="bold",
    )

    for idx, gene in enumerate(BIOMARKERS):
        ax = axes1.flat[idx]
        K = thresholds[gene]
        sub_mask = dataset[gene] < K
        supra_mask = ~sub_mask

        kmf = KaplanMeierFitter()

        for group_name, mask, color in [
            ("Sub-threshold", sub_mask, GROUP_COLORS["Sub-threshold"]),
            ("Supra-threshold", supra_mask, GROUP_COLORS["Supra-threshold"]),
        ]:
            grp = dataset[mask]
            if len(grp) < 3:
                continue
            kmf.fit(grp["time_days"], grp["event"], label=f"{group_name} (n={len(grp)})")
            kmf.plot_survival_function(ax=ax, color=color, lw=2, ci_alpha=0.15)

        # Log-rank test
        sub_grp = dataset[sub_mask]
        supra_grp = dataset[supra_mask]
        if len(sub_grp) >= 3 and len(supra_grp) >= 3:
            lr = logrank_test(
                sub_grp["time_days"], supra_grp["time_days"],
                sub_grp["event"], supra_grp["event"],
            )
            p_val = float(lr.p_value)
            ax.text(
                0.95, 0.95,
                f"Log-rank p = {p_val:.4f}" + (" *" if p_val < 0.05 else ""),
                transform=ax.transAxes, fontsize=9, color="#b38600",
                ha="right", va="top",
                bbox=dict(facecolor=AX_BG, edgecolor=SPINE_C, alpha=0.9),
            )
        else:
            p_val = np.nan

        records.append({
            "Biomarker": gene,
            "Comparison": "Sub vs Supra threshold",
            "N_sub": int(sub_mask.sum()),
            "N_supra": int(supra_mask.sum()),
            "Events_sub": int(dataset.loc[sub_mask, "event"].sum()),
            "Events_supra": int(dataset.loc[supra_mask, "event"].sum()),
            "K": round(K, 4),
            "p_value": round(p_val, 6) if not np.isnan(p_val) else np.nan,
        })

        ax.legend(fontsize=8, facecolor=AX_BG, labelcolor="black", edgecolor=SPINE_C)
        style_axes(ax, title=f"{gene} (K={K:.2f})",
                   xlabel="Days", ylabel="Survival Probability")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path1 = os.path.join(OUTPUT_DIR, "step14_km_curves.png")
    plt.savefig(path1, dpi=150, bbox_inches="tight", facecolor=FIG_BG)
    plt.close()
    print(f"[fig]  {path1}")

    # --- Figure 2: SR-zone vs deep sub-threshold ---
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    fig2.patch.set_facecolor(FIG_BG)
    fig2.suptitle(
        "Kaplan-Meier Survival: SR-Zone vs Deep Sub-threshold vs Supra\n"
        "SR zone = expression within σ* of threshold K",
        color="black", fontsize=13, fontweight="bold",
    )

    for idx, gene in enumerate(BIOMARKERS):
        ax = axes2.flat[idx]
        K = thresholds[gene]
        sigma_star = sigma_stars[gene]
        lower_bound = K - sigma_star

        sr_zone = (dataset[gene] >= lower_bound) & (dataset[gene] < K)
        deep_sub = dataset[gene] < lower_bound
        supra = dataset[gene] >= K

        kmf = KaplanMeierFitter()
        groups = [
            ("SR zone", sr_zone, ZONE_COLORS["SR zone"]),
            ("Deep sub-threshold", deep_sub, ZONE_COLORS["Deep sub-threshold"]),
            ("Supra-threshold", supra, ZONE_COLORS["Supra-threshold"]),
        ]

        for name, mask, color in groups:
            grp = dataset[mask]
            if len(grp) < 3:
                continue
            kmf.fit(grp["time_days"], grp["event"], label=f"{name} (n={len(grp)})")
            kmf.plot_survival_function(ax=ax, color=color, lw=2, ci_alpha=0.15)

        # Log-rank: SR-zone vs deep
        sr_grp = dataset[sr_zone]
        deep_grp = dataset[deep_sub]
        if len(sr_grp) >= 3 and len(deep_grp) >= 3:
            lr2 = logrank_test(
                sr_grp["time_days"], deep_grp["time_days"],
                sr_grp["event"], deep_grp["event"],
            )
            p_val2 = float(lr2.p_value)
            ax.text(
                0.95, 0.95,
                f"SR vs Deep: p = {p_val2:.4f}" + (" *" if p_val2 < 0.05 else ""),
                transform=ax.transAxes, fontsize=9, color="#2ca02c",
                ha="right", va="top",
                bbox=dict(facecolor=AX_BG, edgecolor=SPINE_C, alpha=0.9),
            )
        else:
            p_val2 = np.nan

        records.append({
            "Biomarker": gene,
            "Comparison": "SR-zone vs Deep sub-threshold",
            "N_sub": int(sr_zone.sum()),
            "N_supra": int(deep_sub.sum()),
            "Events_sub": int(dataset.loc[sr_zone, "event"].sum()),
            "Events_supra": int(dataset.loc[deep_sub, "event"].sum()),
            "K": round(K, 4),
            "p_value": round(p_val2, 6) if not np.isnan(p_val2) else np.nan,
        })

        ax.legend(fontsize=8, facecolor=AX_BG, labelcolor="black", edgecolor=SPINE_C)
        style_axes(ax, title=f"{gene} (K={K:.2f}, σ*={sigma_star:.2f})",
                   xlabel="Days", ylabel="Survival Probability")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path2 = os.path.join(OUTPUT_DIR, "step14_km_sr_zone.png")
    plt.savefig(path2, dpi=150, bbox_inches="tight", facecolor=FIG_BG)
    plt.close()
    print(f"[fig]  {path2}")

    return pd.DataFrame(records)


def save_tables(summary):
    """Save survival summary CSV and LaTeX."""
    csv_path = os.path.join(OUTPUT_DIR, "step14_survival_summary.csv")
    summary.to_csv(csv_path, index=False)
    print(f"[save] {csv_path}")

    tex_path = os.path.join(OUTPUT_DIR, "step14_survival_summary.tex")
    with open(tex_path, "w") as f:
        f.write("% Step 14 — Survival Analysis Summary\n\n")
        f.write("\\begin{table}[htbp]\\centering\n")
        f.write(
            "\\caption{Kaplan-Meier survival analysis: sub-threshold vs supra-threshold "
            "expression groups across all TCGA-LIHC stages. "
            "Log-rank $p$-values test the null hypothesis of equal survival distributions.}\n"
        )
        f.write("\\label{tab:survival}\n")
        f.write("\\begin{tabular}{llcccccc}\\toprule\n")
        f.write(
            "Biomarker & Comparison & $N_1$ & Events$_1$ & "
            "$N_2$ & Events$_2$ & $K$ & $p$ \\\\\n\\midrule\n"
        )
        for _, row in summary.iterrows():
            p_str = f"{row['p_value']:.4f}" if not np.isnan(row["p_value"]) else "---"
            f.write(
                f"{row['Biomarker']} & {row['Comparison']} & "
                f"{row['N_sub']} & {row['Events_sub']} & "
                f"{row['N_supra']} & {row['Events_supra']} & "
                f"{row['K']:.2f} & {p_str} \\\\\n"
            )
        f.write("\\bottomrule\n\\end{tabular}\\end{table}\n")
    print(f"[save] {tex_path}")


# ==================================================
# ENTRY POINT
# ==================================================


def run_step14():
    """Run the complete Step 14 workflow."""
    print("=" * 62)
    print("  STEP 14 — Survival / Clinical Outcome Correlation")
    print("=" * 62)

    print("\n[step] Building survival dataset (all stages) ...")
    dataset = build_survival_dataset()

    print("\n[step] Loading thresholds and σ* ...")
    thresholds, sigma_stars = load_thresholds_and_sigma_star()
    for bm in BIOMARKERS:
        print(f"  {bm}: K={thresholds[bm]:.4f}  σ*={sigma_stars[bm]:.3f}")

    print("\n[step] Running Kaplan-Meier analysis ...")
    summary = km_analysis(dataset, thresholds, sigma_stars)

    print("\n[step] Saving tables ...")
    save_tables(summary)

    # Print summary
    print("\n" + "=" * 80)
    print("  STEP 14 — SURVIVAL SUMMARY")
    print("=" * 80)
    for _, row in summary.iterrows():
        sig = " ***" if row["p_value"] < 0.001 else (" **" if row["p_value"] < 0.01 else (" *" if row["p_value"] < 0.05 else ""))
        print(
            f"  {row['Biomarker']:5s}  {row['Comparison']:35s}  "
            f"N=({row['N_sub']},{row['N_supra']})  p={row['p_value']:.4f}{sig}"
        )

    print(f"\n  Outputs in {OUTPUT_DIR}/")
    print("=" * 62)
    return summary


if __name__ == "__main__":
    run_step14()
