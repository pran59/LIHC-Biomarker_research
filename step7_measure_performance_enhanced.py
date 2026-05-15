"""
step7_measure_performance.py  [ENHANCED]
=========================================
Activation Probability vs Noise — Monte-Carlo SR Analysis.

Multi-biomarker version:
  • AFP   — biological anchor
  • DKK1  — weak / borderline signal
  • GPC3  — non-AFP LIHC comparator
  • MDK   — structure / stability comparator

Outputs
-------
  • sr_activation_by_noise_type.png      — full-signal activation curves, all biomarkers
  • sr_activation_subthreshold.png       — sub-threshold activation curves, all biomarkers
  • stochastic_resonance_table_<type>.csv
  • stochastic_resonance_combined.csv

Noise models:
  • Gaussian / AWGN
  • Ornstein-Uhlenbeck
  • Levy-stable
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from noise_utils import (
    DEFAULT_LEVY_ALPHA,
    DEFAULT_LEVY_BETA,
    DEFAULT_OU_STEP,
    DEFAULT_OU_TAU,
    generate_noise,
)

# ==================================================
# SELF-CONTAINED NOISE MODELS
# ==================================================

NOISE_LABELS = {
    "gaussian": "Gaussian (white)",
    "ou":       "Ornstein-Uhlenbeck (colored)",
    "levy":     "Levy-stable (heavy-tailed)",
}

NOISE_COLORS = {
    "gaussian": "#1f77b4",
    "ou":       "#2ca02c",
    "levy":     "#ff7f0e",
}

OU_TAU  = DEFAULT_OU_TAU
OU_STEP = DEFAULT_OU_STEP
LEVY_ALPHA = DEFAULT_LEVY_ALPHA
LEVY_BETA = DEFAULT_LEVY_BETA


# ==================================================
# FILE PATHS
# ==================================================

BASE    = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "output")
os.makedirs(OUT_DIR, exist_ok=True)

AFP_STATS_FILE = os.path.join(OUT_DIR, "tcga_lihc_afp_statistics.json")
THRESHOLD_FILE = os.path.join(OUT_DIR, "gene_activation_threshold_K.json")

OUT_COMBINED_CSV = os.path.join(OUT_DIR, "stochastic_resonance_combined.csv")
OUT_FIG_MAIN     = os.path.join(OUT_DIR, "sr_activation_by_noise_type.png")
OUT_FIG_SUB      = os.path.join(OUT_DIR, "sr_activation_subthreshold.png")


# ==================================================
# LOAD DATA
# ==================================================

with open(AFP_STATS_FILE) as f:
    biomarker_data = json.load(f)
with open(THRESHOLD_FILE) as f:
    k_data = json.load(f)

BIOMARKER_ORDER = ["AFP", "DKK1", "GPC3", "MDK"]
ROLE_LABELS = {
    "AFP": "Biological anchor",
    "DKK1": "Weak / borderline signal",
    "GPC3": "Non-AFP comparator",
    "MDK": "Structure / stability comparator",
}

if "biomarkers" in biomarker_data:
    biomarker_values = {
        gene: np.array(biomarker_data["biomarkers"][gene]["values"], dtype=float)
        for gene in BIOMARKER_ORDER
    }
else:
    biomarker_values = {"AFP": np.array(biomarker_data["afp_values"], dtype=float)}
    BIOMARKER_ORDER = ["AFP"]

if "biomarker_thresholds" in k_data:
    biomarker_thresholds = {
        gene: float(k_data["biomarker_thresholds"][gene]["activation_threshold_K"])
        for gene in BIOMARKER_ORDER
    }
else:
    biomarker_thresholds = {"AFP": float(k_data["activation_threshold_K"])}


# ==================================================
# MODEL PARAMETERS
# ==================================================

HILL_N          = 4
DECISION_THRESH = 0.5
NOISE_LEVELS    = np.linspace(0, 6, 40)
NUM_TRIALS      = 500
NOISE_TYPES     = ["gaussian", "ou", "levy"]


def hill(x_in, K, n=HILL_N):
    """Hill activation with absolute-value guard for noisy negative values."""
    xn = np.abs(x_in) ** n
    return xn / (K ** n + xn)


def run_activation_sweep(signal: np.ndarray, threshold_k: float) -> dict:
    """Return activation-probability curves for all noise types."""
    n_samples = len(signal)
    results = {}

    for nt in NOISE_TYPES:
        act_mean = []
        act_std = []

        for sigma in NOISE_LEVELS:
            trial_probs = []
            for trial in range(NUM_TRIALS):
                eta = generate_noise(
                    nt,
                    n_samples,
                    sigma,
                    tau=OU_TAU,
                    h=OU_STEP,
                    levy_alpha=LEVY_ALPHA,
                    levy_beta=LEVY_BETA,
                    seed=trial,
                )
                x_noisy = signal + eta
                y = hill(x_noisy, threshold_k)
                trial_probs.append(np.mean(y > DECISION_THRESH))

            arr = np.array(trial_probs)
            act_mean.append(arr.mean())
            act_std.append(arr.std())

        act_mean = np.array(act_mean)
        act_std = np.array(act_std)
        results[nt] = {
            "mean": act_mean,
            "std": act_std,
            "sigma_star": NOISE_LEVELS[np.argmax(act_mean)],
        }

    return results


def write_csv_outputs(results_full: dict, results_sub: dict) -> None:
    """Save per-noise and combined SR summaries for all biomarkers."""
    combined_rows = []

    for nt in NOISE_TYPES:
        csv_path = os.path.join(OUT_DIR, f"stochastic_resonance_table_{nt}.csv")
        with open(csv_path, "w") as f:
            f.write("biomarker,population,sigma,mean_activation,std_activation\n")

            for gene in BIOMARKER_ORDER:
                for population, source in [("full", results_full), ("sub_threshold", results_sub)]:
                    if gene not in source or nt not in source[gene]:
                        continue

                    r = source[gene][nt]
                    for sigma, mean_val, std_val in zip(NOISE_LEVELS, r["mean"], r["std"]):
                        f.write(
                            f"{gene},{population},{sigma:.6f},{mean_val:.6f},{std_val:.6f}\n"
                        )
                        combined_rows.append(
                            (gene, population, nt, sigma, mean_val, std_val)
                        )

    with open(OUT_COMBINED_CSV, "w") as f:
        f.write("biomarker,population,noise_type,sigma,mean_activation,std_activation\n")
        for gene, population, nt, sigma, mean_val, std_val in combined_rows:
            f.write(
                f"{gene},{population},{nt},{sigma:.6f},{mean_val:.6f},{std_val:.6f}\n"
            )


def plot_activation_panels(results: dict, outfile: str, population_label: str) -> None:
    """Plot one panel per biomarker, with all noise types overlaid."""
    ncols = 2
    nrows = int(np.ceil(len(BIOMARKER_ORDER) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 9), sharex=True, sharey=False)
    axes = np.atleast_1d(axes).ravel()

    fig.suptitle(
        f"Stochastic Resonance: Activation Probability vs Noise Intensity\n"
        f"({population_label}; shaded band = ±1 SD over {NUM_TRIALS} Monte-Carlo trials;\n"
        f"Levy alpha = {LEVY_ALPHA}, beta = {LEVY_BETA})",
        fontsize=13,
        fontweight="bold",
    )

    for idx, gene in enumerate(BIOMARKER_ORDER):
        ax = axes[idx]
        for nt in NOISE_TYPES:
            if gene not in results or nt not in results[gene]:
                continue

            r = results[gene][nt]
            ax.plot(NOISE_LEVELS, r["mean"], color=NOISE_COLORS[nt], lw=2, label=NOISE_LABELS[nt])
            ax.fill_between(
                NOISE_LEVELS,
                r["mean"] - r["std"],
                r["mean"] + r["std"],
                alpha=0.12,
                color=NOISE_COLORS[nt],
            )
            ax.axvline(r["sigma_star"], color=NOISE_COLORS[nt], linestyle=":", lw=1.0, alpha=0.7)

        ax.set_title(
            f"{gene}  |  K={biomarker_thresholds[gene]:.3f}\n({ROLE_LABELS.get(gene, 'Biomarker')})",
            fontsize=10,
        )
        ax.set_xlabel("Noise Intensity (σ)", fontsize=10)
        ax.set_ylabel("Activation Probability", fontsize=10)
        ax.grid(True, alpha=0.35)
        ax.legend(fontsize=8)

    for idx in range(len(BIOMARKER_ORDER), len(axes)):
        axes[idx].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(outfile, dpi=300)


# ==================================================
# MAIN ANALYSIS
# ==================================================

print("Biomarker activation analysis loaded:")
for gene in BIOMARKER_ORDER:
    x = biomarker_values[gene]
    K = biomarker_thresholds[gene]
    x_sub = x[x < K]
    x_supra = x[x >= K]
    print(
        f"  {gene}: N={len(x)}  |  K={K:.4f}  |  "
        f"sub-threshold={len(x_sub)} ({len(x_sub)/len(x):.1%})  |  "
        f"supra-threshold={len(x_supra)} ({len(x_supra)/len(x):.1%})"
    )

results_full = {}
results_sub = {}

for gene in BIOMARKER_ORDER:
    x = biomarker_values[gene]
    K = biomarker_thresholds[gene]
    x_sub = x[x < K]

    print(f"\nRunning Monte-Carlo sweep on full signal for {gene} …")
    results_full[gene] = run_activation_sweep(x, K)
    for nt in NOISE_TYPES:
        r = results_full[gene][nt]
        print(
            f"  {NOISE_LABELS[nt]:30s}  σ* = {r['sigma_star']:.3f}  "
            f"peak AP = {r['mean'].max():.4f}"
        )

    print(f"Running sub-threshold analysis for {gene} …")
    if len(x_sub) == 0:
        results_sub[gene] = {}
        print("  No sub-threshold samples — skipping.")
    else:
        results_sub[gene] = run_activation_sweep(x_sub, K)
        for nt in NOISE_TYPES:
            r = results_sub[gene][nt]
            print(f"  {NOISE_LABELS[nt]:30s}  σ*(sub) = {r['sigma_star']:.3f}")

write_csv_outputs(results_full, results_sub)
print(f"\nCSVs saved to {OUT_DIR}")

plot_activation_panels(results_full, OUT_FIG_MAIN, "Full signal")
print(f"Figure saved → {OUT_FIG_MAIN}")

plot_activation_panels(results_sub, OUT_FIG_SUB, "Sub-threshold regime")
print(f"Figure saved → {OUT_FIG_SUB}")


# ==================================================
# PRINT SUMMARY TABLE
# ==================================================

print("\n" + "=" * 70)
print("ACTIVATION PROBABILITY — SUMMARY")
print("=" * 70)
for gene in BIOMARKER_ORDER:
    K = biomarker_thresholds[gene]
    print(f"\n{gene}  |  K = {K:.4f}  |  {ROLE_LABELS.get(gene, 'Biomarker')}")
    print(f"{'Noise Type':28s}  {'σ*':6s}  {'Peak AP':8s}  {'AP at σ=0':10s}")
    print("-" * 70)
    for nt in NOISE_TYPES:
        r = results_full[gene][nt]
        print(
            f"{NOISE_LABELS[nt]:28s}  "
            f"{r['sigma_star']:6.3f}  "
            f"{r['mean'].max():8.4f}  "
            f"{r['mean'][0]:10.4f}"
        )

print(f"\n  Hill coefficient n       = {HILL_N}")
print(f"  Monte-Carlo trials       = {NUM_TRIALS}")
print(f"  Decision threshold       = {DECISION_THRESH}")
print(f"  Levy alpha / beta        = {LEVY_ALPHA} / {LEVY_BETA}")
print("\n✓ step7_measure_performance.py [enhanced] complete.")
