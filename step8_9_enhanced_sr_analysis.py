"""
step8_9_enhanced_sr_analysis.py
================================
Multi-biomarker stochastic-resonance analysis for:
  • AFP   — biological anchor
  • DKK1  — weak / borderline signal
  • GPC3  — non-AFP LIHC comparator
  • MDK   — structure / stability comparator

For each biomarker and each noise model (AWGN vs OU vs Levy-stable), the script computes:
  1. Activation Probability   (ensemble mean ± SD over the empirical cohort)
  2. Mutual Information       (bits, ensemble mean ± SD over the empirical cohort)
  3. Fisher Information       (output-level, biophysical approximation)

Fixes applied vs previous version
-----------------------------------
  FIX 1 — OU diffusion term aligned to a = exp(-h/tau) form in both
           generate_ou_vector and generate_ou_matrix, matching Methods text
           and corrected step6:  diffusion = sigma * sqrt(1 - a²)

  FIX 2 — FI seed formula replaced with hash-based scheme to eliminate
           seed collisions across (gene, noise_type, sigma_idx) combinations:
           fi_seed = int(hash((gene, nt, sigma_idx)) % (2**31))

  FIX 3 — Noise type key renamed from "gaussian" to "awgn" in NOISE_TYPES,
           NOISE_LABELS, NOISE_COLORS, and both dispatch functions, matching
           corrected step6 and ensuring CSV rows are consistent downstream.

  FIX 4 — Legend labels in all figures now correctly display
           "AWGN (white Gaussian)" and "OU (colored / correlated)"
           by using NOISE_LABELS[nt] consistently everywhere.

  EXTENSION — Levy-stable heavy-tailed noise is now included alongside
           AWGN and OU using a shared CMS-based generator.

Output-level Fisher Information model
--------------------------------------
FI is evaluated at the nonlinear gene-circuit output, not the raw input.

For each biomarker a weak-signal operating point is defined:
    theta_ref = mean(x_sub)   where x_sub = values below threshold K

A short stochastic output trajectory is simulated:
    y_t = Hill(theta_ref + eta_t; K, n)
    z_t = 1[y_t > d]
    C   = sum_t z_t

FI is computed under a Gaussian closure of the activation count C:
    I_out(theta; sigma) ~= (d mu_C / d theta)^2 / v_eff
                         + 0.5 * (d v_C / d theta)^2 / v_eff^2
    v_eff = Var(C) + v0

This second-order closure is used as the same output-space approximation for
all noise types, including Levy perturbations.

Outputs
-------
  • sr_activation_noise.png
  • sr_mutual_info_noise.png
  • sr_fisher_info_noise.png
  • sr_metrics_combined.png
  • sr_full_results.csv
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score

from noise_utils import (
    DEFAULT_LEVY_ALPHA,
    DEFAULT_LEVY_BETA,
    DEFAULT_OU_STEP,
    DEFAULT_OU_TAU,
    generate_noise,
)


# ==================================================
# NOISE MODELS
# ==================================================

# FIX 3: key is "awgn" throughout (was "gaussian")
NOISE_TYPES = ["awgn", "ou", "levy"]
NOISE_LABELS = {
    "awgn": "AWGN (white Gaussian)",
    "ou":   "OU (colored / correlated)",
    "levy": "Levy-stable (heavy-tailed)",
}
NOISE_COLORS = {
    "awgn": "#2196F3",
    "ou":   "#E91E63",
    "levy": "#F59E0B",
}

OU_TAU = DEFAULT_OU_TAU
OU_STEP = DEFAULT_OU_STEP
LEVY_ALPHA = DEFAULT_LEVY_ALPHA
LEVY_BETA = DEFAULT_LEVY_BETA


def generate_noise_vector(
    noise_type: str, n: int, sigma: float, seed: int = 0
) -> np.ndarray:
    """Dispatch vector noise generation."""
    return generate_noise(
        noise_type,
        n,
        sigma,
        tau=OU_TAU,
        h=OU_STEP,
        levy_alpha=LEVY_ALPHA,
        levy_beta=LEVY_BETA,
        seed=seed,
    )


def generate_noise_matrix(
    noise_type: str,
    shape: tuple,
    sigma: float,
    seed: int = 0,
) -> np.ndarray:
    """Dispatch matrix noise generation."""
    return generate_noise(
        noise_type,
        shape,
        sigma,
        tau=OU_TAU,
        h=OU_STEP,
        levy_alpha=LEVY_ALPHA,
        levy_beta=LEVY_BETA,
        seed=seed,
    )


# ==================================================
# PATHS
# ==================================================

BASE    = os.path.dirname(__file__)
OUT_DIR = os.path.join(BASE, "output")
os.makedirs(OUT_DIR, exist_ok=True)

STATS_FILE     = os.path.join(OUT_DIR, "tcga_lihc_afp_statistics.json")
THRESHOLD_FILE = os.path.join(OUT_DIR, "gene_activation_threshold_K.json")

OUT_ACT      = os.path.join(OUT_DIR, "sr_activation_noise.png")
OUT_MI       = os.path.join(OUT_DIR, "sr_mutual_info_noise.png")
OUT_FI       = os.path.join(OUT_DIR, "sr_fisher_info_noise.png")
OUT_COMBINED = os.path.join(OUT_DIR, "sr_metrics_combined.png")
OUT_CSV      = os.path.join(OUT_DIR, "sr_full_results.csv")


# ==================================================
# LOAD DATA
# ==================================================

with open(STATS_FILE) as f:
    biomarker_data = json.load(f)
with open(THRESHOLD_FILE) as f:
    threshold_data = json.load(f)

BIOMARKER_ORDER = ["AFP", "DKK1", "GPC3", "MDK"]
ROLE_LABELS = {
    "AFP":  "Biological anchor",
    "DKK1": "Weak / borderline signal",
    "GPC3": "Non-AFP comparator",
    "MDK":  "Structure / stability comparator",
}

if "biomarkers" in biomarker_data:
    biomarker_values = {
        gene: np.array(biomarker_data["biomarkers"][gene]["values"], dtype=float)
        for gene in BIOMARKER_ORDER
    }
else:
    biomarker_values = {"AFP": np.array(biomarker_data["afp_values"], dtype=float)}
    BIOMARKER_ORDER  = ["AFP"]

if "biomarker_thresholds" in threshold_data:
    biomarker_thresholds = {
        gene: float(threshold_data["biomarker_thresholds"][gene]["activation_threshold_K"])
        for gene in BIOMARKER_ORDER
    }
else:
    biomarker_thresholds = {"AFP": float(threshold_data["activation_threshold_K"])}


# ==================================================
# MODEL PARAMETERS
# ==================================================

HILL_N           = 4
DEC_THRESH       = 0.5
NUM_TRIALS       = 200
NOISE_LEVELS     = np.linspace(0, 6, 40)
MI_BINS          = 12

FI_WINDOW_STEPS  = 50
FI_NUM_TRIALS    = 5000
FI_EPS_SCALE     = 0.10
FI_VARIANCE_FLOOR = 0.5


# ==================================================
# HELPER FUNCTIONS
# ==================================================

def hill(x_in: np.ndarray, threshold_k: float, n: int = HILL_N) -> np.ndarray:
    """Hill activation with absolute-value guard for noisy negative inputs."""
    xn = np.abs(x_in) ** n
    return xn / (threshold_k ** n + xn)


def discretize(values: np.ndarray, bins: int = MI_BINS) -> np.ndarray:
    _, edges = np.histogram(values, bins=bins)
    return np.digitize(values, edges[:-1])


def mutual_information_bits(x_disc: np.ndarray, y_disc: np.ndarray) -> float:
    return mutual_info_score(x_disc, y_disc) / np.log(2.0)


def weak_signal_reference(
    signal: np.ndarray, threshold_k: float
) -> tuple:
    """
    Return weak-signal operating point theta_ref, finite-difference epsilon,
    and number of sub-threshold samples used to define it.
    """
    sub_signal = signal[signal < threshold_k]
    if len(sub_signal) == 0:
        sub_signal = signal

    theta_ref = float(np.mean(sub_signal))
    spread    = float(np.std(sub_signal))
    epsilon   = max(0.05, FI_EPS_SCALE * spread)
    return theta_ref, epsilon, len(sub_signal)


def activation_count_stats(
    theta: float, threshold_k: float, eta: np.ndarray
) -> tuple:
    """
    Simulate noisy output trajectory and return mean and variance of
    the activation count C over the observation window.
    """
    y      = hill(theta + eta, threshold_k)
    z      = (y > DEC_THRESH).astype(float)
    counts = z.sum(axis=1)
    return float(counts.mean()), float(counts.var(ddof=1))


def output_fisher_information(
    theta_ref: float,
    threshold_k: float,
    sigma: float,
    noise_type: str,
    epsilon: float,
    seed: int = 0,
) -> float:
    """
    Output-level Fisher Information under a Gaussian closure of activation
    count C over a short observation window:

        I_out ~= (d mu_C / d theta)^2 / v_eff
               + 0.5 * (d v_C / d theta)^2 / v_eff^2
        v_eff  = Var(C) + FI_VARIANCE_FLOOR

    The same noise matrix is reused for all three theta perturbations
    (common random numbers) to reduce finite-difference variance.
    """
    eta = generate_noise_matrix(
        noise_type,
        (FI_NUM_TRIALS, FI_WINDOW_STEPS),
        sigma,
        seed=seed,
    )

    mean_minus, var_minus = activation_count_stats(theta_ref - epsilon, threshold_k, eta)
    mean_mid,   var_mid   = activation_count_stats(theta_ref,           threshold_k, eta)
    mean_plus,  var_plus  = activation_count_stats(theta_ref + epsilon, threshold_k, eta)

    d_mean   = (mean_plus - mean_minus) / (2.0 * epsilon)
    d_var    = (var_plus  - var_minus)  / (2.0 * epsilon)
    var_eff  = var_mid + FI_VARIANCE_FLOOR

    return float(
        (d_mean ** 2) / var_eff
        + 0.5 * (d_var ** 2) / (var_eff ** 2)
    )


# ==================================================
# MAIN SWEEP
# ==================================================

results = {
    gene: {
        nt: {"act_mean": [], "act_std": [],
             "mi_mean":  [], "mi_std":  [],
             "fi":       []}
        for nt in NOISE_TYPES
    }
    for gene in BIOMARKER_ORDER
}

fi_metadata = {}

for gene_idx, gene in enumerate(BIOMARKER_ORDER):
    signal      = biomarker_values[gene]
    threshold_k = biomarker_thresholds[gene]
    theta_ref, epsilon, sub_count = weak_signal_reference(signal, threshold_k)

    fi_metadata[gene] = {
        "theta_ref":     theta_ref,
        "epsilon":       epsilon,
        "sub_count":     sub_count,
        "patient_count": len(signal),
        "threshold_k":   threshold_k,
    }

    print(
        f"\nBiomarker: {gene}  |  N={len(signal)}  |  K={threshold_k:.4f}"
        f"  |  theta_ref={theta_ref:.4f}  |  sub-threshold={sub_count}"
    )

    for noise_idx, nt in enumerate(NOISE_TYPES):
        print(f"  Noise type: {NOISE_LABELS[nt]}")

        for sigma_idx, sigma in enumerate(NOISE_LEVELS):
            act_trials = []
            mi_trials  = []

            for trial in range(NUM_TRIALS):
                eta     = generate_noise_vector(nt, len(signal), sigma, seed=trial)
                x_noisy = signal + eta
                y       = hill(x_noisy, threshold_k)

                act_trials.append(np.mean(y > DEC_THRESH))

                x_disc = discretize(x_noisy)
                y_bin  = (y > DEC_THRESH).astype(int)
                mi_trials.append(mutual_information_bits(x_disc, y_bin))

            # FIX 2: hash-based seed — no collisions across (gene, nt, sigma_idx)
            fi_seed = int(hash((gene, nt, sigma_idx)) % (2 ** 31))

            results[gene][nt]["act_mean"].append(np.mean(act_trials))
            results[gene][nt]["act_std"].append(np.std(act_trials))
            results[gene][nt]["mi_mean"].append(np.mean(mi_trials))
            results[gene][nt]["mi_std"].append(np.std(mi_trials))
            results[gene][nt]["fi"].append(
                output_fisher_information(
                    theta_ref, threshold_k, sigma, nt, epsilon, seed=fi_seed
                )
            )

            if (sigma_idx + 1) % 10 == 0:
                print(
                    f"    sigma={sigma:.2f}"
                    f"  AP={results[gene][nt]['act_mean'][-1]:.3f}"
                    f"  MI={results[gene][nt]['mi_mean'][-1]:.4f}"
                    f"  FI={results[gene][nt]['fi'][-1]:.4f}"
                )

        for key in results[gene][nt]:
            results[gene][nt][key] = np.array(results[gene][nt][key], dtype=float)

print("\n✓ Multi-biomarker SR sweep complete.")


# ==================================================
# SAVE CSV
# ==================================================

with open(OUT_CSV, "w") as f:
    f.write(
        "biomarker,noise_type,sigma,act_mean,act_std,mi_mean,mi_std,"
        "fisher_info,threshold_k,fi_theta_ref,fi_epsilon,fi_sub_count,"
        "fi_window_steps,fi_num_trials,fi_variance_floor\n"
    )
    for gene in BIOMARKER_ORDER:
        meta = fi_metadata[gene]
        for nt in NOISE_TYPES:
            for idx, sigma in enumerate(NOISE_LEVELS):
                f.write(
                    f"{gene},{nt},{sigma:.6f},"
                    f"{results[gene][nt]['act_mean'][idx]:.6f},"
                    f"{results[gene][nt]['act_std'][idx]:.6f},"
                    f"{results[gene][nt]['mi_mean'][idx]:.6f},"
                    f"{results[gene][nt]['mi_std'][idx]:.6f},"
                    f"{results[gene][nt]['fi'][idx]:.6f},"
                    f"{meta['threshold_k']:.6f},{meta['theta_ref']:.6f},"
                    f"{meta['epsilon']:.6f},{meta['sub_count']},"
                    f"{FI_WINDOW_STEPS},{FI_NUM_TRIALS},{FI_VARIANCE_FLOOR:.6f}\n"
                )
print(f"CSV saved -> {OUT_CSV}")


# ==================================================
# PLOTTING HELPERS
# ==================================================

def errorband(ax, x_vals, y_mean, y_std, color, label):
    ax.plot(x_vals, y_mean, "o-", color=color, label=label, ms=3, lw=1.8)
    ax.fill_between(
        x_vals, y_mean - y_std, y_mean + y_std, alpha=0.15, color=color
    )


def biomarker_panel_title(gene: str) -> str:
    meta = fi_metadata[gene]
    return (
        f"{gene} | {ROLE_LABELS[gene]}\n"
        f"K={meta['threshold_k']:.2f}, theta_ref={meta['theta_ref']:.2f}"
    )


def plot_metric_grid(
    outfile: str,
    metric_key: str,
    ylabel: str,
    title: str,
    use_errorband: bool = False,
    std_key: str = None,
) -> None:
    """Plot one panel per biomarker for a single metric."""
    ncols = 2
    nrows = int(np.ceil(len(BIOMARKER_ORDER) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 9), sharex=True)
    axes = np.atleast_1d(axes).ravel()

    # FIX 4: collect legend handles from NOISE_LABELS to guarantee correct labels
    legend_handles = []
    legend_labels  = []

    for idx, gene in enumerate(BIOMARKER_ORDER):
        ax = axes[idx]
        for nt in NOISE_TYPES:
            y_vals = results[gene][nt][metric_key]
            if use_errorband and std_key is not None:
                errorband(
                    ax, NOISE_LEVELS, y_vals,
                    results[gene][nt][std_key],
                    NOISE_COLORS[nt], NOISE_LABELS[nt],
                )
            else:
                ax.plot(
                    NOISE_LEVELS, y_vals,
                    color=NOISE_COLORS[nt], lw=2,
                    label=NOISE_LABELS[nt],
                )
            # collect once from the first biomarker panel
            if idx == 0:
                legend_handles.append(ax.lines[-1])
                legend_labels.append(NOISE_LABELS[nt])

        ax.set_title(biomarker_panel_title(gene), fontsize=10)
        ax.set_xlabel("Noise Intensity (sigma)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, alpha=0.35)

    for idx in range(len(BIOMARKER_ORDER), len(axes)):
        axes[idx].axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.0)
    # FIX 4: use legend_labels drawn directly from NOISE_LABELS dict values
    fig.legend(
        legend_handles, legend_labels,
        loc="upper center", ncol=2, frameon=False,
        bbox_to_anchor=(0.5, 0.955),
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(outfile, dpi=300)


# ==================================================
# FIGURES
# ==================================================

plot_metric_grid(
    OUT_ACT,
    metric_key="act_mean",
    ylabel="Activation Probability",
    title="Activation Probability vs sigma",
    use_errorband=True,
    std_key="act_std",
)
print(f"Figure saved -> {OUT_ACT}")

plot_metric_grid(
    OUT_MI,
    metric_key="mi_mean",
    ylabel="Mutual Information (bits)",
    title="Mutual Information vs sigma",
    use_errorband=True,
    std_key="mi_std",
)
print(f"Figure saved -> {OUT_MI}")

plot_metric_grid(
    OUT_FI,
    metric_key="fi",
    ylabel="Output Fisher Information",
    title=(
        "Output-Level Fisher Information vs sigma\n"
        f"(activation-count window T={FI_WINDOW_STEPS}, OU tau={OU_TAU}, "
        f"Levy alpha={LEVY_ALPHA})"
    ),
)
print(f"Figure saved -> {OUT_FI}")

# Combined 3-panel figure
fig, axes = plt.subplots(len(BIOMARKER_ORDER), 3, figsize=(16, 14), sharex=True)
fig.suptitle(
    "Stochastic Resonance Analysis — Multi-Biomarker AWGN vs OU vs Levy",
    fontsize=15, fontweight="bold",
)

column_titles = [
    "Activation Probability",
    "Mutual Information (bits)",
    "Output Fisher Information",
]

for row, gene in enumerate(BIOMARKER_ORDER):
    for col, metric_key in enumerate(["act_mean", "mi_mean", "fi"]):
        ax = axes[row, col]
        for nt in NOISE_TYPES:
            if metric_key in ("act_mean", "mi_mean"):
                std_key = "act_std" if metric_key == "act_mean" else "mi_std"
                errorband(
                    ax, NOISE_LEVELS,
                    results[gene][nt][metric_key],
                    results[gene][nt][std_key],
                    NOISE_COLORS[nt],
                    # FIX 4: label drawn from NOISE_LABELS, not hardcoded string
                    NOISE_LABELS[nt],
                )
            else:
                ax.plot(
                    NOISE_LEVELS,
                    results[gene][nt][metric_key],
                    color=NOISE_COLORS[nt],
                    lw=2,
                    label=NOISE_LABELS[nt],
                )

        if row == 0:
            ax.set_title(column_titles[col], fontsize=12)

        ax.set_xlabel("Noise Intensity (sigma)", fontsize=10)
        ax.grid(True, alpha=0.35)

        ylabel_map = {
            0: f"{gene}\nActivation",
            1: f"{gene}\nMI (bits)",
            2: f"{gene}\nFI",
        }
        ax.set_ylabel(ylabel_map[col], fontsize=10)

# FIX 4: legend pulled from axes[0,0] — labels guaranteed correct via NOISE_LABELS
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="upper center", ncol=2, frameon=False,
    bbox_to_anchor=(0.5, 0.975),
)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(OUT_COMBINED, dpi=300)
print(f"Combined figure saved -> {OUT_COMBINED}")


# ==================================================
# OPTIMAL SIGMA SUMMARY
# ==================================================

print("\n" + "=" * 86)
print("OPTIMAL NOISE INTENSITY (sigma*) BY BIOMARKER")
print("=" * 86)
print(
    f"{'Biomarker':10s}  {'Noise':24s}  "
    f"{'sigma*(AP)':10s}  {'sigma*(MI)':10s}  {'sigma*(FI)':10s}"
)
print("-" * 86)

for gene in BIOMARKER_ORDER:
    for nt in NOISE_TYPES:
        sigma_ap = NOISE_LEVELS[np.argmax(results[gene][nt]["act_mean"])]
        sigma_mi = NOISE_LEVELS[np.argmax(results[gene][nt]["mi_mean"])]
        sigma_fi = NOISE_LEVELS[np.argmax(results[gene][nt]["fi"])]
        print(
            f"{gene:10s}  {NOISE_LABELS[nt]:24s}  "
            f"{sigma_ap:10.3f}  {sigma_mi:10.3f}  {sigma_fi:10.3f}"
        )

print(f"\nLevy alpha / beta = {LEVY_ALPHA} / {LEVY_BETA}")

print(
    "\nFI is computed at the nonlinear gene-circuit output, not the raw noisy input."
    f"\ntheta_ref = mean(sub-threshold signal) per biomarker;"
    f" activation-count window T = {FI_WINDOW_STEPS} steps."
    f"\nVariance floor v0 = {FI_VARIANCE_FLOOR:.2f} regularizes near-deterministic switching."
)
print("\n✓ step8_9_enhanced_sr_analysis.py complete.")
