"""
step6_enhanced_noise.py
========================
Enhanced noise analysis: Gaussian vs Ornstein-Uhlenbeck vs Levy-stable.

Replaces step6_add_noise.py.

Outputs
-------
  • noise_comparison_distributions.png   — histogram overlay across biomarkers/noise types
  • noise_autocorrelation.png            — ACF comparison (shows OU is correlated)
  • noise_effect_table.csv / .txt        — summary statistics
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
# NOISE MODELS
# ==================================================

NOISE_TYPES  = ["gaussian", "ou", "levy"]
NOISE_LABELS = {
    "gaussian": "Gaussian (white)",
    "ou":       "Ornstein-Uhlenbeck (colored / correlated)",
    "levy":     "Levy-stable (heavy-tailed)",
}
NOISE_COLORS = {
    "gaussian": "#2196F3",
    "ou":       "#E91E63",
    "levy":     "#F59E0B",
}

# ==================================================
# PATHS  —  edit if your folder layout differs
# ==================================================

BASE     = os.path.dirname(__file__)
OUT_DIR  = os.path.join(BASE, "output")
os.makedirs(OUT_DIR, exist_ok=True)

AFP_STATS_FILE = os.path.join(OUT_DIR, "tcga_lihc_afp_statistics.json")

OUT_HIST = os.path.join(OUT_DIR, "noise_comparison_distributions.png")
OUT_ACF  = os.path.join(OUT_DIR, "noise_autocorrelation.png")
OUT_TXT  = os.path.join(OUT_DIR, "noise_effect_table.txt")
OUT_CSV  = os.path.join(OUT_DIR, "noise_effect_table.csv")

# ==================================================
# LOAD DATA
# ==================================================

with open(AFP_STATS_FILE) as f:
    biomarker_data = json.load(f)

BIOMARKER_ORDER = ["AFP", "DKK1", "GPC3", "MDK"]
ROLE_LABELS = {
    "AFP": "Biological anchor",
    "DKK1": "Weak/borderline signal",
    "GPC3": "Non-AFP comparator",
    "MDK": "Structure/sensitivity comparator",
}

if "biomarkers" in biomarker_data:
    biomarker_values = {
        gene: np.array(biomarker_data["biomarkers"][gene]["values"], dtype=float)
        for gene in BIOMARKER_ORDER
    }
else:
    biomarker_values = {"AFP": np.array(biomarker_data["afp_values"], dtype=float)}
    BIOMARKER_ORDER = ["AFP"]

print("Biomarker samples loaded:")
for gene in BIOMARKER_ORDER:
    print(f"  {gene}: {len(biomarker_values[gene])}")

# ==================================================
# PARAMETERS
# ==================================================

SIGMA_VALUES = [1.0, 2.0, 3.0]
OU_TAU       = DEFAULT_OU_TAU
OU_STEP      = DEFAULT_OU_STEP
LEVY_ALPHA   = DEFAULT_LEVY_ALPHA
LEVY_BETA    = DEFAULT_LEVY_BETA
SEED         = 0

# ==================================================
# GENERATE NOISY SIGNALS + COLLECT STATS
# ==================================================

def _kurt(arr):
    """Excess kurtosis (0 = Gaussian)."""
    std = arr.std()
    if std == 0:
        return 0.0
    mu4 = np.mean((arr - arr.mean())**4)
    return mu4 / std**4 - 3.0


def robust_autocorrelation(arr: np.ndarray, max_lag: int) -> np.ndarray:
    """
    Winsorise extreme jumps before estimating ACF so Levy outliers do not
    dominate the normalisation.
    """
    centered = arr - np.median(arr)
    clip = np.percentile(np.abs(centered), 99)
    if clip > 0:
        centered = np.clip(centered, -clip, clip)

    acf = np.correlate(centered, centered, mode="full")
    acf = acf[len(acf) // 2 :]
    if np.isclose(acf[0], 0.0):
        return np.zeros(max_lag)
    return acf[:max_lag] / acf[0]

rows = []          # for table output
noisy = {}         # noisy[biomarker][noise_type][sigma] = x_noisy
noise_only = {}    # noise_only[noise_type] = noise array (at sigma=2)

for gene in BIOMARKER_ORDER:
    x = biomarker_values[gene]
    n = len(x)
    noisy[gene] = {}
    for nt in NOISE_TYPES:
        noisy[gene][nt] = {}
        for sigma in SIGMA_VALUES:
            eta = generate_noise(
                nt,
                n,
                sigma,
                tau=OU_TAU,
                h=OU_STEP,
                levy_alpha=LEVY_ALPHA,
                levy_beta=LEVY_BETA,
                seed=SEED,
            )
            x_noisy = x + eta
            noisy[gene][nt][sigma] = x_noisy

            rows.append({
                "biomarker":  gene,
                "noise_type": nt,
                "sigma":      sigma,
                "mean":       float(np.mean(x_noisy)),
                "std":        float(np.std(x_noisy)),
                "min":        float(np.min(x_noisy)),
                "max":        float(np.max(x_noisy)),
                "kurtosis":   float(_kurt(x_noisy)),
            })

for nt in NOISE_TYPES:
    n = len(biomarker_values[BIOMARKER_ORDER[0]])
    noise_only[nt] = generate_noise(
        nt,
        n,
        sigma=2.0,
        tau=OU_TAU,
        h=OU_STEP,
        levy_alpha=LEVY_ALPHA,
        levy_beta=LEVY_BETA,
        seed=SEED,
    )

# ==================================================
# SAVE TABLES
# ==================================================

header = "biomarker\tnoise_type\tsigma\tmean\tstd\tmin\tmax\tkurtosis"
lines  = [header]
for r in rows:
    lines.append(
        f"{r['biomarker']}\t{r['noise_type']}\t{r['sigma']:.2f}\t"
        f"{r['mean']:.4f}\t{r['std']:.4f}\t"
        f"{r['min']:.4f}\t{r['max']:.4f}\t{r['kurtosis']:.4f}"
    )

with open(OUT_TXT, "w") as f:
    f.write("\n".join(lines))

with open(OUT_CSV, "w") as f:
    f.write("biomarker,noise_type,sigma,mean,std,min,max,kurtosis\n")
    for r in rows:
        f.write(
            f"{r['biomarker']},{r['noise_type']},{r['sigma']:.6f},"
            f"{r['mean']:.6f},{r['std']:.6f},"
            f"{r['min']:.6f},{r['max']:.6f},{r['kurtosis']:.6f}\n"
        )

print(f"Tables saved → {OUT_TXT}, {OUT_CSV}")

# ==================================================
# FIGURE 1: DISTRIBUTION COMPARISON (histogram overlay)
# ==================================================

sigma_plot = 2.0
fig, axes = plt.subplots(len(BIOMARKER_ORDER), len(NOISE_TYPES),
                         figsize=(16, 14), sharey=False)
fig.suptitle(
    "Effect of Gaussian, OU, and Levy-stable Noise on LIHC Biomarker Signals\n"
    f"(sigma = {sigma_plot}; Levy alpha = {LEVY_ALPHA}, beta = {LEVY_BETA})",
    fontsize=14, fontweight="bold"
)

for row_idx, gene in enumerate(BIOMARKER_ORDER):
    x = biomarker_values[gene]
    for col_idx, nt in enumerate(NOISE_TYPES):
        ax = axes[row_idx, col_idx] if len(BIOMARKER_ORDER) > 1 else axes[col_idx]
        pooled = np.concatenate([x, noisy[gene][nt][sigma_plot]])
        lo, hi = np.percentile(pooled, [1, 99])
        bins = np.linspace(lo, hi, 30)
        ax.hist(x, bins=bins, alpha=0.5, color="gray", label=f"Clean {gene}")
        ax.hist(noisy[gene][nt][sigma_plot], bins=bins, alpha=0.6,
                color=NOISE_COLORS[nt], label=NOISE_LABELS[nt])
        ax.set_xlabel(f"{gene} Expression", fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.set_title(f"{gene}: {NOISE_LABELS[nt]}\n({ROLE_LABELS.get(gene, 'Biomarker')})",
                     fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.4)

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig(OUT_HIST, dpi=300)
print(f"Figure saved → {OUT_HIST}")

# ==================================================
# FIGURE 2: AUTO-CORRELATION COMPARISON
# Demonstrates that OU noise is temporally correlated
# whereas Gaussian noise is not.
# ==================================================

max_lag = 40
fig, ax = plt.subplots(figsize=(8, 5))

for nt in NOISE_TYPES:
    acf = robust_autocorrelation(noise_only[nt], max_lag)
    ax.plot(
        range(max_lag),
        acf[:max_lag],
        label=NOISE_LABELS[nt],
        color=NOISE_COLORS[nt],
        linewidth=2,
    )

ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xlabel("Lag", fontsize=12)
ax.set_ylabel("Normalised Autocorrelation", fontsize=12)
ax.set_title(
    "Autocorrelation of Noise Types\n"
    f"(OU: tau = {OU_TAU}; Levy: alpha = {LEVY_ALPHA}; white noises stay near zero lag)",
    fontsize=12,
)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(OUT_ACF, dpi=300)
print(f"Figure saved → {OUT_ACF}")

print("\n✓ step6_enhanced_noise.py complete.")
