"""
STEP 5: Synthetic Gene Circuit Model
Hill-function based gene activation using LIHC biomarker signals.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==================================================
# FILE PATHS
# ==================================================

AFP_STATS_FILE = "/Users/pranitabarik/Desktop/Cancer_Research/Coding/output/tcga_lihc_afp_statistics.json"
THRESHOLD_FILE = "/Users/pranitabarik/Desktop/Cancer_Research/Coding/output/gene_activation_threshold_K.json"

# FIX 5: No spaces in filename
OUTPUT_FIGURE = "/Users/pranitabarik/Desktop/Cancer_Research/Coding/output/Synthetic_Gene_Circuit_fig_1.png"

# ==================================================
# STEP 1: LOAD BIOMARKER DATA
# ==================================================

with open(AFP_STATS_FILE, "r") as f:
    biomarker_data = json.load(f)

BIOMARKER_ORDER = ["AFP", "DKK1", "GPC3", "MDK"]

# NOTE: DKK1 role label is a working assumption — validate against
# expression stats (mean, std, SNR) before finalizing for the paper.
ROLE_LABELS = {
    "AFP":  "Biological anchor",
    "DKK1": "Candidate weak/borderline signal (validate)",
    "GPC3": "Non-AFP comparator",
    "MDK":  "Structure/sensitivity comparator",
}

if "biomarkers" in biomarker_data:
    # FIX 2: Guard against missing gene keys with a descriptive error
    biomarker_values = {}
    for gene in BIOMARKER_ORDER:
        if gene not in biomarker_data["biomarkers"]:
            raise KeyError(
                f"Gene '{gene}' not found in biomarker stats file. "
                "Check Step 3 output — it may have been excluded due to all-NaN values."
            )
        raw = np.array(biomarker_data["biomarkers"][gene]["values"], dtype=float)

        # FIX 1: Strip NaNs at load time, report count
        nan_count = int(np.sum(np.isnan(raw)))
        clean = raw[~np.isnan(raw)]
        if nan_count > 0:
            print(f"  [{gene}] Dropped {nan_count} NaN value(s) before Hill computation.")

        # FIX 8: Assert all values are non-negative (log2(RSEM+1) guarantee)
        if np.any(clean < 0):
            raise ValueError(
                f"Negative expression values found for {gene}. "
                "Hill function requires x >= 0. Check expression data source."
            )

        biomarker_values[gene] = clean
else:
    print("WARNING: Multi-biomarker block not found. Falling back to AFP only.")
    biomarker_values = {"AFP": np.array(biomarker_data["afp_values"], dtype=float)}
    BIOMARKER_ORDER = ["AFP"]

print("Biomarker signals loaded")
for gene in BIOMARKER_ORDER:
    print(f"  {gene}: {len(biomarker_values[gene])} samples (post NaN-filter)")

# ==================================================
# STEP 2: LOAD ACTIVATION THRESHOLDS K
# ==================================================

with open(THRESHOLD_FILE, "r") as f:
    k_data = json.load(f)

if "biomarker_thresholds" in k_data:
    biomarker_thresholds = {
        gene: float(k_data["biomarker_thresholds"][gene]["activation_threshold_K"])
        for gene in BIOMARKER_ORDER
    }
else:
    biomarker_thresholds = {"AFP": float(k_data["activation_threshold_K"])}

print("Activation thresholds loaded")
for gene in BIOMARKER_ORDER:
    print(f"  {gene}: K = {biomarker_thresholds[gene]:.4f}")

# ==================================================
# STEP 3: DEFINE HILL FUNCTION
# ==================================================

def hill_function(x, K, n):
    """
    Hill function for gene activation.
    x : input signal (must be >= 0)
    K : half-activation threshold (75th percentile of early-stage expression)
    n : Hill coefficient (cooperativity)
    Returns values in [0, 1].
    """
    return (x**n) / (K**n + x**n)

# ==================================================
# STEP 4: COMPUTE GENE ACTIVATION
# ==================================================

# Hill coefficient n=4 used here; literature justification or sensitivity
# analysis across n=2,3,4 should be included before paper submission.
# n=1 or 2 is more typical for single TF binding; n=4 implies high cooperativity.
n = 4

biomarker_activation = {}
for gene in BIOMARKER_ORDER:
    x = biomarker_values[gene]
    K = biomarker_thresholds[gene]
    biomarker_activation[gene] = hill_function(x, K, n)
    print(f"{gene} activation computed: "
          f"min={biomarker_activation[gene].min():.4f}, "
          f"max={biomarker_activation[gene].max():.4f}, "
          f"mean={biomarker_activation[gene].mean():.4f}")

# ==================================================
# STEP 5: VISUALIZE RESPONSE
# ==================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("Synthetic Gene Circuit Response by Biomarker\n"
             f"(Hill coefficient n={n}, threshold K = 75th percentile)",
             fontsize=14)

for i, (ax, gene) in enumerate(zip(axes.flat, BIOMARKER_ORDER)):
    x = biomarker_values[gene]
    y = biomarker_activation[gene]
    K = biomarker_thresholds[gene]

    # FIX 4: Overlay smooth Hill curve on scatter for interpretability
    x_range = np.linspace(0, x.max() * 1.1, 500)
    y_curve = hill_function(x_range, K, n)

    ax.scatter(x, y, alpha=0.4, s=15, color="steelblue", label="Patient data")
    ax.plot(x_range, y_curve, color="darkorange", linewidth=2, label=f"Hill curve (n={n})")
    ax.axvline(K, color="red", linestyle="--", linewidth=1.5, label=f"K = {K:.2f}")
    ax.set_xlabel(f"{gene} Expression [log2(RSEM+1)]")
    ax.set_ylabel("Gene Activation Output (y)")
    ax.set_title(f"{gene} — {ROLE_LABELS.get(gene, 'Biomarker')}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# FIX 3: Turn off any unused axes (handles AFP-only fallback gracefully)
for j in range(len(BIOMARKER_ORDER), len(axes.flat)):
    axes.flat[j].set_visible(False)

plt.tight_layout()
os.makedirs(os.path.dirname(OUTPUT_FIGURE), exist_ok=True)
plt.savefig(OUTPUT_FIGURE, dpi=300, bbox_inches="tight")
plt.close()
print(f"Figure saved to: {OUTPUT_FIGURE}")