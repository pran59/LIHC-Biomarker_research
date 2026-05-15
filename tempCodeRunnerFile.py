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
OUTPUT_FIGURE = "/Users/pranitabarik/Desktop/Cancer_Research/Coding/output/Synthetic Gene Circuit_fig_1.png"

# ==================================================
# STEP 1: LOAD BIOMARKER DATA
# ==================================================

with open(AFP_STATS_FILE, "r") as f:
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

print("Biomarker signals loaded")
for gene in BIOMARKER_ORDER:
    print(f"  {gene}: {len(biomarker_values[gene])} samples")

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
    Hill function for gene activation
    x : input signal
    K : activation threshold
    n : cooperativity
    """
    return (x**n) / (K**n + x**n)

# ==================================================
# STEP 4: COMPUTE GENE ACTIVATION
# ==================================================

n = 4  # cooperativity (can test n=3,4,5)

biomarker_activation = {}
for gene in BIOMARKER_ORDER:
    x = biomarker_values[gene]
    K = biomarker_thresholds[gene]
    biomarker_activation[gene] = hill_function(x, K, n)

print("Gene activation computed")

# ==================================================
# STEP 5: VISUALIZE RESPONSE
# ==================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("Synthetic Gene Circuit Response by Biomarker", fontsize=14)

for ax, gene in zip(axes.flat, BIOMARKER_ORDER):
    x = biomarker_values[gene]
    y = biomarker_activation[gene]
    K = biomarker_thresholds[gene]

    ax.scatter(x, y, alpha=0.6)
    ax.axvline(K, color="red", linestyle="--", label=f"K = {K:.2f}")
    ax.set_xlabel(f"{gene} Expression")
    ax.set_ylabel("Gene Activation Output (y)")
    ax.set_title(f"{gene} ({ROLE_LABELS.get(gene, 'Biomarker')})")
    ax.legend()
    ax.grid(True)

plt.tight_layout()
os.makedirs(os.path.dirname(OUTPUT_FIGURE), exist_ok=True)
plt.savefig(OUTPUT_FIGURE, dpi=300, bbox_inches="tight")
plt.close()
print(f"Figure saved to: {OUTPUT_FIGURE}")
