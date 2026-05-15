"""
STEP 4: Define gene activation threshold (K)
Using data-driven biomarker expression from TCGA-LIHC.

AFP remains the primary biomarker for downstream compatibility, while
thresholds are also computed for GPC3, DKK1, and MDK so the paper can
position them with distinct biological roles.
"""

import json
import numpy as np

# ==================================================
# FILE PATHS
# ==================================================

AFP_STATS_FILE = "/Users/pranitabarik/Desktop/Cancer_Research/Coding/output/tcga_lihc_afp_statistics.json"
OUTPUT_FILE = "/Users/pranitabarik/Desktop/Cancer_Research/Coding/output/gene_activation_threshold_K.json"

# ==================================================
# STEP 1: LOAD BIOMARKER STATISTICS
# ==================================================

with open(AFP_STATS_FILE, "r") as f:
    biomarker_data = json.load(f)

# NOTE: Role labels are working assumptions for paper positioning.
# DKK1's "weak-signal" label must be validated against actual expression
# statistics (mean, std, SNR) from Step 3 output before finalizing.
BIOMARKER_ROLES = {
    "AFP":  "Primary anchor biomarker that grounds the framework biologically.",
    "DKK1": "Candidate weak-signal biomarker for SR gain testing (validate against expression stats).",
    "GPC3": "Established LIHC comparator to show framework is not AFP-specific.",
    "MDK":  "Secondary comparator to assess biomarker-dependent SR sensitivity.",
}

# FIX 4: Warn explicitly if falling back to AFP-only mode
if "biomarkers" in biomarker_data:
    biomarker_values = {
        gene: np.array(info["values"], dtype=float)
        for gene, info in biomarker_data["biomarkers"].items()
        if gene in BIOMARKER_ROLES
    }
else:
    print("WARNING: Multi-biomarker block not found in stats file. "
          "Falling back to AFP only. GPC3, DKK1, MDK will be excluded.")
    biomarker_values = {"AFP": np.array(biomarker_data["afp_values"], dtype=float)}

print("Biomarker data loaded")
for gene, values in biomarker_values.items():
    # FIX 1 (partial): report NaN count at load time
    nan_count = int(np.sum(np.isnan(values)))
    print(f"  {gene}: {len(values)} early-stage patients ")

# ==================================================
# STEP 2: DEFINE ACTIVATION THRESHOLD K
# ==================================================

# 75th percentile: ~75% of early-stage patients are sub-threshold,
# relying on SR noise to cross K. Adjust and run sensitivity analysis
# across 70–85% range before finalizing for the paper.
PERCENTILE = 75

biomarker_thresholds = {}

for gene, values in biomarker_values.items():

    # FIX 3: Guard against empty array
    clean_values = values[~np.isnan(values)]
    if len(clean_values) == 0:
        raise ValueError(
            f"No valid (non-NaN) expression values for {gene}. "
            "Cannot compute threshold. Check Step 3 output."
        )

    # FIX 1: Use nanpercentile to safely ignore any residual NaNs
    K = float(np.nanpercentile(clean_values, PERCENTILE))

    biomarker_thresholds[gene] = {
        "role_in_paper":         BIOMARKER_ROLES.get(gene, ""),
        "threshold_definition":  "Percentile-based",
        "percentile_used":       PERCENTILE,
        "activation_threshold_K": K,
        "patient_count":         int(len(clean_values)),
    }
    print(f"{gene} activation threshold K "
          f"(at {PERCENTILE}th percentile, n={len(clean_values)}): {K:.4f}")

# ==================================================
# STEP 3: VALIDATE PRIMARY BIOMARKER PRESENT
# ==================================================

primary_gene = "AFP"
if primary_gene not in biomarker_thresholds:
    raise ValueError("Primary biomarker AFP not found in threshold set.")

# ==================================================
# STEP 4: SAVE THRESHOLD
# ==================================================

output = {
    "threshold_definition":  "Percentile-based",
    "percentile_used":       PERCENTILE,
    "primary_biomarker":     primary_gene,
    # Top-level AFP key preserved for downstream script compatibility
    "activation_threshold_K": biomarker_thresholds[primary_gene]["activation_threshold_K"],
    # FIX 2: Safe key access with fallback
    "units":                 biomarker_data.get("expression_units", "log2(RSEM + 1)"),
    "biomarker_thresholds":  biomarker_thresholds,
}

with open(OUTPUT_FILE, "w") as f:
    json.dump(output, f, indent=4)

print(f"\nActivation thresholds saved to: {OUTPUT_FILE}")