"""
Step 3: Extract biomarker expression from TCGA-LIHC (UCSC Xena HiSeqV2)
and compute data-driven mean and standard deviation for early-stage patients.

Primary biomarker remains AFP for downstream compatibility, while the
same extraction now also captures GPC3, DKK1, and MDK from the same
expression matrix.
"""

import json
import pandas as pd
import numpy as np
import os

# ==================================================
# FILE PATHS (EDIT ONLY THESE IF NEEDED)
# ==================================================

CLINICAL_FILE = "/Users/pranitabarik/Desktop/Cancer_Research/Coding/data/clinical.project-tcga-lihc.json"
EXPRESSION_FILE = "/Users/pranitabarik/Desktop/Cancer_Research/Coding/data/TCGA.LIHC.sampleMap:HiSeqV2.tsv"
OUTPUT_FILE = "/Users/pranitabarik/Desktop/Cancer_Research/Coding/output/tcga_lihc_afp_statistics.json"
BIOMARKERS = ["AFP", "GPC3", "DKK1", "MDK"]

# ==================================================
# FIX 1: Unified early-stage list (matches Step 1 exactly)
# ==================================================

VALID_EARLY_STAGES = [
    "Stage I", "Stage IA", "Stage IB",
    "Stage II", "Stage IIA", "Stage IIB"
    # NOTE: "Stage IIC" removed — not a valid AJCC stage for HCC/LIHC
]

# ==================================================
# STEP 1: LOAD CLINICAL JSON
# ==================================================

with open(CLINICAL_FILE, "r") as f:
    clinical_data = json.load(f)

print("Clinical data loaded")

# ==================================================
# STEP 2: EXTRACT EARLY-STAGE PATIENT IDS
# ==================================================

early_stage_ids = []

for patient in clinical_data:
    patient_id = patient.get("submitter_id")
    diagnoses = patient.get("diagnoses", [])

    if diagnoses:
        raw_stage = diagnoses[0].get("ajcc_pathologic_stage", "")
        if raw_stage:
            stage = raw_stage.strip()
            # FIX 1 applied: use full exhaustive stage list
            if stage in VALID_EARLY_STAGES:
                early_stage_ids.append(patient_id)

early_stage_ids = list(set(early_stage_ids))  # remove duplicates

# FIX 4 (partial): count patients skipped due to missing diagnosis
skipped = sum(1 for p in clinical_data if not p.get("diagnoses", []))
print(f"Early-stage patients identified: {len(early_stage_ids)}")
print(f"Patients skipped (no diagnosis data): {skipped}")

# ==================================================
# STEP 3: LOAD GENE EXPRESSION FILE (HiSeqV2)
# ==================================================

expr_df = pd.read_csv(EXPRESSION_FILE, sep="\t")

print("Expression file loaded")
print("Matrix shape (genes x samples):", expr_df.shape)

# First column contains gene names → set as index
expr_df.set_index(expr_df.columns[0], inplace=True)

# ==================================================
# STEP 4: EXTRACT BIOMARKER GENE EXPRESSION
# ==================================================

missing = [gene for gene in BIOMARKERS if gene not in expr_df.index]
if missing:
    raise ValueError(
        f"Biomarker gene(s) not found in expression matrix: {', '.join(missing)}"
    )

print("Biomarker genes confirmed present in matrix:")
for gene in BIOMARKERS:
    print(f"  {gene}: {expr_df.loc[gene].shape[0]} samples")

# TCGA sample barcodes → first 12 characters = patient ID
patient_columns = [col[:12] for col in expr_df.columns]


def extract_early_stage_values(gene_name):
    """Return early-stage expression values for one biomarker gene."""
    gene_series = expr_df.loc[gene_name].copy()
    gene_series.index = patient_columns

    early_values = []
    multi_sample_patients = 0  # FIX 4: track multi-sample patients

    for pid in early_stage_ids:
        if pid in gene_series.index:
            value = gene_series[pid]

            # FIX 4: Log multi-sample averaging
            if isinstance(value, pd.Series):
                multi_sample_patients += 1
                value = value.mean()

            # FIX 6: Skip NaN values explicitly
            if np.isnan(float(value)):
                continue

            early_values.append(float(value))

    if multi_sample_patients > 0:
        print(f"  [{gene_name}] Multi-sample patients averaged: {multi_sample_patients}")

    return np.array(early_values, dtype=float)


# ==================================================
# STEP 5: MATCH BIOMARKERS WITH EARLY-STAGE PATIENTS
# ==================================================

biomarker_results = {}

for gene in BIOMARKERS:
    values = extract_early_stage_values(gene)
    biomarker_results[gene] = {
        # FIX 2: ddof=1 for sample standard deviation
        "mean_x0": float(np.mean(values)),
        "std_delta": float(np.std(values, ddof=1)),
        "values": values.tolist(),
        "patient_count": int(len(values)),
    }
    print(f"{gene} → matched patients: {len(values)}, "
          f"mean: {biomarker_results[gene]['mean_x0']:.4f}, "
          f"std: {biomarker_results[gene]['std_delta']:.4f}")

# ==================================================
# STEP 6: COMPUTE DATA-DRIVEN STATISTICS
# ==================================================

print("\n=== DATA-DRIVEN BIOMARKER PARAMETERS ===")
for gene in BIOMARKERS:
    print(f"{gene} mean (x0): {biomarker_results[gene]['mean_x0']:.4f}")
    print(f"{gene} std (delta): {biomarker_results[gene]['std_delta']:.4f}")

afp_results = biomarker_results["AFP"]

# ==================================================
# STEP 7: SAVE RESULTS
# ==================================================

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

output = {
    # FIX 5: Corrected expression units label
    "expression_source": "TCGA-LIHC UCSC Xena HiSeqV2",
    "expression_units": "log2(RSEM + 1)",
    "selected_biomarkers": BIOMARKERS,

    # FIX 3: Per-biomarker patient counts instead of AFP-only reference
    "early_stage_patient_counts": {
        gene: biomarker_results[gene]["patient_count"] for gene in BIOMARKERS
    },
    "biomarkers": biomarker_results,

    # Preserve AFP-specific keys for downstream scripts
    "afp_mean_x0": afp_results["mean_x0"],
    "afp_std_delta": afp_results["std_delta"],
    "afp_values": afp_results["values"],
}

with open(OUTPUT_FILE, "w") as f:
    json.dump(output, f, indent=4)

print(f"\nBiomarker statistics saved to: {OUTPUT_FILE}")