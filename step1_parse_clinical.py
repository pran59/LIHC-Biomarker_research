"""
STEP 1: Parse TCGA-LIHC clinical data
Goal: Identify early-stage (Stage I & II) patients
"""

import json
import os

# ==================================================
# FILE PATHS (EDIT IF NEEDED)
# ==================================================

BASE_DIR = os.path.dirname(__file__)
CLINICAL_FILE = os.path.join(BASE_DIR, "data", "clinical.project-tcga-lihc.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "output", "early_stage_patient_ids.json")

# ==================================================
# STEP 1: LOAD CLINICAL JSON
# ==================================================

with open(CLINICAL_FILE, "r") as f:
    clinical_data = json.load(f)

print("Clinical data loaded")
print("Total patients in file:", len(clinical_data))

# ==================================================
# STEP 2: EXTRACT EARLY-STAGE PATIENTS (CORRECTED)
# ==================================================

early_stage_patients = []

# Exhaustive list of TCGA early stages
valid_early_stages = [
    "Stage I", "Stage IA", "Stage IB", 
    "Stage II", "Stage IIA", "Stage IIB", "Stage IIC"
]

for patient in clinical_data:
    patient_id = patient.get("submitter_id")
    diagnoses = patient.get("diagnoses", [])

    if diagnoses:
        # Extract, remove trailing/leading spaces, and handle None
        raw_stage = diagnoses[0].get("ajcc_pathologic_stage", "")
        if raw_stage: 
            stage = raw_stage.strip() 

            if stage in valid_early_stages:
                early_stage_patients.append({
                    "patient_id": patient_id,
                    "stage": stage
                })

print("Early-stage patients found:", len(early_stage_patients))

# ==================================================
# STEP 3: SAVE OUTPUT
# ==================================================

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

with open(OUTPUT_FILE, "w") as f:
    json.dump(early_stage_patients, f, indent=4)

print(f"Early-stage patient list saved to: {OUTPUT_FILE}")

patient_ids = [p["patient_id"] for p in early_stage_patients]

print("Unique patients:", len(set(patient_ids)))
print("Duplicate entries:", len(patient_ids) - len(set(patient_ids)))
