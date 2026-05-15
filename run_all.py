"""
ENHANCED SR PIPELINE — README
==============================

File Structure
--------------
noise_utils.py                  ← shared Gaussian / OU / Levy generators
step1_parse_clinical.py          ← unchanged
step4_define_threshold_K.py      ← unchanged
step5_gene_circuit_model.py      ← unchanged (but fix plt.savefig at end)
step6_enhanced_noise.py          ← REPLACES step6_add_noise.py
step7_measure_performance_enhanced.py ← activation probability SR sweep
step8_9_enhanced_sr_analysis.py  ← REPLACES step8 + step9 (adds Fisher Info)
step10_sr_peak_characterization.py ← optimal noise identification / sigma* extraction
step11_multi_biomarker_panel.py  ← label-free multi-marker fusion analysis
step12_robustness_sensitivity.py ← parameter sensitivity across n, K percentile, tau
step13_roc_auc_framing.py        ← ROC / AUC clinical framing (tumour vs normal)
step14_survival_correlation.py   ← Kaplan-Meier survival correlation (all stages)
step15_statistical_rigor.py      ← bootstrap CIs, effect sizes, BH correction, Table 1

Run Order
---------
python step1_parse_clinical.py         # produces early_stage_patient_ids.json
# (steps 2-3 are your expression extraction — not included here)
python step4_define_threshold_K.py     # produces gene_activation_threshold_K.json
python step5_gene_circuit_model.py     # produces gene circuit plots
python step6_enhanced_noise.py         # Gaussian vs OU vs Levy comparison
python step7_measure_performance_enhanced.py # activation-probability SR sweep
python step8_9_enhanced_sr_analysis.py # main SR metrics: AP + MI + FI
python step10_sr_peak_characterization.py # sigma* extraction, summary table, heatmaps
python step11_multi_biomarker_panel.py # panel fusion vs single-marker baselines
python step12_robustness_sensitivity.py # robustness sweep across n, percentile, tau
python step13_roc_auc_framing.py       # ROC/AUC clinical framing
python step14_survival_correlation.py  # Kaplan-Meier survival analysis
python step15_statistical_rigor.py     # bootstrap CIs, effect sizes, Table 1

Quick install (if missing packages)
-------------------------------------
pip install numpy scipy matplotlib scikit-learn lifelines statsmodels

Paper Figure Map
----------------
Figure 1 → sr_metrics_combined.png           (4 biomarkers × 3 metrics: AP, MI, FI vs σ)
Figure 2 → noise_autocorrelation.png          (OU correlation proof; Levy stays white)
Figure 3 → step10_sr_curves_annotated.png     (MI curves with sigma* markers)
Figure 4 → step10_sigma_star_heatmap.png      (sigma* per biomarker × noise type)
Figure 5 → step10_mi_peak_heatmap.png         (MI at sigma*)
Figure 6 → step10_sr_gain_heatmap.png         (delta MI over baseline)
Figure 7 → step11_panel_activation_curves.png (panel activation curves)
Figure 8 → step11_panel_mi_curves.png         (panel MI curves)
Figure 9 → step11_panel_mi_heatmap.png        (panel vs single peak MI)
Figure 10 → step11_panel_sr_gain_bar.png      (panel delta MI vs best single)
Figure 11 → step12_tornado.png                (overall sensitivity range by parameter)
Figure 12 → step12_sigma_star_sensitivity.png (sigma* sensitivity grid)
Figure 13 → step12_delta_mi_sensitivity.png   (delta MI sensitivity grid)
Figure 14 → step12_delta_act_sensitivity.png  (delta activation sensitivity grid)
Figure 15 → step12_sr_hill_sweep.png          (MI overlays for Hill-n sweep)
Figure 16 → step12_sr_percentile_sweep.png    (MI overlays for percentile sweep)
Figure 17 → step12_sr_tau_sweep.png           (MI overlays for OU tau sweep)
Table  1 → step10_sr_peak_summary.csv         (sigma* summary table)
Table  2 → step11_panel_summary.csv           (panel summary table)
Table  3 → step12_sensitivity_summary.csv     (robustness summary table)
Table  4 → sr_full_results.csv                (full SR sweep data)
Figure 18 → step13_auc_vs_sigma.png            (AUC vs noise intensity)
Figure 19 → step13_roc_at_sigma_star.png       (ROC curves at optimal sigma*)
Figure 20 → step13_auc_heatmap.png             (AUC heatmap per biomarker x noise)
Figure 21 → step14_km_curves.png               (Kaplan-Meier sub vs supra threshold)
Figure 22 → step14_km_sr_zone.png              (KM for SR-zone analysis)
Figure 23 → step15_mi_with_ci.png              (MI curves with 95% bootstrap CI)
Figure 24 → step15_act_with_ci.png             (AP curves with 95% bootstrap CI)
Table  5 → step13_auc_summary.csv              (AUC summary with bootstrap CIs)
Table  6 → step14_survival_summary.csv         (survival analysis summary)
Table  7 → step15_effect_sizes.csv             (effect sizes with CIs)
Table  8 → step15_table1.csv                   (publication Table 1)
Table  9 → step15_multiple_testing.csv         (BH-corrected p-values)

Key Paper Contributions (what each script adds)
------------------------------------------------
step6:
  → Addresses reviewer: "Is real biological noise Gaussian?"
  → OU noise = membrane/ion-channel fluctuations (cite Faisal 2008)
  → Levy-stable noise = bursty rare-jump fluctuations / heavy tails

step8_9 (Fisher Info):
  → Theoretical rigour: output-level FI quantifies detector resolution
  → FI is now evaluated on the nonlinear gene-circuit output, not just x + η
  → FI peak at σ* complements the MI-based SR finding across three noise families

step10 (Peak Characterisation):
  → Extracts sigma* where MI is maximised for every biomarker × noise family
  → Adds bootstrap confidence intervals and publication-ready tables
  → Produces the core quantitative heatmaps reviewers will expect

step11 (Multi-Biomarker Panel):
  → Compares multi-marker fusion circuits against every single-marker baseline
  → Uses real matched early-stage biomarker vectors from step 3
  → Keeps the analysis label-free for now; ROC/AUC is intentionally deferred to step 13

step12 (Robustness / Sensitivity):
  → Tests whether the sigma* and SR-gain conclusions stay stable when n, K, and tau move
  → Recomputes percentile-based thresholds directly from the matched biomarker vectors
  → Stays consistent with the current label-free plan by tracking activation and MI, not ROC
"""

# ==================================================
# RUN ALL STEPS IN SEQUENCE
# ==================================================

import subprocess, sys, os

SCRIPTS = [
    "step6_enhanced_noise.py",
    "step7_measure_performance_enhanced.py",
    "step8_9_enhanced_sr_analysis.py",
    "step10_sr_peak_characterization.py",
    "step11_multi_biomarker_panel.py",
    "step12_robustness_sensitivity.py",
    "step13_roc_auc_framing.py",
    "step14_survival_correlation.py",
    "step15_statistical_rigor.py",
]

if __name__ == "__main__":
    base = os.path.dirname(__file__)
    for script in SCRIPTS:
        path = os.path.join(base, script)
        print(f"\n{'='*60}")
        print(f"Running: {script}")
        print("="*60)
        result = subprocess.run([sys.executable, path], capture_output=False)
        if result.returncode != 0:
            print(f"ERROR in {script} — stopping.")
            sys.exit(1)
    print("\n\n✓ All steps complete. Check output/ directory for figures and CSVs.")
