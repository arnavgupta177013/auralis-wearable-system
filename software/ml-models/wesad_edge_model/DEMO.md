# DEMO — 10-minute walkthrough of the stress detection model (for the project guide)

Everything shown below was run on 2026-11-11 (clean re-run of the whole pipeline from deleted outputs; NOTES.md
"Clean-state verification"). Expected outputs are copied from those runs. Commands run from
`software\ml-models\wesad_edge_model` in PowerShell after `$py = "..\..\..\.venv\Scripts\python"`.
"Stress" here = the WESAD laboratory stressor (TSST). It is **not** panic or anxiety, and nothing here is a medical device.

| min | section |
|---|---|
| 0-1 | 1 dataset and features |
| 1-2 | 2 preprocessing and exploratory analysis |
| 2-3.5 | 3 algorithm choice (incl. HRV dropped, personalisation did not help) |
| 3.5-4 | 4 training code |
| 4-5 | 5 LOSO and why random splits are wrong |
| 5-6 | 6 metrics and plots |
| 6-6.5 | 7 comparison with baselines |
| 6.5-8 | 8 prediction on new inputs |
| 8-8.5 | 9 saved model files |
| 8.5-9 | 10 independent loading and the C header |
| 9-10 | 11 limitations and ESP32 feasibility |

## 1. Dataset and features
- WESAD, Empatica E4 **wrist** signals of 15 subjects (S2-S11, S13-S17; 12 male, 3 female, 24-35 years): BVP 64 Hz,
  EDA 4 Hz (µS), TEMP 4 Hz, ACC 32 Hz. Labels 1 baseline, 2 stress (TSST), 3 amusement; everything else excluded.
  Read-only at `C:\datasets\WESAD`, never in the repo.
- Binary task: stress vs non-stress (baseline + amusement).
- Windows 20 / 30 / 60 s, 5 s step, kept if >= 90 % one label. 30 s: 6421 windows, 5509 scored (2541 baseline, 1923
  stress, 1045 amusement). The first 5 min of baseline is the calibration period and is never scored.
- 14 window features (HR/HRV, EDA level/slope/SCR, TEMP, ACC). The deployed model uses 3:
  `hr_mean` (bpm), `eda_slope_rel` (EDA slope / wearer's calibration SD), `scr_count_rel` (SCR peaks with prominence
  >= the wearer's calibration phasic SD). Exact definitions: `feature_spec.json`.

```powershell
& $py -m src.features --data "C:\datasets\WESAD"       # ~2 min -> results/metrics/features_w{20,30,60}.csv
```
Expected (tail of the JSON it prints): `"60": {"total": 6196, "calib": 720, "excluded_overlap_calib": 920, "scored": 5276, ... "scored_stress_fraction": 0.3503}`.

## 2. Preprocessing and exploratory analysis
- Causal filters only (they can run on a stream): BVP Butterworth band-pass 0.7-3.5 Hz (SOS with state), EDA low-pass
  1 Hz, tonic = causal EMA (8 s), phasic = EDA - tonic. Chunked == one-pass filtering exactly (asserted).
- Peaks with parabolic interpolation; IBI kept if 0.33-1.5 s and within 30 % of the window median.
- Quality flags are recorded, never used to delete data.

```powershell
& $py -m src.explore
```
Show: `results/graphs/class_balance_w30.png`, `feature_distributions_universal_w30.png`, `subject_shifts_w30.png`,
`time_shortcut.png`, `quality_flags.png`. Key findings (NOTES.md Stage 2): HR rises in stress in 15/15 subjects;
**minutes since session start alone predicts stress (time-only F1 0.95)**, so time is never an input;
motion flags are almost all stress windows (TSST = speaking/gesturing).

## 3. Algorithm choice and justification
- **Pre-registered selection** (rules written before the experiments, NOTES.md "Step C pre-registration"): 192
  configurations (LR, decision tree, random forest, XGBoost x 2 feature schemes x 8 subsets x 3 windows), only
  hardware-transferable features. Best mean F1 = XGB 60 s (0.868). 35 configurations are statistically as good
  (CI overlap, worst subject within 0.05); the rule then picks the **smallest**: logistic regression on
  [hr_mean, eda_slope, scr_count], 30 s, **7 parameters** (`stepc_selection.json`).
- Model D = the same with the gain-invariant EDA features `eda_slope_rel`, `scr_count_rel`, because the prototype
  reads finger EDA in mV, not µS (gain-dependent features change under a realistic gain spread: F1 0.841 -> 0.791).
- **HRV was dropped, with evidence** (Step A, pre-registered criteria): wrist HRV was compared with chest-ECG HRV in the
  same windows. RMSSD within-subject correlation 0.051 (20 s); wrist RMSSD RISES under stress (87 -> 115 ms) while
  chest RMSSD falls (45 -> 25 ms). Verdict "keep HR only" at 20/30/60 s. Removing HRV changed 30 s F1 by -0.019 to
  +0.003 (universal LR 0.852 with vs 0.840 without; personal z 0.905 vs 0.908; relative-only 0.845 vs 0.826), and
  whatever wrist "HRV" adds is PPG timing noise, not vagal HRV. Show `results/graphs/hrv_check_bland_altman_w20.png`,
  files `hrv_check_agreement.csv`, `hrv_check_verdict.json`.
- **Personalisation did not help**, with evidence: personal z-scoring of the deployable features is on average
  0.032 F1 WORSE than relative-only (27 better, 69 worse of 96 paired configurations; never significant,
  `stepc_wilcoxon.csv`). Per-wearer thresholds also hurt (mean F1 0.740 vs 0.792). The apparent personal-z gain in
  Step B (0.91-0.93) needed absolute SCL/temperature "drift" features, which track time-in-session. The published
  baseline's personal-vs-universal tests were not significant either (p 0.847 / 0.730).

## 4. Training code
| file | role |
|---|---|
| `src/preprocessing.py`, `src/features.py` | loading, causal filters, windows, features, calibration flags |
| `src/checks.py` | leakage assertions run on EVERY outer and inner fold |
| `src/train.py`, `src/runner.py` | LOSO harness: inner GroupKFold(7) for C and threshold, refit, score held-out once |
| `src/stepc.py`, `src/small_models.py` | 192-configuration grid and pre-registered selection |
| `src/final_model.py` | final LOSO for R and D, deployment fit on all 15 subjects, saves `models/` |
Pipeline = median imputer -> standard scaler -> `LogisticRegression(class_weight="balanced")`; C from {0.001, 0.01,
0.1, 1, 10} by inner PR-AUC (D's deployment model: C = 0.001); threshold = max F1 on pooled inner out-of-fold scores.
```powershell
& $py -m src.final_model --data "C:\datasets\WESAD"     # ~1.5 min
```
Expected: `reproduction of Step C: PASS {"R_OP1_max_abs_f1_diff_vs_stepc": 0.0, ... "R_max_abs_score_diff_vs_stepc": 0.0}`.

## 5. LOSO methodology, and why random window splits are wrong here
- 15 folds: train on 14 subjects, test on the 15th, which is touched once (no tuning, no threshold, no floor from it).
- Windows overlap by 25 s of 30 s, and adjacent windows of one person are near-copies. A random window split puts
  near-identical windows in train and test and measures memorisation of the person, not detection on a new wearer.
  A wearable always meets a new wearer, so the subject is the unit of evaluation.
- Leakage tests (8, `tests/test_leakage.py`) inject each violation (subject overlap, calibration window scored,
  time feature, scaler fitted on test rows, ...) and check that the harness catches it. Shuffled labels -> ROC-AUC
  0.529 over 20 seeds (chance).
```powershell
& $py -m unittest tests.test_leakage tests.test_stepc -v       # Ran 13 tests ... OK
```

## 6. Metrics and plots
Per subject then mean ± SD, subject-bootstrap 95 % CI (2000 resamples): F1, sensitivity, specificity, precision,
balanced accuracy, ROC-AUC, PR-AUC, false alarms/h (per window and merged events), worst subject, latency.

| model (LOSO, 30 s) | F1 | sensitivity | specificity | false-alarm events/h | worst subject |
|---|---|---|---|---|---|
| R, OP1 (research reference) | 0.841 ± 0.132 [0.769, 0.903] | 0.927 | 0.820 | 16.9 | S13 0.574 |
| D, OP1 | 0.779 ± 0.217 [0.663, 0.871] | 0.799 | 0.892 | 18.7 | S2 0.140 |
| **D shipped, OP1 + 4-of-5** | **0.762 ± 0.244 [0.629, 0.863]** | 0.745 | 0.929 | **7.6** | S2 0.033 |
(from `results/metrics/final_results.csv`; full table with all CIs in README.md). Plots: `final_roc_pr.png`,
`final_confusion_pooled.png`, `final_confusion_per_subject_D.png`, `final_per_subject_spread.png`,
`final_worst_subjects.png`, `final_train_vs_validation.png` (no overfitting: train and validation curves meet),
`final_calibration.png` (scores are NOT calibrated probabilities).

## 7. Comparison with baselines (30 s, Step B harness, `stepb_results.csv`)
| method | mean F1 | note |
|---|---|---|
| majority class | 0.000 | always "no stress" |
| HR threshold (universal) | 0.632 ± 0.250 | worst S6 0.031 |
| HR z threshold (personal) | 0.717 ± 0.181 | worst S2 0.297 |
| LR relative-only (selected R) | 0.841 ± 0.132 | |
| time only | 0.953 ± 0.062 | NOT a candidate: shows the protocol shortcut |
The published frozen baseline (60 s, 15 s step, non-causal filters, transductive SD floor) reported RF universal
0.908 / RF personal 0.846. Under the corrected protocol the same detectors give 0.904 / 0.749
(`results/paper_numbers.md`, every value traced to its source file).

## 8. Prediction on new inputs (`src/predict.py`)
Three input modes: `--features-csv`, `--signals CAL_DIR TEST_DIR` (raw BVP + EDA CSVs with units in the header),
`--demo-wesad S`. Inputs are validated (units, ranges, sampling rates, flat/railed/clipped PPG, calibration quality);
bad input -> "INPUT REJECTED" and exit code 2.
```powershell
"hr_mean,eda_slope_rel,scr_count_rel`n64,0,0`n108,0.05,6`n95,0,1" | Set-Content -Encoding utf8 $env:TEMP\demo.csv
& $py -m src.predict --features-csv $env:TEMP\demo.csv
```
Expected (SYNTHETIC rows):
```
t_start_s t_end_s hr_mean eda_slope_rel scr_count_rel  score threshold     alert smoothed_alert note
        0      30    64.0        0.0000             0 0.1868    0.5161 no stress      no stress
        5      35   108.0        0.0500             6 0.8238    0.5161    stress      no stress
       10      40    95.0        0.0000             1 0.5005    0.5161 no stress      no stress
Windows: 3; withheld: 0; window alerts 'stress': 1; smoothed alerts 'stress': 0
```
(smoothed alert needs 4 of the last 5 windows). Demo on WESAD:
```powershell
& $py -m src.predict --demo-wesad S5 --data "C:\datasets\WESAD"
```
Expected: `Calibration: 279.4-579.4 s of the stream, sigma_lp_cal 0.141514 uS, sigma_ph_cal 0.0539902 uS, median HR 68.8 bpm, 54 calibration windows. Checks passed.`
... `Windows: 1130; withheld: 199; window alerts 'stress': 225; smoothed alerts 'stress': 160`, and smoothed alerts by
true label: stress 106 of 125, baseline 0 of 175, amusement 0 of 70, meditation 2 of 150.
**S5 is a TRAINING subject of D (D was fitted on all 15 subjects), so this demo is NOT an accuracy estimate**; the
script prints exactly that. The accuracy estimate is the LOSO result in section 6.

## 9. Saved model files (`models/`)
| file | content |
|---|---|
| `stress_lr_D_w30.joblib` | sklearn pipeline + feature order + thresholds + smoothing (1942 B) |
| `stress_lr_D_w30.json` | plain numbers: folded weights, bias, medians, scaler, thresholds, 4-of-5 rule, calibration block (4117 B) |
| `stress_lr_D_w30.h` | C header generated from the JSON (4316 B) |
| `feature_spec.json` | feature names, ORDER, units, exact filter definitions (version stepc-1) |
| `calibration_procedure.md`, `window.json` | 5-min seated rest at first use; 30 s / 5 s / 4-of-5 |
| `metadata.json` | date, library versions, SHA-256 of the 15 WESAD files and derived CSVs, LOSO metrics |
| `roundtrip_windows.csv` | 340 fixed windows with reference scores for the round-trip test |
All model files were regenerated byte-identically in the clean re-run.

## 10. Independent loading and the C header
```powershell
& $py -m src.predict --roundtrip
```
Expected: `ROUNDTRIP PASS {"n_windows": 340, "n_with_nan": 40, "joblib_exact_match": true, "json_max_abs_diff": 3.3306690738754696e-16, "float32_op1_decision_disagreements": 0, "wrong_order_inputs_rejected": 11}`

Without sklearn, from the JSON only (what the C code does: NaN -> median, z = bias + w·x, p = 1/(1+e^-z)):
```powershell
& $py -c "import json,numpy as np; m=json.load(open('models/stress_lr_D_w30.json')); x=np.array([[64,0,0],[108,0.05,6],[95,0,1]],float); p=1/(1+np.exp(-(m['bias_folded']+x@np.array(m['weights_folded'])))); print(np.round(p,4), p>=m['threshold_op1'])"
```
Expected: `[0.1868 0.8238 0.5005] [False  True False]` (same as section 8).
C header excerpt (`models/stress_lr_D_w30.h`): `AURALIS_FEATURE_ORDER "hr_mean,eda_slope_rel,scr_count_rel"`,
`AURALIS_WEIGHTS = {0.04105883713231878, 0.09791634615711936, 0.20033436124755286}`, `AURALIS_BIAS = -4.098981700232388`,
`AURALIS_THRESHOLD_OP1 = 0.5160913415113687`, `AURALIS_SMOOTH_M 4`, `AURALIS_SMOOTH_N 5`. A C harness compiled with
MSVC on this PC reproduces the Python scores on 300 real windows (float: max difference 1.04e-7, 0 alert differences).

## 11. Limitations and ESP32 feasibility
Limitations (full list in README.md): stress not panic/anxiety; 15 young healthy subjects, one lab, one stressor;
E4 wrist signals, not the prototype's fingertip MAX30102 / finger GSR in mV / MAX30205 / MPU6050; the session-time
shortcut cannot be separated on WESAD; S2 and S3 poorly detected, S13 depends on the configuration; D's estimate is
optimistic (selected after seeing results); ~7.6 merged false alarms/h extrapolated from ~20 min of non-stress per
subject; no PPG noise rejection beyond the flat/railed/clipped check; the score is not a probability; not a medical device.

ESP32: **nothing has run on an ESP32 and no firmware exists.**
- MEASURED: model files 1942 / 4117 / 4316 B; ESP32 object for score + smoother 154 B .text + 28 B .literal
  (cross-compiled only, not run); C host test 0 alert differences; streaming reference 0 alert differences on 5509
  windows (float64 and float32). PC timings (Python 25.6 µs per window JSON scorer) are PC numbers, not ESP32 numbers,
  and the streaming reference was ~2.3x slower in a second run (`esp32_resources.csv`, committed vs the saved
  re-run copy; cause not investigated).
- ESTIMATED: RAM 15.1 kB (64/4 Hz) to 24.4 kB (100/10 Hz) of ~520 KB; CPU 0.07-0.67 % of one 240 MHz core (cycles per
  op ASSUMED); model constants 36 B flash.

## Questions the guide is likely to ask (honest answers)
- **"Is 0.76 F1 good?"** It is a LOSO estimate on 15 lab subjects, optimistic because the configuration was chosen
  after seeing results, with 7.6 false-alarm events per hour and two subjects (S2, S3) largely missed. Usable as a
  research prototype, not as a product claim.
- **"Why not the RF / XGBoost that scored higher?"** The best (XGB 0.868) is not statistically better than 35 others;
  the pre-registered rule takes the smallest qualifying model (7 parameters). The paper's RF (200 trees, depth 8) is
  ~174 kB estimated, over the 100 kB budget.
- **"Why does the published baseline show 0.908?"** Different protocol: 60 s windows, non-causal filters, an SD floor
  that uses the test subject's own session, 50 partly-calibration windows scored. Under the corrected protocol the
  same RF universal gives 0.904 and RF personal 0.749 (`results/paper_numbers.md`).
- **"Does it detect anxiety or panic?"** No. WESAD contains a lab stressor only.
- **"Does it work on our hardware?"** Unknown. Different sensors and body sites; no prototype data may be used before
  IEC approval. A post-approval validation plan is in `results/metrics/esp32_feasibility.md`.
- **"Could the model just be learning time-in-session?"** Possibly in part: time alone gives F1 0.95 on WESAD, and no
  WESAD control can separate the two. Time is never an input, but this cannot be ruled out on this dataset.
- **"Is the score a probability?"** No; it is uncalibrated (class-balanced training). Only the alert is meaningful.
- **"Is everything reproducible?"** All features, LR results and model files reproduce byte-for-byte. One exception:
  the frozen baseline RF's PR-AUC differs in the last digits between runs (multi-threaded prediction); F1 is unaffected.
  PC timing measurements are not reproducible.
- **"What is the 'drift' problem?"** Absolute EDA level and skin temperature drift over the session, so they encode
  time; the personal-z advantage in Step B disappeared without them.
