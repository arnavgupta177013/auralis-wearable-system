# CLAUDE.md — wesad_edge_model

Standing rules for all work in this folder.

## Project
- Auralis wrist wearable: ESP32-WROOM-32E, MAX30102 (PPG), Grove GSR, MAX30205 (skin temp), MPU6050 (IMU).
- Goal: a stress detection model trained on public WESAD wrist (Empatica E4) data, later deployed on the ESP32.
- WESAD stress = lab stressor (TSST), NOT panic or anxiety. Every report and plot says "stress detection".
  (The Vishwakarma deck and IEC proforma use "panic/anxiety" wording; this model does not support that claim.)

## Data and target
- Dataset at `C:\datasets\WESAD` (outside the repo; also junction `data\raw\WESAD`). Never commit it.
  Never delete, move or recurse-delete anything under `data\`.
- Target = WESAD label: 1 baseline, 2 stress, 3 amusement. All other labels are excluded.

## Evaluation (non-negotiable)
- Subject-independent only: LOSO or GroupKFold by subject. Never random window splits (75 % overlap leaks).
- Tuning, threshold and feature selection use training subjects only (inner GroupKFold).
  The held-out subject is touched once.
- Personal baseline = first 5 min of each subject's baseline condition, excluded from scoring for ALL methods.

## Reporting
- Never fabricate or round up metrics. Do not claim anything works until it has run.
- Report mean +/- SD across subjects, per-subject results, and subject-level bootstrap 95 % CIs.
- Metrics: sensitivity, specificity, precision, F1, balanced accuracy, ROC-AUC, PR-AUC,
  false alarms per hour, and the worst subject.

## ESP32 constraints
- ~520 KB SRAM; model <= 100 kB int8; 20 s window target.
- Features must be portable to streaming C (prefer causal filters; no filtfilt in deployed features).

## Code style
- Modular, student-readable. No ESP32 firmware in this phase.
- Keep NOTES.md updated after every stage.
- Baseline code in `../wesad_baseline/` is frozen (only the Wilcoxon addition was allowed).

## Hardware and paper alignment
- Prototype v0.1 (breadboard): ESP32 + MAX30102 PPG (FINGERTIP on the breadboard), MAX30205 skin temperature,
  MPU6050 IMU, generic 3-pin GSR module with electrodes on the index/middle FINGERS, read on GPIO34 in mV via
  `analogReadMilliVolts` (the ESP32 ADC is nonlinear; mV is not µS without calibration).
- Training data is WESAD Empatica E4 at the WRIST. PPG (fingertip vs wrist) and EDA (finger vs wrist, mV vs µS)
  are therefore domain-shifted; no model result transfers to the prototype until checked on prototype data.
- Wrist HRV (rmssd, sdnn, pnn50) is excluded from every deployable model: Step A found a within-person correlation
  with chest ECG of 0.05 (RMSSD rmcorr, 20 s). The prototype serial log also shows RMSSD 250-337 ms at HR ~70 bpm,
  which is not physiological (reported by the user; log not in the repo).
- The paper draft (docs/, if present) compares fixed vs personal-baseline detectors on WESAD. Every number in it
  must trace to a file in `results/` or `../wesad_baseline/out/`. Never invent or round a number.
- No human recordings (including prototype recordings) may be used as study data before IEC approval.
