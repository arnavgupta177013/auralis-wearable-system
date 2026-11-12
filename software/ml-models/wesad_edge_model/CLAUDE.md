# CLAUDE.md — wesad_edge_model

Standing rules for all work in this folder.

## Project
- Auralis wrist wearable (planned): ESP32-WROOM-32E, MAX30102 (PPG), Grove GSR, MAX30205 (skin temp), MPU6050 (IMU).
  Bench prototype v0.2 (checked on hardware 2026-11-12) differs: see "Hardware and paper alignment".
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
- Modular, student-readable. ESP32 firmware for model D lives in `firmware/esp32/auralis_model_d/` (C port of
  `src/stream_ref.py`). If a feature definition, filter or model file changes here, update that port and re-run its
  parity test (`firmware/esp32/auralis_model_d/test/`).
- Keep NOTES.md updated after every stage.
- Baseline code in `../wesad_baseline/` is frozen (only the Wilcoxon addition was allowed).

## Hardware and paper alignment
- Prototype v0.2 (breadboard, parts checked on hardware 2026-11-12):
  - PPG: green board marked "MAX30100/30102"; the chip is a **MAX30100** (PART_ID register 0x11, 16-bit), FINGERTIP.
    The board pulls SDA/SCL/INT up to its own ~1.8 V (three 4.7 kΩ "472" resistors); the ESP32 cannot read that, so
    2.2 kΩ pull-ups SDA/SCL -> 3V3 are fitted (bus measured 2.9 V).
  - IMU: GY-521 board sold as MPU6050 carries an **MPU6500** (WHO_AM_I 0x70); libraries that check for 0x68 reject it.
  - GSR: generic 3-pin module (OUT/VCC/GND), electrodes on the index/middle FINGERS, GPIO34 in mV via
    `analogReadMilliVolts` (ESP32 ADC nonlinear; mV is not µS). Grove-type GSR modules measure skin RESISTANCE: the
    output FALLS as conductance rises. The relative EDA features are invariant only to a POSITIVE linear gain, so raw
    mV would invert eda_slope_rel and make scr_count_rel count recoveries. The firmware converts mV to conductance
    (Seeed formula, open-circuit reading at power-on). Direction and linearity are UNVALIDATED.
  - MAX30205 and the haptic motor are not fitted yet (alert LED on GPIO25 instead).
- Training data is WESAD Empatica E4 at the WRIST. PPG (fingertip vs wrist) and EDA (finger vs wrist, mV vs µS)
  are therefore domain-shifted; no model result transfers to the prototype until checked on prototype data.
- Wrist HRV (rmssd, sdnn, pnn50) is excluded from every deployable model: Step A found a within-person correlation
  with chest ECG of 0.05 (RMSSD rmcorr, 20 s). The prototype serial log also shows RMSSD 250-337 ms at HR ~70 bpm,
  which is not physiological (reported by the user; log not in the repo).
- The paper draft (docs/, if present) compares fixed vs personal-baseline detectors on WESAD. Every number in it
  must trace to a file in `results/` or `../wesad_baseline/out/`. Never invent or round a number.
- No human recordings (including prototype recordings) may be used as study data before IEC approval.
