# wesad_edge_model — stress detection from wrist signals (WESAD), for a future ESP32 wearable

**Stress detection only.** WESAD "stress" is a laboratory stressor (TSST: public speaking + mental arithmetic). This
model does not detect panic or anxiety, gives no probability of stress, and is not a medical device.
Rules for this folder: `CLAUDE.md`. Full log of every decision and number: `NOTES.md`. Walk-through: `DEMO.md`.

## What is here
| | |
|---|---|
| Deployed model **D** | logistic regression on 3 features `[hr_mean, eda_slope_rel, scr_count_rel]`, 30 s window, 5 s step, OP1 threshold 0.5161 + causal 4-of-5 smoothing; needs a 5-min seated-rest calibration at first use (`models/`) |
| Reference model **R** | same model with gain-dependent EDA features `[hr_mean, eda_slope, scr_count]`: the research result, not deployable |
| Evaluation | leave-one-subject-out (LOSO) on 15 subjects; tuning, thresholds and feature choices on training subjects only (inner GroupKFold) |
| Inference | `src/predict.py` (features CSV, raw signal CSVs, or a WESAD demo) |
| C export | `models/stress_lr_D_w30.h`, generated from the model JSON; host-tested in C, ESP32 compile-only |

## Setup (Windows PowerShell)
From the repository root (`auralis-wearable-system`):
```powershell
py -3.11 -m venv .venv                                  # versions used: Python 3.11.6
.\.venv\Scripts\python -m pip install -r software\ml-models\wesad_edge_model\requirements.txt
# versions used for the committed results: numpy 2.4.6, scipy 1.17.1, pandas 3.1.1, scikit-learn 1.9.1,
# xgboost 3.2.0, joblib 1.6.0, matplotlib 3.11.2 (recorded in models/metadata.json)
cd software\ml-models\wesad_edge_model
```
Optional: Visual Studio 2022 C++ build tools (the C harness test is SKIPPED without a host C compiler) and
`xtensa-esp32-elf-gcc` from ESP-IDF (ESP32 object size; skipped if absent).

## Dataset (read-only)
WESAD lives OUTSIDE the repository at `C:\datasets\WESAD` (folders `S2` ... `S17`, each with `S<n>.pkl`). The repo has a
junction `data\raw\WESAD -> C:\datasets\WESAD`; `data\` is gitignored. Nothing in this project writes to the dataset.
Never delete, move or recurse-delete anything under `data\`. On a new machine, from the repo root:
`New-Item -ItemType Junction -Path data\raw\WESAD -Target C:\datasets\WESAD`.

## Run: one command per stage (from `software\ml-models\wesad_edge_model`)
`$py = "..\..\..\.venv\Scripts\python"` first, then run each line with `& $py ...`. Times are from the 2026-11-11
clean re-run on a 12-core PC (indicative only).

| # | stage | command | writes | time |
|---|---|---|---|---|
| 1 | features (20/30/60 s windows, causal filters) | `& $py -m src.features --data "C:\datasets\WESAD"` | `results/metrics/features_w*.csv` | ~2 min |
| 2 | exploratory analysis (EDA) | `& $py -m src.explore` | class balance, distributions, time shortcut | <1 min |
| 3 | chest-ECG check of wrist HR/HRV | `& $py -m src.hrv_check --data "C:\datasets\WESAD"` | `hrv_check_*` | ~3 min |
| 4 | leakage + Step C tests | `& $py -m unittest tests.test_leakage tests.test_stepc -v` | nothing | ~3 min |
| 5 | Step B harness grid (LR, baselines, shuffled-label check) | `& $py src\train.py --windows 20 30 60` | `stepb_*` | ~5 min |
| 6 | feature spec (rewrites `feature_spec.json`, identical bytes) | `& $py -m src.feature_spec` | `feature_spec.json` | s |
| 7 | protocol order | `& $py -m src.protocol --data "C:\datasets\WESAD"` | `stepc_label_segments.csv`, `stepc_condition_order.csv` | s |
| 8 | Step C grid (192 small-model configurations) | `& $py -m src.stepc --phase grid` | `stepc_*_grid.csv`, `stepc_selection.json` | ~17-33 min |
| 9 | hardware parity | `& $py -m src.hw_parity --table`; `& $py -m src.hw_parity --eda --data "C:\datasets\WESAD"`; `& $py -m src.hw_parity --device 30 --data "C:\datasets\WESAD"` | `hardware_feasibility.csv`, `features_eda_w*.csv`, `features_hw_w30__*.csv` | ~5 min |
| 10 | paper reconciliation (reads `../wesad_baseline/out`, never writes it) | `& $py -m src.paper_reconcile` | `stepc_paper_*` | ~10 min |
| 11 | Step C follow-up, then combine | `& $py -m src.stepc --phase followup`; `& $py -m src.stepc --phase combine` | `stepc_*_followup.csv`, `stepc_combined_*.csv` | ~6 min |
| 12 | final models R and D | `& $py -m src.final_model --data "C:\datasets\WESAD"` | `models/*`, `final_*` | ~1.5 min |
| 13 | predict: round trip | `& $py -m src.predict --roundtrip` | nothing | s |
| 14 | predict: demo | `& $py -m src.predict --demo-wesad S5 --data "C:\datasets\WESAD"` | nothing | s |
| 15 | streaming parity (sample-by-sample reference) | `& $py -m src.stream_ref --data "C:\datasets\WESAD"` | `stream_*` | 1-2 min |
| 16 | PPG gate evaluation (Step F) | `& $py -m src.ppg_gate --data "C:\datasets\WESAD"` | `ppg_gate_*` | ~1 min |
| 17 | C export, C host test, resources | `& $py -m src.export_esp32 --all` | `models/stress_lr_D_w30.h`, `esp32_*`, `build/` | <1 min |
| 18 | paper outputs | `& $py -m src.paper_outputs` | `results/paper_numbers.md`, `paper_*` | s |
| — | all tests | `& $py -m unittest tests.test_leakage tests.test_stepc tests.test_final_model tests.test_predict tests.test_stream_esp32 -v` | nothing | ~4-5 min |

Baseline (frozen, `../wesad_baseline/`): re-run only into a folder OUTSIDE the repo, e.g.
`& $py ..\wesad_baseline\wesad_baseline.py --data "C:\datasets\WESAD" --out $env:TEMP\baseline_out --task binary`.

## Results (LOSO, 15 held-out subjects, 30 s windows, 5 s step)
Mean ± SD across subjects [subject-bootstrap 95 % CI, 2000 resamples]; copied from
`results/metrics/final_results.csv` (shown to 3 decimals, rates to 1; full precision in the CSV).

| model | operating point | F1 | sensitivity | specificity | precision | PR-AUC | ROC-AUC | false-alarm events/h (merged) | FP windows/h | worst subject (F1) |
|---|---|---|---|---|---|---|---|---|---|---|
| R (research reference, not deployed) | OP1 | 0.841 ± 0.132 [0.769, 0.903] | 0.927 [0.863, 0.976] | 0.820 [0.680, 0.927] | 0.802 [0.696, 0.889] | 0.948 [0.917, 0.976] | 0.977 [0.964, 0.988] | 16.9 [10.4, 24.2] | 129.3 [52.9, 230.1] | S13 0.574 |
| D (deployment candidate) | OP1 | 0.779 ± 0.217 [0.663, 0.871] | 0.799 [0.660, 0.912] | 0.892 [0.808, 0.961] | 0.846 [0.762, 0.923] | 0.951 [0.922, 0.976] | 0.974 [0.960, 0.986] | 18.7 [8.6, 30.5] | 77.9 [27.9, 138.3] | S2 0.140 |
| **D shipped (default)** | OP1 + 4-of-5 smoothing | 0.762 ± 0.244 [0.629, 0.863] | 0.745 [0.600, 0.867] | 0.929 [0.867, 0.980] | 0.895 [0.821, 0.958] | 0.951 [0.922, 0.976] | 0.974 [0.960, 0.986] | 7.6 [2.6, 13.5] | 50.8 [14.7, 95.8] | S2 0.033 |

Median alert latency after stress onset: 29.1 s (OP1), 44.1 s (shipped); no stress episode missed; worst case for the
shipped model 196 s (S2). **D's expected performance is its LOSO result above, and it is optimistic**: the
configuration was chosen after seeing held-out results from 192 candidates (NOTES.md Step C/D). The deployed model
itself is fitted on all 15 subjects and has no held-out estimate of its own.

## ESP32 feasibility: nothing has run on an ESP32
Firmware for the bench prototype is in `firmware/esp32/auralis_model_d/` (2026-11-12): a C port of
`src/stream_ref.py` that matched StreamD on synthetic signals (0 alert differences in 227 windows, float32 and
float64; host gcc). It has **not** been run on the ESP32 yet, and the model has **not** been validated on ESP32
hardware or prototype signals. Source:
`results/metrics/esp32_resources.csv`, `esp32_c_test.json`; write-up `results/metrics/esp32_feasibility.md`.

**MEASURED** (on this PC, or by compiling only):

| quantity | value | how |
|---|---|---|
| model files | joblib 1942 B, JSON 4117 B, C header 4316 B | file size |
| ESP32 object code, score + smoother | .text 154 B, .literal 28 B, .data 0, .bss 0 | xtensa-esp32-elf-gcc 13.2.0 -Os, **compile only, not run** |
| C header vs Python, 300 real WESAD windows | float build max \|dp\| 1.04e-7, double 3.3e-16; 0 alert differences | MSVC host build, run on this PC |
| streaming reference vs offline features, 5509 windows | 0 alert differences (float64 and float32) | `stream_parity_*.csv` |
| peak-detector work per window | ~2030 scan steps, ~72 local maxima, ~14436 prominence steps | instrumented reference (platform-independent counts) |
| PC timings (committed run) | JSON scorer 25.6 µs, joblib 622 µs per window; streaming reference 12.5 / 20.2 µs per sample (float64 / float32 emulated); C alert 2.9 / 3.4 ns | this PC; **not ESP32 timings** |

PC timings are not ESP32 timings and are not stable between runs; no tolerance is claimed. In the 2026-11-11
re-run (files saved outside the repo in `C:\auralis_verification_backup\regenerated_timing_run2\results\metrics\`,
same names as the committed `esp32_resources.csv` / `stream_parity_calibration.csv`) the streaming reference took
28.66 / 46.48 µs per sample (×2.29 / ×2.30), the JSON scorer 30.7 µs (×1.20), joblib 752.0 µs (×1.21), the C alert
2.68 / 3.93 ns (×0.93 / ×1.16). The cause of the slowdown was not investigated (NOTES.md "Clean-state verification").

**ESTIMATED** (arithmetic from buffer sizes and operation counts; assumptions stated):

| quantity | 64 Hz BVP / 4 Hz EDA (WESAD rates) | 100 Hz PPG / 10 Hz GSR (prototype rates) |
|---|---|---|
| RAM, feature front end + model | 15061 B (2.83 % of ~520 KB) | 24405 B (4.58 %) |
| CPU, one core at 240 MHz, 5-20 cycles per op (ASSUMED) | 0.07-0.30 % | 0.17-0.67 % |
| flash, model constants / DSP constants | 36 B / 116 B | same |
| flash, feature + gate + calibration code | 2-8 KB (GUESS; not written in C yet) | same |
RAM excludes the RTOS, WiFi/BT stacks, sensor drivers and logging.

## Limitations (read before quoting any number)
- **Stress, not panic or anxiety.** One lab stressor (TSST). The Auralis deck/IEC wording ("panic", "anxiety") is not
  supported by this model.
- **15 young, healthy subjects, one lab, one stressor** (WESAD; 12 male, 3 female, 24-35 years).
- **Different sensors and sites.** Training data are Empatica E4 WRIST signals (BVP 64 Hz, EDA in µS). The bench
  prototype (v0.2) uses a FINGERTIP **MAX30100** PPG (green MAX30100/30102 board, part ID 0x11; MAX30102 planned),
  a generic finger GSR module read in mV through the nonlinear ESP32 ADC, and an MPU6500 (GY-521 board sold as
  MPU6050); the MAX30205 is not fitted yet. Nothing has been checked on prototype data (none may be used before IEC
  approval).
- **GSR polarity.** Grove-type GSR modules output a voltage that FALLS as skin conductance rises (they measure
  resistance). The relative EDA features tolerate only a positive linear gain, so the firmware converts mV to
  conductance before the features; the conversion (Seeed formula) is UNVALIDATED on this module.
- **Session-time shortcut cannot be separated on WESAD.** Minutes since baseline start alone gives LOSO F1 0.94-0.95,
  higher than every physiological model; the stress-vs-amusement control cannot separate physiology from protocol
  position.
- **Poorly detected subjects.** S2 and S3 are poorly detected by D (shipped F1 0.033 and 0.446: missed stress);
  S13 depends on the configuration (R: false alarms, F1 0.574; D: 0.855 at OP1).
- **Optimistic estimate.** D's expected performance is its LOSO result, selected after seeing results.
- **False alarms:** about 7.6 merged events per hour for the shipped model, extrapolated from only about 20 minutes
  of non-stress data per subject.
- **PPG quality:** no PPG noise rejection beyond the flat/railed/clipped check (Step G). Band-limited noise passes
  the quality rule and yields a fake heart rate (documented expected-failure test).
- **The score is not a probability** (uncalibrated; only the thresholded alert is meaningful).
- **Not a medical device**; no diagnosis.
- Wrist HRV was dropped (does not track chest ECG); personalisation (z-score or per-wearer threshold) did not help on
  average (NOTES.md Steps A, C).
- Reproducibility: all features, LR results and model files reproduce byte-for-byte; the frozen baseline random
  forest's PR-AUC in the paper-reconciliation table is not reproducible in the last digits (multi-threaded
  `predict_proba`; NOTES.md "Clean-state verification").
