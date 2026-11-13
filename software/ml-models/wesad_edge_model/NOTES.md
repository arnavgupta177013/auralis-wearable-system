# NOTES — wesad_edge_model

Running log. Read this plus CLAUDE.md to continue in a fresh chat.
All numbers below come from real runs; nothing is estimated unless marked GUESS.

## Commits so far
- `1682090` baseline: Wilcoxon test added + `wesad_baseline/out/` results committed
- `dd54657` wesad_edge_model: CLAUDE.md + empty module skeleton
- `d4b9f8e` Initial import from ZIP (rest of repo; .venv/ and data/ excluded)

## Wording
WESAD "stress" = lab stressor (TSST: public speaking + mental arithmetic). It is NOT panic or anxiety.
The Vishwakarma Stage-2 deck ("Panic and Anxiety Detection", classes rest/exercise/anxiety/panic) and the
IEC proforma ("Anxiety Monitoring") use panic/anxiety wording. This model can only support "stress detection".

---

## Stage 0 — existing baseline (`software/ml-models/wesad_baseline/`, frozen)
Setup: E4 wrist signals, 60 s windows, 15 s step, purity >= 90 %, 15 subjects, LOSO.
2061 windows; 255 calibration windows (first 5 min of baseline, 17 per subject) excluded from scoring
for all methods -> 1806 scored. Features: 14 (HR/HRV, EDA, TEMP, ACC). Personal = z-score vs calibration.
Only change made to the baseline: additive `wilcoxon_tests()` (paired, two-sided, Holm).

### Binary (stress vs non-stress), mean +/- SD across 15 subjects
| method | F1 | accuracy |
|---|---|---|
| rule_universal | 0.674 +/- 0.221 | 0.767 +/- 0.191 |
| rule_personal | 0.723 +/- 0.219 | 0.814 +/- 0.135 |
| rf_universal | 0.908 +/- 0.114 | 0.937 +/- 0.083 |
| rf_personal | 0.846 +/- 0.236 | 0.918 +/- 0.107 |
| xgb_universal | 0.856 +/- 0.251 | 0.916 +/- 0.112 |
| xgb_personal | 0.821 +/- 0.272 | 0.899 +/- 0.116 |

### Three-class (baseline / stress / amusement), macro F1
| method | macro F1 | accuracy |
|---|---|---|
| rule_universal | 0.437 +/- 0.154 | 0.628 +/- 0.171 |
| rule_personal | 0.490 +/- 0.100 | 0.679 +/- 0.106 |
| rf_universal | 0.628 +/- 0.168 | 0.725 +/- 0.148 |
| rf_personal | 0.678 +/- 0.123 | 0.808 +/- 0.087 |
| xgb_universal | 0.646 +/- 0.184 | 0.750 +/- 0.149 |
| xgb_personal | 0.628 +/- 0.091 | 0.785 +/- 0.082 |

### Wilcoxon signed-rank, per-subject F1, personal minus universal (n = 15, Holm across 3)
| task | pair | median diff | W | p raw | p Holm | scipy method |
|---|---|---|---|---|---|---|
| binary | rule | -0.009 | 56 | 0.847 | 1.000 | exact |
| binary | rf | 0.000 | 47 | 0.730 | 1.000 | normal approx (1 zero diff) |
| binary | xgb | -0.013 | 47 | 0.730 | 1.000 | normal approx (1 zero diff) |
| three | rule | -0.011 | 46 | 0.454 | 0.909 | exact |
| three | rf | +0.062 | 29 | 0.083 | 0.250 | exact |
| three | xgb | +0.013 | 59 | 0.978 | 0.978 | exact |
No personal-vs-universal difference is significant. In binary, universal RF beats personal RF on the mean.
The baseline lacks AUC, specificity, false alarms/h and CIs; uses 60 s windows and non-causal filtfilt.

---

## Stage 1 — inspection (no code written; scratch scripts only)

### 1. Per-subject data (from the .pkl files)
- 15 subjects: S2–S11, S13–S17 (S1, S12 absent from release). 12 male, 3 female (S8, S11, S17); age 24–35.
- Wrist rates confirmed by length/duration: BVP 64 Hz, EDA 4 Hz, TEMP 4 Hz, ACC 32 Hz (3-axis, 1/64 g).
  For every subject all four wrist streams have exactly the same duration as the 700 Hz label vector.
- Session duration 87.1–117.9 min. **No NaNs** in any wrist signal.
- Minutes per label (all subjects): baseline 19.0–20.0, stress 10.3–12.1, amusement 6.0–6.4,
  meditation 12.2–13.4 (2 segments), label 0 (transient) 34–65, labels 5–7 1.9–3.5.
- Baseline is ONE contiguous segment of >= 19 min for every subject -> 5-min calibration always available.
- Baseline is ALWAYS the first condition. Order alternates: 1-2-4-3-4 (S2,S3,S6,S9,S11,S14,S16) or
  1-3-4-2-4 (S4,S5,S7,S8,S10,S13,S15,S17).
- Flat / saturation: no flat 10 s BVP or EDA blocks; ACC clipping (|raw| >= 127) <= 0.08 % of samples;
  EDA never < 0.045 µS, max 15.9 µS (no saturation); TEMP within 28.97–35.97 °C. TEMP has
  <= 0.8 % flat 10 s blocks (likely quantisation, harmless).
- Odd / noteworthy:
  - Very low EDA (baseline mean < 0.35 µS): S2, S4, S8, S9, S14, S16. Hyper-high: S7, S13 (~5 µS), S11 (~3 µS).
  - EDA hardly rises with stress in S14 (0.29 -> 0.34 µS) and falls in S17 (1.37 -> 1.05 µS).
  - Wrist TEMP mostly lower in stress than baseline (session-time / room confound, see below).
  - S15, S16 skin temp ~30 °C (cooler than others).
  - S10 resting HR ~98 bpm (median per-subject resting HR range 59–100 bpm).
  - Subject readme notes: S3 baseline in a sunny workplace; S6 "not really stressed" by TSST;
    S15 did not believe the TSST cover story; S8, S16 felt cold in the stress room; S5 may have slept
    in meditation 1; S8 had a stressful day already.

### 2. Worst subjects in the baseline (binary, per-subject F1)
- Lowest rf_personal: S2 0.311, S3 0.340, S14 0.612, S17 0.800, S15 0.857.
- Lowest rf_universal: S14 0.600, S10 0.703, S6 0.865, S2 0.896.
- Personal much worse than universal: S3 (-0.621), S2 (-0.584), S17 (-0.200), S15 (-0.108).
  Personal much better: S10 (+0.297), S6 (+0.135).
- Re-run of those folds (unmodified baseline functions) — confirmed failure mode:
  - S2 personal: TP 7, FN 30, FP 1. S3 personal: TP 8, FN 31, FP 0. -> **missed stress**, not false alarms.
    Universal got S2 TP 30/FN 7, S3 TP 37/FN 2.
  - Median personalised z in stress windows, S2+S3 vs others: scl_mean 0.01 vs 13.7, scr_count 0.4 vs 4.8,
    hr_mean 2.6 vs 6.5. Relative to their own calibration, S2/S3 show almost no EDA rise during stress.
  - S15, S17 personal: extra false alarms on amusement (S15 13, S17 7) plus 9 missed stress windows in S17.
  - S10 universal: 38 false alarms on baseline (resting HR ~98 bpm); personal removes all of them.
    This is the clearest case for a personal baseline.
  - S14 universal: 24/42 stress missed (weak EDA responder); personal catches more but adds 17 amusement FPs.
- Calibration-window checks:
  - Window count: every subject has exactly 17 calibration windows -> "too few windows" is ruled out as a cause.
  - ppg_quality (calib): S2 0.804, S3 0.717 vs median 0.838; motion_energy (calib): S2 0.048 (2nd highest),
    S3 0.031 vs median 0.026. S13/S15 have similar or worse calib quality and are not failures.
    Correlation of (rf_personal − rf_universal) with calib ppg_quality r = 0.32, with calib motion r = −0.28
    (n = 15, weak). The data do not show calibration quality as the main cause.
  - Calibration SD is tiny for some subjects, so z-scores explode (e.g. S14 HR z in stress = 50.8,
    S10 SCL z = 319, S11 TEMP z in amusement = −213). The personal feature scale varies hugely by subject.
  - Early-baseline drift: in 11/15 subjects, SCL in the rest of baseline is lower than in the calibration
    period (z −1.5 to −6.3) -> calibration EDA is elevated relative to later rest.
  - GUESS (not confirmed): S3's calibration EDA was raised by the warm/sunny baseline room (readme), so
    stress EDA looks "normal" relative to it. For S2 the cause is unknown (no readme note); possibly
    electrode settling in the first minutes. Either way, z-scoring against the first 5 min is fragile.

### 3. Class balance (60 s baseline windows, scored = calibration excluded)
- Baseline 869, stress 614, amusement 323 (total 1806). Binary: stress 614 (34.0 %) vs non-stress 1192 (66.0 %).
- Per subject stress fraction 0.325–0.357 (very even). Including calibration: baseline 1124.
- Imbalance is moderate; amusement is the minority in the 3-class task (17.9 %).

### 4. Dataset/baseline vs deck design — mismatches that affect ML design
| item | WESAD / baseline | Auralis deck | ML consequence |
|---|---|---|---|
| PPG | E4 BVP 64 Hz, green LED, processed unitless | MAX30102 100 Hz, red/IR | Train at 64 Hz; device must downsample 100->64 Hz (or features must be rate-independent: IBI-based). 64 Hz = 15.6 ms beat-timing resolution -> RMSSD/pNN50 coarse; use peak interpolation. Red/IR wrist PPG is more motion-sensitive than green -> SQI needed. |
| EDA | E4 4 Hz, µS, wrist electrodes | Grove GSR V2.0 10 Hz, raw ADC on GPIO34, 316L wrist pads | Downsample device to 4 Hz. Grove gives an ADC value, not µS, without calibration -> absolute SCL does not transfer; prefer relative/slope/phasic features. Note this conflicts with universal (absolute) features winning on WESAD. |
| TEMP | E4 IR thermopile 4 Hz | MAX30205 contact sensor 1 Hz | Downsample WESAD TEMP to 1 Hz. Absolute temp offset differs between sensors -> temp_mean non-transferable; slope only. Also session-time/room confound (temp falls across the session). |
| ACC | E4 32 Hz, ±2 g, 1/64 g units | MPU6050 50 Hz, range configurable | Resample device to 32 Hz, use g units, set ±2 g (or clip) to match. |
| Window | 60 s, 15 s step | 20 s (alert latency <= 30 s); deck also mentions 30 s for HRV | HRV over 20 s (~20–30 beats) is noisy; SCR counts are small. Must compare 20/30/60 s. |
| Filtering | `filtfilt` (zero-phase, non-causal) for BVP and EDA; centred moving-average EDA tonic (`uniform_filter1d`) | streaming ESP32 | Replace with causal SOS IIR (`sosfilt` with state) and causal EMA tonic. Per-window statistics computed at window end (mean, slope, peak count) are causal and fine. |
| Labels | 90 % purity windows; transitions dropped | continuous stream | Offline scores exclude transitions; real use will not. Report this caveat. |
| Classes | baseline / stress / amusement (+ meditation, unused) | rest / exercise / anxiety / panic | WESAD has no exercise, anxiety or panic. Participants are seated (median calib motion 0.026 g) -> no motion confounds to learn; exercise rejection must come from IMU gating, untested on WESAD. |
| Personal baseline | z-score vs first 5 min, right after sensor placement and before stress | z-score vs wearer's own rest | Fragile (drift, tiny SDs, see §2). Real-life calibration timing differs from WESAD. Not shown to help here. |
| Rule fallback | single HR threshold | HR +30 bpm in 20 s AND EDA +30 % AND low motion | Deck rule is untested; evaluate it in Stage 2. |
| Model | RF 200 trees depth 8; XGB 200 trees depth 4 | <= 100 kB int8, TFLite-Micro, ~520 KB SRAM | GUESS: RF-200/depth-8 likely > 100 kB; must measure. TFLite-Micro does not run trees natively -> trees need C codegen (e.g. emlearn/micromlgen) or use LR/small MLP. Size must be a selection constraint. |
| Population | 15 healthy, mostly male, ~27 y, Germany | Indian users, later clinical | Generalisation risk; out of scope for WESAD. |

---

## Stage 2 plan — features + exploratory analysis (PROPOSED, not implemented)
Goal: build streaming-portable features and find out which window length / normalisation / model works
under LOSO, without assuming personalisation helps.

1. Config (one place, e.g. `src/config.py` or dataclass): `WIN_S in {20, 30, 60}`, `STEP_S` (open question),
   `PURITY = 0.9`, `CALIB_MIN = 5`, target rates PPG 64 / EDA 4 / TEMP 1 / ACC 32 Hz.
2. `preprocessing.py`: load pkl -> wrist signals; downsample TEMP 4->1 Hz; causal SOS filters with explicit
   state; window generator that labels windows by purity; calibration flag (first 5 min of baseline,
   windows fully inside it).
3. `features.py`: HR, IBI-based HRV (with parabolic peak interpolation), PPG SQI; EDA causal-EMA tonic level,
   slope, phasic SCR count/amp; TEMP slope (+ temp_mean only as a flagged "absolute" feature);
   ACC magnitude mean, motion energy. Tag every feature as absolute vs relative.
   Unit test: causal-streaming implementation equals batch implementation on the same samples.
4. Normalisation variants (all tested for EVERY model and window):
   (a) universal raw; (b) personal z-score (as baseline); (c) personal delta (subtract calib mean only);
   (d) personal robust z (median/IQR with SD floor); (e) universal + personal concatenated;
   (f) relative-only feature set (drop absolute SCL/temp/HR) to mimic hardware transfer.
5. Models: HR rule, deck rule (HR +30 bpm AND EDA +30 % AND low motion), logistic regression,
   size-capped RF, size-capped XGB. Record model size in bytes for each.
6. Evaluation (`evaluate.py`): outer LOSO; inner GroupKFold by subject on the training subjects for
   hyperparameters, threshold and feature selection; held-out subject used once. Metrics: sensitivity,
   specificity, precision, F1, balanced accuracy, ROC-AUC, PR-AUC, false alarms/hour, worst subject;
   mean +/- SD, per-subject table, subject-level bootstrap 95 % CI. Paired Wilcoxon + Holm for
   personal-vs-universal per model and window.
7. Exploratory checks (notebook + `results/graphs`): per-subject feature distributions by label;
   calibration vs later-baseline drift; time-in-session confound test (classifier on t_start_s alone and on
   temp only — if they score well, that's a shortcut, not stress physiology); EDA responder analysis
   (S14, S17, S2, S3).
8. Outputs: `results/metrics/*.csv|json`, `results/graphs/*.png` titled "stress detection".
   Update this file with real numbers after the run.

### Decisions (user, 2026-11-10)
- Window step: fixed 5 s for all window lengths.
- Primary task: binary stress vs non-stress; three-class is secondary.
- Model ranking: PR-AUC. Decision threshold: chosen to maximise F1 (inner CV, training subjects only).
- False alarms/hour: merge consecutive false-positive windows into one alarm event; also report the raw
  per-window FP rate.
- Stage 2 reduced grid: normalisation = universal, personal z-score, relative-only (only these three).

---

## Stage 2 — features + exploratory analysis (DONE, real run 2026-11-10)
Code: `src/preprocessing.py`, `src/features.py`, `src/explore.py` (explore.py added for the EDA outputs).
Run from `software/ml-models/wesad_edge_model`:
`python -m src.features --data "C:\datasets\WESAD"` (~100 s) then `python -m src.explore`.
Outputs: `results/metrics/` (CSV/JSON), `results/graphs/` (PNG). 57 files, all verified non-empty.

### Pipeline as built
- Model rates: BVP 64 Hz, EDA 4 Hz, TEMP 1 Hz (4->1 Hz block mean), ACC 32 Hz in g.
- One causal forward pass over each whole recording: BVP Butterworth band-pass 0.7–3.5 Hz (order 3 per edge,
  SOS with state); EDA Butterworth low-pass 1 Hz (order 2) -> tonic = EMA (tau 8 s), phasic = EDA − tonic.
  Chunked filtering == one-pass filtering: max abs difference 0.0 (asserted per subject).
- Windows 20/30/60 s, fixed 5 s step, purity >= 90 %, labels 1/2/3. Peak times use parabolic interpolation.
- Calibration: `calib` = window fully inside first 5 min of baseline (stats only);
  `in_calib_period` = overlaps it -> never scored. `assert_calibration_never_scored()` runs in features.py
  and explore.py. (The old baseline scored windows that partly overlapped the calibration period.)
- 14 features: hr_mean, hr_std, rmssd, sdnn, pnn50, scl_mean, eda_std, eda_slope, scr_count, scr_amp_mean,
  temp_mean, temp_slope, acc_mag_mean, motion_energy. Relative-only set (10): drops scl_mean, eda_std,
  scr_amp_mean, temp_mean. Caveat: scr_count (0.01 µS threshold) and eda_slope (µS/s) still depend on EDA gain.
- Personal z floor: SD = max(own calibration SD, median calibration SD of the OTHER subjects). Label-free and
  test-subject-free. (Baseline floor used 0.1 × SD over all of the subject's windows incl. stress = mild leak.)

### Window counts (15 subjects; scored = not overlapping calibration)
| window | total | calib (stats) | excluded (overlap calib) | scored | baseline | stress | amusement | stress share |
|---|---|---|---|---|---|---|---|---|
| 20 s | 6479 | 840 (56/subj) | 907 | 5572 | 2565 | 1942 | 1065 | 34.9 % |
| 30 s | 6421 | 810 (54/subj) | 912 | 5509 | 2541 | 1923 | 1045 | 34.9 % |
| 60 s | 6196 | 720 (48/subj) | 920 | 5276 | 2457 | 1848 | 971 | 35.0 % |
Per-subject stress share 0.34–0.37 (20 s). Windows overlap heavily (5 s step) -> not independent samples.

### Causal vs zero-phase (filtfilt per window, same peak code), all windows
| window | HR bias (bpm) | HR mean abs diff | HR r | RMSSD bias (ms) | RMSSD mean abs diff | RMSSD r | SDNN bias (ms) | pNN50 bias |
|---|---|---|---|---|---|---|---|---|
| 20 s | −0.96 | 3.54 | 0.944 | −29.2 | 42.5 | 0.525 | −13.2 | −0.106 |
| 30 s | −0.91 | 2.90 | 0.962 | −29.4 | 38.9 | 0.583 | −12.8 | −0.104 |
| 60 s | −0.88 | 2.13 | 0.981 | −29.1 | 34.7 | 0.659 | −11.7 | −0.103 |
HR agrees closely; HRV does not (zero-phase mean RMSSD ~126–129 ms, causal ~29 ms lower). Both RMSSD values are far
above typical resting RMSSD, so wrist-PPG beat jitter likely dominates both. GUESS (unverified): the causal
version is less noisy because it has no per-window edge transients. Which is closer to truth is UNKNOWN —
can be checked against WESAD chest ECG (not done).

### Quality flags (flagged, never removed), % of all windows
| window | PPG (<50 % good IBIs / NaN HR / too few beats) | motion (SD of acc magnitude > 0.1 g) | EDA flat (range < 0.005 µS) | EDA low (< 0.05 µS) | any |
|---|---|---|---|---|---|
| 20 s | 171 (2.6 %) | 549 (8.5 %) | 4 (0.06 %) | 0 | 682 (10.5 %) |
| 30 s | 133 (2.1 %) | 542 (8.4 %) | 0 | 0 | 630 (9.8 %) |
| 60 s | 88 (1.4 %) | 498 (8.0 %) | 0 | 0 | 560 (9.0 %) |
- **Motion flags are almost all stress windows**: 20 s scored: 494/1942 stress (25.4 %) vs 21/2565 baseline (0.8 %),
  7/1065 amusement. TSST = speaking/gesturing. Motion is a stress proxy in WESAD; gating out "motion" windows on
  the device would also gate out stress, and motion features may learn "talking", not physiology.
- PPG flags: 4.9 % of stress vs 1.4 % of baseline windows (20 s). Most PPG flags: S3 29, S2 26, S11 24, S5 23.

### Personal z-score investigation (the 50–319 values)
- Cause: the calibration SD of slow features is tiny because the 5-min rest is very stable and windows overlap
  (S16 scl_mean calib SD 0.007 µS, S10 0.010 µS, S4 eda_std 0.001 µS) while stress shifts SCL by 1–2 µS.
  No-floor stress z: S16 scl 256, S10 scl 211, S4 scl 114, S4 eda_std 106, S11 temp −70 (20 s).
- With the leave-subject-out floor (20 s): max |z| scl_mean 20.6, temp_mean 45.3, hr_mean 11.2, eda_std 3.7,
  scr_count 3.6, scr_amp 4.0; floor applied for 7–8 of 15 subjects per feature. Remaining large values for
  scl_mean / temp_mean / hr_mean are real large shifts, not divide-by-near-zero.
- Old baseline-style floor gives similar or smaller numbers but uses the subject's own stress data (leak).

### Per-subject shifts, stress minus scored baseline (median, personal z, 20 s)
- Consistent across subjects: hr_mean up 15/15 (median +3.0 z), hr_std up 15/15, eda_std up 15/15,
  scl_mean up 14/15 (+6.4), scr_count up 14/15 (+2.2, 1 zero), scr_amp up 14/15.
- temp_mean down 12/15 (median −11.0) and temp_slope down 13/15 — consistent with session-time/room drift.
- Inconsistent: eda_slope 8 up / 7 down; sdnn 11/4; acc_mag_mean 9/6.
- S14: no EDA response at all (scl 0.4, scr_count 0.0) but the largest HR rise (+10.7). S17: SCL falls (−1.3).

### Time-in-session shortcut (LOSO, binary, depth-3 tree, one input, scored windows)
| input | 20 s PR-AUC | 30 s PR-AUC | 60 s PR-AUC | 20 s ROC-AUC |
|---|---|---|---|---|
| minutes since baseline start | 0.950 ± 0.088 | 0.951 ± 0.086 | 0.965 ± 0.074 | 0.979 ± 0.045 |
| hr_mean | 0.779 ± 0.168 | 0.800 ± 0.184 | 0.820 ± 0.191 | 0.868 ± 0.113 |
| scl_mean | 0.695 ± 0.291 | 0.697 ± 0.293 | 0.701 ± 0.298 | 0.728 ± 0.332 |
| temp_mean | 0.578 ± 0.254 | 0.578 ± 0.255 | 0.587 ± 0.256 | 0.657 ± 0.272 |
(chance PR-AUC ≈ 0.35). **Time alone beats every physiological feature.** The fixed protocol is a strong
shortcut: never give a model time, and treat slowly drifting features (temp_mean, scl_mean, temp_slope) as
suspect — they can encode "time since sensor placement" rather than stress.

### Other EDA
- Resting HR median per subject 56.0–98.6 bpm (20 s windows, all baseline windows incl. calibration).
- Spearman |ρ| >= 0.7 (20 s): rmssd–sdnn 0.89, rmssd–pnn50 0.84, sdnn–pnn50 0.76, hr_std–rmssd 0.71,
  scl_mean–eda_std 0.80, scl_mean–scr_amp 0.70, eda_std–scr_count 0.84, eda_std–scr_amp 0.87,
  scr_count–scr_amp 0.91. Redundant HRV and EDA groups -> candidates for pruning (ESP32 size).

### Implications for Stage 3 (modelling) — proposals, not decisions
- Keep universal vs personal_z vs relative_only for every model (universal may still win).
- Report results with and without motion features, and with and without temp_mean/scl_mean, because of the
  motion-stress and time-drift confounds above.
- Consider validating wrist HR/HRV against chest ECG before trusting HRV features.

---

## Step A — chest-ECG reference check for wrist HR/HRV
Purpose: decide whether wrist HRV (RMSSD, SDNN, pNN50) is trustworthy enough to stay in the model.
The chest ECG (RespiBAN, 700 Hz) is a REFERENCE ONLY and is never a model input.

### Pre-registered keep/drop criteria (written 2026-11-10 BEFORE any chest-ECG number was computed; not to be changed)
Comparison = wrist value (current causal features in `features_w{W}.csv`) minus chest-ECG value, same window.
Evaluated separately at each window length (20, 30, 60 s) and in each condition (baseline, stress, amusement).
Windows used: every window in `features_w{W}.csv` (labels 1/2/3, calibration windows included) where the wrist
value is not NaN and the chest window is valid (>= 90 % of chest RR intervals pass the artefact check).
"RM LoA" = Bland-Altman 95 % limits of agreement with the repeated-measures variance correction
(Bland & Altman 2007: between-subject + within-subject variance of the differences).
"rmcorr" = repeated-measures (within-subject) correlation: both variables centred on each subject's mean, then Pearson.

HR (`hr_mean`, bpm) passes if, in EVERY condition:
- H1 pooled Pearson r >= 0.80
- H2 |bias| <= 3 bpm
- H3 RM 95 % LoA inside [-10, +10] bpm
- and H4 rmcorr over all conditions together >= 0.70

Each HRV feature (`rmssd` ms, `sdnn` ms, `pnn50` fraction) passes if, in EVERY condition:
- V1 pooled Spearman rho >= 0.50 (Spearman: HRV is skewed)
- V3 |bias| <= 10 ms (pNN50: <= 0.10)
- V4 RM 95 % LoA inside [-40, +40] ms (pNN50: [-0.30, +0.30])
- and V2 rmcorr over all conditions together >= 0.40

Verdict per window length:
- **keep HRV**: HR passes H1-H4 and all three HRV features pass V1-V4.
- **keep HRV with caveats**: HR passes; every HRV feature passes V1 and V2 (it tracks the reference) but at least one
  fails V3 or V4 (offset / scatter). HRV stays; values are not interpreted as physiological ms; a no-HRV variant is
  also run in Step B.
- **keep HR only**: HR passes; any HRV feature fails V1 or V2. Drop rmssd, sdnn, pnn50 (hr_std is reported but is
  not part of this decision). Step B runs a no-HRV variant of every scheme and uses it as the main configuration.
- If HR itself fails any of H1-H4: HR stays (core feature) but the verdict is "keep with caveats (HR not verified)",
  and HRV is judged as above.
If the verdict is not "keep HRV" at ANY window length, Step B runs the no-HRV variant at ALL window lengths.

### Method as built (`src/hrv_check.py`, real run 2026-11-10)
Run: `python -m src.hrv_check --data "C:\datasets\WESAD"` (~3 min). Outputs: `results/metrics/hrv_check_*.csv|json`,
`results/graphs/hrv_check_bland_altman_w{20,30,60}.png`, `results/graphs/hrv_check_ecg_peaks.png`.
- R-peaks: chest ECG band-pass 5–20 Hz (order 2, zero-phase: offline reference only), polarity chosen per subject
  (all 15 came out positive), `find_peaks` with 300 ms refractory period and a per-10 s adaptive height
  (35 % of the block's 98th percentile), peak moved to the 0.5–40 Hz ECG maximum within ±50 ms + parabolic refinement.
  RR artefact check: 0.3–2.0 s and within 20 % of the running median of 11 RR.
- Peak quality checked by (a) good-RR fraction per subject × condition: 0.836 (S7 stress) to 1.000, median 0.99;
  (b) `hrv_check_ecg_peaks.png`: an 8 s stress excerpt per subject, inspected by eye — every visible R wave has
  exactly one dot, no T-wave or noise detections.
- Chest windows: the exact `(subject, t_start_s, t_end_s)` grid of `features_w{W}.csv` (merge asserted one-to-one).
  Chest window valid if >= 90 % of its RR pass; invalid: 403 (20 s), 357 (30 s), 257 (60 s) of ~6400 windows.
  Chest RMSSD/pNN50 use successive differences of ADJACENT good RR only.
- Alignment: the WESAD pkl streams are pre-synchronised by the dataset authors; chest sample i = i/700 s, wrist
  sample j = j/64 s, both from 0. Asserted per subject that chest and wrist durations are equal. Check: lag that
  maximises the correlation of 4 Hz instantaneous HR (chest vs causally filtered wrist BVP) = +1.25 to +2.5 s
  (median +1.75 s; wrist behind chest) in all 15 subjects. Expected from causal BVP filter group delay (0.32 s at
  1.2 Hz, 0.75 s at 0.8 Hz) + pulse transit time (~0.2–0.3 s, literature value) ≈ 0.5–1 s; the remaining ~1 s is
  UNEXPLAINED (GUESS: E4 internal BVP processing delay or a small sync offset). Sensitivity: re-pairing every wrist
  window with the chest window shifted by 1.75 s changes nothing material (20 s HR stress MAE 8.15 -> 8.05 bpm,
  RMSSD rmcorr 0.051 -> 0.052; `hrv_check_agreement_lag_shifted.csv`). The lag does not explain the disagreement.
- Repeated measures: LoA with the Bland & Altman 2007 variance-components correction (they are almost identical to
  the naive LoA here because within-subject variance dominates); rmcorr for within-subject tracking. Pooled
  correlations are descriptive; windows within a subject overlap and are NOT independent, so no p-values are given.

### Results — wrist minus chest, pooled over 15 subjects (all windows incl. calibration; `hrv_check_agreement.csv`)
| W | feature | condition | n | bias | RM 95 % LoA | Pearson | Spearman | rmcorr | MAE |
|---|---|---|---|---|---|---|---|---|---|
| 20 | hr_mean | baseline | 3330 | +1.48 bpm | [−10.1, 13.0] | 0.870 | 0.885 | 0.638 | 2.60 |
| 20 | hr_mean | stress | 1732 | −1.08 | [−29.4, 27.3] | 0.739 | 0.730 | 0.516 | 8.15 |
| 20 | hr_mean | amusement | 1013 | +1.38 | [−11.5, 14.3] | 0.808 | 0.861 | 0.530 | 2.73 |
| 20 | hr_mean | all | 6075 | +0.73 | [−17.5, 18.9] | 0.874 | 0.889 | 0.856 | 4.20 |
| 20 | rmssd | baseline | 3330 | +41.5 ms | [−42.0, 124.9] | 0.319 | 0.316 | 0.149 | 42.7 |
| 20 | rmssd | stress | 1732 | +89.4 | [−5.9, 184.7] | 0.249 | 0.276 | 0.065 | 89.4 |
| 20 | rmssd | amusement | 1013 | +38.2 | [−42.4, 118.8] | 0.530 | 0.569 | 0.341 | 39.3 |
| 20 | rmssd | all | 6075 | +54.6 | [−41.8, 151.0] | 0.202 | 0.207 | 0.051 | 55.4 |
| 20 | sdnn | stress | 1732 | +41.5 ms | [−19.0, 102.0] | 0.380 | 0.375 | 0.217 | 42.4 |
| 20 | sdnn | all | 6075 | +24.8 | [−29.7, 79.3] | 0.507 | 0.509 | 0.447 | 26.9 |
| 20 | pnn50 | stress | 1732 | +0.466 | [−0.01, 0.94] | 0.188 | 0.156 | 0.022 | 0.467 |
| 20 | pnn50 | all | 6075 | +0.252 | [−0.24, 0.74] | 0.350 | 0.321 | 0.145 | 0.268 |
| 60 | hr_mean | baseline | 3305 | +0.73 | [−5.4, 6.9] | 0.959 | 0.949 | 0.730 | 1.41 |
| 60 | hr_mean | stress | 1694 | −1.50 | [−25.6, 22.6] | 0.799 | 0.795 | 0.605 | 6.56 |
| 60 | rmssd | all | 5939 | +57.5 | [−20.9, 135.9] | 0.282 | 0.261 | 0.053 | 57.6 |
| 60 | sdnn | all | 5939 | +22.7 | [−21.9, 67.3] | 0.621 | 0.618 | 0.522 | 23.9 |
30 s rows are in the CSV (mostly between 20 and 60 s, but not on every row, e.g. amusement HR rmcorr 0.530 / 0.547 / 0.421).
- Chest RMSSD falls with stress (baseline 45 -> stress 25 ms, the expected physiology); wrist RMSSD RISES
  (87 -> 115 ms). Wrist pNN50 in stress 0.57 vs chest 0.10. Wrist HRV moves in the WRONG direction under stress:
  it is dominated by beat-timing noise, which grows with motion/stress, not by vagal HRV.
- Per subject (20 s, all conditions) wrist RMSSD Spearman vs chest ranges −0.43 (S11) to 0.53 (S9); bias +31 to +93 ms.
- hr_std (reported only, not in the criteria) also fails to track in stress: Spearman 0.06 (20 s), bias +6.7 bpm.
- Stress HR error (20 s, MAE): S11 26.3, S5 14.4, S7 10.7, S2 9.7 bpm vs S14 2.7, S6 3.4. In stress windows the mean
  abs HR error is 12.5 bpm with flag_motion vs 6.8 without, and 18.0 with flag_ppg vs 7.7 without. The Bland-Altman
  plot shows the large stress errors are mostly wrist HR far BELOW chest at chest HR 100–120 bpm (GUESS: missed
  beats during speaking/gesturing).

### Verdict against the pre-registered criteria
| W | HR H1–H4 | failed HR checks | HRV tracks (V1+V2) | HRV value agreement (V3+V4) | verdict |
|---|---|---|---|---|---|
| 20 | fail | H3 LoA in all 3 conditions; H1 stress (r 0.739); H4 rmcorr 0.856 passes | none of rmssd/sdnn/pnn50 | none | **keep HR only** (HR with caveats) |
| 30 | fail | H3 all conditions; H1 stress (0.771) | none | none | **keep HR only** (HR with caveats) |
| 60 | fail | H3 stress + amusement; H1 stress (0.799) | none | none | **keep HR only** (HR with caveats) |
- HR: unbiased on average (|bias| <= 1.5 bpm everywhere) and tracks within-subject change (rmcorr 0.86–0.91), but
  individual windows can be far off, especially in stress (LoA ±25–29 bpm). Kept, with that caveat.
- HRV (rmssd, sdnn, pnn50): fails V1 and V2 at every window length; wrist RMSSD has the wrong sign of change
  under stress. **Dropped as physiological HRV.** Per the pre-registered rule, Step B runs a no-HRV variant of every
  scheme at all window lengths and uses it as the main configuration. (A model may still find the wrist "HRV"
  predictive, but then it measures PPG noise/motion during the TSST, not HRV.)

### What could NOT be verified
- Whether the extra ~1 s of wrist lag is an E4 internal delay or a sync offset (no ground-truth sync marker used).
- Chest-ECG R-peak accuracy beyond visual inspection of 15 excerpts and the RR artefact rate (no annotated beats).
- EDA, TEMP and ACC features: no comparable reference (chest EDA/TEMP are at a different body site).
- Whether HRV from the Auralis MAX30102 (red/IR, 100 Hz) would be better or worse than the E4.
- hr_std was not part of the decision; it shows the same failure pattern in stress but is kept in the no-HRV variant
  because the criteria did not cover it.

---

## Step B — validation harness and baselines (design fixed 2026-11-10, BEFORE running)
Code: `src/checks.py` (leakage assertions), `src/schemes.py` (feature schemes, fold-aware personal z),
`src/baselines.py` (majority, HR thresholds, time-only reference), `src/evaluate.py` (metrics, bootstrap, Wilcoxon,
plots), `src/train.py` (LOSO harness, CLI). Tests: `tests/test_leakage.py` (stdlib unittest; pytest not installed).
Binary task only: stress (label 2) vs non-stress (labels 1 and 3).

### Design choices
- **Window population**: training AND scoring use only `scored` windows (not overlapping the first 5 min of
  baseline). Calibration windows (fully inside the first 5 min) are used ONLY to compute each subject's personal
  mean/SD; windows that partly overlap the calibration period are used for nothing. Train and test therefore come
  from the same population.
- **Outer loop**: LOSO over 15 subjects. **Inner loop**: GroupKFold(7) by subject on the 14 training subjects.
  Hyperparameters ranked by mean inner-fold PR-AUC; decision threshold = max F1 on the pooled inner out-of-fold
  predictions of the chosen setting; final model refit on all 14 training subjects; held-out subject scored once.
- **Personal z**: (x − own calib mean) / max(own calib SD, floor); floor = median calibration SD of the TRAINING
  subjects of that fold (inner-training subjects inside inner folds). The test subject's own calibration windows
  (unlabelled rest data, as a device would have) give its mean/SD, never the floor.
- **Feature sets**: universal = 14 features; personal_z = same 14, z-scored; relative_only = hr_mean, hr_std, rmssd,
  sdnn, pnn50, scr_count, eda_slope, temp_slope, acc_mag_mean, motion_energy. Factors: motion (acc_mag_mean,
  motion_energy) on/off for all schemes; drift (temp_mean, scl_mean) on/off for universal and personal_z only
  (relative_only never has them). no-HRV variant (drop rmssd, sdnn, pnn50) for every scheme, because Step A said
  "keep HR only". 20 logistic-regression configurations per window length.
- **Main configurations** (used for the Wilcoxon tests and per-subject F1 plots): each scheme with motion ON,
  drift ON (where applicable), **no-HRV** (Step A verdict).
- **Models**: (a) majority class; (b) universal HR threshold (score = hr_mean); (c) personal HR z threshold
  (score = personal-z hr_mean, fold-aware floor); (d) logistic regression, class_weight balanced,
  C in {0.001, 0.01, 0.1, 1, 10}, median imputer + standard scaler inside the pipeline (fitted on training rows only).
  Time-only reference: decision tree (depth tuned in {1, 2, 3, 4} by inner folds) on minutes since baseline start,
  labelled "not a candidate". Seeds fixed (0).
- **False alarms/hour**: non-stress time = (number of scored non-stress windows) × 5 s step. Reported per FP window
  and per alarm event (consecutive FP windows on the 5 s grid merged into one event).
- **Best LR configuration** (for ROC/PR curves and the PPG-flag sensitivity analysis): highest mean per-subject
  PR-AUC across all LR configurations and window lengths. It is chosen after seeing held-out results, so its
  numbers are optimistically selected; it is a reporting choice, not a tuning step.
- **Shuffled-label check**: universal, motion on, drift on, no-HRV, 20 s. Labels permuted within each training
  subject in every outer fold (inner folds see the shuffled labels too); test labels untouched. Pass if the mean
  per-subject PR-AUC is within ±0.10 of the mean per-subject stress prevalence; otherwise STOP (leakage warning).
- **Statistics**: paired Wilcoxon signed-rank on per-subject F1 (n = 15), the three main configurations pairwise,
  Holm across the 3 pairs within each window length. Exact distribution if there are no zero differences and no
  tied |differences|, otherwise normal approximation; the method used is reported per test.

### Step B run 1 (2026-11-10): STOPPED at the shuffled-label sanity check
- Leakage unit tests: 8/8 pass (`python -m unittest tests.test_leakage -v`); each deliberate violation fires.
- Shuffled-label check (LR universal +motion +drift −HRV, 20 s, labels permuted within each training subject,
  seed 0): mean per-subject PR-AUC 0.450 ± 0.245 vs mean prevalence 0.348 -> difference 0.102 > tolerance 0.10
  -> **FAILED the pre-set rule; run stopped as designed (exit code 2). Main grid NOT run. Criterion NOT changed.**
- What the data show (`stepb_shuffled_label_per_subject.csv`): mean ROC-AUC 0.468 (median 0.359); per-subject
  ROC-AUC spans 0.003 (S6) to 0.941 (S13), 7/15 subjects above 0.5. Every fold's F1-optimal threshold predicted all
  windows as stress (sensitivity 1.0, specificity 0.0 for all 15).
- GUESS (not verified): no consistent learned signal (ROC-AUC not above 0.5), but the features separate stress so
  strongly within each subject that any random weight direction ranks a subject's stress windows almost perfectly
  forwards or backwards, and average precision is asymmetric (perfect ranking -> 1.0, inverted -> ~0.2), so mean
  PR-AUC sits above prevalence. Needs the user's decision before going further.

### Shuffled-label criterion CHANGED AFTER SEEING the first result (user decision, 2026-11-10)
- First result (run 1, seed 0): mean per-subject PR-AUC 0.450 vs mean prevalence 0.348 (difference 0.102 > 0.10)
  -> failed the original rule.
- Why the original rule was set: the brief said "confirm PR-AUC falls to about the stress prevalence (~0.35)",
  because an uninformative ranking has an expected average precision equal to prevalence. I set ±0.10 as "about".
  That rule used ONE shuffle and assumed PR-AUC is symmetric around prevalence; it did not anticipate that, with
  strongly separable features, per-subject PR-AUC can be pushed above prevalence without any learned signal.
- New criterion (written BEFORE the multi-seed run): repeat the shuffled-label LOSO for the same configuration
  (LR universal +motion +drift −HRV, 20 s, labels permuted within each training subject) with 20 shuffle seeds
  (0–19; seed 0 = run 1). Model, C grid, inner folds and threshold selection unchanged.
  **PASS if** the mean of the 20 per-seed mean ROC-AUCs lies in [0.45, 0.55] **AND** the 95 % interval of the seed
  means (2.5th–97.5th percentile of the 20 values) contains 0.5. A mean consistently above 0.5 = leakage warning.
- The distribution of per-seed mean PR-AUC is reported as the EMPIRICAL NULL; the real model's PR-AUC is compared
  with this null, not with prevalence.

### Shuffled-label check, 20 seeds (real run 2026-11-10) — PASSED under the changed criterion
Run: `python src\train.py --windows 20 --shuffle-only` (~1 min). Files: `stepb_shuffled_label_seeds.csv`,
`stepb_shuffled_label_per_subject.csv` (seed × subject, incl. AP of the reversed score), `stepb_shuffled_label.json`.
| quantity (per-seed mean over 15 subjects) | mean | SD | min | max | 2.5–97.5 % of seeds |
|---|---|---|---|---|---|
| ROC-AUC | 0.529 | 0.064 | 0.417 | 0.664 | 0.427 – 0.630 |
| PR-AUC (= empirical null) | 0.512 | 0.049 | 0.436 | 0.629 | 0.437 – 0.602 |
- Mean prevalence 0.348. Seed 0 reproduces run 1 exactly (ROC 0.468, PR 0.450).
- Criterion: mean 0.529 in [0.45, 0.55] ✓; interval [0.427, 0.630] contains 0.5 ✓ -> PASS.
- Caveat: 0.529 is ~2 standard errors above 0.5 if the 20 seeds are treated as independent (SE 0.064/√20 = 0.014);
  they are not fully independent (same data, same folds). 7/20 seed means are below 0.5, 13 above. Not "consistently
  above 0.5", but not exactly 0.5 either. GUESS: chance, given the very high per-fold variance; not verified.
- Folds (300 = 20 seeds × 15 subjects): 45 % have ROC-AUC < 0.5; 37.7 % are extreme (< 0.2 or > 0.8); only 22 % lie
  in 0.4–0.6. **The bimodal pattern repeats.** Every subject swings from near-inverted to near-perfect across seeds
  (e.g. S4 0.007–0.986, S7 0.010–0.990, S14 0.035–0.996); per-subject mean ROC-AUC across seeds 0.41–0.65.
- Asymmetry, tested directly: mean PR-AUC is 0.684 in folds with ROC-AUC > 0.5 and 0.301 in folds with ROC-AUC < 0.5;
  the average of AP(score) and AP(−score) is 0.488 — well above prevalence 0.348 although the two directions are
  exact opposites. So for these strongly separable features, chance PR-AUC per subject is ~0.49–0.51, not 0.35.
- **The explanation holds** on both counts: (1) a random weight direction ranks each subject's stress windows
  strongly forwards or backwards (bimodal, flips with seed), and (2) average precision is asymmetric, so its chance
  level here is ~0.51. Not verified: WHY the per-subject separation is so strong for a random direction (GUESS:
  several features shift together in stress, so most directions in feature space correlate with stress).
- Consequence: real-model PR-AUC is compared with the empirical null (mean 0.512, max 0.629), not with 0.35.

### Grid update before the full run (user decision, 2026-11-10, written BEFORE running the grid)
- HRV (rmssd, sdnn, pnn50) removed from ALL main configurations (Step A verdict).
- hr_std added as an on/off factor (it failed the same way as HRV in stress, Step A).
- Main LR grid per window: universal and personal_z: motion × drift × hr_std (8 each); relative_only: motion ×
  hr_std (4) -> 20 main configurations, all without HRV.
- With-HRV REFERENCE (not a candidate, labelled "(HRV reference)"): one per scheme, motion on, drift on, hr_std on.
- Main configurations for Wilcoxon/plots stay as pre-specified: motion on, drift on, no HRV, hr_std ON
  (the feature set used in the shuffled-label check). hr_std-off results are in the combined table.
- Best LR configuration = highest mean per-subject PR-AUC among the 60 main (no-HRV) configurations only.
- Real PR-AUC is compared with the shuffled-label empirical null (mean 0.512, 97.5 % 0.602, max 0.629), which
  strictly applies to LR universal +motion +drift +hr_std −HRV at 20 s.

### Step B results — full grid (real run 2026-11-10)
Run: `python src\train.py --windows 20 30 60` (~6 min, 12 cores): 20-seed shuffled check (pass, identical to the
diagnostic run) then 81 LOSO jobs (27 specs × 3 windows). All 17 output files verified non-empty; the metric
checklist (every required metric, mean, SD, 2000-resample subject-bootstrap CI, pooled confusion matrix, both
false-alarm rates, worst subject) is filled for all 81 rows. Leakage assertions ran on every outer and inner fold
without firing. Files: `stepb_results.csv` (combined table), `stepb_per_subject.csv`, `stepb_folds.csv` (chosen C /
depth and threshold per fold), `stepb_wilcoxon.csv`, `stepb_sensitivity_ppg.csv`, `stepb_pr_auc_vs_null.csv`,
`stepb_predictions_best.csv`, `stepb_summary.json`; plots `stepb_per_subject_f1_w{20,30,60}.png`,
`stepb_config_comparison_{pr_auc,f1}.png`, `stepb_roc_pr_best.png`.
(The per-subject and comparison plots were redrawn from the saved CSVs after a layout-only fix in evaluate.py
— clipped title/legend; no numbers changed.)

Mean ± SD across 15 held-out subjects [subject-bootstrap 95 % CI]. FA = false-alarm events/h (merged) / FP windows/h.
| W | configuration | F1 | PR-AUC | ROC-AUC | sens | spec | FA events/h / windows/h | worst (F1) |
|---|---|---|---|---|---|---|---|---|
| 20 | majority class | 0.000 | 0.348 | 0.500 | 0.000 | 1.000 | 0 / 0 | all 0 |
| 20 | HR threshold (universal) | 0.635 ± 0.205 [0.534, 0.732] | 0.831 ± 0.151 | 0.884 | 0.726 | 0.724 | 24.3 / 198.4 | S6 0.179 |
| 20 | HR z threshold (personal) | 0.696 ± 0.157 [0.617, 0.768] | 0.831 ± 0.151 | 0.884 | 0.728 | 0.826 | 27.4 / 125.2 | S2 0.348 |
| 20 | LR universal +motion +drift +hr_std −HRV (main) | 0.815 ± 0.140 [0.747, 0.879] | 0.942 ± 0.069 [0.905, 0.973] | 0.970 | 0.842 | 0.886 | 23.6 / 81.8 | S9 0.535 |
| 20 | LR personal z +motion +drift +hr_std −HRV (main) | 0.903 ± 0.154 [0.817, 0.961] | 0.964 ± 0.072 [0.927, 0.993] | 0.982 | 0.918 | 0.950 | 9.7 / 35.9 | S3 0.403 |
| 20 | LR relative-only +motion +hr_std −HRV (main) | 0.807 ± 0.112 [0.750, 0.859] | 0.926 ± 0.076 [0.885, 0.959] | 0.963 | 0.854 | 0.850 | 25.8 / 107.9 | S13 0.595 |
| 20 | LR personal z −motion +drift −hr_std −HRV (best F1 at 20 s) | 0.919 ± 0.159 [0.827, 0.974] | 0.978 ± 0.048 | 0.989 | 0.915 | 0.976 | 6.9 / 17.4 | S3 0.372 |
| 20 | time only (NOT a candidate) | 0.941 ± 0.073 [0.902, 0.973] | 0.953 ± 0.082 | 0.980 | 0.940 | 0.969 | 1.6 / 22.1 | S9 0.776 |
| 30 | HR threshold (universal) | 0.632 ± 0.250 | 0.862 ± 0.145 | 0.902 | 0.741 | 0.716 | 16.5 / 204.5 | S6 0.031 |
| 30 | HR z threshold (personal) | 0.717 ± 0.181 | 0.862 ± 0.145 | 0.902 | 0.749 | 0.844 | 17.8 / 112.3 | S2 0.297 |
| 30 | LR universal main | 0.840 ± 0.132 [0.777, 0.900] | 0.956 ± 0.058 | 0.977 | 0.871 | 0.892 | 16.2 / 78.0 | S9 0.571 |
| 30 | LR personal z main | 0.908 ± 0.134 [0.837, 0.961] | 0.969 ± 0.071 | 0.984 | 0.920 | 0.952 | 6.5 / 34.8 | S3 0.494 |
| 30 | LR relative-only main | 0.826 ± 0.120 [0.766, 0.881] | 0.947 ± 0.065 | 0.974 | 0.868 | 0.869 | 18.5 / 94.6 | S15 0.560 |
| 30 | time only (NOT a candidate) | 0.953 ± 0.062 | 0.969 ± 0.062 | 0.984 | 0.948 | 0.977 | 1.6 / 16.4 | S17 0.781 |
| 60 | HR threshold (universal) | 0.648 ± 0.247 | 0.902 ± 0.132 | 0.929 | 0.749 | 0.713 | 9.6 / 206.7 | S6 0.016 |
| 60 | HR z threshold (personal) | 0.756 ± 0.155 | 0.902 ± 0.132 | 0.929 | 0.797 | 0.827 | 10.3 / 124.9 | S9 0.487 |
| 60 | LR universal main | 0.883 ± 0.097 [0.834, 0.929] | 0.964 ± 0.056 | 0.984 | 0.915 | 0.909 | 8.4 / 65.4 | S13 0.679 |
| 60 | LR personal z main | 0.902 ± 0.127 [0.836, 0.956] | 0.978 ± 0.054 | 0.988 | 0.914 | 0.948 | 4.0 / 37.7 | S3 0.531 |
| 60 | LR relative-only main | 0.866 ± 0.111 [0.806, 0.918] | 0.971 ± 0.047 | 0.985 | 0.890 | 0.905 | 9.8 / 68.2 | S15 0.624 |
| 60 | **best LR: personal z −motion +drift −hr_std −HRV** | 0.916 ± 0.123 [0.846, 0.965] | 0.994 ± 0.014 [0.986, 1.000] | 0.997 | 0.911 | 0.967 | 3.1 / 23.8 | S3 0.522 |
| 60 | time only (NOT a candidate) | 0.952 ± 0.074 | 0.965 ± 0.075 | 0.984 | 0.947 | 0.976 | 1.0 / 17.3 | S17 0.768 |
"main" = motion on, drift on (where applicable), hr_std on, no HRV. All 81 rows (incl. accuracy, balanced accuracy,
precision, pooled TP/FP/TN/FN, CIs for every metric) are in `stepb_results.csv`. The best LR was picked after seeing
held-out results, so its numbers are optimistically selected.

What the results show (data only):
- **Every LR configuration beats both HR baselines** on mean PR-AUC and F1 at every window length, and both HR
  baselines beat the majority class. Personal-z HR and universal HR have IDENTICAL PR-AUC and ROC-AUC (a per-subject
  z-score is a positive linear transform of hr_mean within a subject, so the within-subject ranking is unchanged);
  personal z only moves the threshold, which raises F1 (0.635 -> 0.696 at 20 s) and specificity.
- **Empirical null**: every candidate 20 s mean PR-AUC (0.83–0.98) is above the shuffled-label null max (0.629);
  empirical p = 1/21 = 0.048 for all, which is the smallest value 20 seeds can give.
- **Drift features drive the personal-z advantage**: personal z with drift (temp_mean, scl_mean) F1 0.90–0.93;
  without drift 0.76–0.82, i.e. no better than universal or relative-only. Universal changes little with drift
  (20 s: 0.815 vs 0.802). Personal-z drift features = change of SCL/temperature since the first 5 min of the
  session, and Stage 2 showed time-in-session alone separates stress (time-only F1 0.94–0.95 here, higher than
  EVERY candidate mean F1). GUESS (not verified): part of the personal-z +drift gain is the protocol-time shortcut,
  not stress physiology. The no-drift configurations (F1 ~0.78–0.86) are the more conservative estimates.
- **HRV removal costs nothing**: with-HRV references vs the same main config, 20 s F1: universal 0.818 vs 0.815,
  personal z 0.905 vs 0.903, relative-only 0.812 vs 0.807; same pattern at 30/60 s.
- hr_std on/off and motion on/off change mean F1 by at most ~0.08, in no consistent direction (largest: universal
  60 s +drift, hr_std on 0.883 vs off 0.808). Longer windows help universal and relative-only (F1 0.81 -> 0.87–0.88
  from 20 to 60 s), not personal z (0.90 at all three).
- **False alarms**: non-stress time per subject is only 0.31–0.34 h (~20 min), so every FA/h value is extrapolated
  from about 20 min per subject. Merged events/h are 3–10x lower than FP windows/h (FPs come in runs). Best 20 s
  candidate: 6.9 events/h; universal main 20 s: 23.6 events/h.
- **PPG-flag sensitivity** (best LR, 60 s): excluding the 83 PPG-flagged windows (5276 -> 5193) changes F1 0.916 ->
  0.918, PR-AUC 0.994 -> 0.996, events/h 3.14 -> 3.17. PPG-flagged windows do not drive the result.

Wilcoxon signed-rank on per-subject F1, main LR configurations (n = 15, Holm across 3 pairs per window):
| W | pair | median diff | W | p | p Holm | method |
|---|---|---|---|---|---|---|
| 20 | universal − personal z | −0.098 | 21 | 0.026 | 0.051 | exact |
| 20 | universal − relative-only | +0.028 | 43 | 0.359 | 0.359 | exact |
| 20 | personal z − relative-only | +0.074 | 16 | 0.010 | **0.031** | exact |
| 30 | universal − personal z | −0.074 | 21 | 0.026 | 0.051 | exact |
| 30 | universal − relative-only | +0.017 | 53 | 0.720 | 0.720 | exact |
| 30 | personal z − relative-only | +0.046 | 17 | 0.013 | **0.037** | exact |
| 60 | universal − personal z | −0.023 | 42 | 0.330 | 0.991 | exact |
| 60 | universal − relative-only | +0.010 | 42 | 0.510 | 1.000 | normal approx (1 zero diff) |
| 60 | personal z − relative-only | +0.021 | 49 | 0.561 | 1.000 | exact |
Personal z (WITH drift features) beats relative-only at 20 and 30 s after Holm; universal vs personal z is borderline
(p Holm 0.051) at 20/30 s; nothing is significant at 60 s. Given the drift caveat above, this is not evidence that
personal normalisation itself helps.

Worst subjects (confusion matrices from `stepb_per_subject.csv`; causes are GUESSES unless stated):
- The typical failure is a misplaced THRESHOLD, not bad ranking: worst subjects still have ROC-AUC 0.85–0.98.
- S3, personal z (+drift): 20 s TP 32 / FN 93 / FP 2 (ROC 0.943); 60 s TP 43 / FN 75 / FP 1 -> missed stress.
  Same failure as the Stage 1 baseline (S3 shows little EDA rise relative to its own calibration). GUESS: elevated
  calibration-period EDA (sunny baseline room, readme).
- S2, personal z (−drift): 60 s TP 0 / FN 113 / FP 2 although ROC-AUC 0.984; 20 s TP 11 / FN 109 -> missed stress.
  Cause unknown (Stage 1 also flagged S2 with personal normalisation).
- S13, universal (60 s): TP 124 / FN 0 / FP 117 / TN 115 (spec 0.50); relative-only 20 s: FP 177 / TN 68 -> false
  alarms. GUESS: S13 has very high EDA (~5 µS, Stage 1), so the population threshold sits too low for S13.
- S9, universal (20 s): TP 46 / FN 79 (sens 0.37) -> missed stress; under the best LR: 0 FN but 25 FP, all amusement.
- S6, HR thresholds: 60 s TP 1 / FN 120 (ROC 0.915) -> S6 stress HR stays below the population HR threshold.
  GUESS: the readme says S6 was "not really stressed" by the TSST.

### Step B summary answers
- Time-only shortcut: F1 0.94–0.95 / PR-AUC 0.95–0.97, a higher F1 than every candidate. Protocol order is a very
  strong shortcut on WESAD; time is never a model input (asserted), but slow drift features can proxy it.
- Shuffled-label check: run 1 failed the original PR-AUC rule; the 20-seed diagnostic passed the changed ROC-AUC
  rule; chance PR-AUC for this feature set is ~0.51, not 0.35 (see above).
- Step A verdict carried into Step B: keep HR only; HRV dropped from main configurations; removing it costs no F1.

---

## Step 0 — hardware context and paper-vs-code differences (recorded 2026-11-11, before any Step C experiment)
Hardware context is in CLAUDE.md ("Hardware and paper alignment"). The paper draft is NOT in `docs/` (only the IEC
proforma and the Vishwakarma deck are there), so the draft content below is as described by the user, not re-read.

### Known differences between the paper draft and the code
1. **SD floor.** Draft: personal z-score SD floor = 10 % of the subject's own overall SD. That SD is computed over
   the subject's whole session (all labelled windows, incl. stress), so for the held-out subject it uses test-session
   data (transductive). The Step B harness uses a training-only floor (median calibration SD of the training subjects
   of that fold). The frozen baseline code does what the draft says (`personalise()`: `0.1 * g[FEATURES].std()`).
2. **Calibration exclusion.** Draft: calibration windows are excluded for all methods. The baseline script only
   excludes windows fully inside the first 5 min (`calib`), so windows that partly overlap the first 5 min were
   still scored: 50 of its 1806 scored windows (3-4 per subject, all baseline label). It also TRAINS on the
   calibration windows of the training subjects (the harness trains on scored windows only).
3. **Causality.** Draft: all features need only filtering, peak detection and running statistics. The baseline used
   zero-phase `filtfilt` (BVP band-pass, EDA low-pass) and a centred moving average (`uniform_filter1d`) for the EDA
   tonic, neither of which can run on a stream. The harness uses causal SOS filters and a causal EMA.
4. **HRV.** The draft lists RMSSD/SDNN/pNN50 as features. Step A showed wrist HRV does not track chest ECG
   (RMSSD rmcorr 0.05, wrong sign of change under stress); these features measure PPG beat-timing noise.
5. **Session-time shortcut.** The draft does not mention it. Stage 2/Step B: minutes since baseline start alone
   gives F1 0.94-0.95 (LOSO), higher than every physiological model; slow drift features (scl_mean, temp_mean) can
   proxy it.

Other code differences found while checking (not in the draft as described): 15 s step (harness 5 s); RF/XGB
predict at probability 0.5 (harness: threshold from inner folds); HR rule threshold searched over a 91-quantile grid
on training windows (harness: exact F1 maximum on pooled inner out-of-fold scores); TEMP used at 4 Hz (harness 1 Hz);
peak times not interpolated (harness: parabolic interpolation); missing HR -> "no alert" in the rule (harness:
training-median fill); RF seed 42 with `min_samples_leaf=3`; the baseline's calibration windows start from the first
baseline WINDOW (which may begin a few seconds before the first baseline sample), the harness's from the first
baseline sample (15 of the baseline's 255 calibration windows are not fully inside the harness's calibration period).

---

## Step C — PRE-REGISTRATION (written 2026-11-11 BEFORE any Step C experiment; not to be changed after results)
Binary task stress (2) vs non-stress (1, 3) unless stated. Same harness as Step B: LOSO outer, GroupKFold(7) by
subject inner, scored windows only, leakage assertions on every fold, seeds 0, bootstrap 2000 subject resamples.

### 1. Deployable features and schemes
- Deployable pool D, fixed order (= `src/feature_spec.py` order): `hr_mean, hr_std, eda_slope, scr_count,
  temp_slope, acc_mag_mean, motion_energy`.
- Excluded from every deployable model: HRV `rmssd, sdnn, pnn50`; absolute EDA/temperature level `scl_mean`,
  `eda_std` (= the baseline's `scl_std`), `temp_mean`; and `scr_amp_mean` (absolute SCR amplitude in µS; the
  harness already classed it as absolute — my interpretation, stated here before running).
- Two PRIMARY schemes: `relative_only` (raw D) and `personal_z` restricted to D (training-only SD floor).
- REFERENCES only (labelled, never selected): `universal_all` and `personal_z_all` = all 11 non-HRV features incl.
  scl_mean, eda_std, scr_amp_mean, temp_mean ("drift/absolute references"); HR thresholds; majority; time-only.

### 2. Factors and models
- Factors on D: hr_std on/off x motion (acc_mag_mean + motion_energy) on/off x temp_slope on/off -> 8 subsets per
  scheme (hr_mean, eda_slope, scr_count always in) -> 16 primary configurations per model per window.
- Models (median imputer inside the pipeline for all; class-balanced; seed 0; grids ordered simplest first, ties ->
  first):
  - LR (reference model, still eligible): C in {0.001, 0.01, 0.1, 1, 10}, standard scaler.
  - Decision tree: max_depth in {2, 3, 4, 5} x min_samples_leaf in {100, 20}.
  - Random forest: n_estimators in {10, 30} x max_depth in {3, 5}; min_samples_leaf 20, max_features sqrt.
  - XGBoost: n_estimators in {20, 50} x max_depth in {2, 3} x learning_rate in {0.1, 0.3};
    scale_pos_weight = n_neg / n_pos of the training rows; n_jobs 1.
  - Parameters ranked by mean inner-fold PR-AUC (harness `tune()` logic).
- Size per outer-fold model: parameter count = LR n_feat + 1 (scaler folded into weights); trees sum over trees of
  (2 x internal nodes + leaves) (+1 XGB base score); plus n_feat imputer medians; plus n_feat SD-floor values for
  personal z. Trees / total nodes recorded. Estimated C size = flat-array layout (LR: 4 B per float; tree node
  12 B = float32 threshold-or-leaf-value + int16 feature + int16 left + int16 right + 2 B padding) + 4 B per
  preprocessing constant; compiled code size is NOT measured (no Xtensa toolchain) and is labelled "estimated".

### 3. Selection rule
- Eligible pool: the 192 primary configurations (2 schemes x 8 subsets x 4 models x 3 windows) at operating
  point OP1, global threshold, no smoothing.
- Best = highest mean per-subject F1 (ties -> higher mean PR-AUC).
- A configuration qualifies if its subject-bootstrap F1 CI overlaps the best's CI (its upper bound >= best's lower
  bound) AND its worst-subject F1 >= best's worst-subject F1 - 0.05.
- Choose the qualifying configuration with the fewest parameters (median over the 15 outer-fold models), then the
  smallest window, then the higher mean F1. Secondary (reported, not the recommendation): same rule restricted to
  tree models. The chosen estimate is optimistic because it is selected on held-out results (to be stated).

### 4. Windows, operating points, post-processing (computed for every configuration; reported for candidates)
- Windows 20 / 30 / 60 s, 5 s step.
- OP1 (primary): threshold = max F1 on pooled inner out-of-fold (OOF) scores. OP2: lowest threshold whose pooled
  inner OOF specificity >= 0.90.
- D, per-wearer threshold: thr_w = mean + k * max(SD, 0.01) of the model scores on the wearer's own calibration
  windows (unlabelled rest); alert if score > thr_w; k in {0.5, 1, 2, 3, 4, 6, 8, 12, 16} chosen by pooled inner
  OOF F1 (inner validation subjects' calibration windows scored by the inner model).
- E, causal smoothing on OP1 predictions: alert at window t if >= m of the windows starting at t, t-5 s, ...,
  t-5(n-1) s are positive (missing windows count as negative); (m, n) in {(1,1), (2,2), (2,3), (3,3), (3,4),
  (3,5), (4,5)} chosen by pooled inner OOF F1 (ties -> smaller n).
- Alert latency: stress onset = first sample of label 2 in the WESAD label vector; latency = t_end of the first
  alerted scored stress window - onset; subjects with no alert are counted as missed. Also the purity-limited
  minimum (first scored stress window). Reported per window length.

### 5. Other analyses (fixed now)
- A (paper reconciliation, REFERENCE): four detectors (fixed HR rule, personal HR z rule, RF universal, RF
  personal; RF 200 trees, depth 8, min_samples_leaf 3, balanced) at 60 s in the harness: (i) paper floor
  = max(calib SD, 0.1 x the subject's SD over all its windows) — TRANSDUCTIVE; (ii) training-only floor; (iii)
  deployable D only. RF reported at the harness threshold (inner F1) and at probability 0.5 (paper). Attribution
  ladder from the published numbers: P1 baseline code re-run in-process on its own features.csv (must reproduce
  P0); P2 = P1 re-scored without windows overlapping the first 5 min; P3 = also not trained on calibration/overlap
  windows; P4 = also training-only floor; P5 = harness (inner-fold threshold) on the baseline features; N15 =
  harness on the causal Step B features restricted to the baseline's 15 s grid; N5 = full 5 s grid. Wilcoxon
  personal vs universal (rule, RF) per variant, Holm across the 2 pairs.
- C(i) stress vs amusement (labels 2 vs 3 only, scored windows, same protocol re-tuned on this task) for: the best
  configuration of each model family (by mean F1, OP1), the selected configuration, the drift references with the
  same model and window, and time-only. C(ii) order of stress vs amusement from the label vectors; time-only
  LOSO per-subject results split by order group, for the binary and the stress-vs-amusement task.
- F statistics: per model and window, universal_all vs personal_z(D) vs relative_only(D), all factors ON, OP1,
  paired Wilcoxon on per-subject F1, Holm across the 3 pairs. Three-class (baseline/stress/amusement) macro F1 for
  the selected configuration only (argmax, class-balanced, params by inner mean macro F1).
- G3 EDA invariance: per subject gain log-uniform in [0.3, 3] and offset uniform in [-0.5, 0.5] µS (seed 0),
  applied to raw EDA before the causal filters; a feature "changes" if max |diff| > 1e-6 x its SD. Redefinitions
  fixed now: `scr_count_rel` = phasic peaks with prominence >= 1.0 x SD of the wearer's phasic EDA over the 5-min
  calibration period; `eda_slope_rel` = eda_slope / SD of the wearer's low-passed EDA over the calibration period.
  Re-evaluate the selected configuration and its other-scheme counterpart: original vs redefined, clean vs
  perturbed (all subjects perturbed).
- G4 (if runtime allows): EDA quantised with LSB 20 µS / 4096; TEMP rounded to 0.1 °C; BVP resampled to 32 and
  100 Hz; ACC to 50 Hz; features recomputed at those rates; test subject only modified (model trained on original
  features); report feature movement and F1 change for the selected configuration.

---

## Step C — results (real runs 2026-11-11; design = pre-registration above, unchanged)
All numbers: mean ± SD across 15 held-out subjects [subject-bootstrap 95 % CI, 2000 resamples], LOSO, OP1 unless
stated. Every row of every table has the full metric set (checklist passed on all 1112 rows of
`stepc_combined_results.csv`; leakage assertions on every outer and inner fold never fired; 13/13 unit tests pass:
`tests/test_leakage.py` 8 + `tests/test_stepc.py` 5, incl. "run_loso_ext picks the same C / threshold / scores as
train.run_loso" and "flipping all held-out labels leaves held-out predictions unchanged").
Files (results/metrics): `stepc_combined_results.csv` (one combined table: grid B/D/E/F, follow-up C/G3/G4, Part A
ladder), `stepc_combined_per_subject.csv`, `stepc_model_size.csv`, `stepc_selection.json`, `stepc_wilcoxon.csv`,
`stepc_paper_*.csv`, `stepc_order_*.csv`, `stepc_three_class*.{json,csv}`, `stepc_eda_*.csv`,
`stepc_device_feature_shift.csv`, `hardware_feasibility.csv`, `stepc_worst_subjects.csv`,
`stepc_smoothing_grid_selected_descriptive.csv`; `feature_spec.json` (module root; version `stepc-1` is recorded in
every results row); `results/paper_numbers.md`.

### A. Paper reconciliation (60 s, REFERENCE; `stepc_paper_ladder.csv`, `stepc_paper_wilcoxon.csv`)
P1 (baseline code re-run in-process on its own features.csv) reproduces the published per-subject F1 exactly
(max |diff| 0) and my replicate loop reproduces its predictions exactly (asserted). Mean F1:
| step (one change each) | rule univ. | rule pers. | RF univ. | RF pers. | cause |
|---|---|---|---|---|---|
| P0 published = P1 reproduced | 0.674 | 0.723 | 0.908 | 0.846 | — |
| P1s RF seed 0 instead of 42 | | | 0.877 | 0.840 | RF seed noise only (S14 0.600 -> 0.000) |
| P2 drop the 50 overlap windows from scoring | 0.677 | 0.724 | 0.909 | 0.846 | diff 2, scoring: negligible |
| P3 also not trained on calib/overlap windows | 0.709 | 0.727 | 0.878 | 0.819 | diff 2, training |
| P4 training-only SD floor | 0.709 | 0.744 | 0.878 | 0.833 | diff 1 |
| P5 harness bookkeeping, inner-F1 thr (RF at p 0.5) | 0.713 | 0.738 | 0.870 (0.878) | 0.820 (0.823) | small |
| N15 causal features, same 15 s grid | 0.664 | 0.757 | 0.868 (0.869) | 0.785 (0.808) | diff 3 |
| N5 (ii) 5 s step, training-only floor | 0.648 | 0.756 | 0.904 (0.902) | 0.749 (0.789) | step size |
| N5 (i) paper floor (TRANSDUCTIVE) | 0.648 | 0.733 | 0.904 | 0.794 (0.797) | diff 1 in new pipeline |
| N5 (iii) deployable features D only | 0.648 | 0.756 | 0.843 (0.864) | 0.812 (0.831) | diff 4 + absolute features |
- Attribution is sequential (order-dependent), and RF steps are confounded with RF seed noise, which is as large as
  most steps (seed alone: −0.031 for RF universal). Every RF delta of about 0.03 or less is within that noise. Nothing
  is unexplained between P0 and P1. Not separated (so unattributed within the step): the three causal changes inside
  N15 (filters, parabolic peak interpolation, TEMP 1 Hz) and the three bookkeeping changes inside P5.
- Known diff 1 (floor): in the baseline pipeline the transductive floor did NOT favour personal (the training-only
  floor is +0.017 rule / +0.014 RF); in the causal 5 s pipeline it raises RF personal by +0.045 and lowers the rule
  by −0.023.
- Known diff 4/5: removing HRV + scl_mean/eda_std/scr_amp_mean/temp_mean lowers RF universal 0.904 -> 0.843 and
  RAISES RF personal 0.749 -> 0.812 (the S2/S3/S14 failures shrink).
- **Wilcoxon/Holm conclusions do not change**: personal vs universal is not significant at any step (smallest:
  N5_ii RF p 0.100, Holm 0.201; published p 0.847 / 0.730, Holm 1.0). At N5_ii RF personal is below RF universal
  (median −0.094).

### B. Small deployable models (192 primary configurations; `stepc_results_grid.csv`, `stepc_model_size.csv`)
Best feature subset per model x scheme x window (OP1 mean F1; worst subject):
| W | LR rel | LR pz | DT rel | DT pz | RF rel | RF pz | XGB rel | XGB pz |
|---|---|---|---|---|---|---|---|---|
| 20 | 0.818 (S13 0.58) | 0.794 (S2 0.29) | 0.810 (S13 0.55) | 0.762 (S2 0.41) | 0.808 (S13 0.54) | 0.808 (S2 0.32) | 0.820 (S13 0.58) | 0.826 (S2 0.42) |
| 30 | **0.841** (S13 0.57) | 0.804 (S2 0.21) | 0.771 (S14 0.27) | 0.801 (S2 0.30) | 0.845 (S13 0.58) | 0.834 (S2 0.35) | 0.850 (S13 0.65) | 0.835 (S2 0.43) |
| 60 | 0.866 (S15 0.62) | 0.823 (S2 0.21) | 0.850 (S13 0.53) | 0.752 (S3 0.05) | 0.833 (S13 0.55) | 0.834 (S2 0.30) | **0.868** (S15 0.62) | 0.831 (S2 0.23) |
rel = relative-only D, pz = personal z D. Baselines (20/30/60 s): HR rule 0.635/0.632/0.648, personal HR z
0.696/0.717/0.756; time-only 0.941/0.953/0.952 (NOT a candidate). Drift references (all 11 non-HRV features): LR
universal 0.815/0.840/0.883, LR personal z 0.903/0.908/0.902 (identical to Step B), tree models 0.66-0.90 with
worst subjects down to 0.000.
Size (median over the 15 outer-fold models; kB ESTIMATED from a flat-array layout, compiled code not measured):
LR 7-22 parameters, 0.03-0.09 kB; DT 45-89 parameters, 27-55 nodes, 0.3-0.7 kB; RF 10-30 trees, 432-1606 nodes,
5-19 kB; XGB 20-50 trees, 300-692 nodes, 3.5-8.8 kB. All are far below the 100 kB budget. The paper's RF-200 /
depth 8: median 174 kB (baseline features) / 232 kB (harness features), over budget.
Factors (mean F1 with the factor on minus off, over the 192): hr_std −0.004, motion −0.002, temp_slope −0.014.

### Pre-registered selection (`stepc_selection.json`)
- Best deployable: **XGB relative-only, all 7 D features, 60 s**: F1 0.868 ± 0.121 [0.800, 0.922], PR-AUC 0.973,
  worst S15 0.617, 493 parameters (~4.0 kB est.).
- 35 configurations qualify (CI overlap and worst-subject F1 >= 0.567): 15 LR, 12 XGB, 7 RF, 1 DT.
- **Selected: LR relative-only {hr_mean, eda_slope, scr_count}, 30 s**: 7 parameters (3 weights + bias, 3 imputer
  medians; ~28 B est.). F1 0.841 ± 0.132 [0.769, 0.903]; accuracy 0.859; balanced accuracy 0.874; precision 0.802;
  sensitivity 0.927; specificity 0.820; ROC-AUC 0.977; PR-AUC 0.948; FP windows 129.3/h; merged false-alarm events
  16.9/h; pooled TP/FP/TN/FN 1787/647/2939/136; worst S13 F1 0.574. C = 0.001 (the smallest grid value) in 10 of 15
  folds: the grid edge, not extended (pre-registered).
- Trees only (secondary): DT relative-only {hr_mean, hr_std, eda_slope, scr_count, acc_mag_mean, motion_energy},
  60 s, F1 0.784 [0.713, 0.849], worst S13 0.602, 49 parameters (~0.36 kB).
- **This estimate is optimistic**: the configuration was chosen after seeing held-out results from 192 candidates.

### D/E/F. Operating points (selected configuration, 30 s)
| rule | F1 | sens | spec | precision | FA events/h | FP windows/h | median latency | worst |
|---|---|---|---|---|---|---|---|---|
| OP1 global, inner max-F1 | 0.841 | 0.927 | 0.820 | 0.802 | 16.9 | 129.3 | 29.1 s | S13 0.574 |
| OP2 inner specificity >= 0.90 | 0.779 | 0.779 | 0.889 | 0.860 | 12.6 | 79.8 | 29.1 s | S2 0.405 |
| D per-wearer (k = 1 in 11/15 folds) | 0.679 | 0.798 | 0.783 | 0.681 | 22.9 | 156.1 | 30.0 s, 1 missed | S13 0.000 |
| E smoothing (4 of 5 in 10/15 folds) | 0.838 | 0.886 | 0.851 | 0.839 | 8.6 | 107.4 | 42.8 s | S13 0.569 |
Means over all 192 primary configurations: OP1 F1 0.792, FA 17.8/h; OP2 F1 0.760, spec 0.897; per-wearer F1 0.740,
FA 23.6/h; smoothed F1 0.797, FA 9.2/h.
- OP2 holds its target on held-out subjects (spec 0.889 for the selected, 0.897 on average) at a large sensitivity cost.
- The per-wearer threshold (D) HURTS on average. Worst subjects for the selected config: S13 OP1 TP 128 / FN 0 /
  FP 190 / TN 52 -> per-wearer TP 0 / FN 128 (S13's own rest windows score 0.764 vs the global threshold 0.457, so
  mean + 1 SD lies above its stress scores); S2 F1 0.690 -> 0.456 (sensitivity 0.58 -> 0.31); S3 0.800 -> 0.699.
  It helps when the model ranks a wearer well but the population threshold is wrong: RF relative-only 30 s
  0.845 -> 0.875 [0.845, 0.905], S13 0.582 -> 0.927, S2 0.805 -> 0.784, S3 0.857 -> 0.852 (that RF result is picked
  post hoc from 192 configurations; it is not a selection).
- Smoothing (E) halves merged false alarms (16.9 -> 8.6 events/h) for no F1 change and +14 s median latency.
  Descriptive grid on held-out data (NOT used for any choice): `stepc_smoothing_grid_selected_descriptive.csv`.
- Alert latency (median over primary configurations, OP1): 20 s 22.1 s, 30 s 29.1 s, 60 s 57.1 s, equal to the
  purity-limited minimum (the first fully labelled stress window ends about W s after onset). Smoothed: 32.8 / 40.8 /
  61.7 s. No configuration missed a whole stress episode at OP1. Caveat: with ~10-26 false-alarm events per hour, an
  alert in the first stress window is partly chance.
- Wilcoxon (per model x window, Holm over 3 pairs; `stepc_wilcoxon.csv`): 2 of 36 tests significant, both DT 60 s:
  universal (all, reference) beats personal z (D) (median +0.130, p Holm 0.045) and relative-only (D) (+0.043,
  p Holm 0.045). Personal z (D) vs relative-only (D) is never significant (p Holm >= 0.27).
- Three-class (selected configuration only, `stepc_three_class.json`): macro F1 0.463 ± 0.080 [0.422, 0.503],
  accuracy 0.501, worst S13 0.287. Pooled: 1625 of 2541 baseline windows predicted as amusement. The 3-feature model
  cannot separate baseline from amusement (the published RF personal three-class, with drift and HRV features: 0.678).

### C. Time-shortcut controls
- (ii) Order, from the label vectors (`stepc_condition_order.csv`): stress first 1-2-4-3-4 (S2, S3, S6, S9, S11, S14,
  S16; n 7), amusement first 1-3-4-2-4 (S4, S5, S7, S8, S10, S13, S15, S17; n 8); exactly one stress segment each.
  But in minutes since baseline onset: amusement 24-34 (amusement first) or 66-88 (stress first); stress 30-43
  (stress first) or 50-69 (amusement first). Only 30-34 min overlaps, so absolute time still separates them.
- Time-only (LOSO, depth-tuned tree) per order group: binary F1 0.924-0.965 in both groups at every window; stress vs
  amusement F1 0.957-0.965 in both groups, ROC-AUC 0.969-0.981 (`stepc_order_time_only.csv`).
- (i) Stress vs amusement (labels 2 vs 3; "always stress" already gives F1 0.786, so read ROC-AUC): selected LR 30 s
  F1 0.943 [0.911, 0.971], ROC-AUC 0.986; best XGB 60 s 0.950, ROC 0.989; RF 0.908; DT 0.902; drift references
  0.78-0.93; time-only 0.959 / 0.964 (ROC 0.973 / 0.974). Exploratory (not pre-registered): motion off changes
  little (best LR 0.952 -> 0.952, best XGB 0.950 -> 0.944).
- **Can conclude**: the deployable models separate stress from amusement (another post-baseline condition) with
  ROC-AUC 0.97-0.99, also without motion, so they are not only detecting "not baseline". The personal-z drift
  references do worse on this task than relative-only (S2, S3 fail), consistent with drift features carrying
  session-position information. **Cannot conclude**: that the models ignore time. Time-only does just as well, so
  this control cannot separate physiology from protocol position; WESAD has no condition order that breaks the
  time-condition link. TSST stress also differs from amusement in speaking, posture and task, not only in arousal.

### G. Hardware parity
- G1: `src/feature_spec.py` -> `feature_spec.json` (16 features, fixed order, units, window/step, input rates, exact
  filters). Every Step C model's feature list is asserted to follow the spec order.
- G2: `hardware_feasibility.csv`. NOT TRANSFERABLE: HRV (3), scl_mean, eda_std, scr_amp_mean, temp_mean, and raw
  eda_slope and scr_count (gain-dependent). UNVALIDATED: hr_mean, hr_std, eda_slope_rel, scr_count_rel, temp_slope,
  acc_mag_mean, motion_energy. VALIDATED: none (no prototype data may be used before IEC approval).
- G3: per-subject gain (log-uniform 0.3-3) and offset (±0.5 µS), `stepc_eda_gain_offset.csv`. Every original EDA
  feature changes (`stepc_eda_invariance.csv`; scr_count in 46-74 % of windows; eda_slope Pearson clean vs perturbed
  0.72-0.76). The redefined `eda_slope_rel` and `scr_count_rel` change in 0 windows. Re-evaluation at 30 s (F1):
  | | original features | redefined (gain/offset-invariant) |
  |---|---|---|
  | relative-only LR, clean | 0.841 [0.769, 0.903], worst S13 0.574 | 0.779 [0.663, 0.871], worst S2 0.140 |
  | relative-only LR, perturbed | 0.791 [0.716, 0.862], worst S7 0.612 | 0.779 (identical) |
  | personal z LR, clean | 0.804 [0.696, 0.889], worst S2 0.209 | 0.796 [0.664, 0.897], worst S2 0.017 |
  | personal z LR, perturbed | 0.791 (the SD floor is in absolute training units) | 0.796 (identical) |
  The selected model's 0.841 assumes every wearer has E4-like EDA gain; with a realistic gain spread it is 0.791,
  and the transferable (redefined) version is 0.779 with S2 failing.
- G4 (selected configuration; held-out subject's features recomputed, model trained on E4-rate features;
  `stepc_device_feature_shift.csv`): the E4-identity self-check reproduces F1 0.841 exactly. EDA 12-bit 0.835,
  TEMP 0.1 °C 0.841 (temp_slope moves a median 0.29 SD, r 0.91, but it is not in the selected model), BVP 32 Hz
  0.836, BVP 100 Hz 0.840, ACC 50 Hz 0.841, all together 0.834. Sampling rate and quantisation are not the problem;
  site, sensor and units are (G2, G3).

### Interpretation (honest)
- **Where models are weak.** Across the 192 primary configurations mean F1 is 0.673-0.868 and worst-subject F1
  0.00-0.67. False alarms are high for a wearable: 5.7-40.0 merged events per hour of non-stress time (median by
  window 26.0 / 16.6 / 9.7 at 20 / 30 / 60 s; extrapolated from ~20 min of non-stress data per subject); smoothing
  roughly halves it. Three-class performance is poor (macro F1 0.46). For the selected model the worst subjects
  still rank well (S13 ROC-AUC 0.977, S2 0.944, S3 0.935), so its failures are threshold placement, not ranking;
  this is not universal (the lowest per-subject ROC-AUC in the grid is 0.40).
- **Which subjects fail and why.** Relative-only: S13 (worst in 55 of 96 relative-only configurations), by FALSE
  ALARMS. Measured: S13's rest windows score 0.76 under the selected model vs a 0.46 threshold. GUESS: S13's high
  resting HR (87.5 bpm) and very high EDA make its rest look like population stress. Also S14 (no EDA response,
  Stage 1) and S15. Personal z: S2 (61 of 96) and S3 (25), by MISSED STRESS, the same failure as in the Stage 1
  baseline and Step B. GUESS for S3: calibration EDA raised by the sunny baseline room (readme). S2: cause unknown.
- **Stress vs amusement.** Deployable models separate them (ROC-AUC up to 0.99) without motion, but time-only does
  equally well, so WESAD cannot show that the separation is physiological rather than protocol position.
- **Does personalisation help without drift features? No.** Personal z (D) minus relative-only (D): mean −0.032 F1
  over 96 paired configurations (27 better, 69 worse), never significant. Per-wearer thresholds also hurt on average
  (0.740 vs 0.792). The Step B personal-z advantage (0.90) needed scl_mean/temp_mean (drift) features. Personal
  normalisation is still one route to gain invariance for EDA (G3), and the per-wearer threshold rescues S13 for RF,
  so it helps specific wearers, not the average.
- **Measured vs estimated.** Measured: all F1/AUC/false-alarm/latency/CI values, node and parameter counts, feature
  shifts, invariance. Estimated: model size in kB (flat-array layout, 12 B per node; no compiled code). Extrapolated:
  false alarms per hour (~20 min of non-stress per subject). Assumed: the [a]-[d] placeholder mapping in
  `results/paper_numbers.md` (the draft is not in docs/). Not validated at all: anything on the prototype (finger EDA
  in mV, fingertip PPG, MAX30205 contact temperature, breadboard IMU).
- **Paper.** `results/paper_numbers.md` lists published and corrected values with source files. Published RF
  universal (0.908) survives the correction (0.904); RF personal does not (0.846 -> 0.749); the personal HR rule
  improves (0.723 -> 0.756); no personal-vs-universal difference is significant in either version. The paper text
  was not edited.

### Recommended configuration (pre-registered rule)
**Logistic regression, relative-only, features [hr_mean, eda_slope, scr_count] (feature_spec order), 30 s window,
5 s step, C and threshold from inner GroupKFold (OP1), optional 4-of-5 causal smoothing.** LOSO F1 0.841
[0.769, 0.903], optimistic because it was selected on held-out results. For the prototype, eda_slope and scr_count
must be replaced by eda_slope_rel and scr_count_rel (G3), which gives F1 0.779 [0.663, 0.871] with S2 at 0.140.
That is the number to quote for hardware transfer, and it is still unvalidated on finger EDA / fingertip PPG.

### Re-run (from software/ml-models/wesad_edge_model, repo venv; data read-only)
```
..\..\..\.venv\Scripts\python -m src.feature_spec
..\..\..\.venv\Scripts\python -m src.protocol --data "C:\datasets\WESAD"
..\..\..\.venv\Scripts\python -m unittest tests.test_leakage tests.test_stepc -v
..\..\..\.venv\Scripts\python -m src.stepc --phase grid                                  (~17 min, 12 cores)
..\..\..\.venv\Scripts\python -m src.hw_parity --table
..\..\..\.venv\Scripts\python -m src.hw_parity --eda --data "C:\datasets\WESAD"
..\..\..\.venv\Scripts\python -m src.hw_parity --device 30 --data "C:\datasets\WESAD"    (30 = selected window)
..\..\..\.venv\Scripts\python -m src.paper_reconcile
..\..\..\.venv\Scripts\python -m src.stepc --phase followup                              (~6 min)
..\..\..\.venv\Scripts\python -m src.stepc --phase combine
..\..\..\.venv\Scripts\python -m src.paper_outputs
```

---

## Step D — final model: DECISIONS (written 2026-11-11 AFTER seeing the Step C results, BEFORE any Step D run)
These choices were made after seeing Step C held-out results. They are not pre-registered in the Step C sense, and
every Step D number inherits the Step C selection optimism. Nothing below is changed after the Step D run.

### Two configurations (same model and window; only the two EDA features differ)
- **Config R (reference, the pre-registered selection)**: logistic regression, relative-only, features
  [hr_mean, eda_slope, scr_count] (feature_spec order), 30 s window, 5 s step, class-balanced, median imputer +
  standard scaler inside the pipeline. **Reported as the research result. Not deployed**, because eda_slope (µS/s)
  and scr_count (fixed 0.01 µS prominence) change with sensor gain (G3) and the prototype reads mV, not µS.
- **Config D (deployment candidate)**: identical, with eda_slope_rel and scr_count_rel instead (gain/offset-invariant,
  defined relative to the wearer's 5-min rest calibration; definitions in feature_spec.json). Features:
  [hr_mean, eda_slope_rel, scr_count_rel]. **The deployment model needs that 5-min calibration at first use.**
  Its LOSO result is the honest estimate for the prototype (still unvalidated on prototype hardware).
  In LOSO the held-out subject's calibration constants come from its own unlabelled first 5 min of rest, as on a
  device. Step C already ran exactly this configuration (G3 "redefined, clean": OP1 F1 0.779 [0.663, 0.871]).
- Reproduction checks (asserted): the Step D LOSO of R must reproduce the Step C grid row (OP1 F1 0.8406) and D must
  reproduce the Step C G3 "redef clean" row (OP1 F1 0.7794) to 1e-9; otherwise STOP.

### Regularisation C
- C is chosen from the **pre-registered grid only** {0.001, 0.01, 0.1, 1, 10} (mean inner-fold PR-AUC, ties ->
  smallest C), in LOSO and for the deployment model.
- C = 0.001 (the smallest grid value, the grid edge) was chosen in **10 of 15** outer folds for R
  (`stepc_folds_grid.csv`) and in **15 of 15** for D (`stepc_folds_followup.csv`, G3 redef clean). The optimum may lie
  below the grid.
- **Post-hoc sensitivity run** (labelled post-hoc, NOT used for any selection, the shipped model keeps the
  pre-registered grid): same LOSO with C in {1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10}.

### Operating points (all thresholds from training subjects only)
- OP1: threshold = max F1 on pooled inner out-of-fold scores.
- OP2: lowest threshold with pooled inner OOF specificity >= 0.90.
- **OP1 + 4-of-5 smoothing**: alert at window t if >= 4 of the 5 OP1 decisions at t, t-5 s, ..., t-20 s are positive
  (causal; `runner.smooth`). **m-of-n is FIXED at 4-of-5** (it was the inner-fold choice in 10/15 folds for R in
  Step C) so the firmware has one constant. The Step C inner-chosen m-of-n is reported as a labelled reference row.
- **Default = OP1 + 4-of-5 smoothing.** Why: in Step C (R) it halved merged false alarms (16.9 -> 8.6 events/h) for
  about the same F1 (0.841 -> 0.838) at +14 s median latency, and false alarms are the main usability failure of a
  wearable alert. OP2 cost more sensitivity (0.927 -> 0.779) than it saved in false alarms (12.6 events/h).
  The per-wearer threshold is not used (it hurt on average in Step C).

### Experiments (fixed now)
1. Final LOSO for R and D (15 outer folds, GroupKFold(7) inner, scored windows only, leakage assertions on every
   fold): all required metrics with 2000-resample subject-bootstrap 95 % CIs, false alarms per hour per window and
   merged; pooled ROC and PR curves, calibration (reliability) curve of the pooled held-out probabilities, pooled and
   per-subject confusion matrices, per-subject metric spread, worst subjects S13, S2, S3 (score time courses).
2. Train vs validation from the inner folds only (the held-out subject is never used): (a) validation curve over the
   C grid: PR-AUC / ROC-AUC on inner-train rows vs inner-validation rows, and F1 on inner-validation at the inner-train
   max-F1 threshold; (b) learning curve: n in {2, 4, 6, 8, 10, 12} inner-training subjects (5 random draws, seed 0) at
   the outer fold's chosen C, train vs validation PR-AUC.
3. Deployment model D: inner GroupKFold(7) over all 15 subjects chooses C (pre-registered grid), OP1/OP2 thresholds
   from the pooled inner OOF scores; refit on all 15 subjects. **It has no held-out estimate; its expected performance
   is the LOSO result of D.**
4. Save to `models/`: joblib pipeline; plain JSON with folded coefficients (weights, bias, imputer medians, scaler
   means/SDs, thresholds, smoothing m-of-n); `feature_spec.json`; calibration procedure; window/step; metadata JSON
   (date, library versions, SHA-256 of the 15 WESAD .pkl files and the derived feature CSVs, LOSO metrics R and D).
5. Round-trip test in a fresh process: predictions of the reloaded joblib equal the in-memory model exactly on a
   fixed set of windows; the JSON (pure numpy) reproduces them; the feature order matches feature_spec.json; a
   wrongly ordered input is rejected.
6. `results/metrics/final_results.csv` (R and D x OP1 / OP2 / OP1+4of5).

## Step D — results (real run 2026-11-11; design = decisions above, unchanged)
Run: `python -m src.final_model --data "C:\datasets\WESAD"` (~80 s, 12 cores). Code: `src/final_model.py` (driver; reuses
`runner.run_loso_ext / tune_ext / evaluate_ops / smooth`, `stepc.tabulate`, `stepc_followup.eda_variant`,
`train.fit_checked / metric_checklist`; none of them changed), `src/predict.py` (host reference predictor, was a stub),
`tests/test_final_model.py`. 18/18 unit tests pass (`test_leakage` 8, `test_stepc` 5, `test_final_model` 5).
Leakage assertions ran on every outer and inner fold and never fired. Metric checklist passed for all 6 rows of
`final_results.csv` and all 10 rows of `final_results_supplement.csv` (every metric, SD, 2000-resample CI, pooled
confusion matrix, both false-alarm rates, worst subject). All 25 output files verified non-empty.
- **Reproduction (asserted)**: R reproduces Step C exactly (max per-subject F1 difference 0.0 at OP1, OP2 and inner
  m-of-n smoothing; max score difference 0.0); D reproduces the Step C G3 "redef clean" row exactly (0.0).
- Files: `results/metrics/final_results.csv` (R, D x OP1 / OP2 / OP1+4of5), `final_results_supplement.csv` (post-hoc
  C grid + inner-m-of-n reference rows), `final_per_subject.csv`, `final_folds.csv`, `final_predictions.csv`,
  `final_worst_subjects.csv`, `final_train_val_curve.csv`, `final_learning_curve.csv`, `final_calibration_curve.csv`,
  `final_summary.json`; graphs `final_roc_pr.png`, `final_calibration.png`, `final_confusion_pooled.png`,
  `final_confusion_per_subject_{R,D}.png`, `final_per_subject_spread.png`, `final_worst_subjects.png`,
  `final_train_vs_validation.png`; `models/` (below).

### 1. Final LOSO, 30 s, 5 s step (mean ± SD across 15 held-out subjects [subject-bootstrap 95 % CI])
| config | operating point | F1 | sensitivity | specificity | precision | FA events/h (merged) | FP windows/h | worst (F1) |
|---|---|---|---|---|---|---|---|---|
| R | OP1 | 0.841 ± 0.132 [0.769, 0.903] | 0.927 [0.863, 0.976] | 0.820 [0.680, 0.927] | 0.802 | 16.9 [10.4, 24.2] | 129.3 [52.9, 230.1] | S13 0.574 |
| R | OP2 (spec ≥ 0.90) | 0.779 ± 0.173 [0.691, 0.858] | 0.779 | 0.889 | 0.860 | 12.6 [5.4, 22.1] | 79.8 | S2 0.405 |
| R | OP1 + 4-of-5 | 0.834 ± 0.136 [0.763, 0.899] | 0.875 | 0.861 | 0.844 | 8.6 [4.0, 14.2] | 100.4 | S2 0.596 |
| D | OP1 | 0.779 ± 0.217 [0.663, 0.871] | 0.799 [0.660, 0.912] | 0.892 [0.808, 0.961] | 0.846 | 18.7 [8.6, 30.5] | 77.9 [27.9, 138.3] | S2 0.140 |
| D | OP2 (spec ≥ 0.90) | 0.778 ± 0.218 [0.660, 0.871] | 0.802 | 0.886 | 0.844 | 18.3 [8.5, 29.7] | 82.3 | S2 0.140 |
| **D** | **OP1 + 4-of-5 (default, shipped)** | **0.762 ± 0.244 [0.629, 0.863]** | 0.745 [0.600, 0.867] | 0.929 [0.867, 0.980] | 0.895 | **7.6 [2.6, 13.5]** | 50.8 [14.7, 95.8] | S2 0.033 |

(Corrected 2026-11-11: the FP windows/h CI low of the shipped row was written as 14.8; `final_results.csv` has
14.7499..., i.e. 14.7. Documentation rounding error only; the CSV was always right.)

Threshold-free (same at every operating point): R PR-AUC 0.948 [0.917, 0.976], ROC-AUC 0.977 [0.964, 0.988];
D PR-AUC 0.951 [0.922, 0.976], ROC-AUC 0.974 [0.960, 0.986]. Accuracy / balanced accuracy: R OP1 0.859 / 0.874,
D OP1 0.860 / 0.845, D default 0.866 / 0.837 (all CIs in the CSV). Pooled confusion (TP/FP/TN/FN): R OP1
1787/647/2939/136; D OP1 1547/388/3198/376; D default 1444/253/3333/479. Pooled over all windows (one threshold across
subjects) ROC-AUC is R 0.931 / D 0.932 and AP 0.873 / 0.900, below the per-subject means because subjects' score levels
differ. Alert latency: median 29.1 s (OP1, = purity-limited minimum) and 44.1 s (4-of-5), no stress episode missed by any
row; worst-case D default latency 196 s (S2). Reference rows (supplement): inner-chosen m-of-n (Step C rule; D picked
3-of-5 in 12/15 folds) gives D F1 0.780, 10.1 events/h; R 0.838, 8.6 events/h.
- **The default behaves differently for D than for R.** For R, 4-of-5 cost 0.007 F1; for D it costs 0.017 F1 and
  0.054 sensitivity (0.799 -> 0.745) and drives S2 from 0.140 to 0.033, while cutting merged false alarms 18.7 ->
  7.6 events/h. The default was fixed before this run and is NOT changed after seeing it; the inner-CV choice for D
  (3-of-5, F1 0.780, 10.1 events/h) is reported as a reference only.
- OP2 is almost identical to OP1 for D (OP1 thresholds 0.498-0.543, OP2 0.505-0.533 across folds): D's inner OOF specificity at the max-F1
  threshold is already about 0.9.

### Post-hoc C sensitivity (extended grid {1e-5 ... 10}; NOT used for selection)
- R: the extended grid picks the same C in all 15 folds (10 x 0.001, 2 x 0.01, 2 x 0.1, 1 x 10); results identical.
- D: chosen C 1e-4 (6 folds), 1e-3 (6), 1e-5 (3); OP1 F1 0.777 vs 0.779, PR-AUC 0.951 vs 0.951, events/h 18.9 vs 18.7.
- **The grid edge does not matter**: validation curves are flat over C (below). The shipped model keeps C = 0.001.

### 2. Training vs validation (inner folds only; 15 outer x 7 inner = 105 folds; `final_train_vs_validation.png`)
- Validation curve over C: PR-AUC inner-train vs inner-validation, R 0.907-0.908 vs 0.921-0.923; D 0.917-0.918 vs
  0.925-0.927, flat across the whole grid. F1 at the inner-train max-F1 threshold: train 0.830-0.835, validation
  R 0.813-0.816, D 0.798-0.801 (gap 0.02-0.03, the cost of moving a threshold to new subjects).
- Learning curve (5 random draws per size, the fold's chosen C): at 2 training subjects train PR-AUC exceeds validation
  (R 0.940 vs 0.908; D 0.937 vs 0.891) -> overfitting with very few subjects; the curves meet at about 6 subjects and are
  flat from 8 to 12 (R 0.908 vs 0.924, D 0.917 vs 0.927 at 12).
- Reading: **no overfitting at the used training size, and no gain from weaker or stronger regularisation**. The 3-feature
  linear model is limited by bias and by per-subject threshold placement, not variance; more subjects of the same kind
  are unlikely to raise ranking much. Validation PR-AUC above train is expected here: train AP is pooled over 12
  subjects, validation AP over 2, and pooling more subjects with different score levels lowers AP (mechanism stated,
  not separately tested).

### Calibration of probabilities (`final_calibration.png`, `final_calibration_curve.csv`)
The probabilities are NOT calibrated (class-balanced training). D: predicted 0.17-0.32 -> observed stress fraction
0.00-0.08; predicted 0.72-0.86 -> observed 0.93-0.99 (R similar). Only the thresholded alert is meaningful; the score
must not be shown to a wearer as "probability of stress". No recalibration was done.

### Worst subjects (`final_worst_subjects.csv`, `final_worst_subjects.png`)
- **S13** (R's worst): R OP1 TP 128 / FN 0 / FP 190 / TN 52 (spec 0.215) -> false alarms; median baseline score 0.554 vs
  threshold 0.457. **D fixes it**: TP 121 / FN 7 / FP 34 / TN 208, F1 0.855 (default 0.865), baseline median 0.403 vs
  threshold 0.533. Measured: S13 calibration HR 87.4 bpm, sigma_lp_cal 0.641 µS (high EDA). Making EDA relative to the
  wearer's own rest removes S13's high absolute EDA response from the score.
- **S2** (D's worst): D OP1 TP 9 / FN 109 / FP 2 (sens 0.076), F1 0.140; default 2/118 detected, F1 0.033. ROC-AUC
  is 0.951: the ranking is fine, the scores are low. Median stress score 0.377 vs threshold 0.516 (baseline 0.234).
  R does better on S2 (OP1 F1 0.690). Same subject that failed every calibration-relative (personal z) method since
  Stage 1. GUESS: S2's stress EDA change is small relative to its own calibration variability; cause unknown.
- **S3**: R OP1 F1 0.800 (39 FP) -> D 0.530 (TP 48 / FN 75, sens 0.390), default 0.446. ROC-AUC 0.948. Measured:
  S3's sigma_lp_cal = 1.400 µS, the largest of the 15 subjects (median 0.142), which shrinks its eda_slope_rel.
  GUESS: the sunny/warm baseline room (readme) raised calibration EDA variability.
- Other D weaknesses: merged false alarms at OP1 S9 66.3/h, S4 58.0/h, S10 44.8/h (R: 12.1, 0.0, 12.0); F1 S4 0.703
  (R 0.996), S6 0.724 (R 0.935). D trades S13's false alarms for misses in S2/S3 and false alarms in S4/S9/S10.

### 3-4. Deployment model D (all 15 subjects; `models/`)
- Inner GroupKFold(7) over all 15 subjects (`runner.tune_ext`): C = 0.001 (grid edge, see post-hoc above), inner mean
  PR-AUC 0.909, OP1 threshold 0.5161 (pooled inner OOF F1 0.806; the threshold was chosen on these scores, so this F1 is
  optimistic), OP2 threshold 0.5189. Inner CV preferred 3-of-5 smoothing; 4-of-5 is shipped (decision above).
  Resubstitution on the training data (NOT an estimate): PR-AUC 0.916, ROC-AUC 0.944, F1 0.823.
- **This model has no held-out estimate. Its expected performance is the LOSO result of D (default: F1 0.762
  [0.629, 0.863], 7.6 merged false alarms/h, sensitivity 0.745, specificity 0.929)**, itself optimistic (configuration
  chosen after Step C) and measured on E4 wrist data only.
- Coefficients (standardised): hr_mean 0.736, eda_slope_rel 0.012, scr_count_rel 0.583, bias −0.211. **eda_slope_rel
  contributes almost nothing**: the deployed model is in effect HR + relative SCR count. Folded (raw units): 0.04106 per
  bpm, 0.09792 per (calibration-SD/s), 0.20033 per SCR, bias −4.0990; imputer medians 78.68 bpm, −0.000803, 1.0.
  7 parameters + 2 per-wearer calibration constants (sigma_lp_cal, sigma_ph_cal); ~28 B of floats (estimated).
- Saved: `stress_lr_D_w30.joblib` (pipeline + feature order + thresholds + m-of-n), `stress_lr_D_w30.json` (folded and
  unfolded coefficients, medians, scaler means/SDs, thresholds and logit thresholds, 4-of-5 rule, calibration block,
  window/step, input rates), `feature_spec.json` (asserted identical to the module-root spec, version stepc-1),
  `calibration_procedure.md` (what the wearer does for 5 min; exactly which statistics), `window.json`,
  `metadata.json` (date, git HEAD before this commit, Python/library versions, SHA-256 of the 15 WESAD .pkl files and
  the 4 derived input CSVs, model-file checksums, LOSO metrics R and D, reproduction check), `roundtrip_windows.csv`.

### 5. Round trip
Fresh process (`python -m src.predict --roundtrip`, also run by `tests/test_final_model.py`): 340 fixed windows (300
real scored windows, seed 0, plus 40 SYNTHETIC rows = real windows with one or all features set to NaN, flagged
`synthetic_nan`, because no scored 30 s window has a missing D feature and the device must still handle NaN).
Reloaded joblib probabilities equal the in-memory model **exactly** (array_equal) and the OP1/OP2 decisions match; the
JSON in pure numpy differs by at most 3.3e-16; float32 (as on the ESP32) changes 0 of 5509 OP1 decisions (max probability
difference 2.1e-7). Feature order matches feature_spec.json; 11 wrongly formatted inputs are rejected (reversed, swapped,
bare array, missing column, renamed column, for both scorers; reversed order vs the spec).

### 6. Interpretation (honest)
- **Research result (R)**: LOSO F1 0.841 [0.769, 0.903] at OP1 with 16.9 merged false alarms/h; selected after seeing
  Step C held-out results, so optimistic. Not deployable (gain-dependent EDA features).
- **Deployment estimate (D)**: F1 0.762 [0.629, 0.863] at the default (OP1 + 4-of-5), 7.6 false-alarm events per hour of
  non-stress time, 25 % of stress windows missed on average and nearly all of S2's stress missed. The ranking quality is
  as good as R (PR-AUC 0.951 vs 0.948); the loss is threshold placement for individual wearers.
- **Measured**: every LOSO metric, CI, confusion matrix, latency, curve and coefficient above; exactness of the saved
  model. **Estimated / extrapolated**: false alarms per hour (about 20 min of non-stress data per subject), model size in
  bytes. **Assumed**: the 60 s warm-up before calibration (WESAD had 26-643 s; untested); the calibration validity rule.
- **Unvalidated on the prototype (nothing transfers until checked on prototype data after IEC approval)**: finger EDA in
  mV through the nonlinear ESP32 ADC (the relative features are invariant only to LINEAR gain/offset, G3); fingertip
  MAX30102 HR vs E4 wrist HR (E4 stress-HR LoA already ±25-29 bpm vs ECG, Step A); whether a wearer's 5-min rest at first
  use resembles the WESAD seated baseline; real-life stressors other than the TSST, motion, and conditions WESAD lacks
  (exercise, anxiety, panic: this is stress detection only); a C port of scipy `find_peaks` prominence for scr_count_rel.
- Time shortcut (Step C, C(i)/C(ii)): still unresolved on WESAD; nothing here can separate stress physiology from
  protocol position.

### Re-run (from software/ml-models/wesad_edge_model, repo venv; data read-only)
Prerequisites are the committed Step B/C outputs (`features_w30.csv`, `features_eda_w30.csv`, `stepc_*`); to rebuild
them use the Step C re-run list above. Then:
```
..\..\..\.venv\Scripts\python -m src.final_model --data "C:\datasets\WESAD"      (~80 s; writes results/ and models/)
..\..\..\.venv\Scripts\python -m src.predict --roundtrip                          (fresh-process round trip)
..\..\..\.venv\Scripts\python -m unittest tests.test_leakage tests.test_stepc tests.test_final_model -v
```

---

## Step E — inference CLI `src/predict.py` (2026-11-11; no retraining, models/ unchanged, data read-only)
Extends the Step D host predictor; `--roundtrip`, `load`, `proba_joblib`, `proba_json`, `check_feature_order` are
unchanged. Everything is loaded from `models/`: joblib pipeline, plain JSON (checked against the pipeline on every call,
<= 1e-12), `feature_spec.json` (HR and SCR limits are parsed from its definitions), `window.json` (30 s / 5 s, 4-of-5,
asserted equal to the JSON), the calibration block of the JSON and `calibration_procedure.md` (asserted to describe
both stored statistics), OP1 threshold 0.5161.

### What model D needs (from the code, not assumed)
- Inputs, in order: `hr_mean` (bpm; BVP 64 Hz, causal 0.7-3.5 Hz band-pass, IBI rule 0.33-1.5 s -> only 40.0-181.8 bpm
  possible), `eda_slope_rel` (1/s; EDA 4 Hz LP slope / sigma_lp_cal), `scr_count_rel` (phasic peaks, prominence >=
  sigma_ph_cal, >= 1 s apart). Both EDA features need the 5-min rest calibration; hr_mean does not. TEMP and ACC are NOT
  model inputs.
- Input modes: `--features-csv` (header = the 3 names in spec order, optional leading `t_start_s`); `--signals CAL TEST`
  with BVP + EDA only (minimum needed), ACC optional and used only for the motion check (TEMP not accepted: unused);
  `--demo-wesad S` runs the same stream code on a WESAD subject. Non-model rates are resampled (resample_poly) to
  64 / 4 Hz with a warning (UNVALIDATED).
- Output per window: features, raw score, threshold, window alert, 4-of-5 smoothed alert, note. No probability or
  percentage is printed (score uncalibrated, Step D). The required disclaimer line is printed on every run, plus a
  "NOT validated" block (finger mV EDA, fingertip PPG, prototype hardware).

### Validation rules (reject = exit code 2, "INPUT REJECTED - check failed: <rule>", problem, what to do)
- Features CSV: missing columns, wrong order, unexpected/duplicate columns, ragged rows, empty/header-only, non-numeric,
  NaN/inf, hr_mean outside 40.0-181.8 bpm (physiological 30-220 also stated), hr_mean that looks like ms or Hz,
  scr_count_rel not a whole number 0-30, |eda_slope_rel| > 23.2 (10 x WESAD max |value|), t_start_s off the 5 s grid.
  Values inside those limits but outside the WESAD range of the 6421 windows are noted, not rejected.
- Signals: header must state the unit (`eda_uS` / `eda_mV`; `acc_*_g`); EDA > 100 "uS" or > 3300 "mV" -> ADC counts /
  wrong units; ACC median |a| outside 0.3-4 g -> raw LSB; sampling rate missing or header/flag mismatch; NaN gaps;
  signal lengths differ by > 1 s; calibration < 300 s; test < 30 s; railed (> 1 % of samples repeating the segment
  min/max; WESAD ~0.004 %; mV >= 3100 or <= 0). Calibration: flat EDA (range < 0.005 uS, features.py), mean EDA < 0.05 uS,
  sigma 0 or not finite (saved rule), > 50 % of calibration windows with flag_ppg (WESAD max 0.148), > 25 % with
  motion_energy > 0.1 g (WESAD max 0.185), sigma outside [WESAD min / 10, WESAD max x 10] uS (warning if only outside
  [min, max]); warning if < 60 s precede the calibration (saved procedure: >= 60 s warm-up, untested); warning if no ACC.
- Per test window: withheld (counted as negative for smoothing) for flag_ppg / NaN HR, flat or low EDA, > 2 % railed;
  motion > 0.1 g is noted only (training scored those windows).
- These thresholds were written for this script and are NOT validated on any device.

### Checks run (real output)
- `python -m unittest tests.test_leakage tests.test_stepc tests.test_final_model tests.test_predict -v`: 70 tests,
  69 ok + 1 expected failure (below), 119.5 s. The 18 earlier tests are unchanged and pass.
- Stored-output agreement: `--features-csv` on the 300 real rows of `models/roundtrip_windows.csv` (fresh process)
  reproduces `p_ref` exactly (tolerance 0) and `alert_op1_ref`. `--demo-wesad S5` and `--signals` on S5 exported to CSV
  reproduce all 370 scored S5 windows of `features_w30.csv` / `features_eda_w30.csv` (max difference 0.0; test tolerance
  rtol 1e-9, atol 1e-12) and the 17 S5 rows of `p_ref` (difference 0.0; tolerance 1e-12). There is no held-out
  prediction of D to compare with: `final_predictions.csv` holds LOSO fold-model scores, and those models were not saved.
- Two fresh processes give byte-identical output (features CSV, synthetic signals, S5 demo).
- S5 demo: 1130 windows after calibration, 199 withheld for low PPG quality (9 of them in the 370 scored windows = the 9
  `flag_ppg` windows of the training CSV; most of the rest in meditation / transient segments). S5 is a TRAINING
  subject; the demo is not an accuracy estimate.

### Known limitation found (not fixed; `test_KNOWN_LIMITATION_band_limited_noise_passes_ppg_quality`, expectedFailure)
The training PPG quality rule is scale-invariant: band-passed Gaussian noise passes it (synthetic: median ppg_quality
0.69, fake HR ~131 bpm, 1 of 60 windows flagged); random-interval pulses: 0 of 60 flagged; large noise bursts
added to a clean pulse: 4-57 % of windows flagged depending on burst length and spacing. Only "no pulse" inputs
(slow wander, too few beats) are flagged. A real signal quality index (or the MAX30102 DC level for finger detection)
is needed before any device use; any new rule must be checked against WESAD so it does not drop the evaluated windows.

### Run (from software/ml-models/wesad_edge_model, repo venv; data read-only)
```
..\..\..\.venv\Scripts\python -m src.predict --demo-wesad S5 --data "C:\datasets\WESAD"
..\..\..\.venv\Scripts\python -m src.predict --features-csv FILE.csv
..\..\..\.venv\Scripts\python -m src.predict --signals CAL_DIR TEST_DIR [--bvp-fs 64 --eda-fs 4 --acc-fs 32]
..\..\..\.venv\Scripts\python -m unittest tests.test_leakage tests.test_stepc tests.test_final_model tests.test_predict -v
```

---

## Step F — PPG gate: PRE-REGISTRATION (written 2026-11-11 BEFORE any gate number was computed; not changed after)
Scope: a proposed signal-quality gate for the hr_mean input only. Model, features and `predict.py` are NOT changed.
- Gate (per 30 s window, causally band-passed BVP): G0 existing flag_ppg; G1 flat (raw peak-to-peak 0) or railed
  (> 2 % of raw samples repeat the window max/min); G2 r_ac = highest local maximum of the normalised autocorrelation
  over lags 0.33-1.5 s must be >= T_AC; G3 autocorrelation HR (60/lag, or 2 x that) must agree with hr_mean within
  HR_AGREE (relative). Flagged if any of G0-G3 fires.
- Reference for real windows: chest ECG HR (Step A, `hrv_check_chest_windows_w30.csv`): good = ECG valid and |wrist -
  chest| <= 5 bpm; bad = wrist NaN or > 15 bpm; otherwise ambiguous.
- Split: TUNE = S2 S4 S6 S8 S10 S13 S15 S17, EVAL = S3 S5 S7 S9 S11 S14 S16. T_AC = 1st percentile of r_ac and
  HR_AGREE = 99th percentile of the G3 disagreement over TUNE good windows that pass G0 and G1. No noise sets a threshold.
- SYNTHETIC noise: tuning seeds 0-99 (white Gaussian) inspected only; test sets use seeds >= 1000 and 10 fault types
  (white Gaussian/uniform, pink, brown, spikes, random-interval pulses, flat, railed full scale, clipped pulse, periodic
  1.6 Hz motion) + 2 clean-pulse controls. Expected and reported in advance: a periodic in-band artefact cannot be
  separated from a pulse by PPG alone.

## Step F — ESP32 feasibility: results (real runs 2026-11-11; no retraining, models/ unchanged, data read-only)
No firmware. Nothing has run on an ESP32. The 7 saved files in `models/` have the same SHA-256 before and after; the
only new file there is the generated `models/stress_lr_D_w30.h`. `wesad_baseline/` and `predict.py` are not changed.
Full write-up, including the measured-vs-estimated table, the arithmetic, the domain shift and the validation plan:
`results/metrics/esp32_feasibility.md`.

### A. Streaming parity (`src/stream_ref.py`; `stream_parity_*.csv`, `stream_windows_{float64,float32}.csv`)
- **How it works:** a sample-by-sample reference. It uses a biquad cascade (transposed DF-II, same operation order
  as scipy sosfilt, steady-state start), an EMA, 30 s ring buffers, and Welford calibration statistics. Every 5 s,
  window features come from the rings through a hand-written port of scipy find_peaks (local maxima -> distance ->
  prominence, same order). The arithmetic type is a parameter (float64 / float32). Deviations D1-D7 are in its
  docstring.
- **Run:** all 15 subjects, 5509 scored windows, about 65 s on 12 processes.
- **float64:** max |diff| hr_mean 8.5e-14 bpm, eda_slope_rel 1.4e-14, scr_count_rel 0. ppg_quality and n_beats are
  identical. Calibration sigmas differ by <= 5.6e-15.
- **float32:** hr_mean median |diff| 7.0e-6 bpm, max 0.383 bpm. The max is one window (S14 2195 s, stress): one beat
  was lost to rounding (48 -> 47 peaks). The next largest is 7.3e-5. eda_slope_rel max 1.9e-5, scr_count_rel 0,
  sigmas <= 3.8e-6 uS.
- **Alerts:** 0 / 5509 window alerts and 0 / 5509 4-of-5 smoothed alerts differ, in float64 AND float32. flag_ppg
  differs in 0 windows, and logit-threshold vs p-threshold disagree in 0.
- **Peak detector, not tuned:** it differs from scipy only in the order of equal-height peaks (D1). On SYNTHETIC
  quantised signals the stable order differs in 90 / 100 cases, while scipy's argsort order matches 100 / 100 (test).
  On the float, filtered WESAD streams it changed 0 windows. Firmware must therefore keep the filter outputs in
  float32, not integers.
- **D7:** device smoothing over every 5 s window vs the evaluation's gapped table: 44 windows differ (a difference in
  definition).

### B. C export (`src/export_esp32.py`, `models/stress_lr_D_w30.h`, `tests/c/model_harness.c`)
- **Header:** generated from the JSON only. It contains the feature order, window/step, rates, folded weights, bias,
  medians, the OP1 threshold (p and logit), 4-of-5 smoothing, the calibration names and the JSON SHA-256.
  Constants are written with 17 digits, and the test parses them back exactly equal to the JSON.
- **Host test:** compiled with MSVC 14.40 (VS 2022, /O2 /W4 /WX) and RUN on this PC. On the 300 real rows of
  roundtrip_windows.csv, the float build has max |dp| 1.04e-7 and the double build 3.3e-16; 0 alert differences in
  both (smallest |logit - threshold| 0.0046). The 40 synthetic NaN rows: 0 differences. The smoother on 700
  decisions: 0 differences vs runner.smooth.
- **ESP32 cross-compile:** xtensa-esp32-elf-gcc 13.2.0 -Os, compile only, NOT run. Score + smoother = .text 154 B
  + .literal 28 B, no RAM.

### C. Resources (`esp32_resources.csv`; MEASURED = this PC or compile-only, ESTIMATED = arithmetic)
- **MEASURED file sizes:** joblib 1942 B, JSON 4117 B, header 4316 B.
- **MEASURED Python latency per window:** JSON scorer 25.6 us, joblib 622 us.
- **MEASURED streaming reference (Python):** 12.5 us per sample in float64, 20.2 us in float32 emulation. PC timings
  are not ESP32 timings and are not stable between runs; no tolerance is claimed. In the 2026-11-11 re-run the same
  measurement was 28.66 / 46.48 us per sample (x2.29 / x2.30; saved in
  `C:\auralis_verification_backup\regenerated_timing_run2\results\metrics\esp32_resources.csv`); the cause was not
  investigated. Full comparison: "Clean-state verification -> PC timing measurements".
- **MEASURED peak-detector work:** about 2030 scan steps, 72 local maxima and 14436 prominence steps per window.
- **ESTIMATED RAM:** 15.1 kB at 64/4 Hz, 24.4 kB at 100/10 Hz (<= 4.6 % of 520 KB; RTOS and radio stacks excluded).
- **ESTIMATED CPU:** 0.07-0.30 % of one core at 64/4 Hz, 0.17-0.67 % at 100/10 Hz (5-20 cycles per op ASSUMED).
- **ESTIMATED flash:** model constants 36 B, DSP constants 116 B.
- **Verdict: yes, D can run on the ESP32 in compute and memory terms.** Whether it works on fingertip PPG and finger
  GSR in mV is unknown until prototype data exist (after IEC approval). The firmware checklist is in
  esp32_feasibility.md section 5.

### D. PPG gate (`src/ppg_gate.py`; design = the pre-registration above, unchanged; `ppg_gate_*.csv`)
- **Frozen thresholds (TUNE good windows, n = 2794):** T_AC = 0.0871, HR_AGREE = 0.330. Sanity check on tuning
  noise: 18 / 100 white noise flagged.
- **EVAL (S3 S5 S7 S9 S11 S14 S16), existing rule -> proposed gate:**
  - ECG-good windows falsely rejected: 0.22 % (5/2242) -> 1.07 % (24/2242).
  - ECG-bad windows caught: 25.9 % (65/251) -> 28.3 % (71/251).
  - Scored windows flagged: 2.9 % -> 4.1 %; stress windows 4.4 % -> 7.2 %.
  - TUNE good windows: 0.36 % -> 2.1 %.
- **SYNTHETIC test noise (seeds >= 1000), caught by existing -> proposed:**
  - White Gaussian 0 -> 16 / 100; uniform 1 -> 23; pink 2 -> 17; brown 10 -> 21; spikes 37 -> 47; random-interval
    pulses 0 -> 4.
  - Flat 0 -> 10 / 10; railed full scale 0 -> 10 / 10; clipped pulse 0 -> 100 / 100 (all by G1).
  - Periodic 1.6 Hz motion 0 -> 0 / 100 (expected; not separable without ACC).
  - Clean controls flagged: 1 / 200 -> 1 / 200.
- **Reading:** E4 wrist PPG is weakly periodic even when its HR agrees with ECG (median r_ac 0.39-0.48, 1st
  percentile 0.087), so its r_ac overlaps band-passed noise (median 0.12-0.18). The POST-HOC sweep
  (ppg_gate_tradeoff_POSTHOC.csv) looks at EVAL and the test noise and was used for nothing. At T_AC 0.2 it would
  catch 92 % of white noise, but it would also reject 8.1 % of ECG-good windows and flag 34 % of stress windows.
  Withheld windows count as "no alert", so that would suppress stress detection.
- **PROPOSED change to predict.py (not made):**
  (1) add G1 (flat / railed per window: raw ptp 0 or > 2 % of samples at the window max/min) as a withhold rule.
  Evidence: 100 % of flat / railed / clipped SYNTHETIC windows caught, 0 real WESAD windows flagged by G1.
  (2) do NOT add G2 / G3 as withhold rules with E4-derived thresholds. Re-derive them on prototype fingertip PPG
  against ECG after IEC approval (same TUNE/EVAL design), and add a MAX30102 DC-level finger-presence check.
  (3) keep test_KNOWN_LIMITATION_band_limited_noise_passes_ppg_quality as expectedFailure: the limitation is NOT
  solved.

### E. Domain shift and post-IEC validation plan
Written analysis only, nothing run: esp32_feasibility.md sections 6 and 8. Covered by earlier tests: linear EDA gain
and offset (G3), PPG gain (the peak rules are scale-invariant), rates and quantisation (G4, config R only). NOT
covered: ESP32 ADC and GSR-module nonlinearity, finger vs wrist site, fingertip vs wrist PPG, sensor noise, sampling
jitter, the 60 s warm-up, and the home calibration rest. The plan has three parts: bench linearity and timing checks
(no participants), PowerLab-referenced recordings after approval, and pre-registered keep / recalibrate / retrain /
exit rules.

### Tests
`python -m unittest tests.test_leakage tests.test_stepc tests.test_final_model tests.test_predict tests.test_stream_esp32`:
89 tests, OK (expected failures = 1), 181.6 s. That is the 70 earlier tests (unchanged; the Step E known limitation is
still expectedFailure) plus 19 new tests in tests/test_stream_esp32.py: filter / EMA / Welford / slope / find_peaks
equivalence; stream vs offline on SYNTHETIC signals and on WESAD S5; float32; stored parity; header == JSON; C harness
(skipped if no compiler); gate behaviour and threshold reproduction.

### Re-run (from software/ml-models/wesad_edge_model, repo venv; data read-only)
```
..\..\..\.venv\Scripts\python -m src.stream_ref --data "C:\datasets\WESAD"     (~65 s; parity float64 + float32)
..\..\..\.venv\Scripts\python -m src.ppg_gate --data "C:\datasets\WESAD"       (gate evaluation)
..\..\..\.venv\Scripts\python -m src.export_esp32 --all                         (header, C test, resources)
..\..\..\.venv\Scripts\python -m unittest tests.test_leakage tests.test_stepc tests.test_final_model tests.test_predict tests.test_stream_esp32 -v
```

---

## Step G — flat / railed / clipped PPG check added to predict.py (2026-11-11; Step F proposal (1) only)
Only code change: `src/predict.py`. Model, features, `models/`, `ppg_gate.py`, `stream_ref.py` and the C header unchanged.
- Rule (= Step F gate G1, same decision as `ppg_gate.gate()["G1_flat_railed"]`): per 30 s window of RAW BVP at the
  sensor's own rate (before resampling, which would smear a clipped plateau), withhold if peak-to-peak is 0 (or no
  samples) or > 2 % of the samples repeat the WINDOW max or min. `predict.ppg_flat_railed`, constant
  `PPG_RAIL_FRAC_WINDOW = 0.02`; note "flat/railed/clipped PPG"; withheld windows count as negative for smoothing.
  The existing rules (segment-level railed check, window vs SEGMENT min/max) are unchanged.
- NOT added: the G2/G3 periodicity checks (they failed to reject noise with E4-derived thresholds, Step F).
  The band-limited-noise limitation is NOT solved; its test stays expectedFailure.
- Evidence (real runs): all 17292 WESAD 30 s windows on the 5 s grid of the 15 complete recordings (all segments,
  incl. transient/meditation) -> 0 flagged, largest repeated-extreme fraction 0.00104 (S10; limit 0.02). The 6421
  labelled windows in `ppg_gate_real_windows.csv`: 0 flagged. S5 demo: 0 windows with the new note. SYNTHETIC Step F
  test windows: flat 10/10, railed 10/10, clipped 100/100 caught; clean controls 0/200; noise types 0 (expected:
  this is not a noise detector).
- SYNTHETIC intermittent clipping (one minute of a 5-min test segment clipped at half the peak): exactly the 7 windows
  fully inside are withheld, at 64 Hz and at 100 Hz input; the training PPG rule alone accepted all 7 (HR ~66 bpm).
  A flat minute: 7/7 withheld; 2 of them had passed the old rule with a fake HR of 160 bpm (filter ringing).
- Tests: 8 new in `tests/test_predict.py` (class PPGFlatRailed). Full suite: 97 tests, OK (expected failures=1),
  249.9 s. `python -m src.predict --roundtrip`: PASS (unchanged).

---

## Clean-state verification (2026-11-11)
Before deleting: SHA-256 of every file in `models/`, `results/`, the module-root `feature_spec.json` and
`../wesad_baseline/out/` recorded in `verify_hashes_before.csv` (202 files, this folder). 179 generated files deleted
(the 3 `*_run.log` console logs, `results/metrics/esp32_feasibility.md` (hand-written) and `.gitkeep` kept), all
backed up first, then the pipeline re-run from scratch.

### Exception: random-forest PR-AUC is not reproducible in the last digits (paper reconciliation, Part A)
- Where: `stepc_paper_ladder.csv` / `stepc_paper_per_subject.csv`, detector rf_personal (the frozen baseline's RF:
  200 trees, depth 8, `n_jobs=-1`), steps P1 and P2 only; PR-AUC columns only (10 + 2 cells).
- Subject: **S14**. Its per-subject PR-AUC takes one of two values: **0.676362233505** (committed) or
  **0.676695054805** (the clean re-run). Mean PR-AUC P1 0.951396 vs 0.951418, P2 0.951583 vs 0.951605.
- Test (same S14 fold, forest fitted once, 30 multi-threaded `predict_proba` calls): 30 distinct score vectors,
  max |diff| 2.22e-16; PR-AUC 0.676362233505 in **23/30** calls and 0.676695054805 in **7/30**; ROC-AUC 0.829464285714
  and all decisions identical in 30/30; single-threaded gives 0.676362233505. S14 has 13 tied scores.
- Cause: multi-threaded `predict_proba` sums per-tree probabilities in thread-completion order (float addition is not
  associative); the last-bit noise reorders tied scores, and average precision depends on that order. Tree fitting
  is deterministic (random_state 42).
- **Unaffected**: every F1, sensitivity, specificity, precision, accuracy, confusion matrix and ROC-AUC; every other
  detector and step; the deployed model D (logistic regression) and all Step B/C/D LR results.
- Action (user decision): the two committed CSVs were restored from the verified backup (SHA-256 equal to
  `verify_hashes_before.csv`). **This preserves the previously committed reporting values; it does NOT make the
  multithreaded RF calculation deterministic.** A re-run can give either value. Not fixed: `wesad_baseline/` is frozen.
  The regenerated versions and the test are kept outside the repo (`C:\auralis_verification_backup\
  regenerated_rf_nondeterminism\`).

### PC timing measurements: original (committed) vs second run (clean re-run)
Five files differed from the committed versions only in run-dependent fields; every computed value in them was
identical. User decision: the COMMITTED versions were restored from the verified backup (SHA-256 equal to
`verify_hashes_before.csv`); the regenerated versions + SHA-256 are kept outside the repo
(`C:\auralis_verification_backup\regenerated_timing_run2\`). The committed files therefore still show the ORIGINAL
measurements; the second-run values exist only here and in that folder.
- `models/metadata.json`: `created` and `git_head_before_this_commit` only (all checksums, versions, metrics equal).
- `feature_build_summary.json`: `build_seconds` 101.9 (original) vs 118.7 (second run).
- Timings (MEASURED on this PC, Python / MSVC host build; NOT ESP32 timings):

| quantity | original (committed) | second run | ratio |
|---|---|---|---|
| JSON scorer per window | 25.6 us | 30.7 us | 1.20 |
| joblib `predict_proba` per window | 622.2 us | 752.0 us | 1.21 |
| streaming reference per sample, float64 (median of 15 subjects) | 12.5 us | 28.7 us | 2.29 |
| streaming reference per sample, float32 emulation | 20.2 us | 46.5 us | 2.30 |
| streaming run wall time, 15 subjects, 12 processes | 66 s | 118 s (console-observed, see note) | 1.79 |
| C window alert, float | 2.9 ns | 2.68 ns | 0.93 |
| C window alert, double | 3.4 ns | 3.93 ns | 1.16 |

Note on the wall-time row: 66 s is in the saved `results/metrics/stream_parity_run.log`. The 118 s is a
console-observed value from the verification re-run ("stream runs done in 118 s"); that console log was written to a
temporary session folder and was NOT preserved with the saved timing evidence
(`C:\auralis_verification_backup\regenerated_timing_run2\`). It is not independently verifiable from the saved timing
files. All other rows (per-window and per-sample timings) are in the saved `esp32_resources.csv` /
`stream_parity_calibration.csv` / `esp32_c_test.json` of both runs.

- **The ~2.3x streaming slowdown is UNEXPLAINED**: its cause was not investigated (no other job of this session was
  running during that stage). Per subject (`stream_parity_calibration.csv`, both runs): x2.03-2.74 in float64,
  x1.40-2.41 in float32. The earlier "about +-30 % between runs" statement was replaced (user-approved, 2026-11-11) in
  NOTES.md Step F, README.md and `results/metrics/esp32_feasibility.md` by these observed numbers; the "earlier run"
  figures it quoted (32.9 us, 804.6 us, 11.6 / 19.1 us, 2.6 / 4.5 ns) have no saved file.
- Not affected: operation counts (`stream_op_counts.csv` byte-identical), so the ESTIMATED ESP32 RAM / CPU / flash
  figures, which come from counts and buffer sizes, not from PC timings, are unchanged. Nothing has run on an ESP32.

### Result of the verification
202 / 202 files byte-identical to `verify_hashes_before.csv` after the two documented restores (2 RF PR-AUC files,
5 run-dependent files). Without the restores: 195 byte-identical, 5 differing only in timestamps / PC timings, 2 in
RF PR-AUC (thread-order nondeterminism). Baseline re-run (`wesad_baseline.py --out` to a temporary folder, binary and
three-class): all 15 files byte-identical to `../wesad_baseline/out/`. The C build files were written to a scratch
folder (`build/` untouched). No source file was changed by the verification. **The pipeline is NOT fully
deterministic** (RF PR-AUC exception above); all LR results, model D and the feature CSVs reproduced exactly.

### Documentation and paper-number trace (2026-11-11)
- `README.md` rewritten (setup, dataset, one command per stage, final results table copied from
  `final_results.csv`, MEASURED and ESTIMATED ESP32 tables kept separate, limitations). `DEMO.md` added (10-minute
  walk-through; every expected output copied from the clean re-run; the PowerShell one-liners were run as written).
- Paper-number trace: all 36 values in `results/paper_numbers.md` re-derived from their stated source files with
  independent code (not `paper_outputs.py`): 36/36 equal, 36/36 source attributions correct.
- The paper draft IS now in `docs/` (it was not at Step 0). It still has placeholders only (no numbers). It confirms
  the [a]-[d] mapping that `paper_numbers.md` assumed ([a] fixed rule, [b] personal rule, [c] RF universal, [d] RF
  personal). Stale sentences, not edited: `paper_numbers.md` header ("draft is not in docs/"; generated file, would
  change on regeneration) and Step 0 above. Draft-vs-code disagreements (SD floor, calibration exclusion, causal
  filtering, HRV, session-time shortcut, personalisation conclusion, model size, window/step; panic/anxiety wording is
  consistent) are listed privately in `docs/paper_draft_vs_code_2026-11-11.md` (gitignored). The paper was not edited.
- Documentation rounding error fixed: Step D table, shipped row, FP windows/h CI low 14.8 -> 14.7 (CSV value 14.7499).
- `results/metrics/esp32_feasibility.md` (hand-written, protected) line 21 "+-30 % between runs": replaced with the
  user's approval by the observed two-run timings (see "PC timing measurements" above), in a separate commit.

## Bench hardware check and firmware port (2026-11-12)
- Parts identified on the breadboard by register reads (I2C test sketch): PPG board PART_ID 0x11 = **MAX30100**
  (not MAX30102); GY-521 WHO_AM_I 0x70 = **MPU6500** (not MPU6050); GSR = generic 3-pin module. The green
  MAX30100 board pulled SDA/SCL to ~2.0 V (its 4.7 kΩ pull-ups go to the board's 1.8 V rail); 2.2 kΩ pull-ups to
  3V3 raised the bus to 2.9 V and both devices then answered (0x57, 0x68).
- **GSR polarity risk.** Grove-type GSR modules measure resistance; their output falls as conductance rises (Seeed
  wiki example: 503 counts untouched, 256-371 worn). eda_slope_rel and scr_count_rel are invariant only to a
  POSITIVE linear gain (Step C, G3), so raw mV would flip the slope's sign and count SCR recoveries instead of
  responses (scr_count_rel has standardised weight 0.58). Firmware handles the sign; untested on prototype.
- Firmware `firmware/esp32/auralis_model_d/`: C port of `src/stream_ref.py` + the generated model header (unchanged)
  + filter constants generated with the same scipy calls. Host parity test (gcc, float) vs StreamD on synthetic
  signals from `tests/synthetic_signals.py` (60 s warm-up, 300 s calibration, rest, stress-like segment, rest):
  227 windows, 0 alert and 0 smoothed-alert differences vs float64 and float32; max |diff| hr_mean 2.45e-05 bpm,
  eda_slope_rel 7.5e-06, scr_count_rel 0, z 1.1e-06. Device-only parts NOT in that test: MAX30100 100 -> 64 Hz
  linear interpolation (simulated separately: max error 0.07 % of amplitude on a 1.2 Hz sine), the MAX30100 sample
  clock, the GSR conversion, quality gating. Nothing has run on the ESP32.
- Paper: the bench PPG is a MAX30100, not a MAX30102; the outline and draft must say so if hardware is mentioned.
- First bench runs (2026-11-12, developer only; NOT study data, IEC approval pending): PPG gave plausible HR
  (75-86 bpm at rest, IR ~23000 with finger, <1000 without; withheld windows as designed). At rest after
  calibration, scr_count_rel read 8-14 per 30 s window (WESAD median 1) and drove alerts: the 4 Hz GSR samples jumped
  by hundreds of mV (16 back-to-back ADC reads per sample cannot reject 50 Hz mains hum). Firmware changed to
  200 Hz oversampling with a 250 ms mean per 4 Hz sample. The module's open-circuit output floats (650-880 mV),
  so the open-circuit/Seeed-formula conversion was replaced by a measured polarity (light vs firm contact).
  MAX30100lib stalls permanently if its 16-sample FIFO fills (IR stuck at 0 was seen); firmware now resets it.
  None of these fixes has been re-run on the bench yet.

## EDA front end v0.3 (2026-11-13; designed and simulated, not built)
- Bench tests with fixed resistors across the GSR module's electrodes: 12 kOhm and 30 kOhm both read ~2885 mV
  (output at its ceiling), 560 kOhm read the ADC floor (~142 mV), and moving the resistor made jumps of 790-900 mV.
  The ESP32 rail probe (two 10k from 3V3) moved ~24-29 mV per 30 s and did not track the GSR (r ~ 0), so the
  module's narrow, high-gain window, not the supply, explained the noise.
- Decision: ADS1115 + 330 kOhm divider giving uS ratiometrically (AIN3 = 3V3), 10 kOhm NTC on AIN1 for skin
  temperature (MAX30205 dropped). Firmware `firmware/esp32/auralis_model_d` v0.3 implements it; conversions
  host-tested (`test/afe_test.c`), and the whole sketch was run on a PC against an emulated ADS1115 and pulse
  signal: correct uS and deg C, 0-2 SCRs per window at rest when spontaneous responses were simulated.
- Caveat found in that simulation: with NO spontaneous responses during calibration, sigma_ph_cal is pure ADC noise
  and scr_count_rel counts ~20 noise peaks per window. Real skin has responses at rest, but a very calm or dry wearer
  could approach this; it follows from the feature definition (prominence = 1 x sigma_ph_cal).
