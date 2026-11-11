# ESP32 feasibility of model D (stress detection)

Written 2026-11-11 (Step F). This covers model D, a logistic regression on hr_mean, eda_slope_rel and scr_count_rel,
30 s windows with a 5 s step, OP1 threshold plus 4-of-5 smoothing. WESAD "stress" means a lab stressor (TSST). It is
**not** panic or anxiety. Nothing here has run on an ESP32 and no firmware was written. Every number is either
MEASURED (on this PC, or by compiling only) or ESTIMATED (arithmetic, with the arithmetic shown). Sources:
`esp32_resources.csv`, `esp32_c_test.json`, `stream_parity_*.csv`, `ppg_gate_*.csv`.

## 1. Measured vs estimated

| quantity | value | kind | how |
|---|---|---|---|
| models/stress_lr_D_w30.joblib | 1942 B | MEASURED | file size |
| models/stress_lr_D_w30.json | 4117 B | MEASURED | file size |
| models/stress_lr_D_w30.h (generated) | 4316 B | MEASURED | file size (source text, not flash) |
| Python inference per window, JSON scorer | 25.6 µs | MEASURED, this PC | median of 20000 runs, 1 row |
| Python inference per window, joblib `predict_proba` | 622.2 µs | MEASURED, this PC | median of 5000 runs |
| Python streaming reference, per input sample (float64 / float32-emulated) | 12.5 / 20.2 µs | MEASURED, this PC | median over 15 subjects; pure-Python model of the C loop |
| C window alert per call (float / double) | 2.9 / 3.4 ns | MEASURED, this PC | MSVC /O2 harness bench |

PC timings are not stable between runs, and no tolerance is claimed. Two runs have saved timing files: the committed
`esp32_resources.csv` / `esp32_c_test.json` / `stream_parity_calibration.csv` (the values above), and the 2026-11-11
clean re-run, saved outside the repo in `C:\auralis_verification_backup\regenerated_timing_run2\results\metrics\`
(same file names). Second run vs committed: JSON scorer 30.7 vs 25.6 µs (×1.20), joblib 752.0 vs 622.2 µs (×1.21),
streaming reference 28.66 vs 12.54 µs per sample in float64 (×2.29) and 46.48 vs 20.25 µs in float32 (×2.30; per
subject ×2.03-2.74 and ×1.40-2.41, `stream_parity_calibration.csv`), C alert 2.68 vs 2.9 ns float (×0.93) and 3.93 vs
3.4 ns double (×1.16). The cause of the slowdown was not investigated. The "earlier run" figures previously quoted here
(32.9 µs, 804.6 µs, 11.6 / 19.1 µs, 2.6 / 4.5 ns) have no saved file. These are PC timings, not ESP32 timings; the
values above are those in the committed `esp32_resources.csv`.
| ESP32 object code for score + smoother | .text 154 B, .literal 28 B, .data 0, .bss 0 | MEASURED (cross-compile only, NOT run) | xtensa-esp32-elf-gcc 13.2.0 -Os |
| Peak-detector work per window (BVP + EDA) | 2030 scan steps, 72 local maxima, 14436 prominence steps | MEASURED | instrumented `stream_ref`, all 15 subjects (counts do not depend on the platform) |
| RAM, feature front end + model, 64 Hz / 4 Hz | 15061 B (14.7 KiB) | ESTIMATED | sum of buffer sizes, below |
| RAM, feature front end + model, 100 Hz / 10 Hz | 24405 B (23.8 KiB) | ESTIMATED | sum of buffer sizes, below |
| Flash for model constants / DSP constants | 36 B / 116 B | ESTIMATED | 9 and 29 float32 |
| Flash for feature + gate + calibration code | 2-8 KB | ESTIMATED (GUESS) | not written in C yet |
| CPU load, 64 Hz / 4 Hz | 0.07-0.30 % of one core | ESTIMATED | 5-20 cycles per op ASSUMED |
| CPU load, 100 Hz / 10 Hz | 0.17-0.67 % of one core | ESTIMATED | 5-20 cycles per op ASSUMED |

PC timings are **not** ESP32 timings. The ESP32 numbers are arithmetic estimates: none were measured on hardware.

## 2. Arithmetic for the estimates (float32 = 4 B)

**RAM at the WESAD rates (64 Hz BVP, 4 Hz EDA).** N_bvp = 30 s × 64 = 1920; N_eda = 30 × 4 = 120.
- Filtered BVP ring: 1920 × 4 = 7680 B.
- EDA low-passed, phasic and raw rings: 3 × 120 × 4 = 1440 B.
- Peak scratch (worst case N/2 = 960 local maxima): indices uint16, keep flag uint8 and order uint16 = 960 × 5 = 4800 B.
  Measured: about 72 local maxima per window on average, so the worst-case bound is generous.
- Beat times and IBIs: 2 × (1920/21 + 1) × 4 = 736 B.
- Filter states: (3 BVP biquads × 2 + 1 EDA biquad × 2 + EMA 1) × 4 = 36 B.
- Calibration Welford ×2 (n uint32, mean and M2 as double): 40 B. Stored sigmas, smoother byte and counters: 21 B.
- Proposed autocorrelation gate: r(k) for k = 20..97 → 78 × 4 = 308 B. The energies are updated per lag, so no
  prefix array is needed. The Python reference uses a prefix array for readability.
- **Total 15061 B = 2.8 % of 520 KB SRAM.** This excludes FreeRTOS, the WiFi/BT stacks, sensor drivers and logging,
  none of which were measured.

**RAM at the prototype rates (100 Hz MAX30102, 10 Hz GSR)**, if the filters are redesigned at those rates and the
buffers hold 30 s: BVP 3000 × 4 = 12000; EDA 3 × 300 × 4 = 3600; peak scratch 1500 × 5 = 7500; beats
2 × (3000/33 + 1) × 4 = 728; states, calibration and gate as above (gate 120 × 4 = 480). **Total 24405 B = 4.6 % of
520 KB.** The alternative is to resample on the device to 64 / 4 Hz (100→64 is 16/25 and 10→4 is 2/5, polyphase).
That adds an FIR state of a few hundred bytes (not computed) and keeps the trained feature definitions unchanged.

**Operations.**
- Per sample: BVP cascade 3 × (5 mul + 4 add) = 27 FLOP; EDA biquad 9 + EMA 3 = 12 FLOP.
  At 64 / 4 Hz: 27 × 64 + 12 × 4 = 1776 FLOP/s. At 100 / 10 Hz: 2820 FLOP/s.
- Per window (every 5 s), without the gate: two-pass SD 4 × 1920, plus the measured peak-detector steps (scan 2030,
  prominence 14436), sorting n log n with n ≈ 72, the EDA slope 2 × 120, and the model (7 multiply-adds). Total about
  25000 ops per window. At 100 Hz the peak work is scaled by 100/64: about 39000.
- Gate autocorrelation: Σ_k (N − k) for k = 20..97 → 143297 MAC per window at 64 Hz, 349020 at 100 Hz.
- Load = (per-sample ops/s + window ops / 5 s) × cycles per op / 240e6.
  At 64 Hz with an assumed 20 cycles per op: (1776 + (25030 + 143297)/5) × 20 / 240e6 = **0.30 %** of one core.
  At 100 Hz: **0.67 %**.
  The cycles-per-op figure is an ASSUMPTION covering loads, loop overhead and the single-precision FPU. Double
  precision on the ESP32 is emulated in software and should not be used in the per-sample path.

## 3. Verdict: can D realistically run on the ESP32-WROOM-32E?

**Yes, on compute and memory. Whether it works on the prototype's sensors is not known yet.**

Reasons:
1. The model itself is 9 constants and 7 multiply-adds; its compiled code is 182 B (MEASURED cross-compile). With the
   C harness, MSVC builds give 0 / 300 alert differences from Python on the real round-trip windows (float max |Δp|
   1.0e-7, double 3.3e-16). The 40 synthetic NaN rows also match.
2. The feature front end needs about 15-24 KiB of RAM (ESTIMATED, about 3-5 % of SRAM) and under 1 % of one core even
   at a pessimistic 20 cycles per op (ESTIMATED).
3. A sample-by-sample reference that uses only C-portable operations reproduces the offline features on all 15 WESAD
   subjects (§4). In float32 (as on the ESP32 FPU), 0 of 5509 scored-window alerts change, both per window and after
   4-of-5 smoothing.
4. What is NOT shown: that fingertip MAX30102 PPG and finger GSR in mV through the ESP32 ADC produce features that the
   model's thresholds fit (§6). "It can run" is not "it detects stress on the prototype".

## 4. Streaming parity (MEASURED; `python -m src.stream_ref`, all 15 subjects, 5509 scored windows)

| arithmetic | feature | max abs diff | median abs diff | NaN mismatches |
|---|---|---|---|---|
| float64 | hr_mean | 8.5e-14 bpm | 0 | 0 |
| float64 | eda_slope_rel | 1.4e-14 1/s | 1.4e-16 | 0 |
| float64 | scr_count_rel | 0 | 0 | 0 |
| float32 | hr_mean | 0.383 bpm (one window; next largest 7.3e-5) | 7.0e-6 | 0 |
| float32 | eda_slope_rel | 1.9e-5 1/s | 6.8e-8 | 0 |
| float32 | scr_count_rel | 0 | 0 | 0 |

- **Alerts:** 0 / 5509 window alerts and 0 / 5509 smoothed alerts differ, in float64 and in float32. Maximum |Δp| is
  0 (float64) and 0.0031 (float32). The existing flag_ppg differs in 0 windows. The logit-threshold and
  p-threshold decisions agree in every window.
- **The single float32 outlier** is S14 at t = 2195 s, a stress window. One beat was lost to float32 rounding (48 → 47
  peaks), so HR moved 118.81 → 118.43 bpm. Its score of 0.72 is far from the threshold of 0.516, so the alert is
  unchanged. This shows that float32 **can** change the detected peak set. It did so in 1 of 5509 windows here and
  changed no alert.
- **Calibration sigmas (Welford running statistics vs np.std):** max difference 5.6e-15 (float64) and 3.8e-6 µS
  (float32, S13).
- **Device-style smoothing vs evaluation smoothing (deviation D7):** 44 windows differ. On the device every 5 s window
  counts. The offline table has gaps (unlabelled or impure windows count as negative). This is a difference in
  definition, not an error. The firmware behaviour (no gaps) is what the device will do; WESAD metrics used the
  gapped version.
- **The peak detector was not tuned toward a match.** It is a port of scipy's algorithm (same stages, same order). Its
  only behavioural difference is the order of equal-height peaks (D1). On quantised signals with many ties, the stable
  order differs from scipy in 90 / 100 synthetic test cases. With scipy's argsort order it matches in 100 / 100. On
  the float, filtered WESAD streams it changed 0 windows. **Firmware warning:** a fixed-point or integer filter would
  create ties. Keep the filter outputs in float32.

All deviations (D1-D7) are listed in the `src/stream_ref.py` docstring.

## 5. What the firmware must implement (none of it exists yet)

1. **Acquisition at fixed rates with timestamps.** MAX30102 FIFO at 100 Hz (IR channel), GSR on GPIO34 at 10 Hz,
   either resampled to 64 / 4 Hz or with filters redesigned at 100 / 10 Hz. HR depends directly on fs: a 1 % rate
   error is a 1 % HR error.
2. **Filters, each with explicit state.** BVP: Butterworth band-pass 0.7-3.5 Hz, 3 biquads, transposed DF-II, with
   the steady-state start zi = zi_unit × x0. EDA: Butterworth low-pass 1 Hz, 1 biquad. Tonic: EMA, τ = 8 s. Phasic =
   low-passed − tonic. Use float32 (not fixed point; see D1).
3. **30 s ring buffers** for filtered BVP, low-passed EDA, phasic EDA and raw EDA (eda_range / flat check). Features
   are computed every 5 s.
4. **Peak detection** as in `stream_ref.find_peaks_port`: local maxima with plateau midpoints, distance selection by
   height priority, prominence with no window length. BVP: distance 21 samples at 64 Hz, prominence ≥ 0.3 × window
   SD (two-pass). EDA: distance 4 samples at 4 Hz, prominence ≥ 1.0 × sigma_ph_cal.
5. **hr_mean rules:** parabolic sub-sample peak times; IBI kept if 0.33-1.5 s; IBI dropped if |IBI − median| ≥ 0.3 ×
   median; ≥ 5 peaks and ≥ 4 IBIs, else NaN; hr_mean = mean of 60/IBI.
6. **eda_slope_rel:** closed-form least-squares slope over the window ÷ sigma_lp_cal.
7. **Calibration routine:** a 5-min seated rest at first use, after the warm-up. Compute sigma_lp_cal and
   sigma_ph_cal with Welford running statistics. Store them in NVS (flash key-value) with the date and a validity
   flag. Offer re-calibration. Apply the validity checks from `predict.py`: calibration ≥ 300 s; EDA not flat (range
   ≥ 2 mV ASSUMED); sigma finite and > 0; ≤ 50 % of calibration windows with flag_ppg; ≤ 25 % with motion > 0.1 g
   (MPU6050); warn if the warm-up was < 60 s.
8. **60 s warm-up** after power-on or electrode contact, before calibration. This rule is untested (WESAD had
   26-643 s).
9. **Per-window validation (from `predict.py`):** withhold the window, pushing it as 0 to the smoother, if flag_ppg
   or HR is NaN, if EDA is flat or low, or if > 2 % of samples are railed. Note motion > 0.1 g. Reject wrong units,
   meaning ADC counts instead of mV (analogReadMilliVolts).
10. **Model and smoothing:** `models/stress_lr_D_w30.h` (`auralis_window_alert`, `auralis_smooth_push`). No
    probability may be displayed.
11. **Proposed PPG gate (§7):** at present only G1 (flat / railed) is recommended for the device.

## 6. Domain shift WESAD → prototype, per feature

| feature | WESAD (training) | prototype v0.1 | covered by invariance tests | NOT covered |
|---|---|---|---|---|
| hr_mean | E4 wrist PPG (green, reflectance), 64 Hz; vs chest ECG: bias ≤ 1.5 bpm, stress LoA ±25-29 bpm (Step A) | MAX30102 IR at the fingertip, 100 Hz, I2C FIFO, ESP32 timing | PPG gain / units: the peak rules are scale-invariant (prominence relative to SD). Rate: Step C G4 for config R, BVP 100 Hz F1 0.840 vs 0.841 (D not re-run at 100 Hz) | Measurement site (fingertip perfusion, contact pressure, finger motion), sensor noise, ambient light, sampling jitter, the actual fingertip-vs-ECG error |
| eda_slope_rel | E4 wrist EDA in µS, 4 Hz | generic GSR module on finger electrodes, mV via analogReadMilliVolts (nonlinear ADC), 10 Hz | Linear gain and offset: G3, 0 windows changed. 12-bit quantisation: G4 for config R, F1 0.835 | ADC nonlinearity (the relative feature is invariant to LINEAR gain only), the module's own nonlinear mV↔conductance transfer, palmar-finger vs dorsal-wrist SCR dynamics, electrode gel/contact, 10→4 Hz resampling for D, sensor noise relative to sigma_lp_cal |
| scr_count_rel | phasic E4 EDA, prominence ≥ sigma_ph_cal | as above | Linear gain and offset: G3, 0 windows changed | As above. Finger sites typically show larger and more frequent SCRs than the wrist (GUESS, no data), which could shift counts relative to the calibration; noise floor near sigma_ph_cal |
| calibration | first 5 min of the WESAD seated baseline | a 5-min rest at first use, at home or in the lab | none | Whether that rest resembles the WESAD baseline; the 60 s warm-up rule |
| timing | E4 clock; labels at 700 Hz | ESP32 loop / FIFO timestamps | none | Clock drift, dropped FIFO samples, I2C stalls |

## 7. Signal-quality gate (part D summary; details in NOTES.md Step F)

The pre-registered gate has 4 parts. G0 is the existing flag. G1 flags a flat or railed raw window. G2 requires
autocorrelation periodicity r_ac ≥ T_AC. G3 requires the autocorrelation HR to agree with hr_mean. The thresholds were
frozen on the TUNE subjects' ECG-confirmed good windows (T_AC = 0.087, HR_AGREE = 0.33).

| | existing rule | proposed gate |
|---|---|---|
| EVAL good windows (ECG ≤ 5 bpm) falsely rejected | 0.22 % (5 / 2242) | 1.07 % (24 / 2242) |
| EVAL bad windows (> 15 bpm or NaN) caught | 25.9 % (65 / 251) | 28.3 % (71 / 251) |
| EVAL scored windows flagged | 2.9 % | 4.1 % |
| SYNTHETIC white Gaussian noise caught | 0 / 100 | 16 / 100 |
| SYNTHETIC pink / brown / spikes / random-interval pulses caught | 2 / 10 / 37 / 0 of 100 each | 17 / 21 / 47 / 4 of 100 |
| SYNTHETIC flat / railed full scale / clipped pulse caught | 0 / 0 / 0 | 10/10, 10/10, 100/100 (all by G1) |
| SYNTHETIC periodic 1.6 Hz motion caught (expected not separable) | 0 / 100 | 0 / 100 |
| SYNTHETIC clean-pulse controls falsely flagged | 1 / 200 | 1 / 200 |

**Conclusion.** G1 works and should be adopted. G2 and G3 do not solve the noise problem on E4 wrist PPG. Real good
windows are only weakly periodic: median r_ac is 0.39-0.48, with a 1st percentile of 0.087. Band-passed noise sits at
about 0.12-0.18. The post-hoc sweep (`ppg_gate_tradeoff_POSTHOC.csv`) looks at the evaluation data and was used for
nothing. It shows that catching ≥ 92 % of white noise (T_AC 0.2) would reject 8.1 % of ECG-good windows and flag
34 % of stress windows. Withheld windows count as "no alert", so such a gate would mostly suppress stress detection.
G2 and G3 thresholds must be re-derived on prototype fingertip PPG against an ECG reference after IEC approval. A
MAX30102 DC-level / finger-presence check is a separate, sensor-level gate and also needs prototype data.

## 8. Validation plan after IEC approval (PROPOSED; nothing here has been run; no prototype recording may be used before approval)

**Bench (no participants, possible before approval).** These are not human recordings.
1. Feed the GSR input precision resistors spanning about 0.5-30 µS. Record mV through analogReadMilliVolts to get the
   transfer curve. Measure the deviation from a straight line over the range one wearer's EDA spans. If the
   nonlinearity changes eda_slope_rel or scr_count_rel by more than a pre-set amount (proposed: > 5 % of
   sigma-relative units, computed by passing WESAD EDA through the measured curve), firmware must linearise with the
   bench lookup table before computing features.
2. Drive the PPG chain with a function generator or optical phantom at known rates of 40-180 bpm. Required: hr_mean
   within ±1 bpm. Also check the sample timing: FIFO rate error < 0.5 %.
3. Run the C front end (once written) on WESAD signals replayed over serial and compare its output with
   `stream_ref` / `stream_windows_float32.csv`. Required: 0 alert differences.

**Recordings (after approval only).** Participants wear the prototype while ADInstruments PowerLab records reference
ECG (HR) and reference GSR (µS, calibrated amplifier) in sync, with a synchronisation marker at the start and end.
Sessions follow a WESAD-like order: warm-up ≥ 60 s; 5-min seated rest (the calibration); an approved stressor (for
example timed mental arithmetic with evaluation); recovery; a neutral or amusing control; and short scripted motion
segments for the gate. The number of participants, consent wording and stressor are set by the IEC protocol. Sizes
of about 15 match WESAD; this needs a statistical justification in the protocol.

**Comparisons.**
- HR: prototype hr_mean vs PowerLab ECG HR per 30 s window, using Bland-Altman with the repeated-measures correction
  and rmcorr, the same method as Step A. Compare with E4 vs ECG: bias ≤ 1.5 bpm, stress LoA ±25-29 bpm.
- EDA: prototype eda_slope_rel / scr_count_rel vs the same features computed from PowerLab µS (both relative to
  their own calibration). Report ICC / Spearman per window, SCR-count MAE, and the distribution of calibration sigmas.
- Model: window alerts on prototype features vs the protocol labels (stressor vs rest/control), reporting the
  CLAUDE.md metric set, and compared with alerts computed from the PowerLab reference features.
- Gate: re-derive T_AC / HR_AGREE on a TUNE subset of participants using ECG agreement, as in Step F, and report on
  the rest.

**Decision rules (to be pre-registered before the first recording; values PROPOSED).**
- Keep D unchanged if all of these hold: HR rest LoA within ±10 bpm and stressor LoA no wider than E4's (±29 bpm);
  feature ICC ≥ 0.75 against the reference; per-participant ROC-AUC (stressor vs rest) lower CI bound ≥ 0.80; and
  false alarms per hour no worse than the WESAD LOSO estimate (7.6/h, CI upper bound 13.5).
- Recalibrate the threshold only, if the ranking is fine (ROC-AUC criterion met) but the scores are shifted, as with
  S2 in WESAD. The new threshold is chosen on a calibration subset of participants and reported on the others.
- Retrain, if the features agree poorly with the reference (ICC < 0.5) or the ranking fails. Retraining needs enough
  labelled prototype recordings and the same subject-independent evaluation (LOSO, inner CV only).
- Exit (deployable for a further study, still "stress detection" only) when the keep or recalibrate criteria are met
  on participants not used for any threshold, with the gate's false rejection of ECG-good windows ≤ 2 %.
