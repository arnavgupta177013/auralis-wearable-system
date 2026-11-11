# Calibration procedure (deployment model D)

Duration: 300 s (5 min), once at first use (and whenever the electrodes are re-placed).

## What the wearer does
- Put the device on (GSR electrodes, PPG sensor) and keep it on for the whole session.
- Let the filters run before calibrating: in WESAD the causal filters had run for 26-643 s before the calibration period began; at least 60 s (> 5 x the 8 s EMA tau) is RECOMMENDED but this warm-up rule is untested.
- Sit still and at rest for 5 minutes: seated, relaxed, hand still, no talking, no walking, no eating/drinking, comfortable room (WESAD baseline: seated, reading neutral magazines; the calibration period is its first 5 min).
- Do not start calibration during or straight after exercise or a stressful event.

## Statistics computed from the 5-min period
- `sigma_lp_cal`: population SD (ddof 0) of the causally low-passed EDA samples (1 Hz Butterworth order 2 at 4 Hz, SOS state carried from power-on) over the 5-min calibration period (1200 samples at 4 Hz). Device units (mV) are fine: only ratios are used.
- `sigma_ph_cal`: population SD (ddof 0) of the phasic EDA (low-passed EDA minus causal EMA tonic, tau 8 s) over the same 1200 samples.

## Where they are used
- `eda_slope_rel`: least-squares slope (per s) of the low-passed EDA over the window / sigma_lp_cal
- `scr_count_rel`: number of scipy.signal.find_peaks(phasic window, prominence >= 1.0 x sigma_ph_cal, distance >= 4 samples = 1 s) in the window
- `hr_mean`: no calibration

Stored on the device: sigma_lp_cal, sigma_ph_cal.

Validity (UNTESTED): repeat calibration if either sigma is 0 or not finite; there is no validated lower bound on the prototype.

WESAD reference values (E4 wrist, uS; not comparable to the prototype's finger mV): sigma_lp_cal_min 0.007927, sigma_lp_cal_median 0.1415, sigma_lp_cal_max 1.4, sigma_ph_cal_min 0.00184, sigma_ph_cal_median 0.03476, sigma_ph_cal_max 0.1221, note: E4 wrist EDA in uS; NOT comparable to finger mV on the prototype

In WESAD the calibration period is the first 5 min of the baseline condition; the same stream is used afterwards, so the filters are NOT restarted between calibration and use.

Exact filter and feature definitions: `feature_spec.json`. Nothing here is validated on the prototype.
