# auralis_model_d — stress model D on the ESP32 (breadboard v0.3, ADS1115 front end)

**Stress detection only** (WESAD lab stressor, TSST). Not panic or anxiety detection, not a medical device.
Model D is validated on WESAD wrist data only. **On this prototype it is UNVALIDATED** (fingertip PPG, dry or gel EDA electrodes).
No prototype recordings may be used as study data before IEC approval.

## Files
| file | what it is |
|---|---|
| `auralis_model_d.ino` | the sketch: sensors, calibration flow, alert LED |
| `auralis_features.h` | C port of `src/stream_ref.py` (feature front end + model call). Plain C99, no Arduino code |
| `stress_lr_D_w30.h` | model D, copied **unchanged** from `software/ml-models/wesad_edge_model/models/` |
| `dsp_consts.h` | filter coefficients, generated with the same scipy calls as `src/stream_ref.design_constants()` |
| `analog_front_end.h` | EDA conductance (µS) and thermistor (°C) conversions, ADS1115 config word. Plain C99 |
| `test/` | host parity test (C vs Python `StreamD`) and `afe_test.c` (conversions) |

Library: **MAX30100lib** (OXullo Intersecans); only its low-level `MAX30100` class is used, for raw IR samples.
The ADS1115 and the MPU are read with plain `Wire` calls (no other library).

## Hardware assumed
ESP32 DevKit V1 · green MAX30100 board (part ID 0x11) with 2.2 kΩ pull-ups SDA/SCL → 3V3 · GY-521 with MPU6500
(ID 0x70) · ADS1115 on the same I2C bus (0x48–0x4B, found automatically) · alert LED on GPIO25 (330 Ω).

| ADS1115 input | circuit |
|---|---|
| AIN0, EDA | 3V3 → 330 kΩ → AIN0 → electrode A ~ skin ~ electrode B → GND. Max current through skin 3.3 V / 330 kΩ = 10 µA |
| AIN1, skin temperature | 3V3 → 10 kΩ (1 %) → AIN1 → NTC 10 kΩ (2 mm bead) → GND |
| AIN3, supply | wired straight to 3V3: both dividers are computed ratiometrically, so rail drift cancels |

 Pins match `hardware/electronics/draw_schematic.py`.

## Signal path (same rules as the training pipeline)
| model input | on the device |
|---|---|
| `hr_mean` | MAX30100 raw IR, 100 Hz, 16 bit → inverted (pulse peaks up) → **linear-interpolated to 64 Hz** → causal band-pass 0.7–3.5 Hz → 30 s ring → peaks (distance 21, prominence ≥ 0.3 × window SD) → IBI rules → mean 60/IBI |
| `eda_slope_rel` | ADS1115 AIN0 at 128 SPS, ~25 conversions averaged per 250 ms (rejects 50/60 Hz hum) → conductance in **µS** = (Vs − V0)/(330 kΩ · V0), minus the zero level → causal low-pass 1 Hz → LS slope / `sigma_lp_cal` |
| `scr_count_rel` | phasic = low-passed − EMA (τ 8 s) → peaks with prominence ≥ `sigma_ph_cal` |
| calibration | 60 s warm-up, then 300 s seated rest; Welford SDs of low-passed and phasic EDA |
| decision | `auralis_window_alert` (OP1) per 5 s window → 4-of-5 smoother → LED |

**EDA in µS, no polarity question.** The divider gives true skin conductance in µS, the unit the model was trained
on, and it always rises with conductance. The ADS1115's own input impedance (a few MΩ, not measured here) sits in
parallel with the skin and adds a constant conductance. The sketch measures it at power-on with the electrodes off the
skin (the 3 s **zero** step) and subtracts it. An offset would not change the model's features anyway; the zero
matters for absolute µS values and for contact detection (more than 0.05 µS above zero = on skin).

**Skin temperature** is printed and logged (°C) but is not a model input. The seller gives no B value for the
thermistor: `NTC_BETA_K` defaults to 3950 and `TEMP_OFFSET_C` to 0, both to be set from a calibration against a
reference thermometer. The divider drives about 165 µA through the bead, so expect a small self-heating offset
(roughly a tenth of a degree, not measured); the calibration absorbs it.

**PPG watchdog.** MAX30100lib computes the samples to read as `(write_ptr − read_ptr) & 15`; if the 16-sample FIFO
ever fills completely this is 0 and the library never reads again (seen on the bench: IR stuck at 0). The sketch
resets the FIFO after 300 ms without samples and re-initialises the sensor after 2 s.

**Windows are withheld** (decision forced to 0, as the model header requires) when the pulse finger is off or the
sensor is saturated (mean raw IR outside 10000–65000) or EDA contact is lost. Motion is printed but is not a model
input (the training pipeline dropped ACC).

## Re-running the parity test
From `software/ml-models/wesad_edge_model` (Python environment from that folder's README):
```powershell
gcc -O2 -std=c99 -o harness ..\..\..\firmware\esp32\auralis_model_d\test\harness.c -lm
& $py ..\..\..\firmware\esp32\auralis_model_d\test\gen_and_compare.py
```
It writes `bvp.txt`, `eda.txt` and `harness` in the current folder; delete them afterwards. If
`models/stress_lr_D_w30.h` is regenerated, copy it here again.

## Verified (on a PC, not on an ESP32)
`test/gen_and_compare.py` feeds the same synthetic signals (repo `tests/synthetic_signals.py`; warm-up, rest
calibration, rest, a stress-like segment, rest) through `auralis_features.h` (gcc, float) and through
`src/stream_ref.StreamD`:

| vs StreamD | hr_mean max diff | eda_slope_rel | scr_count_rel | z | alert / smoothed differences |
|---|---|---|---|---|---|
| float64 | 2.45e-05 bpm | 4.1e-06 | 0 | 1.1e-06 | 0 / 0 of 227 windows |
| float32 | 1.53e-05 bpm | 7.5e-06 | 0 | 7.2e-07 | 0 / 0 of 227 windows |

`test/afe_test.c` checks the conversions: 1 MΩ, 100 kΩ and 10 MΩ skin give 1, 10 and 0.1 µS, a 3.20 V supply gives
the same answer as 3.30 V, and the NTC at 25 °C and 37 °C reads back 25.00 and 37.00 °C.

The whole sketch was also run on a PC against emulated hardware (a simulated ADS1115 on the I2C bus with 50 Hz hum and
noise, a simulated pulse signal, and an assumed 8 MΩ ADS1115 input impedance): it found the ADS1115, measured the
zero (0.125 µS), detected contact, ran warm-up and calibration, read 2.0 µS and 33.0 °C as simulated, and with
simulated spontaneous skin responses counted 0–2 SCRs per window at rest (no alerts) and alerted during a simulated
stress episode. That checks the code paths and timing, not the real parts.
MAX30100lib method and enum names were checked against the library source. **Nothing has run on the ESP32 yet.**

## Not verified / known gaps
- The 100 → 64 Hz linear interpolation and the MAX30100's own sample clock (assumed exactly 100 Hz) are not in the
  parity test. HR error scales with any clock error.
- The ADS1115 circuit has not been built yet: the I2C code, the zero step and the µS values are untested on hardware.
- If calibration happens to contain almost no spontaneous skin responses (a very calm or very dry-skinned wearer),
  `sigma_ph_cal` comes out tiny and `scr_count_rel` then counts noise wiggles. This follows from the model's feature
  definition (prominence = 1 × `sigma_ph_cal`), not from the device; watch for many SCRs at rest.
- Electrode site and fingertip PPG: domain shift from E4 wrist data, untested.
- `IR_FINGER_MIN` (10000) is a guess: tune it from the `IR` value printed each window.
- Keep `loop()` free of long blocking calls: MAX30100lib reads at most 15 FIFO samples per call and reads nothing
  if the 16-sample FIFO is completely full (≈160 ms at 100 Hz).
