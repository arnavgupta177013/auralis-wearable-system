# auralis_model_d — stress model D on the ESP32 (breadboard v0.2)

**Stress detection only** (WESAD lab stressor, TSST). Not panic or anxiety detection, not a medical device.
Model D is validated on WESAD wrist data only. **On this prototype it is UNVALIDATED** (fingertip PPG, finger GSR).
No prototype recordings may be used as study data before IEC approval.

## Files
| file | what it is |
|---|---|
| `auralis_model_d.ino` | the sketch: sensors, calibration flow, alert LED |
| `auralis_features.h` | C port of `src/stream_ref.py` (feature front end + model call). Plain C99, no Arduino code |
| `stress_lr_D_w30.h` | model D, copied **unchanged** from `software/ml-models/wesad_edge_model/models/` |
| `dsp_consts.h` | filter coefficients, generated with the same scipy calls as `src/stream_ref.design_constants()` |
| `test/` | host parity test (C vs Python `StreamD`) |

Library: **MAX30100lib** (OXullo Intersecans); only its low-level `MAX30100` class is used, for raw IR samples.

## Hardware assumed
ESP32 DevKit V1 · green MAX30100 board (part ID 0x11) with 2.2 kΩ pull-ups SDA/SCL → 3V3 · GY-521 with MPU6500
(ID 0x70) · 3-pin GSR on GPIO34 · alert LED on GPIO25 (330 Ω). Pins match `hardware/electronics/draw_schematic.py`.

## Signal path (same rules as the training pipeline)
| model input | on the device |
|---|---|
| `hr_mean` | MAX30100 raw IR, 100 Hz, 16 bit → inverted (pulse peaks up) → **linear-interpolated to 64 Hz** → causal band-pass 0.7–3.5 Hz → 30 s ring → peaks (distance 21, prominence ≥ 0.3 × window SD) → IBI rules → mean 60/IBI |
| `eda_slope_rel` | GSR mV at 4 Hz (16-read average) → **conductance** (see below) → causal low-pass 1 Hz → LS slope / `sigma_lp_cal` |
| `scr_count_rel` | phasic = low-passed − EMA (τ 8 s) → peaks with prominence ≥ `sigma_ph_cal` |
| calibration | 60 s warm-up, then 300 s seated rest; Welford SDs of low-passed and phasic EDA |
| decision | `auralis_window_alert` (OP1) per 5 s window → 4-of-5 smoother → LED |

**GSR direction.** Grove-type GSR modules measure skin *resistance*: their output falls when the skin conducts more.
The model was trained on conductance (µS), and its EDA features only tolerate a *positive* linear gain. The sketch
therefore converts with Seeed's formula, `G = 1e6 / R`, `R = (1024 + 2x)·10000 / (x_open − x)` (x in 10-bit units,
x_open = reading with nothing touching, measured at power-on). If a module's output instead *rises* when touched, the
sketch detects this and uses mV directly. Either way this is UNVALIDATED: the formula is Seeed's, and the module's
nonlinearity was never measured against a reference.

**Windows are withheld** (decision forced to 0, as the model header requires) when the pulse finger is off or the
sensor is saturated (mean raw IR outside 10000–65000) or GSR contact is lost. Motion is printed but is not a model
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

The sketch itself was compiled only against stub Arduino headers (syntax check). MAX30100lib method and enum
names were checked against the library source. **Nothing has run on the ESP32 yet.**

## Not verified / known gaps
- The 100 → 64 Hz linear interpolation and the MAX30100's own sample clock (assumed exactly 100 Hz) are not in the
  parity test. HR error scales with any clock error.
- GSR conductance conversion, finger site and fingertip PPG: domain shift from E4 wrist data, untested.
- `IR_FINGER_MIN` (10000) is a guess: tune it from the `IR` value printed each window.
- Keep `loop()` free of long blocking calls: MAX30100lib reads at most 15 FIFO samples per call and reads nothing
  if the 16-sample FIFO is completely full (≈160 ms at 100 Hz).
