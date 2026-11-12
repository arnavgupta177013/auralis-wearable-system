/*
  Auralis — stress model D on the ESP32 (breadboard prototype v0.2)
  STRESS DETECTION ONLY (WESAD lab stressor). Not panic or anxiety detection, not a medical device.
  The model is validated on WESAD wrist data only; on this prototype it is UNVALIDATED (finger PPG, finger GSR).

  Hardware (see the Auralis breadboard schematic v0.2):
    MAX30100 pulse board  SDA 21, SCL 22  (0x57, part ID 0x11)   + R5/R6 2.2k pull-ups to 3V3
    MPU6500 (GY-521)      SDA 21, SCL 22  (0x68, ID 0x70)         motion: printed only, not a model input
    GSR module            OUT -> GPIO34                           read as conductance (see GSR below)
    Alert LED             GPIO25 -> 330 ohm -> LED -> GND         (vibration motor later, same pin)

  Library: "MAX30100lib" by OXullo Intersecans (only its low-level MAX30100 class is used, for raw IR samples).

  Files in this folder:
    auralis_features.h   C port of src/stream_ref.py (host-tested: 0 alert differences vs the Python reference)
    stress_lr_D_w30.h    model D, copied unchanged from software/ml-models/wesad_edge_model/models/
    dsp_consts.h         filter coefficients generated from src/preprocessing.py

  What happens after power-on (Serial Monitor 115200):
    1. GSR open-circuit reading: keep fingers OFF the GSR electrodes for 5 s.
    2. Put the finger on the pulse sensor and the GSR electrodes on two fingers of the other hand. Detected automatically.
    3. 60 s warm-up (filters settle), then 5 min CALIBRATION: sit still, relaxed, no talking (LED blinks).
    4. RUNNING: one line every 5 s. LED on = smoothed stress alert (4 of the last 5 windows).
  Serial commands:  c = recalibrate   o = redo GSR open-circuit reading   v = toggle CSV output

  IMPORTANT: no prototype recordings may be used as study data before IEC approval.
*/

#include <Wire.h>
#include "MAX30100.h"
#include "auralis_features.h"

// ------------------------------------------------------------------ pins and settings
const int PIN_SDA = 21, PIN_SCL = 22;
const int PIN_GSR = 34;
const int PIN_ALERT = 25;

const uint32_t IR_FINGER_MIN   = 10000;   // mean raw IR above this = finger on sensor (16-bit; tune from the IR printout)
const uint32_t IR_RAIL         = 65000;   // mean raw IR above this = saturated: lower IR_LED_CURRENT
#define IR_LED_CURRENT  MAX30100_LED_CURR_27_1MA
const uint32_t WARMUP_S        = 60;
const uint32_t CAL_S           = 300;
const float    MOVING_SD_G     = 0.05f;

// ------------------------------------------------------------------ objects and state
MAX30100 ppg;
fe_state FE;
enum Phase { GSR_OPEN, WAIT_WEAR, WARMUP, CALIBRATING, RUNNING };
Phase phase = GSR_OPEN;
uint32_t phaseStartMs = 0;
bool csvOut = false;
bool mpuOk = false;

// ------------------------------------------------------------------ GSR
// The Grove-type GSR module measures skin RESISTANCE: its output FALLS when the skin conducts more (sweat).
// The model needs a signal that RISES with conductance, so the reading is converted with Seeed's formula:
//   R = (1024 + 2x) * 10000 / (x_open - x)  ohm,  x = reading in 10-bit units, x_open = reading with nothing touching
//   G = 1e6 / R  uS
// If this module's output instead RISES when touched, the sketch detects that at step 2 and uses mV directly.
float gsrOpenMv = NAN;
bool  gsrUseRawMv = false;

float readGsrMv() {
  uint32_t s = 0;
  for (int i = 0; i < 16; i++) s += analogReadMilliVolts(PIN_GSR);
  return s / 16.0f;
}

// Returns the EDA value fed to the model (uS, or mV in raw mode). NAN = no skin contact.
float gsrToEda(float mv) {
  if (gsrUseRawMv) return mv;
  float x = mv * 1024.0f / 3300.0f, xo = gsrOpenMv * 1024.0f / 3300.0f;
  if (x >= xo * 0.97f) return NAN;                 // within 3 % of open circuit: no contact
  return 100.0f * (xo - x) / (1024.0f + 2.0f * x); // = 1e6 / R
}

// ------------------------------------------------------------------ PPG: 100 Hz raw IR -> 64 Hz by linear interpolation
uint64_t irIn = 0, bvpOut = 0;
float irPrev = 0;
double irSum = 0; uint32_t irN = 0; uint16_t irLast = 0;   // per-window finger check

void pushIr(uint16_t ir) {
  float cur = -(float)ir;              // raw IR falls with each pulse: invert so systolic peaks point up (as BVP)
  if (irIn == 0) irPrev = cur;
  while (bvpOut * 25 <= irIn * 16) {   // output time m/64 s <= input time n/100 s
    float frac = irIn == 0 ? 1.0f : (float)((int64_t)(bvpOut * 25) - (int64_t)((irIn - 1) * 16)) / 16.0f;
    float y = irPrev + frac * (cur - irPrev);
    if (phase >= WARMUP) fe_push_bvp(&FE, y);
    bvpOut++;
  }
  irPrev = cur; irIn++;
  irSum += ir; irN++; irLast = ir;
}

// ------------------------------------------------------------------ motion (MPU6050 / MPU6500, direct registers)
const uint8_t MPU_ADDR = 0x68;
bool mpuWrite(uint8_t r, uint8_t v) { Wire.beginTransmission(MPU_ADDR); Wire.write(r); Wire.write(v); return Wire.endTransmission() == 0; }
int mpuRead(uint8_t r, uint8_t *b, uint8_t n) {
  Wire.beginTransmission(MPU_ADDR); Wire.write(r);
  if (Wire.endTransmission(false) != 0) return 0;
  Wire.requestFrom(MPU_ADDR, n); int k = 0; while (Wire.available() && k < n) b[k++] = Wire.read(); return k;
}
bool mpuInit() {
  uint8_t id = 0; if (mpuRead(0x75, &id, 1) != 1 || id == 0 || id == 0xFF) return false;
  mpuWrite(0x6B, 0x80); delay(100); mpuWrite(0x6B, 0x01); delay(50);
  mpuWrite(0x1A, 0x04); mpuWrite(0x1C, 0x08); if (id != 0x68) mpuWrite(0x1D, 0x04);
  Serial.printf("Motion sensor OK (chip ID 0x%02X)\n", id); return true;
}
float accSum = 0, accSq = 0; uint32_t accN = 0;
void sampleMotion() {
  uint8_t b[6]; if (mpuRead(0x3B, b, 6) != 6) return;
  float x = (int16_t)(b[0] << 8 | b[1]) / 8192.0f, y = (int16_t)(b[2] << 8 | b[3]) / 8192.0f, z = (int16_t)(b[4] << 8 | b[5]) / 8192.0f;
  float m = sqrtf(x * x + y * y + z * z); accSum += m; accSq += m * m; accN++;
}

// ------------------------------------------------------------------ helpers
void setPhase(Phase p) {
  phase = p; phaseStartMs = millis();
  digitalWrite(PIN_ALERT, LOW);
  switch (p) {
    case GSR_OPEN:    Serial.println("\n[1/4] GSR open-circuit reading: keep fingers OFF the GSR electrodes (5 s)..."); break;
    case WAIT_WEAR:   Serial.println("[2/4] Put a fingertip on the pulse sensor and the GSR electrodes on two fingers."); break;
    case WARMUP:      Serial.printf("[3/4] Worn. Warm-up %lu s: sit still.\n", (unsigned long)WARMUP_S); break;
    case CALIBRATING: Serial.printf("[3/4] CALIBRATION %lu s: sit still and relaxed, hand still, no talking.\n", (unsigned long)CAL_S); break;
    case RUNNING:     Serial.println("[4/4] RUNNING. LED on = smoothed stress alert.");
                      if (csvOut) Serial.println("t_s,hr_mean,eda_slope_rel,scr_count_rel,z,alert,alert_smoothed,ir_mean,eda,motion_sd,withheld");
                      break;
  }
}

void startFeatures() { fe_init(&FE); irIn = bvpOut = 0; }

// ------------------------------------------------------------------ setup / loop
void setup() {
  Serial.begin(115200);
  delay(300);
  Serial.println("\nAuralis model D (stress detection; UNVALIDATED on this prototype)");
  pinMode(PIN_ALERT, OUTPUT); digitalWrite(PIN_ALERT, LOW);
  analogReadResolution(12);
  analogSetPinAttenuation(PIN_GSR, ADC_11db);

  Wire.begin(PIN_SDA, PIN_SCL);
  mpuOk = mpuInit();
  if (!mpuOk) Serial.println("Motion sensor not found (optional).");

  if (!ppg.begin()) {
    Serial.println("MAX30100 NOT FOUND - check SDA/SCL, R5/R6 pull-ups. Halting.");
    while (true) delay(1000);
  }
  ppg.setMode(MAX30100_MODE_HRONLY);
  ppg.setLedsCurrent(IR_LED_CURRENT, MAX30100_LED_CURR_0MA);
  ppg.setLedsPulseWidth(MAX30100_SPC_PW_1600US_16BITS);
  ppg.setSamplingRate(MAX30100_SAMPRATE_100HZ);
  ppg.setHighresModeEnabled(true);
  Serial.println("MAX30100 OK: IR only, 100 Hz, 16 bit (resampled to 64 Hz for the model)");

  setPhase(GSR_OPEN);
}

uint32_t tEda = 0, tImu = 0;
float openSum = 0; uint32_t openN = 0;
uint32_t wornSinceMs = 0;
bool edaContactAll = true;

void loop() {
  // ---- PPG: drain the FIFO (16 samples deep = 160 ms at 100 Hz) ----
  ppg.update();
  uint16_t ir, red;
  while (ppg.getRawValues(&ir, &red)) pushIr(ir);

  uint32_t now = millis();

  // ---- serial commands ----
  if (Serial.available()) {
    char c = Serial.read();
    if (c == 'c' && phase >= WARMUP) { startFeatures(); setPhase(WARMUP); }
    if (c == 'o') { openSum = 0; openN = 0; setPhase(GSR_OPEN); }
    if (c == 'v') { csvOut = !csvOut; Serial.printf("CSV output %s\n", csvOut ? "on" : "off"); }
  }

  // ---- motion 50 Hz ----
  if (mpuOk && now - tImu >= 20) { tImu = now; sampleMotion(); }

  // ---- EDA at exactly 4 Hz ----
  if (now - tEda < 250) return;
  tEda += 250;
  if (now - tEda > 1000) tEda = now;           // fell far behind (e.g. after a reset): resync
  float mv = readGsrMv();

  switch (phase) {
    case GSR_OPEN:
      openSum += mv; openN++;
      if (now - phaseStartMs >= 5000) {
        gsrOpenMv = openSum / openN;
        Serial.printf("GSR open-circuit: %.0f mV\n", gsrOpenMv);
        if (gsrOpenMv < 50) Serial.println("  WARNING: very low. Check GSR VCC/GND/OUT wiring and its trimmer.");
        setPhase(WAIT_WEAR);
      }
      return;

    case WAIT_WEAR: {
      float irMean = irN ? irSum / irN : 0; irSum = 0; irN = 0;
      bool finger = irMean > IR_FINGER_MIN;
      bool touched = fabsf(mv - gsrOpenMv) > 0.05f * gsrOpenMv + 20;   // clearly away from open circuit
      if (finger && touched) {
        if (!wornSinceMs) wornSinceMs = now;
        if (now - wornSinceMs >= 5000) {
          gsrUseRawMv = mv > gsrOpenMv;           // module output RISES with contact: use mV as is
          Serial.printf("GSR worn: %.0f mV (open %.0f mV) -> %s\n", mv, gsrOpenMv,
                        gsrUseRawMv ? "output rises with contact: using mV directly"
                                    : "output falls with contact (measures resistance): converting to conductance");
          wornSinceMs = 0; startFeatures(); setPhase(WARMUP);
        }
      } else {
        wornSinceMs = 0;
        static uint32_t lastHint = 0;
        if (now - lastHint > 3000) {
          lastHint = now;
          Serial.printf("  waiting: pulse finger %s (IR %.0f), GSR %s (%.0f mV)\n",
                        finger ? "OK" : "not detected", irMean, touched ? "OK" : "not touched", mv);
        }
      }
      return;
    }
    default: break;
  }

  // ---- WARMUP / CALIBRATING / RUNNING: feed the model front end ----
  float eda = gsrToEda(mv);
  if (isnan(eda)) {
    edaContactAll = false; eda = 0.0f;
    if (phase == CALIBRATING) {                  // contact lost: these samples would corrupt the sigmas
      Serial.println("GSR contact lost during calibration: restarting calibration.");
      fe_cal_begin(&FE, CAL_S * FE_FS_EDA); setPhase(CALIBRATING);
    }
  }
  if (phase == WARMUP && now - phaseStartMs >= WARMUP_S * 1000) {
    fe_cal_begin(&FE, CAL_S * FE_FS_EDA);
    setPhase(CALIBRATING);
  }
  if (phase == CALIBRATING) digitalWrite(PIN_ALERT, ((now / 250) % 8) == 0);   // short blink every 2 s

  float irMean = irN ? irSum / irN : 0;
  bool fingerOn = irMean > IR_FINGER_MIN && irMean < IR_RAIL;
  fe_window w;
  bool withhold = !fingerOn || !edaContactAll;
  if (!fe_push_eda(&FE, eda, withhold, &w)) return;

  // ---- one window every 5 s ----
  if (phase == CALIBRATING && !FE.calibrating) {
    if (FE.calibrated) {
      Serial.printf("Calibration done: sigma_lp_cal %.6g, sigma_ph_cal %.6g (%s)\n", FE.sigma_lp_cal, FE.sigma_ph_cal,
                    gsrUseRawMv ? "mV" : "uS");
      setPhase(RUNNING);
    } else {
      Serial.println("Calibration INVALID (flat GSR?). Check electrodes; repeating calibration.");
      fe_cal_begin(&FE, CAL_S * FE_FS_EDA);
      setPhase(CALIBRATING);
    }
  }

  float accMean = accN ? accSum / accN : NAN;
  float motionSd = accN > 1 ? sqrtf(fmaxf(0, accSq / accN - accMean * accMean)) : NAN;
  if (phase == RUNNING) digitalWrite(PIN_ALERT, w.alert_smoothed);

  const char *tag = phase == WARMUP ? "WARM" : phase == CALIBRATING ? "CAL " : "RUN ";
  if (csvOut && phase == RUNNING) {
    Serial.printf("%.0f,%.2f,%.5f,%.0f,%.4f,%d,%d,%.0f,%.4f,%.4f,%d\n", w.t_end_s, w.hr_mean, w.eda_slope_rel,
                  w.scr_count_rel, w.z, w.alert, w.alert_smoothed, irMean, eda, motionSd, withhold);
  } else if (!csvOut) {
    Serial.printf("[%s] t=%4.0fs  HR %5.1f  slope_rel %8.4f  SCR %2.0f  z %6.3f  alert %d  SMOOTHED %d  | IR %5.0f  EDA %.3f %s  motion %.3f g%s%s\n",
                  tag, w.t_end_s, w.hr_mean, w.eda_slope_rel, w.scr_count_rel, w.z, w.alert, w.alert_smoothed,
                  irMean, eda, gsrUseRawMv ? "mV" : "uS", motionSd,
                  !fingerOn ? "  [no finger / saturated: withheld]" : "",
                  !edaContactAll ? "  [GSR contact lost: withheld]" : "");
  }
  irSum = 0; irN = 0; edaContactAll = true;
  accSum = accSq = 0; accN = 0;
}
