/*
  Auralis — stress model D on the ESP32 (breadboard prototype v0.3: ADS1115 front end)
  STRESS DETECTION ONLY (WESAD lab stressor). Not panic or anxiety detection, not a medical device.
  The model is validated on WESAD wrist data only; on this prototype it is UNVALIDATED.

  Hardware (see the Auralis breadboard schematic v0.3):
    MAX30100 pulse board  SDA 21, SCL 22  (0x57, part ID 0x11)   + R5/R6 2.2k pull-ups to 3V3
    MPU6500 (GY-521)      SDA 21, SCL 22  (0x68, ID 0x70)         motion: printed only, not a model input
    ADS1115 16-bit ADC    SDA 21, SCL 22  (0x48-0x4B, found automatically)
      AIN0: EDA     3V3 -- 330k -- AIN0 -- electrode A ~ skin ~ electrode B -- GND
      AIN1: TEMP    3V3 -- 10k (1 %) -- AIN1 -- NTC 10k -- GND
      AIN3: SUPPLY  wired straight to 3V3 (ratiometric reference for both dividers)
    Alert LED             GPIO25 -> 330 ohm -> LED -> GND         (vibration motor later, same pin)

  Library: "MAX30100lib" by OXullo Intersecans (only its low-level MAX30100 class is used, for raw IR samples).
  The ADS1115 and MPU are read with plain Wire calls: no other library.

  Files in this folder:
    auralis_features.h   C port of src/stream_ref.py (host-tested: 0 alert differences vs the Python reference)
    stress_lr_D_w30.h    model D, copied unchanged from software/ml-models/wesad_edge_model/models/
    dsp_consts.h         filter coefficients generated from src/preprocessing.py
    analog_front_end.h   EDA (uS) and NTC (deg C) conversions (host-tested: test/afe_test.c)

  What happens after power-on (Serial Monitor 115200):
    1. ZERO, 3 s: keep the EDA electrodes OFF the skin and apart. This measures what the circuit reads with nothing
       connected (mostly the ADS1115's own input impedance) and subtracts it from every later reading.
    2. Put the EDA electrodes on and a fingertip on the pulse sensor. Detected automatically.
    3. 60 s warm-up (filters settle), then 5 min CALIBRATION: sit still, relaxed, no talking (LED blinks).
    4. RUNNING: one line every 5 s. LED on = smoothed stress alert (4 of the last 5 windows).
  Serial commands:  c = recalibrate   z = redo the zero step   v = toggle CSV output

  IMPORTANT: no prototype recordings may be used as study data before IEC approval.
*/

#include <Wire.h>
#include "MAX30100.h"
#include "auralis_features.h"
#include "analog_front_end.h"

// ------------------------------------------------------------------ pins and settings
const int PIN_SDA = 21, PIN_SCL = 22;
const int PIN_ALERT = 25;

const uint32_t IR_FINGER_MIN   = 10000;   // mean raw IR above this = finger on sensor (16-bit; tune from the IR printout)
const uint32_t IR_RAIL         = 65000;   // mean raw IR above this = saturated: lower IR_LED_CURRENT
#define IR_LED_CURRENT  MAX30100_LED_CURR_27_1MA
const uint32_t WARMUP_S        = 60;
const uint32_t CAL_S           = 300;
const float    MOVING_SD_G     = 0.05f;
// Analog front end (ADS1115). Measure your actual resistors and enter them here for best accuracy.
const float R_EDA_OHM        = 330000.0f;  // series resistor of the EDA divider
const float R_NTC_FIXED_OHM  = 10000.0f;   // fixed resistor of the thermistor divider
const float NTC_R0_OHM       = 10000.0f;   // thermistor resistance at NTC_T0_C
const float NTC_T0_C         = 25.0f;
const float NTC_BETA_K       = 3950.0f;    // NOT stated by the seller: calibrate against a reference thermometer
const float TEMP_OFFSET_C    = 0.0f;       // set from that calibration (reference minus reading)
const float EDA_CONTACT_US   = 0.05f;      // above the zero level by more than this = electrodes on skin
const float EDA_MAX_US       = 100.0f;     // above this = electrodes touching each other / shorted

// ------------------------------------------------------------------ objects and state
MAX30100 ppg;
fe_state FE;
enum Phase { ZEROING, WAIT_WEAR, WARMUP, CALIBRATING, RUNNING };
Phase phase = ZEROING;
uint32_t phaseStartMs = 0;
bool csvOut = false;
bool mpuOk = false;

// ------------------------------------------------------------------ ADS1115: EDA, temperature, supply
// One single-shot conversion at a time (128 SPS, ~7.8 ms), never blocking the loop. Within each 250 ms frame
// (one 4 Hz model sample) the supply (AIN3) is converted once, the thermistor (AIN1) once a second, and the EDA
// node (AIN0) the rest of the time, about 25 conversions spread over the frame. Their mean is a 250 ms boxcar:
// 12.5 cycles of 50 Hz mains hum, so hum averages out instead of reaching the model as fake skin responses.
uint8_t adsAddr = 0;
bool    adsBusy = false;
int     adsCh = 0;
uint32_t adsT = 0;
double  edaSumV = 0; uint32_t edaN = 0;
float   supplyV = NAN, tempC = NAN;
bool    needSupply = true, needTemp = true;
uint32_t adsErrors = 0;

bool adsWrite16(uint8_t reg, uint16_t v) {
  Wire.beginTransmission(adsAddr); Wire.write(reg); Wire.write(v >> 8); Wire.write(v & 0xFF);
  return Wire.endTransmission() == 0;
}
bool adsRead16(uint8_t reg, int16_t *v) {
  Wire.beginTransmission(adsAddr); Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom(adsAddr, (uint8_t)2) != 2) return false;
  *v = (int16_t)((Wire.read() << 8) | Wire.read());
  return true;
}
bool adsFind() {
  for (uint8_t a = 0x48; a <= 0x4B; a++) {
    Wire.beginTransmission(a);
    if (Wire.endTransmission() == 0) { adsAddr = a; return true; }
  }
  return false;
}
void adsStart(int ch) {
  adsCh = ch;
  if (adsWrite16(0x01, afe_ads_config(ch))) { adsBusy = true; adsT = millis(); }
  else adsErrors++;
}
// Call every loop: collects a finished conversion and starts the next one.
void adsService(uint32_t now) {
  if (!adsAddr) return;
  if (adsBusy) {
    if (now - adsT < 9) return;                     // 128 SPS conversion = 7.8 ms
    adsBusy = false;
    int16_t raw;
    if (!adsRead16(0x00, &raw)) { adsErrors++; }
    else {
      float v = raw * AFE_ADS_LSB_V;
      if (adsCh == 0)      { edaSumV += v; edaN++; }
      else if (adsCh == 3) { supplyV = v; needSupply = false; }
      else if (adsCh == 1) {
        float r = afe_ntc_ohm(v, supplyV, R_NTC_FIXED_OHM);
        tempC = afe_ntc_celsius(r, NTC_R0_OHM, NTC_BETA_K, NTC_T0_C) + TEMP_OFFSET_C;
        needTemp = false;
      }
    }
  }
  adsStart(needSupply ? 3 : (needTemp && !isnan(supplyV)) ? 1 : 0);
}
// Mean EDA conductance (uS, before the zero is subtracted) over the frame that just ended; starts a new frame.
uint32_t frameCount = 0;
float edaFrameRawUs() {
  float v0 = edaN ? (float)(edaSumV / edaN) : NAN;
  edaSumV = 0; edaN = 0;
  needSupply = true;                                // fresh supply reading every frame
  if (++frameCount % 4 == 0) needTemp = true;       // temperature once a second
  return afe_conductance_uS(v0, supplyV, R_EDA_OHM);
}

float edaZeroUs = 0.0f;                             // conductance read with the electrodes off the skin
// EDA fed to the model (uS above the zero level), or NAN when the electrodes are off the skin or shorted.
float edaSkinUs(float rawUs) {
  if (isnan(rawUs)) return NAN;
  float g = rawUs - edaZeroUs;
  if (g < EDA_CONTACT_US || g > EDA_MAX_US) return NAN;
  return g;
}


// ------------------------------------------------------------------ PPG: 100 Hz raw IR -> 64 Hz by linear interpolation
uint64_t irIn = 0, bvpOut = 0;
uint32_t lastIrMs = 0, ppgStalls = 0;
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
  lastIrMs = millis();
}

void configurePpg() {
  ppg.setMode(MAX30100_MODE_HRONLY);
  ppg.setLedsCurrent(IR_LED_CURRENT, MAX30100_LED_CURR_0MA);
  ppg.setLedsPulseWidth(MAX30100_SPC_PW_1600US_16BITS);
  ppg.setSamplingRate(MAX30100_SAMPRATE_100HZ);
  ppg.setHighresModeEnabled(true);
  ppg.resetFifo();
}

// MAX30100lib reads (write_ptr - read_ptr) & 15 samples: if the 16-sample FIFO ever fills completely, that is 0
// and the library stops reading for good. Reset the FIFO if no sample arrived for 300 ms; re-initialise after 2 s.
void ppgWatchdog(uint32_t now) {
  if (now - lastIrMs < 300) return;
  ppgStalls++;
  if (now - lastIrMs > 2000) {
    Serial.println("MAX30100: no samples for 2 s - re-initialising (check SDA/SCL wires and R5/R6 pull-ups)");
    if (ppg.begin()) configurePpg(); else Serial.println("MAX30100 NOT FOUND on I2C");
  } else {
    ppg.resetFifo();
  }
  lastIrMs = now;                       // wait another period before the next action
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
    case ZEROING:     Serial.println("\n[1/4] ZERO: keep the EDA electrodes OFF the skin and apart (3 s)..."); break;
    case WAIT_WEAR:   Serial.println("[2/4] Put the EDA electrodes on the skin and a fingertip on the pulse sensor."); break;
    case WARMUP:      Serial.printf("[3/4] Worn. Warm-up %lu s: sit still.\n", (unsigned long)WARMUP_S); break;
    case CALIBRATING: Serial.printf("[3/4] CALIBRATION %lu s: sit still and relaxed, hand still, no talking.\n", (unsigned long)CAL_S); break;
    case RUNNING:     Serial.println("[4/4] RUNNING. LED on = smoothed stress alert.");
                      if (csvOut) Serial.println("t_s,hr_mean,eda_slope_rel,scr_count_rel,z,alert,alert_smoothed,ir_mean,eda_uS,eda_range_30s_uS,temp_C,motion_sd,withheld");
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
  Wire.begin(PIN_SDA, PIN_SCL);
  mpuOk = mpuInit();
  if (!mpuOk) Serial.println("Motion sensor not found (optional).");

  if (!ppg.begin()) {
    Serial.println("MAX30100 NOT FOUND - check SDA/SCL, R5/R6 pull-ups. Halting.");
    while (true) delay(1000);
  }
  configurePpg();
  lastIrMs = millis();
  Serial.println("MAX30100 OK: IR only, 100 Hz, 16 bit (resampled to 64 Hz for the model)");

  // MAX30100lib's begin() calls Wire.begin() again and sets 400 kHz: find the ADS1115 after it.
  if (!adsFind()) {
    Serial.println("ADS1115 NOT FOUND (0x48-0x4B) - check its VDD/GND/SDA/SCL. Halting.");
    while (true) delay(1000);
  }
  Serial.printf("ADS1115 OK at 0x%02X: AIN0 EDA, AIN1 temperature, AIN3 supply\n", adsAddr);
  setPhase(ZEROING);
}

uint32_t tEda = 0, tImu = 0;
float zeroSum = 0; uint32_t zeroN = 0;
uint32_t wornSinceMs = 0;
bool edaContactAll = true;

void loop() {
  // ---- PPG: drain the FIFO (16 samples deep = 160 ms at 100 Hz) ----
  ppg.update();
  uint16_t ir, red;
  while (ppg.getRawValues(&ir, &red)) pushIr(ir);

  uint32_t now = millis();
  ppgWatchdog(now);

  // ---- serial commands ----
  if (Serial.available()) {
    char c = Serial.read();
    if (c == 'c' && phase >= WARMUP) { startFeatures(); setPhase(WARMUP); }
    if (c == 'z') { zeroSum = 0; zeroN = 0; startFeatures(); setPhase(ZEROING); }
    if (c == 'v') { csvOut = !csvOut; Serial.printf("CSV output %s\n", csvOut ? "on" : "off"); }
  }

  // ---- motion 50 Hz; ADS1115 conversions back to back ----
  if (mpuOk && now - tImu >= 20) { tImu = now; sampleMotion(); }
  adsService(now);

  // ---- EDA at exactly 4 Hz ----
  if (now - tEda < 250) return;
  tEda += 250;
  if (now - tEda > 1000) tEda = now;           // fell far behind (e.g. after a reset): resync
  float rawUs = edaFrameRawUs();

  switch (phase) {
    case ZEROING: {
      if (now - phaseStartMs >= 500 && !isnan(rawUs)) { zeroSum += rawUs; zeroN++; }   // skip the first 0.5 s
      if (now - phaseStartMs < 3500) return;
      edaZeroUs = zeroN ? zeroSum / zeroN : 0.0f;
      zeroSum = 0; zeroN = 0;
      Serial.printf("Zero: %.3f uS with nothing connected (subtracted from all EDA). Supply %.3f V.\n", edaZeroUs, supplyV);
      if (edaZeroUs > 1.0f) Serial.println("  WARNING: high zero. Were the electrodes touching skin or each other? Send 'z' to redo.");
      setPhase(WAIT_WEAR);
      return;
    }

    case WAIT_WEAR: {
      float irMean = irN ? irSum / irN : 0; uint32_t nS = irN; irSum = 0; irN = 0;
      bool finger = irMean > IR_FINGER_MIN && irMean < IR_RAIL;
      float g = edaSkinUs(rawUs);
      bool edaOk = !isnan(g);
      if (finger && edaOk) {
        if (!wornSinceMs) wornSinceMs = now;
        if (now - wornSinceMs >= 5000) { wornSinceMs = 0; startFeatures(); setPhase(WARMUP); }
      } else {
        wornSinceMs = 0;
        static uint32_t lastHint = 0;
        if (now - lastHint > 3000) {
          lastHint = now;
          if (nS == 0) Serial.printf("  waiting: NO SAMPLES from the MAX30100 (watchdog resets so far: %lu)\n", (unsigned long)ppgStalls);
          else Serial.printf("  waiting: pulse finger %s (IR %.0f), EDA %s (%.3f uS above zero), temp %.2f C\n",
                             finger ? "OK" : (irMean >= IR_RAIL ? "saturated" : "not detected"), irMean,
                             edaOk ? "OK" : "no skin contact", rawUs - edaZeroUs, tempC);
        }
      }
      return;
    }
    default: break;
  }

  // ---- WARMUP / CALIBRATING / RUNNING: feed the model front end ----
  float eda = edaSkinUs(rawUs);
  if (isnan(eda)) {
    edaContactAll = false; eda = 0.0f;
    if (phase == CALIBRATING) {                  // contact lost: these samples would corrupt the sigmas
      Serial.println("EDA contact lost during calibration: restarting calibration.");
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
      Serial.printf("Calibration done: sigma_lp_cal %.6g, sigma_ph_cal %.6g (uS)\n", FE.sigma_lp_cal, FE.sigma_ph_cal);
      Serial.println("  At rest, SCR should mostly read 0-3 per window. Persistent 6+ at rest = noisy EDA (contact/movement).");
      setPhase(RUNNING);
    } else {
      Serial.println("Calibration INVALID (flat EDA?). Check electrodes; repeating calibration.");
      fe_cal_begin(&FE, CAL_S * FE_FS_EDA);
      setPhase(CALIBRATING);
    }
  }

  float accMean = accN ? accSum / accN : NAN;
  float motionSd = accN > 1 ? sqrtf(fmaxf(0, accSq / accN - accMean * accMean)) : NAN;
  if (phase == RUNNING) digitalWrite(PIN_ALERT, w.alert_smoothed);

  const char *tag = phase == WARMUP ? "WARM" : phase == CALIBRATING ? "CAL " : "RUN ";
  if (csvOut && phase == RUNNING) {
    Serial.printf("%.0f,%.2f,%.5f,%.0f,%.4f,%d,%d,%.0f,%.4f,%.4f,%.2f,%.4f,%d\n", w.t_end_s, w.hr_mean, w.eda_slope_rel,
                  w.scr_count_rel, w.z, w.alert, w.alert_smoothed, irMean, eda, w.eda_range, tempC, motionSd, withhold);
  } else if (!csvOut) {
    Serial.printf("[%s] t=%4.0f HR %5.1f slope %7.4f SCR %2.0f z %6.3f a %d SM %d | IR %5.0f EDA %.3f uS range %.3f | %.2f C | mot %.3f%s%s\n",
                  tag, w.t_end_s, w.hr_mean, w.eda_slope_rel, w.scr_count_rel, w.z, w.alert, w.alert_smoothed,
                  irMean, eda, w.eda_range, tempC, motionSd,
                  !fingerOn ? "  [no finger / saturated: withheld]" : "",
                  !edaContactAll ? "  [EDA contact lost: withheld]" : "");
  }
  irSum = 0; irN = 0; edaContactAll = true;
  accSum = accSq = 0; accN = 0;
}
