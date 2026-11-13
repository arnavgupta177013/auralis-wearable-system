/*
 * analog_front_end.h — conversions for the ADS1115 analog front end (EDA and skin temperature).
 * Plain C99, no Arduino code, so it can be host-tested (test/afe_test.c).
 *
 * Circuits (all on the 3V3 rail that also powers the ADS1115):
 *   EDA:  3V3 -- R_EDA (330 k) --+-- electrode A ~~ skin ~~ electrode B -- GND      AIN0 = node
 *   TEMP: 3V3 -- R_NTC_FIXED (10 k) --+-- NTC 10 k -- GND                          AIN1 = node
 *   SUPPLY: AIN3 wired straight to 3V3, so both dividers are computed ratiometrically
 *           (a change in the 3V3 rail cancels out).
 */
#ifndef AURALIS_AFE_H
#define AURALIS_AFE_H

#include <math.h>

/* Skin (plus anything in parallel) conductance in microsiemens from the divider node voltage v0 and the supply vs.
   v0 = vs * Rskin / (Rseries + Rskin)  =>  G = (vs - v0) / (Rseries * v0).
   Returns NAN if the reading is outside the divider's possible range (0 < v0 < vs). */
static inline float afe_conductance_uS(float v0, float vs, float r_series_ohm) {
    if (!(v0 > 0.0f) || !(vs > 0.0f) || !(v0 < vs)) return NAN;
    return (vs - v0) / (r_series_ohm * v0) * 1.0e6f;
}

/* NTC resistance from the divider node v1 (NTC on the GND side). NAN if out of range. */
static inline float afe_ntc_ohm(float v1, float vs, float r_fixed_ohm) {
    if (!(v1 > 0.0f) || !(vs > 0.0f) || !(v1 < vs)) return NAN;
    return r_fixed_ohm * v1 / (vs - v1);
}

/* Beta equation: 1/T = 1/T0 + ln(R/R0)/B, T in kelvin. Returns degrees Celsius. */
static inline float afe_ntc_celsius(float r_ohm, float r0_ohm, float beta_k, float t0_c) {
    if (!(r_ohm > 0.0f)) return NAN;
    float inv_t = 1.0f / (t0_c + 273.15f) + logf(r_ohm / r0_ohm) / beta_k;
    return 1.0f / inv_t - 273.15f;
}

/* ADS1115 single-shot config word: OS=1, MUX = AINch vs GND, PGA +-4.096 V, single-shot, 128 SPS, comparator off. */
static inline unsigned short afe_ads_config(int channel) {
    return (unsigned short)(0x8000u | ((unsigned)(4 + channel) << 12) | 0x0200u | 0x0100u | 0x0080u | 0x0003u);
}
#define AFE_ADS_LSB_V (4.096f / 32768.0f)   /* 125 uV per count at PGA +-4.096 V */

#endif /* AURALIS_AFE_H */
