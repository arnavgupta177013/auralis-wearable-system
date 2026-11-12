/*
 * auralis_features.h — streaming front end for stress model D (WESAD lab stressor, NOT panic or anxiety).
 *
 * C port of software/ml-models/wesad_edge_model/src/stream_ref.py (StreamD), same rules and order:
 *   hr_mean        BVP 64 Hz -> causal band-pass 0.7-3.5 Hz (3 biquads) -> 30 s ring -> find_peaks
 *                  (distance 21, prominence >= 0.3 x window SD) -> parabolic peak times -> IBI rules -> mean 60/IBI
 *   eda_slope_rel  EDA 4 Hz -> causal low-pass 1 Hz (1 biquad) -> 30 s ring -> LS slope (per s) / sigma_lp_cal
 *   scr_count_rel  phasic = low-passed EDA - causal EMA (tau 8 s) -> 30 s ring -> find_peaks
 *                  (distance 4, prominence >= sigma_ph_cal) -> count
 *   sigma_lp_cal, sigma_ph_cal: Welford over the calibration period (population SD).
 * A window is produced every 5 s (20 EDA samples) once 30 s of EDA exist.
 *
 * Plain C99, no Arduino dependencies, so it can be host-tested against the Python reference.
 * Deviations from stream_ref.py: Welford in double (device: 4 Hz only, cost negligible); a window is produced when
 * the EDA ring is full and the BVP ring is full (the device's two sensors run on separate clocks).
 */
#ifndef AURALIS_FEATURES_H
#define AURALIS_FEATURES_H

#include <math.h>
#include <string.h>
#include "dsp_consts.h"
#include "stress_lr_D_w30.h"

#define FE_FS_BVP      64
#define FE_FS_EDA      4
#define FE_WIN_S       30
#define FE_STEP_S      5
#define FE_N_BVP       (FE_WIN_S * FE_FS_BVP)     /* 1920 */
#define FE_N_EDA       (FE_WIN_S * FE_FS_EDA)     /* 120  */
#define FE_PEAK_DIST_BVP 21                        /* int(0.33 * 64) */
#define FE_PEAK_PROM_K_BVP 0.3f
#define FE_PEAK_DIST_EDA 4                         /* int(1 * 4) */
#define FE_SCR_REL_K   1.0f
#define FE_IBI_MIN_S   0.33f
#define FE_IBI_MAX_S   1.5f
#define FE_IBI_DEV     0.3f
#define FE_MIN_PEAKS   5
#define FE_MIN_IBIS    4
#define FE_MAX_PEAKS   (FE_N_BVP / 2 + 1)

/* ------------------------------------------------------------------ per-sample blocks */
typedef struct { float z[AURALIS_BVP_NSEC][2]; int started; } fe_bvp_filter;
typedef struct { float z[2]; int started; } fe_eda_filter;
typedef struct { float y; int started; } fe_ema;
typedef struct { double mean, m2; unsigned long n; } fe_welford;

static inline float fe_bvp_step(fe_bvp_filter *f, float x) {
    if (!f->started) {                       /* steady-state start: z = zi_unit * x0 (scipy sosfilt_zi) */
        for (int s = 0; s < AURALIS_BVP_NSEC; s++) { f->z[s][0] = BVP_ZI[s][0] * x; f->z[s][1] = BVP_ZI[s][1] * x; }
        f->started = 1;
    }
    for (int s = 0; s < AURALIS_BVP_NSEC; s++) {       /* transposed direct form II, scipy op order */
        const float *c = BVP_SOS[s];
        float y = c[0] * x + f->z[s][0];
        f->z[s][0] = c[1] * x - c[3] * y + f->z[s][1];
        f->z[s][1] = c[2] * x - c[4] * y;
        x = y;
    }
    return x;
}

static inline float fe_eda_step(fe_eda_filter *f, float x) {
    if (!f->started) { f->z[0] = EDA_ZI[0] * x; f->z[1] = EDA_ZI[1] * x; f->started = 1; }
    float y = EDA_SOS[0] * x + f->z[0];
    f->z[0] = EDA_SOS[1] * x - EDA_SOS[3] * y + f->z[1];
    f->z[1] = EDA_SOS[2] * x - EDA_SOS[4] * y;
    return y;
}

static inline float fe_ema_step(fe_ema *e, float x) {
    if (!e->started) { e->y = x; e->started = 1; }
    e->y = EMA_A * x + (1.0f - EMA_A) * e->y;
    return e->y;
}

static inline void fe_welford_push(fe_welford *w, float x) {
    w->n++;
    double d = (double)x - w->mean;
    w->mean += d / (double)w->n;
    w->m2 += d * ((double)x - w->mean);
}
static inline float fe_welford_sd(const fe_welford *w) {
    return w->n ? (float)sqrt(w->m2 / (double)w->n) : NAN;
}

/* ------------------------------------------------------------------ peaks (port of scipy find_peaks, see stream_ref) */
static unsigned short fe_pk[FE_MAX_PEAKS];
static unsigned short fe_ord[FE_MAX_PEAKS];
static unsigned char  fe_keep[FE_MAX_PEAKS];

/* x: n samples in time order. Writes kept peak indices to out (sorted), returns count. */
static int fe_find_peaks(const float *x, int n, int distance, float prom_min, unsigned short *out) {
    int m = 0, i = 1, i_max = n - 1;
    while (i < i_max) {                                   /* 1. local maxima, plateau midpoint */
        if (x[i - 1] < x[i]) {
            int ia = i + 1;
            while (ia < i_max && x[ia] == x[i]) ia++;
            if (x[ia] < x[i]) { if (m < FE_MAX_PEAKS) fe_pk[m++] = (unsigned short)((i + ia - 1) / 2); i = ia; }
        }
        i++;
    }
    for (int k = 0; k < m; k++) { fe_ord[k] = (unsigned short)k; fe_keep[k] = 1; }
    for (int a = 1; a < m; a++) {                          /* stable insertion sort by (height, index) */
        unsigned short v = fe_ord[a]; int b = a - 1;
        while (b >= 0 && x[fe_pk[fe_ord[b]]] > x[fe_pk[v]]) { fe_ord[b + 1] = fe_ord[b]; b--; }
        fe_ord[b + 1] = v;
    }
    for (int r = m - 1; r >= 0; r--) {                     /* 2. distance, highest first */
        int j = fe_ord[r];
        if (!fe_keep[j]) continue;
        for (int k = j - 1; k >= 0 && fe_pk[j] - fe_pk[k] < distance; k--) fe_keep[k] = 0;
        for (int k = j + 1; k < m && fe_pk[k] - fe_pk[j] < distance; k++) fe_keep[k] = 0;
    }
    int nout = 0;
    for (int k = 0; k < m; k++) {                          /* 3. prominence */
        if (!fe_keep[k]) continue;
        int p = fe_pk[k]; float xp = x[p], lmin = xp, rmin = xp;
        for (int q = p; q >= 0 && x[q] <= xp; q--) if (x[q] < lmin) lmin = x[q];
        for (int q = p; q <= n - 1 && x[q] <= xp; q++) if (x[q] < rmin) rmin = x[q];
        float base = lmin > rmin ? lmin : rmin;
        if (prom_min <= xp - base) out[nout++] = (unsigned short)p;
    }
    return nout;
}

/* ------------------------------------------------------------------ window features */
static float fe_bvp_win[FE_N_BVP];
static unsigned short fe_bvp_peaks[FE_MAX_PEAKS];
static float fe_ibi[FE_MAX_PEAKS];
static float fe_tmp[FE_MAX_PEAKS];

static void fe_mean_sd(const float *x, int n, float *mu, float *sd) {
    float s = 0.0f; for (int i = 0; i < n; i++) s = s + x[i];
    float m = s / (float)n, q = 0.0f;
    for (int i = 0; i < n; i++) { float d = x[i] - m; q = q + d * d; }
    *mu = m; *sd = sqrtf(q / (float)n);
}

static float fe_median(float *v, int n) {                /* sorts v in place */
    for (int a = 1; a < n; a++) { float t = v[a]; int b = a - 1; while (b >= 0 && v[b] > t) { v[b + 1] = v[b]; b--; } v[b + 1] = t; }
    return (n % 2) ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2.0f;
}

/* hr_mean from one band-passed BVP window (time order). Returns NAN if not enough clean beats. */
static float fe_hr_window(const float *x, int n, int *n_beats, float *quality) {
    *n_beats = 0; *quality = 0.0f;
    float mu, sd; fe_mean_sd(x, n, &mu, &sd);
    if (sd == 0.0f) return NAN;
    int np = fe_find_peaks(x, n, FE_PEAK_DIST_BVP, FE_PEAK_PROM_K_BVP * sd, fe_bvp_peaks);
    *n_beats = np;
    if (np < FE_MIN_PEAKS) return NAN;
    int nt = 0;
    for (int k = 0; k < np; k++) {                         /* parabolic refinement -> times in s */
        int p = fe_bvp_peaks[k];
        if (p > 0 && p < n - 1) {
            float y0 = x[p - 1], y1 = x[p], y2 = x[p + 1];
            float den = y0 - 2.0f * y1 + y2;
            float off = (fabsf(den) > 1e-12f) ? 0.5f * (y0 - y2) / den : 0.0f;
            if (off < -0.5f) off = -0.5f;
            if (off > 0.5f) off = 0.5f;
            fe_tmp[nt++] = ((float)p + off) / (float)FE_FS_BVP;
        }
    }
    int ni = 0;
    for (int k = 1; k < nt; k++) {
        float v = fe_tmp[k] - fe_tmp[k - 1];
        if (v > FE_IBI_MIN_S && v < FE_IBI_MAX_S) fe_ibi[ni++] = v;
    }
    if (ni < FE_MIN_IBIS) return NAN;
    for (int k = 0; k < ni; k++) fe_tmp[k] = fe_ibi[k];
    float med = fe_median(fe_tmp, ni), tol = FE_IBI_DEV * med;
    int ng = 0; float s = 0.0f;
    for (int k = 0; k < ni; k++) if (fabsf(fe_ibi[k] - med) < tol) { s = s + 60.0f / fe_ibi[k]; ng++; }
    *quality = (float)ng / (float)ni;
    if (ng < FE_MIN_IBIS) return NAN;
    return s / (float)ng;
}

static float fe_slope_per_s(const float *x, int n) {
    float ibar = (float)((n - 1) / 2.0), sii = (float)(n * ((double)n * n - 1) / 12.0), s = 0.0f;
    for (int i = 0; i < n; i++) s = s + ((float)i - ibar) * x[i];
    return s / sii * (float)FE_FS_EDA;
}

/* ------------------------------------------------------------------ the streaming processor */
typedef struct {
    float hr_mean, eda_slope_rel, scr_count_rel;   /* model inputs (NAN = missing) */
    float ppg_quality, eda_range, z;
    int   n_beats, calibrated, alert, alert_smoothed;
    float t_end_s;                                  /* EDA-clock time of the window end */
} fe_window;

typedef struct {
    fe_bvp_filter bvp_f; fe_eda_filter eda_f; fe_ema ema;
    float r_bvp[FE_N_BVP]; unsigned int h_bvp; unsigned long n_bvp;
    float r_lp[FE_N_EDA], r_ph[FE_N_EDA], r_raw[FE_N_EDA]; unsigned int h_eda; unsigned long n_eda;
    int calibrating, calibrated, cal_ok; unsigned long cal_stop_at;
    fe_welford w_lp, w_ph;
    float sigma_lp_cal, sigma_ph_cal;
    auralis_smoother sm;
} fe_state;

static void fe_init(fe_state *s) {
    memset(s, 0, sizeof(*s));
    s->sigma_lp_cal = s->sigma_ph_cal = NAN;
    auralis_smoother_reset(&s->sm);
}

/* Start calibration over the next n_samples EDA samples (300 s x 4 Hz = 1200). It ends by itself; check
   s->calibrated afterwards (0 with calibrating == 0 means the sigmas were invalid: repeat calibration). */
static void fe_cal_begin(fe_state *s, unsigned long n_samples) {
    memset(&s->w_lp, 0, sizeof(fe_welford)); memset(&s->w_ph, 0, sizeof(fe_welford));
    s->calibrating = 1; s->calibrated = 0; s->cal_stop_at = s->n_eda + n_samples;
}

/* Ends calibration. Returns 1 if both sigmas are finite and > 0 (else the device must repeat calibration). */
static int fe_cal_end(fe_state *s) {
    s->calibrating = 0;
    float a = fe_welford_sd(&s->w_lp), b = fe_welford_sd(&s->w_ph);
    if (a > 0.0f && b > 0.0f && isfinite(a) && isfinite(b)) {
        s->sigma_lp_cal = a; s->sigma_ph_cal = b; s->calibrated = 1; return 1;
    }
    return 0;
}

static void fe_push_bvp(fe_state *s, float x) {
    s->r_bvp[s->h_bvp] = fe_bvp_step(&s->bvp_f, x);
    s->h_bvp = (s->h_bvp + 1) % FE_N_BVP;
    s->n_bvp++;
}

/* Push one EDA sample. Returns 1 and fills *w when a 30 s window ends here (every 5 s). withhold=1 forces the
   window decision to 0 (failed quality check), as the model header requires. */
static int fe_push_eda(fe_state *s, float x, int withhold, fe_window *w) {
    float lp = fe_eda_step(&s->eda_f, x);
    float ph = lp - fe_ema_step(&s->ema, lp);
    s->r_lp[s->h_eda] = lp; s->r_ph[s->h_eda] = ph; s->r_raw[s->h_eda] = x;
    s->h_eda = (s->h_eda + 1) % FE_N_EDA;
    if (s->calibrating) { fe_welford_push(&s->w_lp, lp); fe_welford_push(&s->w_ph, ph); }
    s->n_eda++;
    if (s->calibrating && s->n_eda == s->cal_stop_at) s->cal_ok = fe_cal_end(s);
    if (s->n_eda < FE_N_EDA || (s->n_eda - FE_N_EDA) % (FE_STEP_S * FE_FS_EDA) != 0) return 0;

    memset(w, 0, sizeof(*w));
    w->t_end_s = (float)s->n_eda / (float)FE_FS_EDA;
    /* BVP window in time order */
    if (s->n_bvp >= FE_N_BVP) {
        for (int i = 0; i < FE_N_BVP; i++) fe_bvp_win[i] = s->r_bvp[(s->h_bvp + i) % FE_N_BVP];
        w->hr_mean = fe_hr_window(fe_bvp_win, FE_N_BVP, &w->n_beats, &w->ppg_quality);
    } else w->hr_mean = NAN;
    /* EDA windows in time order */
    static float lpw[FE_N_EDA], phw[FE_N_EDA];
    float rmin = 1e30f, rmax = -1e30f;
    for (int i = 0; i < FE_N_EDA; i++) {
        int j = (s->h_eda + i) % FE_N_EDA;
        lpw[i] = s->r_lp[j]; phw[i] = s->r_ph[j];
        if (s->r_raw[j] < rmin) rmin = s->r_raw[j];
        if (s->r_raw[j] > rmax) rmax = s->r_raw[j];
    }
    w->eda_range = rmax - rmin;
    w->calibrated = s->calibrated;
    if (s->calibrated) {
        static unsigned short scr[FE_N_EDA / 2 + 1];
        w->eda_slope_rel = fe_slope_per_s(lpw, FE_N_EDA) / s->sigma_lp_cal;
        w->scr_count_rel = (float)fe_find_peaks(phw, FE_N_EDA, FE_PEAK_DIST_EDA, FE_SCR_REL_K * s->sigma_ph_cal, scr);
    } else {
        w->eda_slope_rel = w->scr_count_rel = NAN;          /* calibrating (stream_ref D4) */
    }
    auralis_real xin[AURALIS_N_FEATURES] = { w->hr_mean, w->eda_slope_rel, w->scr_count_rel };
    w->z = auralis_logit(xin);
    w->alert = (s->calibrated && !withhold) ? auralis_window_alert(xin) : 0;
    w->alert_smoothed = auralis_smooth_push(&s->sm, w->alert);
    return 1;
}

#endif /* AURALIS_FEATURES_H */
