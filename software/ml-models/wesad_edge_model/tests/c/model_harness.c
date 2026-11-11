/*
 * Host test harness for models/stress_lr_D_w30.h (stress detection model D).  NOT firmware.
 * Built and run by src/export_esp32.py (python -m src.export_esp32 --c-test).
 *
 *   model_harness score   IN.csv OUT.csv   rows of models/roundtrip_windows.csv -> z, p, window alert per row
 *   model_harness smooth  IN.txt OUT.txt   one 0/1 window decision per line -> 4-of-5 smoothed alert per line
 *   model_harness bench   IN.csv REPS      time REPS passes of auralis_window_alert over the rows (PC timing only)
 *
 * Compile once with AURALIS_REAL=float (as on the ESP32) and once with AURALIS_REAL=double.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "stress_lr_D_w30.h"

#define MAX_ROWS 1000

/* split one CSV line in place; empty field -> NaN */
static int split(char *line, char **f, int maxf) {
    int n = 0;
    char *p = line;
    f[n++] = p;
    while (*p && n < maxf) {
        if (*p == ',') { *p = 0; f[n++] = p + 1; }
        else if (*p == '\n' || *p == '\r') { *p = 0; }
        p++;
    }
    return n;
}

static double num(const char *s) {
    if (!s || !*s) return strtod("nan", NULL);
    return strtod(s, NULL);
}

static int read_rows(const char *path, auralis_real x[][AURALIS_N_FEATURES], char subj[][8], double *t) {
    FILE *fp = fopen(path, "r");
    char line[512], *f[16];
    int n = 0;
    if (!fp) { perror(path); exit(2); }
    if (!fgets(line, sizeof line, fp)) exit(2);
    /* header must start with subject,t_start_s,<feature order> */
    if (strncmp(line, "subject,t_start_s," AURALIS_FEATURE_ORDER, strlen("subject,t_start_s," AURALIS_FEATURE_ORDER)) != 0) {
        fprintf(stderr, "header does not match AURALIS_FEATURE_ORDER: %s", line);
        exit(3);
    }
    while (fgets(line, sizeof line, fp) && n < MAX_ROWS) {
        int k = split(line, f, 16);
        if (k < 5) continue;
        strncpy(subj[n], f[0], 7); subj[n][7] = 0;
        t[n] = num(f[1]);
        for (int i = 0; i < AURALIS_N_FEATURES; i++) x[n][i] = (auralis_real)num(f[2 + i]);
        n++;
    }
    fclose(fp);
    return n;
}

int main(int argc, char **argv) {
    static auralis_real x[MAX_ROWS][AURALIS_N_FEATURES];
    static char subj[MAX_ROWS][8];
    static double t[MAX_ROWS];
    if (argc < 4) { fprintf(stderr, "usage: model_harness score|smooth|bench IN OUT|REPS\n"); return 1; }
    if (strcmp(argv[1], "score") == 0) {
        int n = read_rows(argv[2], x, subj, t);
        FILE *out = fopen(argv[3], "w");
        fprintf(out, "subject,t_start_s,z,p,alert_logit,sizeof_real\n");
        for (int r = 0; r < n; r++) {
            auralis_real z = auralis_logit(x[r]);
            fprintf(out, "%s,%.0f,%.17g,%.17g,%d,%d\n", subj[r], t[r], (double)z, (double)auralis_score(x[r]),
                    auralis_window_alert(x[r]), (int)sizeof(auralis_real));
        }
        fclose(out);
        return 0;
    }
    if (strcmp(argv[1], "smooth") == 0) {
        FILE *in = fopen(argv[2], "r"), *out = fopen(argv[3], "w");
        auralis_smoother s;
        int a;
        auralis_smoother_reset(&s);
        while (fscanf(in, "%d", &a) == 1) fprintf(out, "%d\n", auralis_smooth_push(&s, a));
        fclose(in); fclose(out);
        return 0;
    }
    if (strcmp(argv[1], "bench") == 0) {
        int n = read_rows(argv[2], x, subj, t);
        long reps = atol(argv[3]);
        volatile int sink = 0;
        clock_t c0 = clock();
        for (long k = 0; k < reps; k++)
            for (int r = 0; r < n; r++) sink += auralis_window_alert(x[r]);
        clock_t c1 = clock();
        printf("%.6f ns_per_window_alert (PC, %ld x %d calls, sink %d)\n",
               1e9 * (double)(c1 - c0) / CLOCKS_PER_SEC / ((double)reps * n), reps, n, sink);
        return 0;
    }
    return 1;
}
