#include <stdio.h>
#include <stdlib.h>
#include "../auralis_features.h"
static fe_state S;
int main(int argc, char **argv) {
    if (argc < 5) return 1;
    FILE *fb = fopen(argv[1], "r"), *fe = fopen(argv[2], "r");
    unsigned long cal0 = strtoul(argv[3], 0, 10), cal1 = strtoul(argv[4], 0, 10);
    fe_init(&S);
    double b, e; fe_window w; unsigned long je = 0;
    printf("t_end_s,hr_mean,eda_slope_rel,scr_count_rel,z,alert,alert_smoothed\n");
    while (fscanf(fe, "%lf", &e) == 1) {
        for (int k = 0; k < 16; k++) { if (fscanf(fb, "%lf", &b) != 1) break; fe_push_bvp(&S, (float)b); }
        if (je == cal0) fe_cal_begin(&S, cal1 - cal0);
        if (fe_push_eda(&S, (float)e, 0, &w))
            printf("%.2f,%.9g,%.9g,%.9g,%.9g,%d,%d\n", w.t_end_s, w.hr_mean, w.eda_slope_rel, w.scr_count_rel, w.z, w.alert, w.alert_smoothed);
        je++;
    }
    fprintf(stderr, "sigma_lp_cal %.9g sigma_ph_cal %.9g\n", S.sigma_lp_cal, S.sigma_ph_cal);
    return 0;
}
