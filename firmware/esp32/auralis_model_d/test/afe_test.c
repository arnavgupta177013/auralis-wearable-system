/* Host test of analog_front_end.h:  gcc -O2 -std=c99 -o afe_test afe_test.c -lm && ./afe_test */
#include <stdio.h>
#include <math.h>
#include "../analog_front_end.h"

static int fails = 0;
static void check(const char *what, double got, double want, double tol) {
    int ok = fabs(got - want) <= tol;
    printf("%-44s got %12.6f want %12.6f  %s\n", what, got, want, ok ? "ok" : "FAIL");
    if (!ok) fails++;
}

int main(void) {
    const float vs = 3.30f, rs = 330000.0f;
    /* EDA: skin of 1 MOhm, 100 kOhm, 10 MOhm -> 1, 10, 0.1 uS */
    float rskin[] = {1e6f, 1e5f, 1e7f}; double gwant[] = {1.0, 10.0, 0.1};
    for (int i = 0; i < 3; i++) {
        float v0 = vs * rskin[i] / (rs + rskin[i]);
        char name[64]; snprintf(name, sizeof name, "EDA %.0f ohm -> uS", rskin[i]);
        check(name, afe_conductance_uS(v0, vs, rs), gwant[i], gwant[i] * 1e-4);
    }
    /* ratiometric: same skin, supply 3.20 V instead of 3.30 V -> same conductance */
    float v0 = 3.20f * 1e6f / (rs + 1e6f);
    check("EDA 1 Mohm at 3.20 V supply -> uS", afe_conductance_uS(v0, 3.20f, rs), 1.0, 1e-4);
    check("EDA open circuit (v0 == vs) -> NAN", isnan(afe_conductance_uS(vs, vs, rs)), 1, 0);
    check("EDA shorted (v0 == 0) -> NAN", isnan(afe_conductance_uS(0.0f, vs, rs)), 1, 0);

    /* NTC 10k B3950: 25 C -> 10 kOhm; 37 C -> 10k * exp(B*(1/310.15 - 1/298.15)) */
    float r37 = 10000.0f * expf(3950.0f * (1.0f / 310.15f - 1.0f / 298.15f));
    float temps[] = {25.0f, 37.0f}; float rntc[] = {10000.0f, r37};
    for (int i = 0; i < 2; i++) {
        float v1 = vs * rntc[i] / (10000.0f + rntc[i]);
        float r = afe_ntc_ohm(v1, vs, 10000.0f);
        char name[64]; snprintf(name, sizeof name, "NTC at %.0f C -> C", temps[i]);
        check(name, afe_ntc_celsius(r, 10000.0f, 3950.0f, 25.0f), temps[i], 0.01);
    }
    /* config words: AIN0 0xC383, AIN1 0xD383, AIN3 0xF383 */
    check("ADS config AIN0", afe_ads_config(0), 0xC383, 0);
    check("ADS config AIN1", afe_ads_config(1), 0xD383, 0);
    check("ADS config AIN3", afe_ads_config(3), 0xF383, 0);
    printf("%s (%d failures)\n", fails ? "FAILED" : "ALL PASSED", fails);
    return fails != 0;
}
