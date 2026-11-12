/* GENERATED from software/ml-models/wesad_edge_model/src/preprocessing.py with scipy 1.18.1
 * (same calls as src/stream_ref.design_constants). Do not edit by hand.
 * BVP: butter(3, (0.7, 3.5) Hz, band, fs 64) -> 3 sections; EDA: butter(2, 1 Hz, low, fs 4) -> 1 section.
 * Row format: b0, b1, b2, a1, a2 (a0 = 1). zi rows: steady-state state per unit input (sosfilt_zi). */
#ifndef AURALIS_DSP_CONSTS_H
#define AURALIS_DSP_CONSTS_H
#define AURALIS_BVP_NSEC 3
static const float BVP_SOS[3][5] = {
  {0.0020084438514153171f, 0.0040168877028306342f, 0.0020084438514153171f, -1.7361451801947416f, 0.7569801924705033f},
  {1.0f, 0.0f, -1.0f, -1.7050632314193555f, 0.80011036061156404f},
  {1.0f, -2.0f, 1.0f, -1.9460021738957873f, 0.95111514315558399f}
};
static const float BVP_ZI[3][2] = {
  {0.3835817012049102f, -0.28987565836805129f},
  {-0.38559014505632555f, -0.38559014505632555f},
  {-0.0f, 0.0f}
};
static const float EDA_SOS[5] = { 0.29289321881345243f, 0.58578643762690485f, 0.29289321881345243f, -1.6653345369377353e-16f, 0.17157287525380993f };
static const float EDA_ZI[2] = { 0.70710678118654757f, 0.1213203435596425f };
static const float EMA_A = 0.03076676552365587f;   /* 1 - exp(-1/(8 s * 4 Hz)) */
#endif
