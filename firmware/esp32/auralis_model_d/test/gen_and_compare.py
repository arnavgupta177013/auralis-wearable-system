import sys, json, subprocess, numpy as np, pandas as pd
# Run from software/ml-models/wesad_edge_model after: gcc -O2 -std=c99 -o harness <this folder>/harness.c -lm
# usage: python <this folder>/gen_and_compare.py   (needs ./harness in the current folder)
REPO = "."
sys.path.insert(0, REPO)
from tests import synthetic_signals as ss
from src import stream_ref as sr
# 60 s warm-up, 300 s rest calibration, 300 s rest, 300 s "stress" (HR up, frequent SCRs, rising EDA), 200 s rest
seg = [(360,64,0),(300,66,40),(300,98,8),(200,70,40)]
bvp = np.concatenate([ss.bvp(d,h,64,seed=i) for i,(d,h,_) in enumerate(seg)])
eda_parts=[]; lvl=2.0
for i,(d,h,every) in enumerate(seg):
    x = ss.eda(d, level=lvl, scr_every_s=every, scr_amp=0.05 if every!=8 else 0.12, fs=4, seed=20+i)
    if every==8: x = x + np.linspace(0,0.4,len(x))
    eda_parts.append(x); lvl = x[-1]
eda = np.concatenate(eda_parts)
np.savetxt("bvp.txt", bvp, fmt="%.9g"); np.savetxt("eda.txt", eda, fmt="%.9g")
cal0, cal1 = 240, 240+1200
consts = sr.design_constants(); model = json.load(open(REPO + "/models/stress_lr_D_w30.json"))
import io
out=subprocess.run(["./harness","bvp.txt","eda.txt",str(cal0),str(cal1)],capture_output=True,text=True,check=True)
print(out.stderr.strip())
c=pd.read_csv(io.StringIO(out.stdout))
for name,T in [("float64",float),("float32",np.float32)]:
    p, cal, _ = sr.run_stream(bvp, eda, cal0, cal1, consts, model, T=T)
    n=min(len(p),len(c)); p=p.iloc[:n].reset_index(drop=True); cc=c.iloc[:n]
    print(f"--- C (float) vs Python StreamD {name}: {n} windows; py sigmas {cal}")
    for f in ["hr_mean","eda_slope_rel","scr_count_rel","z"]:
        a,b=p[f].to_numpy(float),cc[f].to_numpy(float)
        both=np.isnan(a)&np.isnan(b); d=np.abs(a-b); d[both]=0
        print(f"  {f:14s} max|diff| {np.nanmax(d):.3g}  NaN mismatches {int((np.isnan(a)!=np.isnan(b)).sum())}")
    print("  alert diffs", int((p.alert_z.to_numpy()!=cc.alert.to_numpy()).sum()),
          " smoothed diffs", int((p.alert_smoothed_device.to_numpy()!=cc.alert_smoothed.to_numpy()).sum()))
print(c.iloc[::12].to_string(index=False))
