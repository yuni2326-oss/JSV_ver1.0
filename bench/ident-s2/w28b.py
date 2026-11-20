"""10-28 타격별 극점(spectral_fit, w22d와 같은 대역·기준) → 20 N 환산(w22e 기울기) → 무리별 중앙값과 무리 안 시간 기울기. 10-26 같은 깃 값과 비교."""
import sys, glob, os, json
sys.path.insert(0, sys.argv[1])
import numpy as np
from specfit import spectral_fit

OUT = sys.argv[2]
fs = 204800.0; ti = 4096
SRC = "/mnt/nas/data/GY_DATA/impact_testw22/impact_testw22"
MODES = [("2041 h2", (2025, 2080), 2, 0), ("2062 h2", (2025, 2080), 2, 1), ("3061 h0", (3045, 3078), 1, 0), ("3489 h1", (3478, 3502), 1, 0),
         ("3631 h1", (3618, 3645), 1, 0), ("3778 h1", (3763, 3795), 1, 0), ("3903 h3", (3890, 3918), 1, 0), ("4274 h2", (4260, 4290), 1, 0),
         ("5113", (5095, 5130), 1, 0), ("6090", (6065, 6112), 1, 0), ("7292 h3", (7272, 7310), 1, 0)]
SL = {"2041 h2": -0.126, "2062 h2": -0.059, "3061 h0": 0.089, "3489 h1": -0.053, "3631 h1": -0.129, "3778 h1": -0.217, "3903 h3": -0.1, "4274 h2": -0.080,
      "5113": -0.026, "6090": -0.117, "7292 h3": 0.022}                       # %/2x, w22e
sec = lambda s: int(s[:2]) * 3600 + int(s[3:5]) * 60 + int(s[6:8])
rows = []
for p in sorted(glob.glob(SRC + "/20261028_*.npz")):
    with np.load(p, allow_pickle=False) as z:
        d = z["data"].astype(float); N = float(z["peak_force_N"]); loc = str(z["location"]); ts = str(z["timestamp"])[11:19]; hit = int(z["hit"])
    f = d[0] - d[0, :ti - 100].mean(); m = d[1] - d[1, :ti - 100].mean()
    if N < 5 or (f[ti + int(0.003 * fs): ti + int(0.2 * fs)] > max(1.0, 0.05 * N)).any():
        continue
    r = dict(g=f"v{loc.split('vane')[1][0]}x{loc.split('xi')[1]}", N=N, ts=ts, t=sec(ts), hit=hit, f={})
    for name, (lo, hi), k, j in MODES:
        fk, zz, amp, rr, e = spectral_fit(m, fs, ti, lo, hi, k=k)
        ok = e < 0.3 and np.all((fk > lo) & (fk < hi)) and (k == 1 or fk[1] - fk[0] > 8)
        r["f"][name] = float(fk[j] * (1 - SL[name] / 100 * np.log2(N / 20))) if ok else None
    rows.append(r)
json.dump(rows, open(os.path.join(OUT, "w28b_hits.json"), "w"))
old = json.load(open(os.path.join(OUT, "w22d_hits.json")))
REF = {}
for name, *_ in MODES:
    for v in (1, 2):
        vals = [r["f"].get(name) * (1 - SL[name] / 100 * np.log2(r["N"] / 20)) for r in old if r["v"] == v and r["f"].get(name)]
        REF[(name, v)] = (np.median(vals), len(vals)) if len(vals) >= 2 else (None, len(vals))
print("20 N medians (n) [sd %] per group, time slope within group [%/10 min] if n>=6; 10-26 same vane xi0.3 for reference")
groups = ["v1x0.3", "v1x0.6", "v2x0.6"]
for name, *_ in MODES:
    line = f" {name:8s}"
    for g in groups:
        D = [(r["t"], r["f"][name]) for r in rows if r["g"] == g and r["f"][name]]
        if len(D) < 2:
            line += f" | {g} -                       "; continue
        t = np.array([d[0] for d in D], float); v = np.array([d[1] for d in D])
        md = np.median(v); res = np.abs(v / md - 1); keep = res < 3 * 1.4826 * np.median(res) + 1e-6
        sl = ""
        if keep.sum() >= 6:
            c = np.polyfit((t[keep] - t[keep].mean()) / 600, v[keep] / md - 1, 1)[0] * 100; sl = f" {c:+.3f}/10m"
        line += f" | {g} {np.median(v[keep]):8.2f}({keep.sum():2d})[{100*np.std(v[keep],ddof=1)/md:.3f}]{sl}"
    r1, n1 = REF[(name, 1)]; r2, n2 = REF[(name, 2)]
    line += f" || 10-26 v1 {r1:8.2f}({n1})" if r1 else " || 10-26 v1 -"
    line += f" v2 {r2:8.2f}({n2})" if r2 else " v2 -"
    print(line)
