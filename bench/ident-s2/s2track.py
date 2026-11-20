"""새 시리즈 타격별 극점(마지막 재타격 3 ms 뒤부터 자유 감쇠) → 깃·시간에 따른 변화. 접촉이면 깃(시각)마다 크게 흔들린다."""
import sys, glob, os, json
sys.path.insert(0, sys.argv[1])
import numpy as np
from specfit import spectral_fit

SRC, OUT = sys.argv[2], sys.argv[3]
fs = 204800.0; ti = 4096
MODES = [("2132", (2110, 2165), 2, 0), ("2141", (2110, 2165), 2, 1), ("3458", (3440, 3478), 1, 0), ("3638", (3620, 3656), 1, 0), ("3803", (3780, 3825), 1, 0),
         ("4221", (4180, 4265), 1, 0), ("7453", (7420, 7490), 1, 0), ("8315", (8285, 8345), 1, 0)]
sec = lambda s: int(s[:2]) * 3600 + int(s[3:5]) * 60 + int(s[6:8])
rows = []
for p in sorted(glob.glob(os.path.join(SRC, "*.npz"))):
    with np.load(p, allow_pickle=False) as z:
        d = z["data"].astype(float); N = float(z["peak_force_N"]); v = int(str(z["location"]).split("vane")[1][0]); ts = str(z["timestamp"])[11:19]
    if N < 5:
        continue
    f = d[0] - d[0, :ti - 100].mean(); m = d[1] - d[1, :ti - 100].mean()
    seg = f[ti + int(0.003 * fs): ti + int(0.2 * fs)]
    o = np.where(seg > max(0.5, 0.03 * N))[0]
    t0 = max(0.005, (o[-1] + int(0.003 * fs)) / fs + 0.003) if o.size else 0.005
    r = dict(v=v, N=N, ts=ts, t=sec(ts), t0=t0, f={})
    for name, (lo, hi), k, j in MODES:
        fk, zz, amp, rr, e = spectral_fit(m, fs, ti, lo, hi, k=k, t0=t0)
        ok = e < 0.3 and np.all((fk > lo) & (fk < hi)) and (k == 1 or fk[1] - fk[0] > 3)
        r["f"][name] = [float(fk[j]), float(zz[j])] if ok else None
    rows.append(r)
json.dump(rows, open(os.path.join(OUT, "s2", "s2track.json"), "w"))
print("per vane: median f [Hz] (n) and zeta %, raw (no force correction); t0 median ms")
for name, *_ in MODES:
    line = f" {name}:"
    meds = []
    for v in range(1, 7):
        D = [r["f"][name] for r in rows if r["v"] == v and r["f"][name]]
        if len(D) >= 2:
            fv = np.median([d[0] for d in D]); meds.append(fv)
            line += f" v{v} {fv:8.2f}({len(D):2d},ζ{100*np.median([d[1] for d in D]):.2f})"
        else:
            line += f" v{v}     -          "
    print(line + (f" | spread {100*(max(meds)-min(meds))/np.median(meds):.2f}%" if len(meds) > 1 else ""))
print("t0 median", np.median([r["t0"] for r in rows]) * 1e3, "ms")
