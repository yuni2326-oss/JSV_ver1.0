"""2번 시편 10-30 타격별 극점(s2track과 같은 대역·기준, 마지막 재타격 3 ms 뒤부터 자유 감쇠) → 20 N 환산(s2fig2 기울기) →
무리별 중앙값·무리 안 시간 기울기, 10-30 시리즈 순서 경향, 닫음 점검(1번 깃 ξ0.3 10-28 대 10-30), 오늘 자료로 다시 잡은 연화 기울기."""
import sys, glob, os, json
sys.path.insert(0, sys.argv[1])
import numpy as np
from specfit import spectral_fit

SRC, OUT = sys.argv[2], sys.argv[3]
fs = 204800.0; ti = 4096
MODES = [("2132", (2110, 2165), 2, 0), ("2141", (2110, 2165), 2, 1), ("3458", (3440, 3478), 1, 0), ("3638", (3620, 3656), 1, 0), ("3803", (3780, 3825), 1, 0),
         ("4221", (4180, 4265), 1, 0), ("7453", (7420, 7490), 1, 0), ("8315", (8285, 8345), 1, 0)]
SL = {"2132": -0.00059, "2141": -0.00059, "3458": -0.00032, "3638": -0.00051, "3803": -0.00036, "4221": -0.00050, "7453": -0.00071, "8315": -0.00046}
sec = lambda s: int(s[:2]) * 3600 + int(s[3:5]) * 60 + int(s[6:8])
JS = os.path.join(OUT, "s2", "s2b_track.json")
if not os.path.exists(JS):
    rows = []
    for p in sorted(glob.glob(os.path.join(SRC, "20261030_*.npz"))):
        with np.load(p, allow_pickle=False) as z:
            d = z["data"].astype(float); N = float(z["peak_force_N"]); loc = str(z["location"]); ts = str(z["timestamp"])[11:19]; hit = int(z["hit"])
            T = z["temperature_C"].astype(float)
        if N < 5:
            continue
        f = d[0] - d[0, :ti - 100].mean(); m = d[1] - d[1, :ti - 100].mean()
        seg = f[ti + int(0.003 * fs): ti + int(0.2 * fs)]
        o = np.where(seg > max(0.5, 0.03 * N))[0]
        t0 = max(0.005, (o[-1] + int(0.003 * fs)) / fs + 0.003) if o.size else 0.005
        r = dict(v=int(loc.split("vane")[1][0]), xi=float(loc.split("xi")[1]), hit=hit, N=N, ts=ts, t=sec(ts), t0=t0, T0=float(T[0]), T1=float(T[1]), f={})
        for name, (lo, hi), k, j in MODES:
            fk, zz, amp, rr, e = spectral_fit(m, fs, ti, lo, hi, k=k, t0=t0)
            ok = e < 0.3 and np.all((fk > lo) & (fk < hi)) and (k == 1 or fk[1] - fk[0] > 3)
            r["f"][name] = [float(fk[j]), float(zz[j])] if ok else None
        rows.append(r)
    json.dump(rows, open(JS, "w"))
rows = json.load(open(JS))
old = json.load(open(os.path.join(OUT, "s2", "s2track.json")))
for r in old:
    r["xi"] = 0.3
c20 = lambda r, name: r["f"][name][0] * (1 - SL[name] * np.log2(r["N"] / 20)) if r["f"].get(name) else None


def stat(D):
    t = np.array([d[0] for d in D], float); v = np.array([d[1] for d in D])
    md = np.median(v); res = np.abs(v / md - 1); keep = res < 3 * 1.4826 * np.median(res) + 1e-6
    sl = np.polyfit((t[keep] - t[keep].mean()) / 600, v[keep] / md - 1, 1)[0] * 100 if keep.sum() >= 6 else np.nan
    return np.median(v[keep]), keep.sum(), 100 * np.std(v[keep], ddof=1) / md if keep.sum() > 2 else np.nan, sl


groups = [(f"v{v}x0.6", v, 0.6) for v in range(1, 7)] + [("v1x0.3", 1, 0.3)]
print("20 N medians [Hz] (n) per group 10-30; last column = 10-28 same vane xi0.3; within-group slope %/10 min where n>=6")
summ = {}
for name, *_ in MODES:
    print(f" {name}:")
    line = "   "
    for g, v, x in groups:
        D = [(r["t"], c20(r, name)) for r in rows if r["v"] == v and r["xi"] == x and r["f"].get(name)]
        if len(D) >= 2:
            md, n, sd, sl = stat(D); summ[(name, g)] = md
            line += f" {g} {md:8.2f}({n:2d},{sd:.3f}%{'' if np.isnan(sl) else f',{sl:+.3f}/10m'})"
        else:
            line += f" {g}    -    "
    print(line)
    line = "    10-28 xi0.3:"
    for v in range(1, 7):
        D = [(r["t"], c20(r, name)) for r in old if r["v"] == v and r["f"].get(name)]
        if len(D) >= 2:
            md, n, sd, sl = stat(D); summ[(name, f"o{v}")] = md
            line += f" v{v} {md:8.2f}({n:2d})"
    print(line)
print("\nclosure vane 1 xi0.3 (20 N): 10-30 minus 10-28 [%]; 10-30 xi0.6 series v6 minus v1 [%]; 10-28 xi0.3 series v6 minus v1 [%]; 10-30 v1x0.3 minus v1x0.6 [%]")
for name, *_ in MODES:
    g = lambda k: summ.get((name, k))
    pct = lambda a, b: f"{100*(a/b-1):+.3f}" if a and b else "   -  "
    print(f"  {name}: closure {pct(g('v1x0.3'), g('o1'))} | 10-30 v6-v1 {pct(g('v6x0.6'), g('v1x0.6'))} | 10-28 v6-v1 {pct(g('o6'), g('o1'))} | 10-30 v1x0.3-v1x0.6 {pct(g('v1x0.3'), g('v1x0.6'))}"
          f" | 10-30 v1x0.6 vs 10-28 v1x0.3 {pct(g('v1x0.6'), g('o1'))}")
# 오늘 자료로 연화 기울기: 무리별 평균 제거 후 log2(N/20)에 회귀
print("\nsoftening re-estimated on 10-30 (group-demeaned, raw f): %/2x [n]; 10-28 value")
for name, *_ in MODES:
    xs, ys = [], []
    for g, v, x in groups:
        D = [(np.log2(r["N"] / 20), r["f"][name][0]) for r in rows if r["v"] == v and r["xi"] == x and r["f"].get(name)]
        if len(D) >= 4:
            a = np.array(D); md = np.median(a[:, 1]); ok = np.abs(a[:, 1] / md - 1) < 0.003
            xs += list(a[ok, 0] - a[ok, 0].mean()); ys += list(a[ok, 1] / md - 1 - (a[ok, 1] / md - 1).mean())
    if len(xs) > 8:
        b = np.polyfit(xs, ys, 1)[0] * 100
        print(f"  {name}: {b:+.3f} [{len(xs)}]  (10-28 {SL[name]*100:+.3f})")
