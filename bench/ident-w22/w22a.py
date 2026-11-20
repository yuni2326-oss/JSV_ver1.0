"""impact_testw22(20:00–20:29, 고무줄 두 줄, 마이크 CH6): 깃 1–6 ξ0.3 품질·깃 안 반복성·깃 사이 주파수 일관성.
승인(ok)·10 N 이상 타격만. 2번 깃은 17:45 S4(고무줄 두 줄, impact_test2)와 FRF를 비교해 매달기가 같은지 본다."""
import sys, glob, os
sys.path.insert(0, sys.argv[1])
import numpy as np
from specfit import spectral_fit

OUT = sys.argv[2]
fs = 204800.0; ti = 4096
NEW = "/mnt/nas/data/GY_DATA/impact_testw22/impact_testw22"
OLD = "/mnt/nas/data/GY_DATA/impact_test2/impact_test2"
n = int(0.3 * fs); fr = np.fft.rfftfreq(n, 1 / fs); ph = np.exp(2j * np.pi * fr * 40 / fs)
wf = np.zeros(n); wf[:40 + int(0.00045 * fs)] = 1
wr = np.zeros(n); nr = int(0.25 * fs); wr[:nr] = 1; nt = int(0.01 * fs); wr[nr - nt:nr] = 0.5 * (1 + np.cos(np.pi * np.arange(nt) / nt))


def load(p):
    with np.load(p, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    for k, v in list(d.items()):
        if isinstance(v, np.ndarray) and v.ndim == 0:
            d[k] = v.item()
    return d


def spec(d):
    f = d["data"][0].astype(float) - d["data"][0, :ti - 100].astype(float).mean()
    m = d["data"][1].astype(float) - d["data"][1, :ti - 100].astype(float).mean()
    F = np.fft.rfft(f[ti - 40: ti - 40 + n] * wf) * ph; X = np.fft.rfft(m[ti - 40: ti - 40 + n] * wr) * ph
    return F, X, m


rows = []
for p in sorted(glob.glob(NEW + "/*.npz")):
    d = load(p)
    if d["quality"] != "ok" or d["peak_force_N"] < 10:
        continue
    F, X, m = spec(d)
    v = int(d["location"].split("vane")[1][0])
    rows.append(dict(v=v, hit=d["hit"], N=d["peak_force_N"], F=F, X=X, mic=m.astype(np.float32), t=d["timestamp"][11:19],
                     pk=np.abs(m[ti:ti + int(0.05 * fs)]).max(), bg=m[-int(2 * fs):].std()))
s4 = []
for p in sorted(glob.glob(OLD + "/*S4_back_vane2_xi0.3*.npz")):
    d = load(p)
    if d["quality"] == "ok" and d["peak_force_N"] >= 10 and d["hit"] != 1:
        F, X, m = spec(d); s4.append(dict(v=2, hit=d["hit"], N=d["peak_force_N"], F=F, X=X))
old = {}
for p in sorted(glob.glob(OLD + "/*.npz")):
    d = load(p)
    if d["quality"] == "ok" and d["peak_force_N"] >= 10 and d["session"] in (1, 2, 3) and d["location"] in ("back_vane2_xi0.3", "back_vane3_xi0.3"):
        F, X, m = spec(d); old.setdefault(d["location"][5:10], []).append(dict(F=F, X=X, N=d["peak_force_N"]))


def avg(rs):
    Sfx = sum(np.conj(r["F"]) * r["X"] for r in rs); Sff = sum(np.abs(r["F"]) ** 2 for r in rs); Sxx = sum(np.abs(r["X"]) ** 2 for r in rs)
    return Sfx / Sff, np.abs(Sfx) ** 2 / (Sff * Sxx)


def frac(a, b):
    return np.abs(np.vdot(a, b)) ** 2 / (np.vdot(a, a).real * np.vdot(b, b).real)


V = range(1, 7)
A = {v: avg([r for r in rows if r["v"] == v]) for v in V}
print("vane  n   time         F med [N]  mic pk/F [Pa/N]  bg 2 s rms [mPa]  coherence 1.5-3 / 3-5 / 5-7.5 / 7.5-10 kHz")
for v in V:
    R = [r for r in rows if r["v"] == v]
    C = A[v][1]
    cs = " / ".join(f"{C[(fr >= lo) & (fr < hi)].mean():.2f}" for lo, hi in ((1500, 3000), (3000, 5000), (5000, 7500), (7500, 10000)))
    print(f"  {v}  {len(R):2d}  {R[0]['t']}-{R[-1]['t']}  {np.median([r['N'] for r in R]):5.1f}      {np.median([r['pk'] / r['N'] for r in R]):.3f}          {np.median([r['bg'] for r in R]) * 1e3:6.1f}         {cs}")

# 깃 안 반복성: 타격별 H1 대 나머지 평균(leave-one-out) FRAC
print("\nwithin-vane: per-hit H1 vs leave-one-out vane mean, FRAC 1.9-7.5 kHz  (median [min]) and outliers (<0.5)")
sel = (fr >= 1900) & (fr <= 7500)
for v in V:
    R = [r for r in rows if r["v"] == v]
    fx = []
    for i, r in enumerate(R):
        H, _ = avg([q for j, q in enumerate(R) if j != i])
        fx.append(frac((r["X"] / r["F"])[sel], H[sel]))
    fx = np.array(fx)
    bad = ", ".join(f"H{R[i]['hit']:03d}({R[i]['N']:.0f} N) {fx[i]:.2f}" for i in np.where(fx < 0.5)[0])
    print(f"  vane {v}: {np.median(fx):.3f} [{fx.min():.3f}]  {bad}")

# 깃 사이 평균 H1 FRAC(서로 다른 타격점이므로 낮아도 정상) + 2번 깃 대 S4(같은 매달기면 높아야)
print("\nvane-average H1 FRAC 1.9-7.5 kHz (different excitation points; only a reference):")
M = np.array([[frac(A[a][0][sel], A[b][0][sel]) for b in V] for a in V])
for a in V:
    print("  " + " ".join(f"{M[a - 1, b - 1]:.2f}" for b in V))
H4, C4 = avg(s4); Ho2, _ = avg(old["vane2"]); Ho3, _ = avg(old["vane3"])
for lo, hi in ((1900, 7500), (1900, 3500), (3500, 5000), (5000, 7500)):
    s = (fr >= lo) & (fr <= hi)
    r_db = lambda a, b: np.corrcoef(20 * np.log10(np.abs(a[s])), 20 * np.log10(np.abs(b[s])))[0, 1]
    print(f"  {lo}-{hi} Hz: new vane2 ~ S4(17:45, rubber, n={len(s4)}) FRAC {frac(A[2][0][s], H4[s]):.3f} r_dB {r_db(A[2][0], H4):.2f} | "
          f"new vane2 ~ string S1-S3 vane2 {frac(A[2][0][s], Ho2[s]):.3f} | new vane3 ~ string vane3 {frac(A[3][0][s], Ho3[s]):.3f} | S4 ~ string vane2 {frac(H4[s], Ho2[s]):.3f}")
mag = lambda H, lo, hi: 10 * np.log10(np.mean(np.abs(H[(fr >= lo) & (fr < hi)]) ** 2))
print("  |H1| band levels [dB re 1 Pa/N] 300-1500 / 3-7 kHz: new vane2 "
      f"{mag(A[2][0], 300, 1500):.1f} / {mag(A[2][0], 3000, 7000):.1f}, S4 {mag(H4, 300, 1500):.1f} / {mag(H4, 3000, 7000):.1f}, string vane2 {mag(Ho2, 300, 1500):.1f} / {mag(Ho2, 3000, 7000):.1f}")

np.savez(os.path.join(OUT, "w22_avgs.npz"), fr=fr, **{f"v{v}_H": A[v][0] for v in V}, **{f"v{v}_C": A[v][1] for v in V}, s4_H=H4, o2_H=Ho2, o3_H=Ho3,
         **{f"v{v}_n": len([r for r in rows if r["v"] == v]) for v in V})

# 추적 모드(타격별 spectral_fit, 20 N 환산)
series = [("2.04 low comp", (2025, 2080), 2, 0, -0.0009), ("2.06 high comp", (2025, 2080), 2, 1, -0.0009), ("2.04/2.06 pair mean", (2025, 2080), 2, "mean", -0.0009),
          ("3.25 single", (3240, 3275), 1, 0, -0.0010), ("3.39 single", (3375, 3405), 1, 0, -0.0005), ("4.33 single", (4310, 4350), 1, 0, -0.0005),
          ("6.14 single", (6125, 6165), 1, 0, -0.0005), ("6.70 single", (6680, 6715), 1, 0, -0.0005), ("9.37 single", (9350, 9385), 1, 0, -0.0005)]
print("\ntracked (per-hit spectral_fit, 20 N corrected): median Hz (sd %, n accepted fits) | relative amplitude of the two comps for pairs")
TR = {}
for name, (lo, hi), k, comp, slope in series:
    line = f" {name:20s}"
    for v in V:
        vals = []; amps = []
        for r in rows:
            if r["v"] != v:
                continue
            f, z, a_, rr, e = spectral_fit(r["mic"].astype(float), fs, ti, lo, hi, k=k)
            if e > 0.25 or not np.all((f > lo) & (f < hi)) or (k == 2 and f[1] - f[0] < 6):
                continue
            val = f.mean() if comp == "mean" else f[comp]
            vals.append(val * (1 - slope * np.log2(r["N"] / 20))); amps.append(np.abs(rr))
        TR[(name, v)] = vals
        if vals:
            line += f" | v{v} {np.median(vals):8.2f} ({100 * np.std(vals, ddof=1) / np.median(vals) if len(vals) > 1 else 0:.3f}%,{len(vals):2d})"
        else:
            line += f" | v{v}    -    "
    allv = np.concatenate([TR[(name, v)] for v in V if TR[(name, v)]])
    meds = np.array([np.median(TR[(name, v)]) for v in V if len(TR[(name, v)]) >= 3])
    print(line + f" || spread of vane medians {100 * (meds.max() - meds.min()) / np.median(meds) if meds.size > 1 else 0:.3f}%")
