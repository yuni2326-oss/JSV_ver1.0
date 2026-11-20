"""10-28 16:34 1번 깃 ξ0.3 닫음 점검(impact_testw22 S2, 14타): 10-26 20:00 같은 깃(S1)과 같은 설치인지(FRF), 극점이 시리즈 첫값으로 돌아왔는지.
타격별 극점은 w22d와 같은 spectral_fit·같은 승인 기준, 20 N 환산 기울기는 w22e처럼 깃(·날짜) 효과를 둔 공통 기울기."""
import sys, glob, os, json
sys.path.insert(0, sys.argv[1])
import numpy as np
from specfit import spectral_fit

OUT = sys.argv[2]
fs = 204800.0; ti = 4096
SRC = "/mnt/nas/data/GY_DATA/impact_testw22/impact_testw22"
n = int(0.3 * fs); fr = np.fft.rfftfreq(n, 1 / fs); ph = np.exp(2j * np.pi * fr * 40 / fs)
wf = np.zeros(n); wf[:40 + int(0.00045 * fs)] = 1
wr = np.zeros(n); nr = int(0.25 * fs); wr[:nr] = 1; nt = int(0.01 * fs); wr[nr - nt:nr] = 0.5 * (1 + np.cos(np.pi * np.arange(nt) / nt))
MODES = [("2041 h2", (2025, 2080), 2, 0), ("2062 h2", (2025, 2080), 2, 1), ("3061 h0", (3045, 3078), 1, 0), ("3489 h1", (3478, 3502), 1, 0),
         ("3631 h1", (3618, 3645), 1, 0), ("3778 h1", (3763, 3795), 1, 0), ("4274 h2", (4260, 4290), 1, 0), ("5113", (5095, 5130), 1, 0),
         ("6090", (6065, 6112), 1, 0), ("7292 h3", (7272, 7310), 1, 0)]


def load(p):
    with np.load(p, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    for k, v in list(d.items()):
        if isinstance(v, np.ndarray) and v.ndim == 0:
            d[k] = v.item()
    return d


def fits(m, N, t0=0.005):
    r = dict(f={}, a={}, e={})
    for name, (lo, hi), k, j in MODES:
        f, zz, amp, rr, e = spectral_fit(m, fs, ti, lo, hi, k=k, t0=t0)
        okfit = e < 0.3 and np.all((f > lo) & (f < hi)) and (k == 1 or f[1] - f[0] > 8)
        r["f"][name] = float(f[j]) if okfit else None; r["a"][name] = float(np.abs(rr[j]) / N) if okfit else None; r["e"][name] = float(e)
    return r


# ---------- A. 새 타격 품질
new = []
for p in sorted(glob.glob(SRC + "/20261028_*xi0.3*.npz")):
    d = load(p)
    assert d["channel_names"][1] == "CH6_mic" and d["physical_channels"][1] == "PXI1Slot5/ai2", d["channel_names"]
    fo = d["data"][0].astype(float); fo -= fo[:ti - 100].mean()
    m = d["data"][1].astype(float); m -= m[:ti - 100].mean()
    # 늦은 재타격: 해머의 1.4 ms 혹을 지나 3–200 ms 안에서 max(1 N, 5 % 피크)를 넘는 첫 힘 표본
    seg = fo[ti + int(0.003 * fs): ti + int(0.2 * fs)]
    late = np.where(seg > max(1.0, 0.05 * d["peak_force_N"]))[0]
    t_late = (late[0] + int(0.003 * fs)) / fs * 1e3 if late.size else None
    F = np.fft.rfft(fo[ti - 40: ti - 40 + n] * wf) * ph; X = np.fft.rfft(m[ti - 40: ti - 40 + n] * wr) * ph
    new.append(dict(hit=d["hit"], N=d["peak_force_N"], q=d["quality"], kind=d["double_kind"], sp=d["second_peak_N"], spms=d["second_peak_ms"], t_late=t_late,
                    late_N=float(seg.max()), pk=np.abs(m[ti:ti + int(0.05 * fs)]).max(), pre=m[:ti - 100].std(), bg=m[-int(2 * fs):].std(),
                    T=d["temperature_C"], ts=d["timestamp"][11:19], F=F, X=X, m=m))
print("A. new hits 2026-10-28 (vane 1 xi0.3, S2):  hit time N quality kind | 2nd peak N @ms | late force >1 N @ms (max 1-200 ms) | mic pk/F | pre-trig rms mPa | T ai0/ai1")
for r in new:
    tl = f"{r['t_late']:6.2f}" if r["t_late"] is not None else "   -  "
    print(f"  H{r['hit']:03d} {r['ts']} {r['N']:5.1f} {r['q']:10s} {r['kind'] or '-':15s} | {r['sp']:4.2f} @{r['spms']:6.2f} | {tl} ({r['late_N']:4.2f} N) | {r['pk']/r['N']:.3f} | {r['pre']*1e3:5.1f} | {r['T'][0]:.2f}/{r['T'][1]:.2f}")

# ---------- B. FRF: 같은 설치인지
A = np.load(os.path.join(OUT, "w22_avgs.npz"), allow_pickle=False)
assert np.allclose(A["fr"], fr)
acc = [r for r in new if r["q"] == "ok" and r["N"] >= 10]


def avg(rs):
    Sfx = sum(np.conj(r["F"]) * r["X"] for r in rs); Sff = sum(np.abs(r["F"]) ** 2 for r in rs); Sxx = sum(np.abs(r["X"]) ** 2 for r in rs)
    return Sfx / Sff, np.abs(Sfx) ** 2 / (Sff * Sxx)


def frac(a, b):
    return np.abs(np.vdot(a, b)) ** 2 / (np.vdot(a, a).real * np.vdot(b, b).real)


Hn, Cn = avg(acc)
print(f"\nB. accepted >=10 N: n={len(acc)} (hits {[r['hit'] for r in acc]}), N median {np.median([r['N'] for r in acc]):.1f}, mic pk/F median {np.median([r['pk']/r['N'] for r in acc]):.3f} Pa/N, "
      f"bg 2 s rms median {np.median([r['bg'] for r in acc])*1e3:.1f} mPa")
print("   coherence 1.5-3 / 3-5 / 5-7.5 / 7.5-10 kHz: " + " / ".join(f"{Cn[(fr >= lo) & (fr < hi)].mean():.2f}" for lo, hi in ((1500, 3000), (3000, 5000), (5000, 7500), (7500, 10000))))
for lo, hi in ((1900, 7500), (1900, 3500), (3500, 5000), (5000, 7500)):
    s = (fr >= lo) & (fr <= hi)
    print(f"   FRAC {lo}-{hi} Hz new v1 vs 10-26 vanes: " + " ".join(f"v{v} {frac(Hn[s], A[f'v{v}_H'][s]):.2f}" for v in range(1, 7)))
s = (fr >= 1900) & (fr <= 7500)
print("   per-hit new H1 vs 10-26 vane-1 mean, FRAC 1.9-7.5 kHz: " + " ".join(f"H{r['hit']:03d} {frac((r['X']/r['F'])[s], A['v1_H'][s]):.2f}" for r in acc))
mag = lambda H, lo, hi: 10 * np.log10(np.mean(np.abs(H[(fr >= lo) & (fr < hi)]) ** 2))
print("   |H1| band levels dB re 1 Pa/N (300-1500 / 1.9-3 / 3-5 / 5-7.5 kHz): new " + " / ".join(f"{mag(Hn, lo, hi):.1f}" for lo, hi in ((300, 1500), (1900, 3000), (3000, 5000), (5000, 7500)))
      + " | 10-26 v1 " + " / ".join(f"{mag(A['v1_H'], lo, hi):.1f}" for lo, hi in ((300, 1500), (1900, 3000), (3000, 5000), (5000, 7500))))

# ---------- C. 극점: 10-26 깃별(w22d) + 10-28 1번 깃
old = json.load(open(os.path.join(OUT, "w22d_hits.json")))
P = json.load(open(os.path.join(OUT, "w22c_slope0.0.json")))
FJ = {"2041 h2": 2041.03, "2062 h2": 2062.35, "3061 h0": 3061.55, "3489 h1": 3489.27, "3631 h1": 3630.59, "3778 h1": 3777.99,
      "4274 h2": 4273.76, "5113": 5113.02, "6090": 6089.85, "7292 h3": 7291.57}
for r in new:
    late_ok = r["t_late"] is None
    t0 = 0.005 if late_ok else max(0.005, r["t_late"] / 1e3 + 0.003)
    r.update(fits(r["m"], r["N"], t0)); r["t0"] = t0
grp_new = [dict(g="28v1", v=1, N=r["N"], ts=r["ts"], hit=r["hit"], f=r["f"], a=r["a"], q=r["q"], kind=r["kind"], t0=r["t0"]) for r in new]
grp_old = [dict(g=f"26v{r['v']}", v=r["v"], N=r["N"], ts=r["ts"], hit=r["hit"], f=r["f"], a=r["a"], q="ok", kind="", t0=0.005) for r in old]
json.dump(dict(new=[{k: v for k, v in r.items()} for r in grp_new]), open(os.path.join(OUT, "w22f_hits.json"), "w"))


def table(use_new, label):
    print(f"\nC. {label}: 20 N medians (n) with slope shared over groups (w22e rule: node vanes < -15 dB and vane 2 for the 2 kHz pair excluded)")
    print("   mode      slope%/2x | 10-26 v1     | 10-28 v1     | diff % (se)       | 10-26 v6   diff new-v6 % | within-group sd %")
    out = {}
    for name, fj in FJ.items():
        p = min(P, key=lambda q: abs(q["f"] - fj))
        rel = 20 * np.log10(np.array(p["absr"]) / max(p["absr"]))
        good = [v for v in range(1, 7) if rel[v - 1] >= -15 and not (name.startswith("20") and v == 2)]
        D = [(r["g"], r["N"], r["f"][name]) for r in grp_old + [q for q in grp_new if use_new(q)] if r["f"][name] and r["v"] in good]
        if not any(d[0] == "28v1" for d in D):
            print(f"   {name:9s} (vane 1 node or no fit)"); continue
        g = np.array([d[0] for d in D]); N = np.array([d[1] for d in D]); f = np.array([d[2] for d in D])
        keep = np.ones(len(f), bool)
        for it in range(3):
            gs = sorted(set(g[keep]))
            Xm = np.column_stack([np.log2(N[keep] / 20)] + [(g[keep] == k).astype(float) for k in gs])
            c, *_ = np.linalg.lstsq(Xm, np.log(f[keep]), rcond=None)
            off = dict(zip(gs, c[1:]))
            r_ = np.log(f) - c[0] * np.log2(N / 20) - np.array([off.get(k, np.nan) for k in g])
            mad = 1.4826 * np.nanmedian(np.abs(r_[keep]))
            keep = np.abs(r_) < 3 * mad
        slope = c[0]; f20 = f * np.exp(-slope * np.log2(N / 20))
        meds = {k: np.median(f20[keep & (g == k)]) for k in sorted(set(g[keep])) if (keep & (g == k)).sum() >= 2}
        sd = np.std(np.concatenate([(f20[keep & (g == k)] / meds[k] - 1) for k in meds]), ddof=len(meds)) * 100
        nn = {k: int((keep & (g == k)).sum()) for k in meds}
        o1, n1 = meds.get("26v1"), meds.get("28v1")
        if o1 is None or n1 is None:
            print(f"   {name:9s} too few"); continue
        d = 100 * (n1 / o1 - 1); se = sd * np.sqrt(1 / nn["26v1"] + 1 / nn["28v1"])
        o6 = meds.get("26v6")
        s6 = f"{o6:8.2f}  {100*(n1/o6-1):+.3f}" if o6 else "    -            "
        print(f"   {name:9s} {slope*100:+.3f}   | {o1:8.2f}({nn['26v1']:2d}) | {n1:8.2f}({nn['28v1']:2d}) | {d:+.3f} ({se:.3f}) | {s6}  | {sd:.3f}")
        out[name] = dict(slope=slope, o1=o1, n1=n1, d=d, se=se, o6=o6, sd=sd, f20_new=[float(x) for x in f20[(g == "28v1")]], keep_new=[bool(x) for x in keep[g == "28v1"]])
    return out


R1 = table(lambda q: q["q"] == "ok" and q["N"] >= 10, "primary = accepted, >=10 N (same rule as 10-26)")
R2 = table(lambda q: q["N"] >= 10, "sensitivity = all >=10 N incl. double hits (window starts 3 ms after a late re-hit)")
R3 = table(lambda q: q["N"] >= 5, "sensitivity = all >=5 N")

# h2 짝 두 성분의 크기 비(같은 깃·같은 마이크면 짝 방향 θ0가 그대로인지)
for lab, rs in (("10-26 v1", [r for r in grp_old if r["v"] == 1]), ("10-28 v1 accepted>=10N", [r for r in grp_new if r["q"] == "ok" and r["N"] >= 10]),
                ("10-28 v1 all>=10N", [r for r in grp_new if r["N"] >= 10])):
    rat = [20 * np.log10(r["a"]["2062 h2"] / r["a"]["2041 h2"]) for r in rs if r["a"]["2041 h2"] and r["a"]["2062 h2"]]
    a1 = [r["a"]["2041 h2"] for r in rs if r["a"]["2041 h2"]]; a2 = [r["a"]["2062 h2"] for r in rs if r["a"]["2062 h2"]]
    print(f"\nD. {lab}: |r|/N 2041 {np.median(a1):.4f}, 2062 {np.median(a2):.4f}; ratio 2062/2041 median {np.median(rat):+.1f} dB (IQR {np.percentile(rat,25):+.1f}..{np.percentile(rat,75):+.1f}, n {len(rat)})")
json.dump(dict(primary=R1, all10=R2, all5=R3), open(os.path.join(OUT, "w22f_tables.json"), "w"), indent=1, default=float)
np.savez(os.path.join(OUT, "w22f_avg.npz"), fr=fr, Hn=Hn, Cn=Cn)
