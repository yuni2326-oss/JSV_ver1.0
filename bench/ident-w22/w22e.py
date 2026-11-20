"""깃 사이 고유주파수 표: 타격별 극점(w22d) → 깃 효과를 둔 연화 기울기(이상치 제외) → 20 N 환산 깃별 중앙값.
마디(공동 적합 잔차가 최대 대비 −15 dB 아래)인 깃은 뺀다."""
import sys, os, json
import numpy as np

OUT = sys.argv[1]
hits = json.load(open(os.path.join(OUT, "w22d_hits.json")))
P = json.load(open(os.path.join(OUT, "w22c_slope0.0.json")))
MODES = {"2041 h2": 2041.03, "2062 h2": 2062.35, "3061 h0": 3061.55, "3489 h1": 3489.27, "3631 h1": 3630.59, "3778 h1": 3777.99,
         "4274 h2": 4273.76, "5113": 5113.02, "6090": 6089.85, "7292 h3": 7291.57}
print("mode       slope[%/2x](n)  | 20 N vane medians (n) ; '-' = node (< -15 dB) or <2 fits | spread over vanes | within-vane sd")
for name, fj in MODES.items():
    p = min(P, key=lambda q: abs(q["f"] - fj))
    rel = 20 * np.log10(np.array(p["absr"]) / max(p["absr"]))
    good = [v for v in range(1, 7) if rel[v - 1] >= -15 and not (name.startswith("20") and v == 2)]
    D = [(r["v"], r["N"], r["f"][name]) for r in hits if r["f"][name] and r["v"] in good]
    v = np.array([d[0] for d in D]); N = np.array([d[1] for d in D]); f = np.array([d[2] for d in D])
    keep = np.ones(len(f), bool)
    for it in range(3):
        vs = sorted(set(v[keep]))
        Xm = np.column_stack([np.log2(N[keep] / 20)] + [(v[keep] == k).astype(float) for k in vs])
        c, *_ = np.linalg.lstsq(Xm, np.log(f[keep]), rcond=None)
        off = dict(zip(vs, c[1:]))
        r_ = np.log(f) - c[0] * np.log2(N / 20) - np.array([off.get(k, np.nan) for k in v])
        mad = 1.4826 * np.nanmedian(np.abs(r_[keep]))
        keep = np.abs(r_) < 3 * mad
    slope = c[0]
    f20 = f * np.exp(-slope * np.log2(N / 20))
    meds = {k: np.median(f20[keep & (v == k)]) for k in sorted(set(v[keep])) if (keep & (v == k)).sum() >= 2}
    sd = np.std(np.concatenate([(f20[keep & (v == k)] / meds[k] - 1) for k in meds]), ddof=len(meds)) * 100
    line = f"{name:9s} {slope*100:+.3f} ({keep.sum()}/{len(f)}) | "
    for k in range(1, 7):
        line += (f"v{k} {meds[k]:8.2f}({(keep & (v == k)).sum():2d}) " if k in meds else f"v{k}    -        ")
    m = np.array(list(meds.values()))
    line += f"| {100*(m.max()-m.min())/np.median(m):.3f}% | {sd:.3f}%"
    print(line)
