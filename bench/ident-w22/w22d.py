"""impact_testw22 타격별 극점 적합(spectral_fit)을 시간 순서로: 깃 사이 주파수 차가 깃 탓(가진점 의존)인지 시간 탓(설치 변화)인지 본다."""
import sys, glob, os, json
sys.path.insert(0, sys.argv[1])
import numpy as np
from specfit import spectral_fit

OUT = sys.argv[2]
fs = 204800.0; ti = 4096
NEW = "/mnt/nas/data/GY_DATA/impact_testw22/impact_testw22"
MODES = [("2041 h2", (2025, 2080), 2, 0), ("2062 h2", (2025, 2080), 2, 1), ("3061 h0", (3045, 3078), 1, 0), ("3489 h1", (3478, 3502), 1, 0),
         ("3631 h1", (3618, 3645), 1, 0), ("3778 h1", (3763, 3795), 1, 0), ("4274 h2", (4260, 4290), 1, 0), ("5113", (5095, 5130), 1, 0),
         ("6090", (6065, 6112), 1, 0), ("7292 h3", (7272, 7310), 1, 0)]
rows = []
for p in sorted(glob.glob(NEW + "/*.npz")):
    with np.load(p, allow_pickle=False) as z:
        if str(z["quality"]) != "ok" or float(z["peak_force_N"]) < 10:
            continue
        m = z["data"][1].astype(float); N = float(z["peak_force_N"]); v = int(str(z["location"]).split("vane")[1][0]); ts = str(z["timestamp"])[11:19]; hit = int(z["hit"])
    m = m - m[:ti - 100].mean()
    r = dict(v=v, N=N, ts=ts, hit=hit, f={}, a={}, e={})
    for name, (lo, hi), k, j in MODES:
        f, zz, amp, rr, e = spectral_fit(m, fs, ti, lo, hi, k=k)
        okfit = e < 0.3 and np.all((f > lo) & (f < hi)) and (k == 1 or f[1] - f[0] > 8)
        r["f"][name] = float(f[j]) if okfit else None; r["a"][name] = float(np.abs(rr[j]) / N) if okfit else None; r["e"][name] = float(e)
    rows.append(r)
json.dump(rows, open(os.path.join(OUT, "w22d_hits.json"), "w"))
t0 = rows[0]["ts"]
sec = lambda s: int(s[:2]) * 3600 + int(s[3:5]) * 60 + int(s[6:8])
for name, *_ in MODES:
    # 연화 기울기(깃 효과를 뺀 뒤): f = c_v · (1 + s·log2(N/20))
    D = [(r["v"], r["N"], r["f"][name], sec(r["ts"]) - sec(t0)) for r in rows if r["f"][name]]
    if len(D) < 8:
        print(name, "too few fits"); continue
    v = np.array([d[0] for d in D]); N = np.array([d[1] for d in D]); f = np.array([d[2] for d in D]); t = np.array([d[3] for d in D])
    Xm = np.column_stack([np.log2(N / 20)] + [(v == k).astype(float) for k in range(1, 7) if (v == k).sum() > 0])
    vs = [k for k in range(1, 7) if (v == k).sum() > 0]
    c, *_ = np.linalg.lstsq(Xm, np.log(f), rcond=None)
    slope = c[0]; off = {k: np.exp(c[1 + i]) for i, k in enumerate(vs)}
    f20 = f * np.exp(-slope * np.log2(N / 20))
    res = np.log(f20) - np.array([np.log(off[k]) for k in v])
    sw = res.std(ddof=len(vs) + 1) * 100
    meds = np.array([off[k] for k in vs])
    nper = {k: int((v == k).sum()) for k in vs}
    se = sw / np.sqrt(np.array([nper[k] for k in vs]))
    chi = (((meds / np.median(meds) - 1) * 100 - ((meds / np.median(meds) - 1) * 100).mean()) ** 2 / se ** 2).sum()
    print(f"{name:8s} slope {slope*100:+.3f}%/2x | within-vane sd {sw:.3f}% | vane offsets (20 N) " + " ".join(f"v{k} {off[k]:8.2f}(n{nper[k]})" for k in vs)
          + f" | spread {100*(meds.max()-meds.min())/np.median(meds):.3f}% | chi2/dof {chi/(len(vs)-1):.1f}")
