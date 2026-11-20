"""h2 짝(2134/2142 Hz) 다시 매달기 비교 — 같은 방법(타격 묶음 적합, 20 N 축)으로:
전 = 10-30 1번 깃 ξ0.6(15:55–16:01, 같은 매달기; H003·H004 딴 자리, H036 잡음 제외), 16:21 ξ0.3(응답 약함, 참고),
후 = S3·S4·S5(ξ0.3). 10-28 1번 깃 ξ0.3도. 대역 2118–2160 Hz, 극점 2개(+3개 비교), 부트스트랩 100."""
import sys, os, glob, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import s2d_pool_lib as L

fs, ti = L.fs, L.ti
SL = -0.00059
extra = []
for p in sorted(glob.glob(os.path.join(L.SRC, "20261030_*_S1_back_vane1_xi0.6_*.npz"))):
    b = os.path.basename(p)
    with np.load(p, allow_pickle=False) as z:
        d = z["data"].astype(float); N = float(z["peak_force_N"]); hit = int(z["hit"]); ts = str(z["timestamp"])[11:19]
    if N < 5 or hit in (3, 4, 36):
        continue
    f = d[0] - d[0, :ti - 100].mean(); m = d[1] - d[1, :ti - 100].mean()
    seg = f[ti + int(0.003 * fs): ti + int(0.2 * fs)]
    o = np.where(seg > max(0.5, 0.03 * N))[0]
    t0 = max(0.005, (o[-1] + int(0.003 * fs)) / fs + 0.003) if o.size else 0.005
    r = dict(file=b, g="10-30 15:55 전 ξ0.6", hit=hit, N=N, ts=ts, t0=t0)
    sg = m[ti + int(t0 * fs): ti + int((t0 + 0.4) * fs)]
    L.spec[b] = np.fft.rfft(sg, L.NFFT) / fs
    extra.append(r)
rows = extra + [r for r in L.rows if r["N"] >= 5]
order = ["10-28 17:08", "10-30 15:55 전 ξ0.6", "10-30 16:21 전", "10-30 19:07 직후", "10-30 19:17 +10분", "10-30 19:28 +21분"]
rng = np.random.default_rng(5)
lo, hi = 2118, 2160
out = {}
for g in order:
    R = [r for r in rows if r["g"] == g]
    f, _ = L.stretched(R[0], lo, hi, SL)
    Ys = [L.stretched(r, lo, hi, SL)[1] for r in R]
    for k in (2, 3):
        best = None
        for p0f in ([2133.8, 2142.3], [2130.0, 2140.0]) if k == 2 else ([2125.0, 2133.8, 2142.3], [2133.8, 2142.3, 2150.0]):
            p, e = L.jfit(f, Ys, lo, hi, k, p0=np.concatenate([p0f, np.full(k, 0.0015)]))
            if best is None or e < best[1]:
                best = (p, e)
        p, e = L.jfit(f, Ys, lo, hi, k)
        if e < best[1]:
            best = (p, e)
        p, e = best
        bs = []
        for b_ in range(100):
            s = rng.choice(len(Ys), len(Ys), replace=True)
            pb, _ = L.jfit(f, [Ys[i] for i in s], lo, hi, k, p0=p)
            bs.append(pb)
        bs = np.array(bs)
        out[(g, k)] = (p, e, bs.std(0, ddof=1), len(Ys))
        print(f"{g:20s} n{len(Ys):2d} k={k}: f = {np.round(p[:k], 2)} ± {np.round(bs[:, :k].std(0, ddof=1), 2)}, "
              f"zeta = {np.round(p[k:] * 1e3, 2)}e-3 ± {np.round(bs[:, k:].std(0, ddof=1) * 1e3, 2)}, err {e:.3f}", flush=True)
json.dump({f"{g}|{k}": dict(f=v[0][:k].tolist(), z=v[0][k:].tolist(), err=v[1], sd=v[2].tolist(), n=v[3]) for (g, k), v in out.items()},
          open(L.OUT + "s2d_h2.json", "w"), ensure_ascii=False, indent=1)
