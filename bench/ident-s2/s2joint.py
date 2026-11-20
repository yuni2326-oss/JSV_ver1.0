"""새 시리즈(impact_testw2s2) 깃 6곳 공동 적합: 깨끗한 타격(재타격 없음, 5 N 이상) 깃 평균 H1, 공통 극점·깃별 잔차 → MPC, h0–h3 에너지, θ0, 부트스트랩."""
import sys, os, json
sys.path.insert(0, sys.argv[1])
import numpy as np
from w22joint import vane_avg, joint_fit, pattern

OUT = sys.argv[2]
Z = np.load(os.path.join(OUT, "s2", "s2_spec.npz"), allow_pickle=False)
fr = Z["fr"]; use = ~Z["late"] & (Z["N"] >= 5)
F = Z["F"][use].astype(complex); X = Z["X"][use].astype(complex); V = Z["vane"][use]; N = Z["N"][use]
slope = float(os.environ.get("SLOPE", "-0.001")); B = int(os.environ.get("NBOOT", "200"))
H = vane_avg(F, X, V, N, slope, fr); Hs = [H[v] for v in range(1, 7)]
FITS = [((2105, 2170), 2), ((3140, 3185), 1), ((3200, 3300), 1), ((3440, 3480), 1), ((3570, 3610), 1), ((3620, 3660), 1), ((3730, 3765), 1), ((3780, 3830), 1),
        ((4150, 4300), 1), ((4930, 5010), 1), ((5580, 5620), 1), ((5690, 5790), 1), ((6380, 6580), 2), ((6740, 6840), 1), ((7060, 7100), 1), ((7400, 7510), 1),
        ((7540, 7580), 1), ((8270, 8360), 1), ((8790, 8830), 1), ((9930, 9990), 1), ((10040, 10130), 1)]
rng = np.random.default_rng(1)
idx = {v: np.where(V == v)[0] for v in range(1, 7)}
boots = []
for b in range(B):
    Hb = []
    for v in range(1, 7):
        s = rng.choice(idx[v], idx[v].size, replace=True)
        Hb.append((np.conj(F[s]) * X[s]).sum(0) / (np.abs(F[s]) ** 2).sum(0))
    boots.append(Hb)
print(f"clean hits per vane: {[int(idx[v].size) for v in range(1, 7)]}; slope {slope}; bootstrap {B}")
print("pole f [Hz] (zeta %) rel dB | MPC [5%] | h0 h1 h2 h3 | best h [boot share] | theta0 (90% half-width) | pattern a (vanes 1-6)")
res = []
for (lo, hi), k in FITS:
    fk, zk, R, err = joint_fit(fr, Hs, lo, hi, k)
    sel = (fr >= lo) & (fr <= hi); f = fr[sel]; w = 2j * np.pi * f; fc = 0.5 * (lo + hi)
    sk = -zk * 2 * np.pi * fk + 1j * 2 * np.pi * fk * np.sqrt(1 - zk ** 2)
    Bm = np.column_stack([1 / (w - s) for s in sk] + [np.ones_like(w), f - fc])
    print(f"== {lo}-{hi} k={k} err {err:.3f}")
    for j in range(k):
        mpc, a, P, th0, alpha = pattern(R[:, j]); hb = int(np.argmax(P))
        bh, bth, bm = [], [], []
        for Hb in boots:
            Rb = np.array([np.linalg.lstsq(Bm, Y[sel], rcond=None)[0][j] for Y in Hb])
            m_, a_, P_, t_, _ = pattern(Rb); bh.append(int(np.argmax(P_))); bm.append(m_)
            if hb in (1, 2):
                bth.append((t_[hb] - th0[hb] + 90 / hb) % (180 / hb) - 90 / hb)
        share = np.mean(np.array(bh) == hb); hw = np.percentile(np.abs(bth), 90) if bth else np.nan
        rel = 20 * np.log10(np.abs(R[:, j]).max() / np.abs(R).max())
        ths = f"{th0[hb]:6.1f} (±{hw:4.1f})" if hb in (1, 2) else "   -          "
        print(f"  {fk[j]:8.2f} ({zk[j]*100:.2f}) {rel:6.1f} | {mpc:.2f} [{np.percentile(bm,5):.2f}] | {P[0]:.2f} {P[1]:.2f} {P[2]:.2f} {P[3]:.2f} | h{hb} [{share:.2f}] | {ths} | "
              + " ".join(f"{x/np.abs(a).max():+5.2f}" for x in a))
        res.append(dict(lo=lo, hi=hi, k=k, f=float(fk[j]), z=float(zk[j]), rel=float(rel), mpc=float(mpc), P=[float(x) for x in P], h=hb, share=float(share),
                        th0=float(th0[hb]) if hb in (1, 2) else None, hw=float(hw) if bth else None, a=[float(x) for x in a / np.abs(a).max()],
                        absr=[float(x) for x in np.abs(R[:, j])], err=float(err)))
json.dump(res, open(os.path.join(OUT, "s2", f"s2joint_slope{slope}.json"), "w"), indent=1)
