"""impact_testw22 확정 대역 공동 적합 + 타격 재표본(부트스트랩)으로 원주 차수·방향·MPC의 흔들림, 깃별 단독 적합 주파수."""
import sys, os, json
sys.path.insert(0, sys.argv[1])
import numpy as np
from w22joint import vane_avg, joint_fit, pattern, TH

OUT = sys.argv[2]
z = np.load(os.path.join(OUT, "w22_spec.npz"), allow_pickle=False)
fr = z["fr"]; F = z["F"].astype(complex); X = z["X"].astype(complex); V = z["vane"]; N = z["N"]
slope = float(os.environ.get("SLOPE", "0")); B = int(os.environ.get("NBOOT", "200"))
H = vane_avg(F, X, V, N, slope, fr); Hs = [H[v] for v in range(1, 7)]
FITS = [((2020, 2090), 2), ((3040, 3080), 1), ((3140, 3180), 1), ((3225, 3295), 4), ((3375, 3415), 2), ((3475, 3510), 1), ((3615, 3650), 1),
        ((3760, 3800), 1), ((3885, 3925), 1), ((4255, 4295), 1), ((4300, 4365), 2), ((5090, 5135), 1), ((5230, 5270), 1), ((6055, 6115), 1),
        ((6120, 6170), 1), ((6420, 6460), 2), ((6675, 6730), 1), ((6830, 6880), 2), ((7075, 7125), 1), ((7265, 7315), 1), ((7805, 7855), 2),
        ((9340, 9385), 1), ((9530, 9590), 1), ((9845, 9910), 1), ((10190, 10260), 2)]
rng = np.random.default_rng(1)
idx = {v: np.where(V == v)[0] for v in range(1, 7)}
# 부트스트랩 평균 H1(대역 전체) 미리 만들기
boots = []
for b in range(B):
    Hb = []
    for v in range(1, 7):
        s = rng.choice(idx[v], idx[v].size, replace=True)
        Hb.append((np.conj(F[s]) * X[s]).sum(0) / (np.abs(F[s]) ** 2).sum(0))
    boots.append(Hb)
res = []
print(f"slope {slope}, bootstrap {B}")
print("pole f [Hz] (zeta %) rel dB | MPC | h0 h1 h2 h3 | best h [boot share] | theta0 h1/h2 [deg] (boot 90% half-width) | pattern a")
for (lo, hi), k in FITS:
    fk, zk, R, err = joint_fit(fr, Hs, lo, hi, k)
    sel = (fr >= lo) & (fr <= hi); f = fr[sel]; w = 2j * np.pi * f; fc = 0.5 * (lo + hi)
    sk = -zk * 2 * np.pi * fk + 1j * 2 * np.pi * fk * np.sqrt(1 - zk ** 2)
    Bm = np.column_stack([1 / (w - s) for s in sk] + [np.ones_like(w), f - fc])
    print(f"== {lo}-{hi} k={k} err {err:.3f}")
    for j in range(k):
        mpc, a, P, th0, alpha = pattern(R[:, j])
        hb = int(np.argmax(P))
        bh, bth, bm = [], [], []
        for Hb in boots:
            Rb = np.array([np.linalg.lstsq(Bm, Y[sel], rcond=None)[0][j] for Y in Hb])
            m_, a_, P_, t_, _ = pattern(Rb)
            bh.append(int(np.argmax(P_))); bm.append(m_)
            if hb in (1, 2):
                d = (t_[hb] - th0[hb] + 90 / hb) % (180 / hb) - 90 / hb      # 부호가 뒤집힌 패턴은 같은 방향(θ0는 180°/h 주기)
                bth.append(d)
        share = np.mean(np.array(bh) == hb)
        hw = np.percentile(np.abs(bth), 90) if bth else np.nan
        rel = 20 * np.log10(np.abs(R[:, j]).max() / np.abs(R).max())
        thstr = f"{th0[hb]:6.1f} (±{hw:4.1f})" if hb in (1, 2) else "   -         "
        print(f"  {fk[j]:8.2f} ({zk[j]*100:.2f}) {rel:6.1f} | {mpc:.2f} [{np.percentile(bm,5):.2f}] | {P[0]:.2f} {P[1]:.2f} {P[2]:.2f} {P[3]:.2f} | h{hb} [{share:.2f}] | {thstr} | "
              + " ".join(f"{x/np.abs(a).max():+5.2f}" for x in a))
        res.append(dict(lo=lo, hi=hi, k=k, f=float(fk[j]), z=float(zk[j]), rel=float(rel), mpc=float(mpc), P=[float(x) for x in P], h=hb, share=float(share),
                        th0=float(th0[hb]) if hb in (1, 2) else None, hw=float(hw) if bth else None, a=[float(x) for x in a / np.abs(a).max()],
                        absr=[float(x) for x in np.abs(R[:, j])], err=float(err)))
json.dump(res, open(os.path.join(OUT, f"w22c_slope{slope}.json"), "w"), indent=1)

# 깃별 단독 적합(k=1 대역 + 2 kHz 짝): 그 깃에서 극점 크기가 최대 대비 −15 dB 이상일 때만
print("\nper-vane fits on the vane-average H1 (only where that vane's |r| >= -15 dB of the max):")
for (lo, hi), k in FITS:
    if k > 2 or ((lo, hi) != (2020, 2090) and k != 1):
        continue
    fk, zk, R, err = joint_fit(fr, Hs, lo, hi, k)
    line = f"  {lo}-{hi} joint " + "/".join(f"{x:.2f}" for x in fk) + " |"
    allf = [[] for _ in range(k)]
    for v in range(1, 7):
        fv, zv, Rv, ev = joint_fit(fr, [Hs[v - 1]], lo, hi, k)
        ok = [20 * np.log10(np.abs(R[v - 1, j]) / np.abs(R[:, j]).max()) >= -15 for j in range(k)]
        line += f" v{v} " + "/".join((f"{fv[j]:.2f}" if ok[j] else "  -   ") for j in range(k))
        for j in range(k):
            if ok[j]:
                allf[j].append(fv[j])
    line += " || spread " + "/".join(f"{100*(max(x)-min(x))/np.median(x):.3f}%(n{len(x)})" if len(x) > 1 else "-" for x in allf)
    print(line)
