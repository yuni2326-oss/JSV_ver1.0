"""10-28 ξ0.6(1·2번 깃)와 ξ0.3(1번 깃) 점검:
(1) 5–10 N 타격을 써도 되는지(코히어런스·타격별 FRAC), (2) 오늘 세 무리 공동 적합(공통 극점, 20 N 연화 보정) 대 10-26 깃 6곳 공동 적합:
극점 주파수 차, 깃 비 v2/v1(ξ0.6 오늘 대 ξ0.3 10-26), 1번 깃 반지름 비 ξ0.6/ξ0.3(오늘), 1번 깃 ξ0.3 날 사이 잔차 비."""
import sys, os
sys.path.insert(0, sys.argv[1])
import numpy as np
from w22joint import joint_fit, vane_avg

OUT = sys.argv[2]
SLOPE = float(os.environ.get("SLOPE", "-0.001"))
T = np.load(os.path.join(OUT, "w28_spec.npz"), allow_pickle=False)
O = np.load(os.path.join(OUT, "w22_spec.npz"), allow_pickle=False)
fr = T["fr"]; assert np.allclose(fr, O["fr"])
F = T["F"].astype(complex); X = T["X"].astype(complex); N = T["N"]; late = T["late"]; vane = T["vane"]; xi = T["xi"]


def gavg(idx, slope=SLOPE):
    Sfx = 0; Sff = 0
    for i in idx:
        sc = 1 + slope * np.log2(N[i] / 20.0); fi = fr * sc
        c = np.conj(F[i]) * X[i]
        Sfx = Sfx + np.interp(fi, fr, c.real) + 1j * np.interp(fi, fr, c.imag); Sff = Sff + np.interp(fi, fr, np.abs(F[i]) ** 2)
    return Sfx / Sff


def coh(idx):
    Sfx = (np.conj(F[idx]) * X[idx]).sum(0); return np.abs(Sfx) ** 2 / ((np.abs(F[idx]) ** 2).sum(0) * (np.abs(X[idx]) ** 2).sum(0))


def frac(a, b):
    return np.abs(np.vdot(a, b)) ** 2 / (np.vdot(a, a).real * np.vdot(b, b).real)


band = (fr >= 1900) & (fr <= 7500)
G = {"v1x0.3": (vane == 1) & (xi == 0.3), "v1x0.6": (vane == 1) & (xi == 0.6), "v2x0.6": (vane == 2) & (xi == 0.6)}
print("(1) usable = no late re-hit. A = >=10 N, B = 5-10 N. coherence 1.9-3 / 3-5 / 5-7.5 kHz; per-hit FRAC 1.9-7.5 kHz vs mean of the other usable hits (median [min])")
for g, msk in G.items():
    A = np.where(msk & ~late & (N >= 10))[0]; B = np.where(msk & ~late & (N >= 5) & (N < 10))[0]; AB = np.concatenate([A, B])
    line = f"  {g}: A {A.size}, B {B.size}"
    for lab, idx in (("A", A), ("B", B), ("A+B", AB)):
        if idx.size >= 2:
            C = coh(idx)
            line += f" | {lab} coh " + "/".join(f"{C[(fr >= lo) & (fr < hi)].mean():.2f}" for lo, hi in ((1900, 3000), (3000, 5000), (5000, 7500)))
    print(line)
    if AB.size >= 3:
        fx = np.array([frac((X[i] / F[i])[band], gavg([j for j in AB if j != i], 0)[band]) for i in AB])
        print(f"      per-hit FRAC A: {np.median(fx[:A.size]) if A.size else np.nan:.2f} [{fx[:A.size].min() if A.size else np.nan:.2f}]  B: {np.median(fx[A.size:]) if B.size else np.nan:.2f} [{fx[A.size:].min() if B.size else np.nan:.2f}]"
              f"  (B hits: " + " ".join(f"{N[i]:.1f}N:{x:.2f}" for i, x in zip(B, fx[A.size:])) + ")")

# (2) 공동 적합
use = ~late & (N >= 5)
Ht = [gavg(np.where(G[g] & use)[0]) for g in G]
Ho = vane_avg(O["F"].astype(complex), O["X"].astype(complex), O["vane"], O["N"], SLOPE, fr); Hos = [Ho[v] for v in range(1, 7)]
FITS = [((2020, 2090), 2), ((3040, 3080), 1), ((3245, 3280), 1), ((3475, 3510), 1), ((3615, 3650), 1), ((3760, 3800), 1), ((3885, 3925), 1),
        ((4255, 4295), 1), ((5090, 5135), 1), ((6055, 6115), 1), ((6830, 6880), 2), ((7265, 7315), 1), ((7805, 7855), 1)]
ID = {2041: "h2", 2062: "h2", 3061: "h0", 3260: "h1", 3489: "h1", 3630: "h1", 3778: "h1", 3902: "h3", 4273: "h2", 5113: "h2?", 6089: "h2", 6854: "h3", 6866: "h3", 7291: "h3", 7823: "h3"}
dB = lambda z: 20 * np.log10(abs(z)); deg = lambda z: np.degrees(np.angle(z))
print(f"\n(2) joint fits (slope {SLOPE}): today groups v1x0.3 / v1x0.6 / v2x0.6 (>=5 N, no late) vs 10-26 six vanes xi0.3")
print("   pole 10-26 -> 10-28 [Hz] (diff %) | |r| rel. to strongest group/vane, dB: today v1x.3 v1x.6 v2x.6 ; 10-26 v1 v2"
      " | v2/v1 at xi0.6 today | v2/v1 at xi0.3 10-26 | v1 xi0.6/xi0.3 today | v1 xi0.3 today/10-26")
rows = []
for (lo, hi), k in FITS:
    ft, zt, Rt, et = joint_fit(fr, Ht, lo, hi, k)
    fo, zo, Ro, eo = joint_fit(fr, Hos, lo, hi, k)
    for j in range(k):
        jo = int(np.argmin(np.abs(fo - ft[j])))
        r1a, r1b, r2b = Rt[0, j], Rt[1, j], Rt[2, j]; o1, o2 = Ro[0, jo], Ro[1, jo]
        mt = np.abs(Rt[:, j]).max(); mo = np.abs(Ro[:, jo]).max()
        tag = ID.get(int(fo[jo]), "")
        print(f"   {fo[jo]:8.2f} -> {ft[j]:8.2f} ({100*(ft[j]/fo[jo]-1):+.3f}) {tag:3s} err {et:.2f}/{eo:.2f} | {dB(r1a/mt):6.1f} {dB(r1b/mt):6.1f} {dB(r2b/mt):6.1f} ; {dB(o1/mo):6.1f} {dB(o2/mo):6.1f}"
              f" | {dB(r2b/r1b):+6.1f} dB {deg(r2b/r1b):+5.0f} | {dB(o2/o1):+6.1f} dB {deg(o2/o1):+5.0f} | {dB(r1b/r1a):+6.1f} dB {deg(r1b/r1a):+5.0f} | {dB(r1a/o1):+6.1f} dB {deg(r1a/o1):+5.0f}")
        rows.append(dict(f_old=float(fo[jo]), f_new=float(ft[j]), z_new=float(zt[j]), tag=tag, Rt=[[float(x.real), float(x.imag)] for x in Rt[:, j]],
                         Ro=[[float(x.real), float(x.imag)] for x in Ro[:, jo]]))
import json
json.dump(rows, open(os.path.join(OUT, f"w28a_slope{SLOPE}.json"), "w"), indent=1)
