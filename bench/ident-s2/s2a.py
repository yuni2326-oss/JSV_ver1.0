"""impact_testw2s2(10-28 17:08–17:41, 마이크를 뒤판 가까이 세움) 깃 1–6 ξ0.3 품질:
쓸 타격(늦은 재타격 없음, 5 N 이상) 수, 마이크 크기·잡음, 코히어런스, 타격별 FRAC(힘·접촉 종류별), 깃 사이 FRAC, 10-26 시리즈와 비교."""
import sys, os
import numpy as np

OUT = sys.argv[1]
Z = np.load(os.path.join(OUT, "s2", "s2_spec.npz"), allow_pickle=False)
O = np.load(os.path.join(OUT, "w22_spec.npz"), allow_pickle=False)
fr = Z["fr"]; F = Z["F"].astype(complex); X = Z["X"].astype(complex); N = Z["N"]; V = Z["vane"]; late = Z["late"]; kind = Z["kind"]; q = Z["q"]
band = (fr >= 1900) & (fr <= 7500)


def avg(idx, FF=F, XX=X):
    Sfx = (np.conj(FF[idx]) * XX[idx]).sum(0); Sff = (np.abs(FF[idx]) ** 2).sum(0); Sxx = (np.abs(XX[idx]) ** 2).sum(0)
    return Sfx / Sff, np.abs(Sfx) ** 2 / (Sff * Sxx)


def frac(a, b):
    return np.abs(np.vdot(a, b)) ** 2 / (np.vdot(a, a).real * np.vdot(b, b).real)


B3 = ((1900, 3000), (3000, 5000), (5000, 7500), (7500, 10000))
use = ~late & (N >= 5)
print("vane time        hits late | usable(>=5N) >=10N recontact | N med | pk/F med Pa/N | pre-trig rms mPa | bg 2 s rms mPa | max|mic| Pa | coherence usable 1.9-3/3-5/5-7.5/7.5-10")
for v in range(1, 7):
    s = V == v; u = np.where(s & use)[0]
    _, C = avg(u)
    print(f"  {v}  {Z['ts'][s][0]}-{Z['ts'][s][-1]} {s.sum():3d} {late[s].sum():3d} | {u.size:3d} {(N[u] >= 10).sum():3d} {sum('recontact' in k for k in kind[u]):3d} | {np.median(N[u]):5.1f} | "
          f"{np.median(Z['pkF'][u]):.3f} | {np.median(Z['pre'][u])*1e3:5.1f} | {np.median(Z['bg'][u])*1e3:5.1f} | {Z['sat'][s].max():6.1f} | "
          + " / ".join(f"{C[(fr >= lo) & (fr < hi)].mean():.2f}" for lo, hi in B3))
print("\nper-hit FRAC 1.9-7.5 kHz vs leave-one-out mean of the vane's usable hits: median [min] (n) by class")
H = {}
for v in range(1, 7):
    u = np.where((V == v) & use)[0]
    fx = np.array([frac((X[i] / F[i])[band], avg([j for j in u if j != i])[0][band]) for i in u])
    cls = {"ok>=10N": (q[u] == "ok") & (N[u] >= 10), "ok 5-10N": (q[u] == "ok") & (N[u] < 10), "recontact": np.array(["recontact" in k for k in kind[u]])}
    line = f"  vane {v}: all {np.median(fx):.2f} [{fx.min():.2f}] ({u.size})"
    for c, m in cls.items():
        if m.sum():
            line += f" | {c} {np.median(fx[m]):.2f} [{fx[m].min():.2f}] ({m.sum()})"
    bad = [f"H{Z['hit'][i]:03d}({N[i]:.0f}N,{kind[i] or q[i]}) {x:.2f}" for i, x in zip(u, fx) if x < 0.5]
    print(line + ("  low: " + ", ".join(bad) if bad else ""))
    H[v] = avg(u)[0]
print("\nvane-average H1 FRAC 1.9-7.5 kHz between vanes (new series):")
for a in range(1, 7):
    print("  " + " ".join(f"{frac(H[a][band], H[b][band]):.2f}" for b in range(1, 7)))
# 10-26 시리즈(마이크 멀리)와 같은 깃 비교
Fo = O["F"].astype(complex); Xo = O["X"].astype(complex); Vo = O["vane"]
print("\nsame vane, 10-26 series (far mic) vs new (near mic): FRAC 1.9-3 / 3-5 / 5-7.5 kHz; |H1| level difference new-old dB in 1.9-3 / 3-5 / 5-7.5 kHz")
for v in range(1, 7):
    Ho = avg(np.where(Vo == v)[0], Fo, Xo)[0]
    fx = [frac(H[v][(fr >= lo) & (fr < hi)], Ho[(fr >= lo) & (fr < hi)]) for lo, hi in B3[:3]]
    lv = [10 * np.log10(np.mean(np.abs(H[v][(fr >= lo) & (fr < hi)]) ** 2) / np.mean(np.abs(Ho[(fr >= lo) & (fr < hi)]) ** 2)) for lo, hi in B3[:3]]
    print(f"  vane {v}: FRAC " + " / ".join(f"{x:.2f}" for x in fx) + " | dB " + " / ".join(f"{x:+.1f}" for x in lv))
print(f"\ntemperature ai0 {Z['T0'].min():.2f}-{Z['T0'].max():.2f}, ai1 {Z['T1'].min():.2f}-{Z['T1'].max():.2f} C")
np.savez(os.path.join(OUT, "s2", "s2_avgs.npz"), fr=fr, **{f"v{v}_H": H[v] for v in range(1, 7)})
