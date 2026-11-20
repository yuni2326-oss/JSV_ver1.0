"""2번 시편 10-30(15:55–16:22) ξ0.6 깃 1–6 + 1번 깃 ξ0.3 닫음 점검 품질:
쓸 타격(늦은 재타격 없음, 5 N 이상) 수, 재접촉, 힘, 응답/힘, 잡음, 코히어런스, 타격별 FRAC, 깃 사이 FRAC, 10-28 같은 자리(1번 깃 ξ0.3) 비교."""
import sys, os
import numpy as np

OUT = sys.argv[1]
Z = np.load(os.path.join(OUT, "s2", "s2b_spec.npz"), allow_pickle=False)
O = np.load(os.path.join(OUT, "s2", "s2_spec.npz"), allow_pickle=False)
fr = Z["fr"]; assert np.allclose(fr, O["fr"])
F = Z["F"].astype(complex); X = Z["X"].astype(complex); N = Z["N"]; V = Z["vane"]; XI = Z["xi"]; late = Z["late"]; kind = Z["kind"]; q = Z["q"]
band = (fr >= 1900) & (fr <= 7500)


def avg(idx, FF=F, XX=X):
    Sfx = (np.conj(FF[idx]) * XX[idx]).sum(0); Sff = (np.abs(FF[idx]) ** 2).sum(0); Sxx = (np.abs(XX[idx]) ** 2).sum(0)
    return Sfx / Sff, np.abs(Sfx) ** 2 / (Sff * Sxx)


def frac(a, b):
    return np.abs(np.vdot(a, b)) ** 2 / (np.vdot(a, a).real * np.vdot(b, b).real)


B3 = ((1900, 3000), (3000, 5000), (5000, 7500), (7500, 10000))
use = ~late & (N >= 5)
G = [(f"v{v}x0.6", (V == v) & (XI == 0.6)) for v in range(1, 7)] + [("v1x0.3", (V == 1) & (XI == 0.3))]
print("group  time             hits late | usable(>=5N) >=10N recontact | N med | pk/F med Pa/N | pre-trig mPa | bg mPa | max|mic| Pa | coherence usable 1.9-3/3-5/5-7.5/7.5-10 | late onset ms (median)")
H = {}
for g, s in G:
    u = np.where(s & use)[0]
    _, C = avg(u)
    tl = Z["tlate"][s & late]
    print(f" {g}  {Z['ts'][s][0]}-{Z['ts'][s][-1]} {s.sum():3d} {late[s].sum():3d} | {u.size:3d} {(N[u] >= 10).sum():3d} {sum('recontact' in k for k in kind[u]):3d} | {np.median(N[u]):5.1f} | "
          f"{np.median(Z['pkF'][u]):.3f} | {np.median(Z['pre'][u])*1e3:5.1f} | {np.median(Z['bg'][u])*1e3:5.1f} | {Z['sat'][s].max():6.1f} | "
          + " / ".join(f"{C[(fr >= lo) & (fr < hi)].mean():.2f}" for lo, hi in B3) + f" | {np.median(tl) if tl.size else np.nan:5.1f}")
    H[g] = avg(u)[0]
print("\nper-hit FRAC 1.9-7.5 kHz vs leave-one-out mean of the group's usable hits: median [min] (n) by class")
for g, s in G:
    u = np.where(s & use)[0]
    fx = np.array([frac((X[i] / F[i])[band], avg([j for j in u if j != i])[0][band]) for i in u])
    cls = {"ok>=10N": (q[u] == "ok") & (N[u] >= 10), "ok 5-10N": (q[u] == "ok") & (N[u] < 10), "recontact": np.array(["recontact" in k for k in kind[u]])}
    line = f"  {g}: all {np.median(fx):.2f} [{fx.min():.2f}] ({u.size})"
    for c, m in cls.items():
        if m.sum():
            line += f" | {c} {np.median(fx[m]):.2f} [{fx[m].min():.2f}] ({m.sum()})"
    bad = [f"H{Z['hit'][i]:03d}({N[i]:.0f}N,{kind[i] or q[i]}) {x:.2f}" for i, x in zip(u, fx) if x < 0.6]
    print(line + ("  low: " + ", ".join(bad) if bad else ""))
print("\ngroup-average H1 FRAC 1.9-7.5 kHz between groups:")
names = [g for g, _ in G]
print("        " + " ".join(f"{g:>6s}" for g in names))
for a in names:
    print(f"  {a} " + " ".join(f"{frac(H[a][band], H[b][band]):6.2f}" for b in names))
# 닫음 점검: 1번 깃 ξ0.3, 10-28 대 10-30
Fo = O["F"].astype(complex); Xo = O["X"].astype(complex); Vo = O["vane"]; uo = ~O["late"] & (O["N"] >= 5)
Ho1 = avg(np.where((Vo == 1) & uo)[0], Fo, Xo)[0]
print("\nclosure: vane 1 xi0.3, 10-28 17:08-11 (clean, n=%d) vs 10-30 16:21-22 (clean, n=%d)" % (((Vo == 1) & uo).sum(), ((V == 1) & (XI == 0.3) & use).sum()))
for lo, hi in B3:
    b = (fr >= lo) & (fr < hi)
    print(f"  {lo}-{hi} Hz: FRAC {frac(H['v1x0.3'][b], Ho1[b]):.3f}, level 10-30 minus 10-28 {10*np.log10(np.mean(np.abs(H['v1x0.3'][b])**2)/np.mean(np.abs(Ho1[b])**2)):+.2f} dB")
for v in range(2, 7):
    Hov = avg(np.where((Vo == v) & uo)[0], Fo, Xo)[0]
    print(f"  (10-28 v{v}x0.3 vs 10-30 v1x0.3 FRAC 1.9-7.5 kHz {frac(Hov[band], H['v1x0.3'][band]):.3f})")
print(f"\ntemperature 10-30 ai0 {Z['T0'].min():.2f}-{Z['T0'].max():.2f}, ai1 {Z['T1'].min():.2f}-{Z['T1'].max():.2f} C; 10-28 ai0 {O['T0'].min():.2f}-{O['T0'].max():.2f}, ai1 {O['T1'].min():.2f}-{O['T1'].max():.2f} C")
print("10-28 vane1 ai0/ai1 median", np.median(O['T0'][Vo == 1]), np.median(O['T1'][Vo == 1]))
np.savez(os.path.join(OUT, "s2", "s2b_avgs.npz"), fr=fr, **{f"{g}_H": H[g] for g in names})
