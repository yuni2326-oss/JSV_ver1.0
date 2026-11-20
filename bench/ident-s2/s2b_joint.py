"""2번 시편 공동 적합 두 개(같은 대역·같은 연화 보정):
 A = 10-30 일곱 무리(ξ0.6 깃 1–6 + 1번 깃 ξ0.3 닫음 점검), B = 10-28 ξ0.3 깃 1–6.
깨끗한 타격 = 늦은 재타격 없음, 5 N 이상, 타격별 FRAC ≥ 0.6(1.9–7.5 kHz, 무리 평균 대비).
극점마다 복소 잔차와 부트스트랩 표준편차를 저장한다(→ s2b_shape.py)."""
import sys, os, json
sys.path.insert(0, sys.argv[1])
import numpy as np
from w22joint import joint_fit, pattern

OUT = sys.argv[2]
SLOPE = float(os.environ.get("SLOPE", "-0.0005")); NB = int(os.environ.get("NBOOT", "200"))
Zn = np.load(os.path.join(OUT, "s2", "s2b_spec.npz"), allow_pickle=False)
Zo = np.load(os.path.join(OUT, "s2", "s2_spec.npz"), allow_pickle=False)
fr = Zn["fr"]; assert np.allclose(fr, Zo["fr"])
band = (fr >= 1900) & (fr <= 7500)
FITS = [((2105, 2170), 2), ((3140, 3190), 1), ((3200, 3300), 1), ((3440, 3485), 1), ((3570, 3615), 1), ((3620, 3665), 1), ((3730, 3770), 1),
        ((3780, 3835), 1), ((4150, 4300), 1), ((4930, 5020), 1), ((5580, 5630), 1), ((5690, 5800), 1), ((6380, 6590), 2), ((6740, 6850), 1),
        ((7060, 7115), 1), ((7400, 7520), 1), ((7540, 7595), 1), ((8270, 8370), 1), ((8790, 8850), 1), ((9880, 9995), 1), ((10040, 10140), 1)]


def frac(a, b):
    return np.abs(np.vdot(a, b)) ** 2 / (np.vdot(a, a).real * np.vdot(b, b).real)


def stretch(Fi, Xi, Ni):
    """타격 하나의 교차·자기 스펙트럼을 20 N 기준 주파수 축으로(f_N = f_20 · sc)."""
    sc = 1 + SLOPE * np.log2(Ni / 20.0); fi = fr * sc
    c = np.conj(Fi) * Xi
    return np.interp(fi, fr, c.real) + 1j * np.interp(fi, fr, c.imag), np.interp(fi, fr, np.abs(Fi) ** 2)


def groups(Z, defs):
    F = Z["F"].astype(complex); X = Z["X"].astype(complex); N = Z["N"]
    use0 = ~Z["late"] & (N >= 5)
    out = []
    for name, msk in defs:
        u = np.where(msk & use0)[0]
        Hm = (np.conj(F[u]) * X[u]).sum(0) / (np.abs(F[u]) ** 2).sum(0)
        fx = np.array([frac((X[i] / F[i])[band], ((np.conj(F[u[u != i]]) * X[u[u != i]]).sum(0) / (np.abs(F[u[u != i]]) ** 2).sum(0))[band]) for i in u])
        drop = u[fx < 0.6]; u = u[fx >= 0.6]
        S = [stretch(F[i], X[i], N[i]) for i in u]
        out.append(dict(name=name, idx=u, drop=[int(Z["hit"][i]) for i in drop], Sfx=np.array([s[0] for s in S]), Sff=np.array([s[1] for s in S])))
    return out


Vn, XIn = Zn["vane"], Zn["xi"]
GA = groups(Zn, [(f"v{v}x0.6", (Vn == v) & (XIn == 0.6)) for v in range(1, 7)] + [("v1x0.3", (Vn == 1) & (XIn == 0.3))])
GB = groups(Zo, [(f"v{v}x0.3", Zo["vane"] == v) for v in range(1, 7)])
for G, lab in ((GA, "A 10-30"), (GB, "B 10-28")):
    print(f"{lab}: clean hits " + ", ".join(f"{g['name']} {len(g['idx'])}" + (f" (dropped H{g['drop']})" if g['drop'] else "") for g in G))
rng = np.random.default_rng(3)


def run(G):
    Hs = [g["Sfx"].sum(0) / g["Sff"].sum(0) for g in G]
    boots = []
    for b in range(NB):
        Hb = []
        for g in G:
            s = rng.choice(len(g["idx"]), len(g["idx"]), replace=True)
            Hb.append(g["Sfx"][s].sum(0) / g["Sff"][s].sum(0))
        boots.append(Hb)
    res = []
    for (lo, hi), k in FITS:
        fk, zk, R, err = joint_fit(fr, Hs, lo, hi, k)
        sel = (fr >= lo) & (fr <= hi); f = fr[sel]; w = 2j * np.pi * f; fc = 0.5 * (lo + hi)
        sk = -zk * 2 * np.pi * fk + 1j * 2 * np.pi * fk * np.sqrt(1 - zk ** 2)
        Bm = np.column_stack([1 / (w - s) for s in sk] + [np.ones_like(w), f - fc])
        Rb = np.array([[np.linalg.lstsq(Bm, Y[sel], rcond=None)[0][:k] for Y in Hb] for Hb in boots])     # (NB, ngroup, k)
        for j in range(k):
            res.append(dict(lo=lo, hi=hi, k=k, j=j, f=float(fk[j]), z=float(zk[j]), err=float(err),
                            R=[[float(x.real), float(x.imag)] for x in R[:, j]],
                            Rsd=[float(x) for x in np.sqrt(np.var(Rb[:, :, j].real, 0) + np.var(Rb[:, :, j].imag, 0))],
                            Rb=[[[float(x.real), float(x.imag)] for x in row] for row in Rb[:40, :, j]]))
    return res


A = run(GA); B = run(GB)
json.dump(dict(slope=SLOPE, A_groups=[g["name"] for g in GA], B_groups=[g["name"] for g in GB], A=A, B=B),
          open(os.path.join(OUT, "s2", "s2b_joint.json"), "w"))
dB = lambda z: 20 * np.log10(abs(z)); deg = lambda z: np.degrees(np.angle(z))
print(f"\nslope {SLOPE}; poles A (10-30) vs B (10-28) [Hz] (zeta %); xi0.6 six-vane pattern (A): MPC, h0-h3, theta0;"
      " closure v1x0.3 A/B: dB, deg, SNR(|R|/sd) A,B; radial v1 xi0.6/xi0.3 (A): dB, deg")
for a, b in zip(A, B):
    RA = np.array([complex(*x) for x in a["R"]]); RB = np.array([complex(*x) for x in b["R"]])
    mpc, pa, P, th0, _ = pattern(RA[:6]); hb = int(np.argmax(P))
    mA = np.abs(RA).max()
    snA = abs(RA[6]) / a["Rsd"][6]; snB = abs(RB[0]) / b["Rsd"][0]
    ths = f"{th0[hb]:6.1f}" if hb in (1, 2) else "   -  "
    print(f" {a['f']:8.2f} ({a['z']*100:.2f}) | {b['f']:8.2f} ({b['z']*100:.2f}) {100*(a['f']/b['f']-1):+.3f}% err {a['err']:.2f}/{b['err']:.2f} | "
          f"MPC {mpc:.2f} {P[0]:.2f} {P[1]:.2f} {P[2]:.2f} {P[3]:.2f} h{hb} {ths} | closure {dB(RA[6]/RB[0]):+5.1f} dB {deg(RA[6]/RB[0]):+5.0f} SNR {snA:4.1f},{snB:4.1f}"
          f" | radial {dB(RA[0]/RA[6]):+5.1f} dB {deg(RA[0]/RA[6]):+5.0f} (v1x0.6 {dB(RA[0]/mA):+5.1f}, v1x0.3 {dB(RA[6]/mA):+5.1f} dB)")
