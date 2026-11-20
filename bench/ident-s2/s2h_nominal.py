"""E2-I FE 기준을 표준 두께(앞판 0.40, 뒤판 0.60 mm, 나머지는 2번 시편 치수)로 바꾼 뒤 측정 극점과 다시 짝짓는다.
1) 옛 FE(1번 시편 치수)·2번 시편 치수 FE의 짝 이름을 꼭짓점 모드형 부분공간 MAC로 새 FE에 옮긴다.
2) 28번과 같은 반지름 비 ρ = c_h(ξ0.6)/c_h(ξ0.3)(뒤판 바깥면 깃 선, 측정 깃 번호 = FE −θ)로 측정 극점마다 같은 차수 FE 후보를 본다.
3) 구역 운동에너지 몫, 빠진 질량을 FE 분포대로 넣은 주파수(41.3186 g)."""
import json
import numpy as np
import pandas as pd

D0 = "/home/super/project-paper3/docs/_generated/data/paper3/"
D1 = "/tmp/claude-1000/-home-super-project/efb54ce3-bcf9-473d-9dc8-eb835c653517/scratchpad/impact1022/out/fe_s2/"
OUT = "/tmp/claude-1000/-home-super-project/efb54ce3-bcf9-473d-9dc8-eb835c653517/scratchpad/impact1022/out/s2/"
import sys
TAG = sys.argv[1] if len(sys.argv) > 1 else "nom_32x16"      # 간격 3.00 mm 기준은 nom_ch300_32x16
SUF = "" if TAG == "nom_32x16" else f"_{TAG}"
M_MEAS = 41.3186
mem = lambda s: [int(x) for x in str(s).split("|")]
z = np.load(D1 + f"shapes_{TAG}.npz", allow_pickle=False)
reg = pd.read_csv(D1 + f"modes_{TAG}.csv").sort_values("f_Hz").reset_index(drop=True)
meta = json.load(open(D1 + f"meta_{TAG}.json"))
p = meta["params_mm"]
a, b = (15.41 - p["colt"]) * 1e-3, 0.036465
V = z["vshape"].astype(float)
members = [mem(m) for m in reg["members"]]
scale = np.sqrt(meta["mass_g"] / M_MEAS)
lines = [f"FE 기준 {TAG}: 앞판 {p['tf']}, 뒤판 {p['tb']}, 깃 {p['tv']}, 간격 {p['ch']}, 테 높이 {p['colh']}, 테 벽 {p['colt']} mm, "
         f"a {a * 1e3:.2f}, b {b * 1e3:.3f} mm; 질량 {meta['mass_g']:.2f} g → 실측 {M_MEAS} g까지 FE 분포대로면 모든 모드 ×{scale:.4f} ({(scale - 1) * 100:+.2f} %)", ""]


def subspace(W, ids):
    Q, _ = np.linalg.qr(W[ids].T.astype(float))
    return Q


QN = [subspace(V, m) for m in members]


def transfer(npz, csv):
    """다른 FE(같은 격자 위상)의 무리 이름 → 새 FE 무리 (번호, MAC)."""
    zz = np.load(npz, allow_pickle=False)
    tt = pd.read_csv(csv)
    VV = zz["vshape"].astype(float)
    assert np.array_equal(zz["conn"], z["conn"])
    out = {}
    for r in tt.itertuples():
        Q0 = subspace(VV, mem(r.members))
        best = max(((np.linalg.svd(Q0.T @ QN[g], compute_uv=False) ** 2).mean(), g) for g in range(len(reg))
                   if abs(reg.f_Hz[g] / r.f_Hz - 1) < 0.25)
        out[r.label] = (best[1], best[0], r.f_Hz)
    return out


old = transfer(D0 + "rev1_n6_shapes_ff32x16h6w62c40.npz", D0 + "rev1_n6_modes_ff32x16h6w62c40.csv")
s2 = transfer(D1 + "shapes_tf400_32x16.npz", D1 + "modes_tf400_32x16.csv")

# 구역 운동에너지 몫
rho_m = 7930.0
X, conn, ck, hub = z["coors"], z["conn"], z["cell_kind"], z["cell_hub"]
g_ = np.array([-1, 1]) / np.sqrt(3)
xi_ = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)
gp = np.array([[q1, q2, q3] for q1 in g_ for q2 in g_ for q3 in g_])
Nsh = np.prod(1 + gp[:, None, :] * xi_[None, :, :], axis=2) / 8
dN = np.stack([xi_[None, :, k] * np.prod(1 + gp[:, None, [m for m in range(3) if m != k]] * xi_[None, :, [m for m in range(3) if m != k]], axis=2) / 8
               for k in range(3)], axis=2)
detJ = np.abs(np.linalg.det(np.einsum("gnk,cnj->cgkj", dN, X[conn])))
Me = rho_m * np.einsum("cg,gi,gj->cij", detJ, Nsh, Nsh)
U = V.reshape(len(V), -1, 3)
Tc = np.einsum("cij,mcik,mcjk->mc", Me, U[:, conn, :], U[:, conn, :])
REG = {"front": ck == 2, "back": (ck == 0) & ~hub, "hub": (ck == 0) & hub, "vanes": ck == 1, "collar": ck == 3}
MASS = {k: float(rho_m * detJ.sum(1)[m].sum() * 1e3) for k, m in REG.items()}
Tm = np.stack([Tc[:, m].sum(1) / Tc.sum(1) for m in REG.values()], 1)
Tg = np.array([Tm[m].mean(0) for m in members])
lines.append("구역 질량 [g]: " + ", ".join(f"{k} {v:.2f}" for k, v in MASS.items()))

# 반지름 비 점(뒤판 바깥면 z = 0, 깃 선 = 격자 loc 0·1 평균)
gi = z["grid_idx"]
per = 1 + 16
rr = np.hypot(X[:, 0], X[:, 1])
bot = np.isclose(X[:, 2], 0.0, atol=1e-9)
loc, sec = gi[:, 1] % per, gi[:, 1] // per


def point(k, xi):
    r_t = a + xi * (b - a)
    W = []
    for l in (0, 1):
        c = np.nonzero(bot & (loc == l) & (sec == k))[0]
        c = c[np.argsort(rr[c])]
        i = np.searchsorted(rr[c], r_t)
        lo_, hi_ = c[i - 1], c[i]
        t = (r_t - rr[lo_]) / (rr[hi_] - rr[lo_])
        W += [(lo_, 0.5 * (1 - t)), (hi_, 0.5 * t)]
    return W


PTS = [point(k, xi) for xi in (0.3, 0.6) for k in range(6)]
Phi = np.array([[sum(w * V[m, 3 * n + 2] for n, w in W) for W in PTS] for m in range(V.shape[0])])
k6 = np.arange(6)
coef = lambda v, h, s=1: (v * np.exp(-1j * s * h * np.radians(60.0) * k6)).mean()
rho = lambda v3, v6, h, s=1: coef(v6, h, s) / coef(v3, h, s)
dB = lambda q: 20 * np.log10(abs(q))
deg = lambda q: np.degrees(np.angle(q))
FE = []
for g, m in enumerate(members):
    h = int(reg.h[g])
    FE.append(dict(g=g, label=reg.label[g], f=float(reg.f_Hz[g]), fm=float(reg.f_Hz[g]) * scale, h=h, rho=rho(Phi[m[0], :6], Phi[m[0], 6:], h)))

# 측정 ρ(거울: 측정 깃 번호 = FE −θ → c_−h), 28번 s2b_radial과 같은 계산
J = json.load(open(OUT + "s2b_joint.json"))


def hshare(v):
    A = np.array([abs(coef(v, h)) ** 2 for h in range(6)])
    P = np.array([A[0], A[1] + A[5], A[2] + A[4], A[3]])
    return P / P.sum()


MEAS = []
for A, B in zip(J["A"], J["B"]):
    RA = np.array([complex(*x) for x in A["R"]]); RB = np.array([complex(*x) for x in B["R"]])
    D = RA[6] / RB[0]
    sn = min(abs(RA[6]) / A["Rsd"][6], abs(RB[0]) / B["Rsd"][0])
    Du = D if sn >= 5 and abs(RB[0]) > 10 ** (-0.5) * np.abs(RB[:6]).max() else 1.0
    v3, v6 = RB[:6] * Du, RA[:6]
    h = int(np.argmax(hshare(v3) + hshare(v6)))
    MEAS.append(dict(f=A["f"], h=h, rho=rho(v3, v6, h, -1)))

# 28번 짝(옛 FE 이름) → 새 FE
PAIR28 = {2133.80: "h2:m2n0", 3163.65: "h3:m3n0", 3249.89: "h0:m0n1", 3458.89: "h1:m1n1", 3638.99: "h2:m4n0", 3747.87: "h3:m3n1",
          3804.07: "h0:m6n1", 4225.49: "h1:m1n1#2", 4966.99: "h2:m4n1", 5617.22: "h0:m6n0", 5744.69: "h1:m5n1", 6438.86: "h2:m2n1",
          8321.24: "h3:m3n1#2", 7462.21: "h2:m8n2"}
lines.append("")
lines.append("옛 FE 이름 → 새 FE(MAC): " + ", ".join(f"{k}→{reg.label[old[k][0]]}({old[k][1]:.3f})" for k in sorted(set(PAIR28.values()))))
lines.append("2번 시편 치수 FE 이름 → 새 FE: " + ", ".join(f"{k}→{reg.label[v[0]]}({v[1]:.3f})" for k, v in s2.items() if v[2] < 9000))
lines.append("")
lines.append("새 FE 모드(9 kHz까지): 이름, f[Hz], 실측 질량 f, h, ρ(dB, °), 운동에너지 몫 앞판/뒤판/허브/깃/테")
for e in FE:
    if e["f"] > 9500:
        break
    T = Tg[e["g"]]
    lines.append(f"  {e['label']:11s} {e['f']:7.1f} {e['fm']:7.1f} h{e['h']} {dB(e['rho']):+6.1f} {deg(e['rho']):+6.0f} | " + " ".join(f"{x:.2f}" for x in T))
lines.append("")
lines.append("측정 극점 | h | 측정 ρ (거울) | 28번 짝 → 새 FE 이름: 실측 질량 f, 측정/FE−1, FE ρ | 같은 차수 후보(±8 %, 실측 질량 f): 측정/FE−1, ρ")
rows = []
for mm in MEAS:
    f = mm["f"]
    s = f" {f:8.2f} | h{mm['h']} | {dB(mm['rho']):+6.1f} dB {deg(mm['rho']):+5.0f}° |"
    row = dict(f=f, h=mm["h"], rho_dB=dB(mm["rho"]), rho_deg=deg(mm["rho"]))
    k28 = min(PAIR28, key=lambda k: abs(k - f))
    if abs(k28 - f) < 0.05:
        g = old[PAIR28[k28]][0]
        e = FE[g]
        s += f" {PAIR28[k28]} → {e['label']}: {e['fm']:.1f}, {100 * (f / e['fm'] - 1):+.1f} %, ρ {dB(e['rho']):+.1f} {deg(e['rho']):+.0f} |"
        row.update(pair28=PAIR28[k28], new=e["label"], f_fe=e["f"], f_fm=e["fm"], err=100 * (f / e["fm"] - 1), fe_rho_dB=dB(e["rho"]), fe_rho_deg=deg(e["rho"]))
    else:
        s += " (미정) |"
    cand = [e for e in FE if e["h"] == mm["h"] and abs(f / e["fm"] - 1) < 0.08]
    s += " ; ".join(f"{e['label']} {e['fm']:.0f} {100 * (f / e['fm'] - 1):+.1f} % ρ {dB(e['rho']):+.1f} {deg(e['rho']):+.0f}" for e in cand)
    row["cands"] = [dict(label=e["label"], f_fm=e["fm"], err=100 * (f / e["fm"] - 1), rho_dB=dB(e["rho"]), rho_deg=deg(e["rho"])) for e in cand]
    lines.append(s)
    rows.append(row)
txt = "\n".join(lines)
print(txt)
open(OUT + f"s2h_nominal{SUF}.txt", "w").write(txt + "\n")
json.dump(dict(tag=TAG, params=p, mass=meta["mass_g"], scale=scale, mass_regions=MASS,
               fe=[dict(label=e["label"], f=e["f"], fm=e["fm"], h=e["h"], rho_dB=dB(e["rho"]), rho_deg=deg(e["rho"]), T=Tg[e["g"]].tolist()) for e in FE],
               meas=rows), open(OUT + f"s2h_nominal{SUF}.json", "w"), ensure_ascii=False, indent=1)
