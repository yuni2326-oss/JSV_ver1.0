"""빠진 질량이 '숨을' 수 있는 최대량 — 선형계획.
가정: FE 강성 오차는 주파수를 내리기만 한다(연속 접합 이상화 = FE가 더 강함). 그러면 각 모드에서
질량 몫 −½ Σ ε_b T_jb 가 측정 차 err_j 보다 더 내려갈 수 없다(err_j ≤ 질량 몫 ≤ 0).
구역 × 반경 띠(b)별 밀도 증가 ε_b ≥ 0, 목적 = Σ ε_b m_b 최대.
3163.65/3249.89(h3:m3n0·h0:m0n1 섞임)는 쌍 평균 하나로, 3639.0(h2:m4n0 섞임)은 뺀다.
"""
import json
import numpy as np
import pandas as pd
from scipy.optimize import linprog

D = "/home/super/project-paper3/docs/_generated/data/paper3/"
KEY = "ff32x16h6w62c40"
OUT = "/tmp/claude-1000/-home-super-project/efb54ce3-bcf9-473d-9dc8-eb835c653517/scratchpad/impact1022/out/s2/"
z = np.load(D + f"rev1_n6_shapes_{KEY}.npz", allow_pickle=False)
X, conn, ck, vsh = z["coors"], z["conn"], z["cell_kind"], z["vshape"].astype(float)
meta = json.load(open(D + f"rev1_n6_modes_{KEY}.json"))
tab = pd.read_csv(D + f"rev1_n6_modes_{KEY}.csv")
rho, a = 7930.0, meta["a"]

g = np.array([-1, 1]) / np.sqrt(3)
xi = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
               [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)
gp = np.array([[p, q, r] for p in g for q in g for r in g])
N = np.prod(1 + gp[:, None, :] * xi[None, :, :], axis=2) / 8
dN = np.stack([xi[None, :, k] * np.prod(1 + gp[:, None, [m for m in range(3) if m != k]]
                                         * xi[None, :, [m for m in range(3) if m != k]], axis=2) / 8
               for k in range(3)], axis=2)
Xe = X[conn]
detJ = np.abs(np.linalg.det(np.einsum("gnk,cnj->cgkj", dN, Xe)))
vol = detJ.sum(1)
Me = rho * np.einsum("cg,gi,gj->cij", detJ, N, N)
U = vsh.reshape(len(vsh), -1, 3)
Tc = np.einsum("cij,mcik,mcjk->mc", Me, U[:, conn, :], U[:, conn, :])
Tc = Tc / Tc.sum(1, keepdims=True)
r_c = np.hypot(Xe[:, :, 0].mean(1), Xe[:, :, 1].mean(1)) * 1e3

kind = np.where(ck == 0, np.where(r_c < a * 1e3, "hub", "back"),
                np.where(ck == 1, "vanes", np.where(ck == 2, "front", "collar")))
edges = [6.3, 10.0, 15.0, 20.0, 25.0, 30.0, 33.5, 36.6]
bands = []
for k in ["hub", "back", "front", "vanes", "collar"]:
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (kind == k) & (r_c >= lo) & (r_c < hi)
        if m.any():
            bands.append((f"{k} r{lo:g}–{hi:g}", m, float(rho * vol[m].sum() * 1e3)))

def obs_T(label):
    mem = [int(s) for s in str(tab[tab.label == label].iloc[0]["members"]).split("|")]
    return np.array([Tc[mem][:, b[1]].sum(1).mean() for b in bands])

meas = [("h2:m2n0", 2138.04), (("h3:m3n0", "h0:m0n1"), (3163.65 + 3249.89) / 2), ("h1:m1n1", 3458.89),
        ("h3:m3n1", 3747.87), ("h0:m6n1", 3804.07), ("h1:m1n1#2", 4225.49), ("h2:m4n1", 4966.99),
        ("h0:m6n0", 5617.22), ("h1:m5n1", 5744.69), ("h2:m2n1", 6438.86), ("h3:m3n1#2", 8321.24)]
fe = tab.set_index("label")["f_Hz"]
A, err, names = [], [], []
for lab, fm_ in meas:
    if isinstance(lab, tuple):
        T = np.mean([obs_T(l) for l in lab], axis=0); f0 = np.mean([fe[l] for l in lab]); nm = "m3n0+m0n1 평균"
    else:
        T = obs_T(lab); f0 = fe[lab]; nm = lab
    A.append(T); err.append(fm_ / f0 - 1); names.append(nm)
A, err = np.array(A), np.array(err)
mb = np.array([b[2] for b in bands])
# 질량 몫(1차) = −½ ε·T ≥ err  ⇔  ½ T·ε ≤ −err
res = linprog(-mb, A_ub=0.5 * A, b_ub=-err, bounds=[(0, None)] * len(bands), method="highs")
eps = res.x
lines = [f"띠 {len(bands)}개, 모드 {len(err)}개(쌍 평균 1). 측정 차 [%]: " + ", ".join(f"{n} {e * 100:+.1f}" for n, e in zip(names, err)),
         f"최대 추가 질량(가정: FE 강성 오차는 주파수를 내리기만) = {-res.fun:.2f} g  (실측 차 2번 {41.3186 - meta['mass_model_g']:.2f} g, 1번 {40.2059 - meta['mass_model_g']:.2f} g)",
         "  배치: " + ", ".join(f"{bands[i][0]} {eps[i] * mb[i]:.2f} g(+{eps[i] * 100:.0f} %)" for i in np.argsort(-eps * mb) if eps[i] * mb[i] > 0.005),
         "  묶는 모드(질량 몫 = 측정 차): " + ", ".join(n for n, s in zip(names, 0.5 * A @ eps + err) if abs(s) < 1e-6)]
# 띠별 단독 상한
lines.append("띠 하나에만 넣을 때 상한 [g]: " + ", ".join(f"{b[0]} {min(-2 * err[j] / A[j, i] for j in range(len(err)) if A[j, i] > 1e-6) * b[2]:.2f}"
                                                for i, b in enumerate(bands)))
lines.append("구역 질량 [g]: " + ", ".join(f"{b[0]} {b[2]:.2f}" for b in bands))
txt = "\n".join(lines)
print(txt)
open(OUT + "s2c_lp.txt", "w").write(txt + "\n")
