"""2번 시편 치수 FE(앞판 0.40, 32×16)로 질량 검토 다시 — 옛 FE와 모드형 MAC로 짝을 옮기고, 29번과 같은 가설·LP.
옛 FE `ff32x16h6w62c40`과 격자 위상이 같으므로 꼭짓점 번호로 바로 비교한다(축퇴쌍은 부분공간 MAC)."""
import json
import numpy as np
import pandas as pd
from scipy.optimize import linprog

D0 = "/home/super/project-paper3/docs/_generated/data/paper3/"
D1 = "/tmp/claude-1000/-home-super-project/efb54ce3-bcf9-473d-9dc8-eb835c653517/scratchpad/impact1022/out/fe_s2/"
OUT = "/tmp/claude-1000/-home-super-project/efb54ce3-bcf9-473d-9dc8-eb835c653517/scratchpad/impact1022/out/s2/"
TAG = "tf400_32x16"
z0 = np.load(D0 + "rev1_n6_shapes_ff32x16h6w62c40.npz", allow_pickle=False)
z1 = np.load(D1 + f"shapes_{TAG}.npz", allow_pickle=False)
t0 = pd.read_csv(D0 + "rev1_n6_modes_ff32x16h6w62c40.csv")
t1 = pd.read_csv(D1 + f"modes_{TAG}.csv")
meta1 = json.load(open(D1 + f"meta_{TAG}.json"))
assert z0["conn"].shape == z1["conn"].shape and np.array_equal(z0["conn"], z1["conn"]), "격자 위상이 다르다"
mem = lambda s: [int(x) for x in str(s).split("|")]


def subspace(V, ids):
    Q, _ = np.linalg.qr(V[ids].T.astype(float))
    return Q


V0, V1 = z0["vshape"].astype(float), z1["vshape"].astype(float)
B1 = [(r.label, r.f_Hz, subspace(V1, mem(r.members))) for r in t1.itertuples()]
match = {}
for r in t0.itertuples():
    Q0 = subspace(V0, mem(r.members))
    best = max(((np.linalg.svd(Q0.T @ Q1, compute_uv=False) ** 2).mean(), lab, f) for lab, f, Q1 in B1
               if abs(f / r.f_Hz - 1) < 0.25)
    match[r.label] = (best[1], best[2], best[0])

# 운동에너지 분율(새 FE, 꼭짓점 삼선형 질량)
rho = 7930.0
X, conn, ck, hub = z1["coors"], z1["conn"], z1["cell_kind"], z1["cell_hub"]
g = np.array([-1, 1]) / np.sqrt(3)
xi = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)
gp = np.array([[p, q, r] for p in g for q in g for r in g])
N = np.prod(1 + gp[:, None, :] * xi[None, :, :], axis=2) / 8
dN = np.stack([xi[None, :, k] * np.prod(1 + gp[:, None, [m for m in range(3) if m != k]] * xi[None, :, [m for m in range(3) if m != k]], axis=2) / 8
               for k in range(3)], axis=2)
detJ = np.abs(np.linalg.det(np.einsum("gnk,cnj->cgkj", dN, X[conn])))
vol = detJ.sum(1)
Me = rho * np.einsum("cg,gi,gj->cij", detJ, N, N)
U = V1.reshape(len(V1), -1, 3)
Tc = np.einsum("cij,mcik,mcjk->mc", Me, U[:, conn, :], U[:, conn, :])
REG = {"front": ck == 2, "back_main": (ck == 0) & ~hub, "hub": (ck == 0) & hub, "vanes": ck == 1, "collar": ck == 3}
MASS = {k: float(rho * vol[m].sum() * 1e3) for k, m in REG.items()}
Tsum = Tc.sum(1)
Tr = {k: Tc[:, m].sum(1) / Tsum for k, m in REG.items()}
T1 = {r.label: {k: float(np.mean([Tr[k][i] for i in mem(r.members)])) for k in REG} for r in t1.itertuples()}

meas = [("h2:m2n0", 2138.04), (("h3:m3n0", "h0:m0n1"), (3163.65 + 3249.89) / 2), ("h1:m1n1", 3458.89),
        ("h3:m3n1", 3747.87), ("h0:m6n1", 3804.07), ("h1:m1n1#2", 4225.49), ("h2:m4n1", 4966.99),
        ("h0:m6n0", 5617.22), ("h1:m5n1", 5744.69), ("h2:m2n1", 6438.86), ("h3:m3n1#2", 8321.24), ("h2:m4n0", 3638.99)]
rows = []
for lab, fm in meas:
    labs = lab if isinstance(lab, tuple) else (lab,)
    news = [match[l] for l in labs]
    f_old = np.mean([t0.set_index("label").loc[l, "f_Hz"] for l in labs])
    f_new = np.mean([m[1] for m in news])
    T = {k: np.mean([T1[m[0]][k] for m in news]) for k in REG}
    rows.append({"pair": "+".join(labs), "new_label": "+".join(m[0] for m in news), "MAC": min(m[2] for m in news),
                 "f_meas": fm, "f_old": f_old, "f_new": f_new, "err_old": 100 * (fm / f_old - 1), "err_new": 100 * (fm / f_new - 1),
                 **{f"T_{k}": v for k, v in T.items()}})
P = pd.DataFrame(rows)
dm = 41.3186 - meta1["mass_g"]
use = P.pair != "h2:m4n0"
lines = [f"새 FE {TAG}: 질량 {meta1['mass_g']:.2f} g (구역 " + ", ".join(f"{k} {v:.2f}" for k, v in MASS.items()) + f"), 실측 41.32 g → 차 {dm:.2f} g ({dm / meta1['mass_g'] * 100:.1f} %)",
         "", P[["pair", "new_label", "MAC", "f_meas", "f_old", "f_new", "err_old", "err_new"] + [f"T_{k}" for k in REG]].to_string(
             index=False, float_format=lambda v: f"{v:.3f}" if abs(v) < 10 else f"{v:.1f}"), ""]
e = P.loc[use, "err_new"].to_numpy() / 100
Tm = P.loc[use, [f"T_{k}" for k in REG]].to_numpy()
uni = (1 / np.sqrt(1 + dm / meta1["mass_g"]) - 1)
lines.append(f"빠진 {dm:.2f} g를 FE 분포대로 → 모든 모드 {uni * 100:+.2f} %; 남는 차(측정 − 보정 FE) [%]: " +
             ", ".join(f"{p} {100 * ((1 + x) / (1 + uni) - 1):+.1f}" for p, x in zip(P.loc[use, "pair"], e)))
res_uni = (1 + e) / (1 + uni) - 1
lines.append(f"  남는 차 평균 {res_uni.mean() * 100:+.2f} %, rms {np.sqrt((res_uni ** 2).mean()) * 100:.2f} % (보정 전 평균 {e.mean() * 100:+.2f}, rms {np.sqrt((e ** 2).mean()) * 100:.2f})")
# 한 기전 상한(29번과 같은 가정)
mech = {"FE 분포대로": list(REG), "앞판": ["front"], "뒤판+허브": ["back_main", "hub"], "깃": ["vanes"], "허브+테": ["hub", "collar"], "테": ["collar"]}
lines.append("")
lines.append("측정 주파수가 허용하는 최대 추가 질량(가정: FE 강성 오차는 주파수를 내리기만):")
for name, S in mech.items():
    ind = np.array([k in S for k in REG], float); mt = sum(MASS[k] for k in S)
    r = -2 * e / (Tm @ ind); j = int(np.argmin(r))
    lines.append(f"  {name:10s}: {max(r[j], 0) * mt:5.2f} g (밀도 +{max(r[j], 0) * 100:.1f} %), 묶는 모드 {P.loc[use, 'pair'].iloc[j]} ({e[j] * 100:+.1f} %)")
mvec = np.array([MASS[k] for k in REG])
lp = linprog(-mvec, A_ub=0.5 * Tm, b_ub=-e, bounds=[(0, None)] * len(REG), method="highs")
lines.append(f"  5구역 자유 배치 LP: {-lp.fun:.2f} g (" + ", ".join(f"{k} {x * MASS[k]:.2f}" for k, x in zip(REG, lp.x) if x * MASS[k] > 0.01) + ")")
txt = "\n".join(lines)
print(txt)
open(OUT + "s2f_mass2.txt", "w").write(txt + "\n")
P.to_csv(OUT + "s2f_mass2_pairs.csv", index=False)
