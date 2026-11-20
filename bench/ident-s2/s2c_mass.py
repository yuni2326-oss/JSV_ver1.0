"""2번 시편 무게(41.3186 g)로 FE 질량 차이 검토 — 구역별 운동·변형에너지 분율로 1차 감도.

T(운동에너지 분율): 꼭짓점 모드형(질량정규 φ의 꼭짓점 값)을 삼선형 hex 일관 질량으로 적분.
  검증: 모드마다 Σ_R T_R ≈ 1(φᵀMφ = 1)이어야 한다(2차 요소의 중간 절점을 버린 근사 오차).
U(변형에너지 분율): FE 표의 Efrac_*(정확, 축퇴군 평균).
"""
import json
import numpy as np
import pandas as pd

D = "/home/super/project-paper3/docs/_generated/data/paper3/"
KEY = "ff32x16h6w62c40"
OUT = "/tmp/claude-1000/-home-super-project/efb54ce3-bcf9-473d-9dc8-eb835c653517/scratchpad/impact1022/out/s2/"
M_MEAS = {"s1": 40.2059, "s2": 41.3186}

z = np.load(D + f"rev1_n6_shapes_{KEY}.npz", allow_pickle=False)
X, conn, ck, vsh = z["coors"], z["conn"], z["cell_kind"], z["vshape"].astype(float)
meta = json.load(open(D + f"rev1_n6_modes_{KEY}.json"))
tab = pd.read_csv(D + f"rev1_n6_modes_{KEY}.csv")
rho, a = 7930.0, meta["a"]

# 삼선형 hex: 2x2x2 가우스, 꼭짓점 순서는 sfepy 3_8(아래 면 0-3, 위 면 4-7)
g = np.array([-1, 1]) / np.sqrt(3)
xi = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
               [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)
gp = np.array([[p, q, r] for p in g for q in g for r in g])
N = np.prod(1 + gp[:, None, :] * xi[None, :, :], axis=2) / 8           # (8 gp, 8 node)
dN = np.stack([xi[None, :, k] * np.prod(1 + gp[:, None, [m for m in range(3) if m != k]]
                                         * xi[None, :, [m for m in range(3) if m != k]], axis=2) / 8
               for k in range(3)], axis=2)                               # (gp, node, 3)
Xe = X[conn]                                                             # (cell, 8, 3)
J = np.einsum("gnk,cnj->cgkj", dN, Xe)
detJ = np.linalg.det(J)                                                  # (cell, gp)
if (detJ <= 0).any():
    # 꼭짓점 순서가 반대면 부호만 뒤집힌다
    detJ = np.abs(detJ)
vol = detJ.sum(1)                                                        # 가우스 가중 1
Me = rho * np.einsum("cg,gi,gj->cij", detJ, N, N)                        # (cell, 8, 8) 스칼라 질량

r_c = np.hypot(Xe[:, :, 0].mean(1), Xe[:, :, 1].mean(1))
regions = {"front": ck == 2, "back_main": (ck == 0) & (r_c >= a), "hub": (ck == 0) & (r_c < a),
           "vanes": ck == 1, "collar": ck == 3}
mass = {k: float(rho * vol[m].sum() * 1e3) for k, m in regions.items()}
print("구역 질량 [g]:", {k: round(v, 3) for k, v in mass.items()}, " 합", round(sum(mass.values()), 3),
      " (표 mass_model_g", round(meta["mass_model_g"], 3), ")")

U = vsh.reshape(len(vsh), -1, 3)                                         # (mode, node, 3)
Tc = np.einsum("cij,mcik,mcjk->mc", Me, U[:, conn, :], U[:, conn, :])    # 셀별 φᵀMφ
Tr = {k: Tc[:, m].sum(1) for k, m in regions.items()}
Ttot = sum(Tr.values())
print("모드별 Σ T (1이어야):", np.round(Ttot[:24], 3))

# 관측량(축퇴군) 단위로 평균
rows = []
for _, o in tab.iterrows():
    mem = [int(s) for s in str(o["members"]).split("|")]
    t = {k: float(np.mean([Tr[k][i] / Ttot[i] for i in mem])) for k in regions}
    rows.append({"label": o["label"], "h": int(o["h"]), "f_FE": float(o["f_Hz"]), "sumT": float(np.mean(Ttot[mem])),
                 **{f"T_{k}": v for k, v in t.items()},
                 **{f"U_{k}": float(o[f"Efrac_{k}"]) for k in regions}})
ft = pd.DataFrame(rows)

# 10-30 공동 적합 극점 → FE 짝(28번 표), 3639.0은 섞임이라 따로
pairs = [(2138.04, "h2:m2n0", "2133.8/2142.3 평균"), (3163.65, "h3:m3n0", ""), (3249.89, "h0:m0n1", ""),
         (3458.89, "h1:m1n1", ""), (3747.87, "h3:m3n1", ""), (3804.07, "h0:m6n1", "반대 부호"),
         (4225.49, "h1:m1n1#2", ""), (4966.99, "h2:m4n1", ""), (5617.22, "h0:m6n0", ""),
         (5744.69, "h1:m5n1", "반대 부호"), (6438.86, "h2:m2n1", ""), (8321.24, "h3:m3n1#2", ""),
         (3638.99, "h2:m4n0", "섞임")]
P = []
for fm_, lab, note in pairs:
    r = ft[ft.label == lab].iloc[0]
    P.append({**r.to_dict(), "f_meas": fm_, "err": fm_ / r.f_FE - 1, "note": note})
P = pd.DataFrame(P)
use = P.note != "섞임"
reg = list(regions)

def show(df):
    cols = ["label", "f_FE", "f_meas", "err"] + [f"T_{k}" for k in reg] + [f"U_{k}" for k in reg] + ["sumT"]
    return df[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}")

lines = [f"구역 질량 [g]: " + ", ".join(f"{k} {v:.2f}" for k, v in mass.items()) + f", 합 {sum(mass.values()):.2f}",
         f"실측: 1번 {M_MEAS['s1']} g, 2번 {M_MEAS['s2']} g → 2번 − FE = {M_MEAS['s2'] - meta['mass_model_g']:.2f} g "
         f"({(M_MEAS['s2'] / meta['mass_model_g'] - 1) * 100:.1f} %)", "", show(P), ""]

# (1) 빠진 질량 4.54 g를 한 구역에 모두 넣을 때(밀도 배율, 모양 고정 레일리 몫): f/f0 = 1/sqrt(1 + ε T_R)
dm = M_MEAS["s2"] - meta["mass_model_g"]
lines.append(f"(1) 빠진 질량 {dm:.2f} g를 한 구역에 고르게 넣으면 — 모드별 예측 Δf/f [%] (측정 err와 비교)")
hdr = "  " + "label".ljust(11) + " err  " + " ".join(k.rjust(9) for k in reg + ["uniform"])
lines.append(hdr)
pred1 = {}
for k in reg + ["uniform"]:
    eps = dm / (mass[k] if k != "uniform" else sum(mass.values()))
    T = P[f"T_{k}"] if k != "uniform" else 1.0
    pred1[k] = 1 / np.sqrt(1 + eps * T) - 1
for i, r in P.iterrows():
    lines.append("  " + r.label.ljust(11) + f"{r.err * 100:+5.1f} " + " ".join(f"{pred1[k][i] * 100 if k != 'uniform' else pred1[k] * 100:+9.2f}" for k in reg + ["uniform"]))
lines.append("")

# (2) 질량만: ε_R ≥ 0 (NNLS), err ≈ −½ Σ ε_R T_R  → 필요한 추가 질량
from scipy.optimize import nnls, lsq_linear
A_T = -0.5 * P.loc[use, [f"T_{k}" for k in reg]].to_numpy()
b = P.loc[use, "err"].to_numpy()
# NNLS on ε (A_T has negative entries, ε ≥ 0)
res = lsq_linear(A_T, b, bounds=(0, np.inf))
eps = res.x
lines.append("(2) 질량만 맞춤(구역별 밀도 증가 ε ≥ 0, 1차): " + ", ".join(f"{k} ε={e:.3f}({e * mass[k]:.2f} g)" for k, e in zip(reg, eps))
             + f" → 추가 질량 합 {sum(e * mass[k] for k, e in zip(reg, eps)):.2f} g (실측 차 {dm:.2f} g), "
             f"잔차 rms {np.sqrt(np.mean((A_T @ eps - b) ** 2)) * 100:.2f} % (err rms {np.sqrt(np.mean(b ** 2)) * 100:.2f} %)")
# (2b) 질량 합을 실측 차로 고정
from scipy.optimize import minimize
mvec = np.array([mass[k] for k in reg])
cons = [{"type": "eq", "fun": lambda e: e @ mvec - dm}]
r2 = minimize(lambda e: np.sum((A_T @ e - b) ** 2), x0=np.full(len(reg), dm / mvec.sum()),
              bounds=[(0, None)] * len(reg), constraints=cons, method="SLSQP")
lines.append("(2b) 질량 합 = {:.2f} g 고정: ".format(dm) + ", ".join(f"{k} {e * mass[k]:.2f} g" for k, e in zip(reg, r2.x))
             + f", 잔차 rms {np.sqrt(np.mean((A_T @ r2.x - b) ** 2)) * 100:.2f} %")

# (3) 강성만: δ_R ≤ 0, err ≈ +½ Σ δ_R U_R
A_U = 0.5 * P.loc[use, [f"U_{k}" for k in reg]].to_numpy()
r3 = lsq_linear(A_U, b, bounds=(-1, 0))
lines.append("(3) 강성만 맞춤(구역별 강성 감소 −1 ≤ δ ≤ 0, 1차): " + ", ".join(f"{k} δ={d:+.3f}" for k, d in zip(reg, r3.x))
             + f", 잔차 rms {np.sqrt(np.mean((A_U @ r3.x - b) ** 2)) * 100:.2f} %")

# (4) 질량 4.54 g 고정 배치 + 강성 맞춤 — 배치 가설별
lines.append("(4) 빠진 질량 배치 가설 + 강성 맞춤(δ ≤ 0):")
hyp = {"없음(질량 무시)": {}, "고르게": {k: dm * mass[k] / mvec.sum() for k in reg},
       "허브+테": {"hub": dm * mass["hub"] / (mass["hub"] + mass["collar"]), "collar": dm * mass["collar"] / (mass["hub"] + mass["collar"])},
       "깃": {"vanes": dm}, "뒤판 전체": {"back_main": dm * mass["back_main"] / (mass["back_main"] + mass["hub"]),
                                     "hub": dm * mass["hub"] / (mass["back_main"] + mass["hub"])},
       "앞판+테": {"front": dm * mass["front"] / (mass["front"] + mass["collar"]), "collar": dm * mass["collar"] / (mass["front"] + mass["collar"])}}
for name, dist in hyp.items():
    e_m = np.zeros(use.sum())
    for k, g_ in dist.items():
        e_m += -0.5 * (g_ / mass[k]) * P.loc[use, f"T_{k}"].to_numpy()
    r4 = lsq_linear(A_U, b - e_m, bounds=(-1, 0))
    rms = np.sqrt(np.mean((A_U @ r4.x + e_m - b) ** 2)) * 100
    lines.append(f"   {name:10s}: 질량 몫 Δf/f {np.min(e_m) * 100:+.1f}…{np.max(e_m) * 100:+.1f} %, 강성 " +
                 ", ".join(f"{k} {d:+.2f}" for k, d in zip(reg, r4.x)) + f" → 잔차 rms {rms:.2f} %")
txt = "\n".join(lines)
print(txt)
open(OUT + "s2c_mass.txt", "w").write(txt + "\n")
P.to_csv(OUT + "s2c_mass_pairs.csv", index=False)
ft.to_csv(OUT + "s2c_mass_fe_fractions.csv", index=False)
