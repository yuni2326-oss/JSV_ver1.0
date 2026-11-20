"""치수 측정 오차로 측정 주파수(2번 시편, 10-30)를 맞출 수 있는지 — 32×16 민감도(s2g_sens.py)로 선형 모형을 만들고 경우별 적합.
모형: ln f_i = ln f_i0 + Σ_j S_ij Δp_j − ½ ln(1 + Σ_r T_ir Δm_r / M_r) + s_E
  Δp = 치수 변화[mm], Δm_r = 구역 r에 더한 비구조 질량[g](≥ 0), T = 기준 FE 운동에너지 몫, s_E = ½ ln(E/E0).
  질량 제약: M0 + Σ_j (∂M/∂p_j) Δp_j + Σ_r Δm_r = 41.3186 g.
선형 모형은 s2g_confirm.py(치수·질량을 넣은 FE)로 확인한다(s2g_check.py)."""
import json
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import importlib.util

SCR = "/tmp/claude-1000/-home-super-project/efb54ce3-bcf9-473d-9dc8-eb835c653517/scratchpad/impact1022/scripts/"
_s = importlib.util.spec_from_file_location("lib", SCR + "s2g_fit_lib.py")
L = importlib.util.module_from_spec(_s); _s.loader.exec_module(L)
D0 = "/home/super/project-paper3/docs/_generated/data/paper3/"
D1 = L.D1
OUT = "/tmp/claude-1000/-home-super-project/efb54ce3-bcf9-473d-9dc8-eb835c653517/scratchpad/impact1022/out/s2/"
M_MEAS = 41.3186
PAIRS, names, fmeas, mem = L.PAIRS, L.names, L.fmeas, L.mem
UNSURE = {"h0:m6n1", "h0:m6n0"}

# 구역 질량·운동에너지 몫(꼭짓점 삼선형 질량, s2f_mass2와 같은 계산)
rho = 7930.0
g = np.array([-1, 1]) / np.sqrt(3)
xi = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)
gp = np.array([[p, q, r] for p in g for q in g for r in g])
Nsh = np.prod(1 + gp[:, None, :] * xi[None, :, :], axis=2) / 8
dN = np.stack([xi[None, :, k] * np.prod(1 + gp[:, None, [m for m in range(3) if m != k]] * xi[None, :, [m for m in range(3) if m != k]], axis=2) / 8
               for k in range(3)], axis=2)
REGS = ["front", "back_main", "hub", "vanes", "collar"]


def regions(z):
    X, conn, ck, hub = z["coors"], z["conn"], z["cell_kind"], z["cell_hub"]
    detJ = np.abs(np.linalg.det(np.einsum("gnk,cnj->cgkj", dN, X[conn])))
    R = {"front": ck == 2, "back_main": (ck == 0) & ~hub, "hub": (ck == 0) & hub, "vanes": ck == 1, "collar": ck == 3}
    return detJ, R, {k: float(rho * detJ.sum(1)[m].sum() * 1e3) for k, m in R.items()}


zB, tB, VB = L.zB, L.tB, L.VB
detJ, RB, MB = regions(zB)
Me = rho * np.einsum("cg,gi,gj->cij", detJ, Nsh, Nsh)
U = VB.reshape(len(VB), -1, 3)
Tc = np.einsum("cij,mcik,mcjk->mc", Me, U[:, zB["conn"], :], U[:, zB["conn"], :])
Tr = np.stack([Tc[:, RB[k]].sum(1) / Tc.sum(1) for k in REGS], 1)          # 모드 × 구역
lab_idx = {lab: mem(r.members) for lab, r in zip(tB.label, tB.itertuples())}
T = np.array([np.mean([Tr[lab_idx[l]].mean(0) for l in p[1]], axis=0) for p in PAIRS])
fB, _ = L.pair_freqs({l: (L.fB_lab[l], 1.0) for l in L.fB_lab})
M0 = sum(MB.values())

# 민감도 실행 (테 벽은 테 바깥지름 30.82를 고정하고 안쪽을 바꾼다)
RUNS = {"tf": ("s_tf375", -0.025), "tb": ("s_tb620", +0.04), "tv": ("s_tv120", +0.2), "ch": ("s_ch318", +0.13),
        "colh": ("s_colh40", -1.0), "colt": ("s_colt425", -0.095)}
PN = list(RUNS)
NP = len(PN)
BASE = {"tf": 0.40, "tb": 0.58, "tv": 1.0, "ch": 3.05, "colh": 5.0, "colt": 0.52}
KOR = {"tf": "앞판", "tb": "뒤판", "tv": "깃", "ch": "간격", "colh": "테 높이", "colt": "테 벽"}
S = np.zeros((len(PAIRS), NP))
dMdp = np.zeros(NP)
dMr = np.zeros((len(REGS), NP))
lines = [f"기준 FE(2번 시편 치수, 32×16): {M0:.2f} g (" + ", ".join(f"{k} {MB[k]:.2f}" for k in REGS) + ")", ""]
lines.append("민감도 [%/mm] (ln f 기울기 × 100) — 행 = 모드, 열 = 치수; 마지막 줄 = 질량 [g/mm]")
for j, (k, (tag, d)) in enumerate(RUNS.items()):
    z, t, V, G = L.load(D1 + f"shapes_{tag}.npz", D1 + f"modes_{tag}.csv")
    f, mac = L.pair_freqs(L.matched(G))
    S[:, j] = (np.log(f) - np.log(fB)) / d
    _, _, Mr = regions(z)
    dMr[:, j] = [(Mr[r] - MB[r]) / d for r in REGS]
    dMdp[j] = (sum(Mr.values()) - M0) / d
    lines.append(f"  {k}: {tag} Δ{d:+} mm, 최소 MAC {mac:.3f}, 질량 {sum(Mr.values()):.2f} g")
Sdf = pd.DataFrame(100 * S, index=names, columns=PN)
Sdf.loc["질량 g/mm"] = dMdp
lines += [Sdf.to_string(float_format=lambda v: f"{v:8.2f}"), ""]
STEP = {"tf": 0.02, "tb": 0.02, "tv": 0.1, "ch": 0.1, "colh": 1.0, "colt": 0.05}
lines.append("캘리퍼 수준 오차 하나가 주는 주파수 변화 [%]: " + ", ".join(f"{KOR[k]} {STEP[k]:+} mm" for k in PN))
lines.append(pd.DataFrame(100 * S * np.array([STEP[k] for k in PN]), index=names, columns=[KOR[k] for k in PN]).to_string(float_format=lambda v: f"{v:6.2f}"))
lines.append("")

# 옛 FE(1번 시편 치수)
zO = np.load(D0 + "rev1_n6_shapes_ff32x16h6w62c40.npz", allow_pickle=False)
tO = pd.read_csv(D0 + "rev1_n6_modes_ff32x16h6w62c40.csv")
VO = zO["vshape"].astype(float)
GO = [(r.label, r.f_Hz, L.subspace(VO, mem(r.members))) for r in tO.itertuples()]
fO, macO = L.pair_freqs(L.matched(GO))
P1 = [0.375 - 0.40, 0.59 - 0.58, 1.2 - 1.0, 3.18 - 3.05, 4.0 - 5.0, 0.425 - 0.52]
pred_old = fB * np.exp(S @ np.array(P1))
lines.append(f"점검: 옛 FE(1번 시편 치수) = 기준 + 선형 민감도(외경 +0.05·보어 +0.025 mm 차는 빠짐, 외경만으로 약 −0.27 %), 최소 MAC {macO:.3f}")
lines.append("  " + ", ".join(f"{n} {100 * (p / o - 1):+.2f}" for n, p, o in zip(names, pred_old, fO)) + " % (예측/실제 − 1)")
lines.append(f"  질량: 예측 {M0 + dMdp @ np.array(P1):.2f} g, 실제 36.78 g")
lines.append("")


def model(q, mass_mode, sE_free):
    dp = q[:NP]
    sE = q[NP] if sE_free else 0.0
    m_add = q[NP + 1:]
    Mr_now = np.array([MB[r] for r in REGS]) + dMr @ dp
    Mstruct = M0 + dMdp @ dp
    if mass_mode == "uniform":
        x = np.full(5, (M_MEAS - Mstruct) / Mr_now.sum())
        Mtot = M_MEAS
    elif mass_mode == "free":
        x = m_add / Mr_now
        Mtot = Mstruct + m_add.sum()
    else:
        x = np.zeros(5)
        Mtot = Mstruct
    lnf = np.log(fB) + S @ dp - 0.5 * np.log1p(T @ x) + sE
    return lnf, Mtot, dp, sE, Mr_now * x, Mstruct


def fit(name, lo, hi, mass_mode, sE_free=False, sE_bound=0.015, use=None):
    use = np.ones(len(PAIRS), bool) if use is None else use
    lo, hi = np.array(lo, float), np.array(hi, float)
    fixed = hi - lo < 1e-12
    lb = np.concatenate([lo, [-sE_bound if sE_free else 0.0], np.zeros(5)])
    ub = np.concatenate([np.where(fixed, lo + 1e-9, hi), [sE_bound if sE_free else 1e-9], np.full(5, 20.0 if mass_mode == "free" else 1e-9)])
    q0 = np.concatenate([np.where(fixed, lo, 0.5 * (lo + hi)), [0.0], np.full(5, 0.5 if mass_mode == "free" else 0.0)])

    def res(q):
        lnf, Mtot, *_ = model(q, mass_mode, sE_free)
        r = 100 * (lnf - np.log(fmeas))[use]
        return np.concatenate([r, [(Mtot - M_MEAS) / 0.01] if mass_mode == "free" else []])
    sol = least_squares(res, np.clip(q0, lb, ub), bounds=(lb, ub), x_scale="jac", max_nfev=20000)
    lnf, Mtot, dp, sE, madd, Ms = model(sol.x, mass_mode, sE_free)
    e = 100 * (np.exp(np.log(fmeas) - lnf) - 1)       # 측정/모형 − 1 [%]
    dims = {k: BASE[k] + dp[j] for j, k in enumerate(PN)}
    at_bound = [KOR[k] for j, k in enumerate(PN) if not fixed[j] and (abs(dp[j] - lo[j]) < 1e-4 or abs(dp[j] - hi[j]) < 1e-4)]
    return {"case": name, "err": e, "use": use, "dims": dims, "E_pct": 100 * (np.exp(2 * sE) - 1), "madd": dict(zip(REGS, madd)),
            "M": Mtot, "M_struct": Ms, "at_bound": at_bound}


def show(c):
    e, use = c["err"], c["use"]
    s = [f"[{c['case']}] 측정/모형 − 1: 평균 {e[use].mean():+.2f} %, rms {np.sqrt((e[use] ** 2).mean()):.2f} %, 범위 {e[use].min():+.1f}~{e[use].max():+.1f} %"]
    s.append("   치수[mm] " + ", ".join(f"{KOR[k]} {v:.3f}" for k, v in c["dims"].items()) + f"; E {c['E_pct']:+.1f} %; 치수 질량 {c['M_struct']:.2f} g → 합계 {c['M']:.2f} g"
             + (f"; 경계에 닿음: {', '.join(c['at_bound'])}" if c["at_bound"] else ""))
    if any(v > 0.005 for v in c["madd"].values()):
        s.append("   더한 질량[g] " + ", ".join(f"{k} {v:.2f}" for k, v in c["madd"].items() if v > 0.005))
    s.append("   모드별 " + ", ".join(f"{n} {x:+.1f}" + ("" if u else "(제외)") for n, x, u in zip(names, e, use)))
    return s


Z = [0.0] * NP
lines.append("== 1번 시편 치수(옛 FE 그대로, 32×16) ==")
e_old = 100 * (fmeas / fO - 1)
e_old_u = 100 * (fmeas / (fO / np.sqrt(M_MEAS / 36.777)) - 1)
e_old_u1 = 100 * (fmeas / (fO / np.sqrt(40.2059 / 36.777)) - 1)
for lab, e in [("질량 보정 없음 (36.78 g)", e_old), ("빠진 4.54 g을 FE 분포대로 (2번 시편 41.32 g)", e_old_u),
               ("빠진 3.43 g을 FE 분포대로 (1번 시편 40.21 g)", e_old_u1)]:
    lines.append(f"[{lab}] 평균 {e.mean():+.2f} %, rms {np.sqrt((e ** 2).mean()):.2f} %, 범위 {e.min():+.1f}~{e.max():+.1f} %")
    lines.append("   모드별 " + ", ".join(f"{n} {x:+.1f}" for n, x in zip(names, e)))
lines.append("")
lines.append("== 선형 모형 적합 (기준 = 2번 시편 치수 FE) ==")
cases = []
cases.append(fit("2번 시편 치수, 질량 보정 없음", Z, Z, "none"))
cases.append(fit("2번 시편 치수 + 빠진 질량 FE 분포대로", Z, Z, "uniform"))
cases.append(fit("1번 시편 두께(외경은 2번) + 빠진 질량 FE 분포대로", P1, P1, "uniform"))
NOM = [0.0, 0.02, 0.0, 0.0, 0.0, 0.0]
cases.append(fit("공칭 판재(앞 0.4·뒤 0.6·깃 1.0) + 빠진 질량 FE 분포대로", NOM, NOM, "uniform"))
# 측정 오차 범위: 앞판 0.37–0.42, 뒤판 0.56–0.62, 깃 0.9–1.3, 간격 2.95–3.20, 테 높이 4.0–5.5, 테 벽 0.43–0.55
LO = [-0.03, -0.02, -0.1, -0.10, -1.0, -0.09]
HI = [+0.02, +0.04, +0.3, +0.15, +0.5, +0.03]
cases.append(fit("치수 = 측정 오차 범위 안 자유, 빠진 질량 FE 분포대로", LO, HI, "uniform"))
cases.append(fit("치수 = 측정 오차 범위 안 자유, 빠진 질량 구역 자유(≥0)", LO, HI, "free"))
use9 = np.array([n not in UNSURE for n in names])
cases.append(fit("치수 오차 범위 안, 질량 분포대로, 짝 불확실 2개(h0:m6n1·h0:m6n0) 제외", LO, HI, "uniform", use=use9))
cases.append(fit("치수 오차 범위 안, 질량 구역 자유, E ±3 %", LO, HI, "free", sE_free=True))
LOw = [-0.10, -0.10, -0.5, -0.5, -2.0, -0.2]
HIw = [+0.15, +0.15, +0.5, +0.5, +2.0, +0.3]
cases.append(fit("치수 범위 넓힘(판 ±0.1–0.15, 깃 ±0.5), 질량 분포대로", LOw, HIw, "uniform"))
for c in cases:
    lines += show(c)
txt = "\n".join(lines)
print(txt)
open(OUT + "s2g_fit.txt", "w").write(txt + "\n")
json.dump([{**{k: v for k, v in c.items() if k not in ("err", "use")}, "err": c["err"].tolist(), "use": c["use"].tolist()} for c in cases]
          + [{"case": "옛 FE 그대로", "err": e_old.tolist()}, {"case": "옛 FE + 41.32 g 분포대로", "err": e_old_u.tolist()},
             {"case": "옛 FE + 40.21 g 분포대로", "err": e_old_u1.tolist()},
             {"names": names, "S_pct_per_mm": (100 * S).tolist(), "params": PN, "step_mm": [STEP[k] for k in PN], "T": T.tolist(), "regions": REGS,
              "fB": fB.tolist(), "fO": fO.tolist(), "fmeas": fmeas.tolist()}],
          open(OUT + "s2g_fit.json", "w"), ensure_ascii=False, indent=1)
