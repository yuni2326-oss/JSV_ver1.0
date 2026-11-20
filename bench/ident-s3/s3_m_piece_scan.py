"""질량 조각(땜납 4 × 1 mm, 20 mg) FEM 훑기와 사진 자리 비교(46번 §7, 11-12).

1) FEM 훑기: 칸마다 조각 하나(6회 대칭) 고리를 FEM 섹터 각 0–60°(1°)·중심 r 32.3–35.3 mm(11-20 넓힘)·축 기울기 ψ(반지름 방향 0°,
   25°, 깃 선과 나란한 50.6°)마다 1차 질량 섭동으로 계산한다. 조각 = 축을 따라 1 mm 간격 5 mg 점 4개.
   점마다 윗면 꼭짓점의 |φ|²을 가우스 가중(σ 1 mm)으로 보간해, 격자 간격(둘레 약 2 mm) 때문에 계단이 지지 않게 한다.
   → fe-nominal/s3_m_piece_scan.csv(약 4분; 있으면 다시 쓰지 않음, --rerun)
2) 비교: 측정 M120(A1-6v 대비)·M240 증분(M120 대비)을 (a) 도안 자리(깃 k에서 20°·40°, r 34.3)와
   (b) 사진 자리(groove-test/m_piece_positions.csv, 그린 깃 선 기준; M120 = 두 사진 평균, M240 = 깃 k+1까지 각에서)로 낸 예측과 맞댄다.
   조각마다 다른 각은 1차 섭동이 더해지므로 η = Σ_i η_고리(θ_i)/6. (c) 그린 선과 실제 깃 사이 공통 어긋남 ε를 함께 맞춘다.
   측정값은 groove-test/out/m_angle_fit.json(s3_m_angle_scan.py가 groove_compare 출력에서 모은 것). σ는 s3_m_angle_scan.py와 같다.
   → groove-test/out/m_piece_fit.json, 화면 표
사용: PYTHONPATH=/home/super/project-paper3 python s3_m_piece_scan.py [--rerun]   (PAPER3_GEOM을 비울 것)"""
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SCAN = os.path.join(HERE, "..", "fe-nominal", "s3_m_piece_scan.csv")
GT = os.path.join(HERE, "..", "groove-test")
LAB = {"h2": "h2:m2n0", "3164": "h3:m3n0", "3250": "h0:m0n1", "3459": "h1:m1n1", "3639": "h2:m4n0", "3748": "h3:m3n1",
       "3804": "h0:m6n1", "8321": "h3:m3n1#2", "5745": "h1:m5n0", "5617": "h0:m6n0", "7093": "h3:m3n2", "7574": "h2:m8n0",
       "6439": "h2:m2n1", "4967": "h2:m4n0#2"}
PSI = (0.0, 25.0, 50.6)
RC = (32.3, 32.8, 33.3, 33.8, 34.3, 34.8, 35.3)    # 11-20(59번): 32.3–35.3으로 넓힘 — 자리 불확도 ±0.5 mm와 r < 33.3 조각을 자르지 않게


def fem_scan():
    from impeller_fingerprint import geometry as geo
    from impeller_fingerprint.rev1 import n1_mesh as nm
    from impeller_fingerprint.rev1 import n6_freefree as n6
    assert geo.ASBUILT.gap_in is None, "G3(간격 일정)로 돌린다 — PAPER3_GEOM을 비울 것"
    t0 = time.time()
    ff = n6.FFSpec(base=nm.FitSpec(n_r=32, n_bay=16, wrap_deg=geo.ASBUILT.wrap_deg), n_hub=6, n_z_collar=4,
                   collar_h=geo.ASBUILT.collar_h, clamp_r=None)
    fm = n6.build(ff)
    md, _ = n6.solve(n6.assemble(fm), 80, n_rigid=ff.n_rigid)
    obs = {o.label: o for o in n6.classify(fm, md)}
    print(f"건전 {md.freqs[0]:.1f}–{md.freqs[-1]:.1f} Hz ({time.time() - t0:.0f}s)", flush=True)
    fs, X = ff.base, fm.coors
    top = np.nonzero(np.isclose(X[:, 2], ff.z_front[1], atol=1e-9) & (fm.r_v >= ff.a + ff.collar_t - 1e-9))[0]
    XT = X[top, :2]
    cam = lambda r: float(fs.camber(np.array([r]))[0])
    G = {}
    for k, lab in LAB.items():
        o = obs[lab]
        U = md.vshape[o.members].reshape(o.deg, -1, 3)[:, top, :]
        G[k] = np.einsum("anc,bnc->nab", U, U)

    def point_G(x, y, sig=1.0e-3):
        d2 = (XT[:, 0] - x) ** 2 + (XT[:, 1] - y) ** 2
        near = d2 < (4 * sig) ** 2
        w = np.exp(-d2[near] / (2 * sig ** 2)); w /= w.sum()
        return {k: np.einsum("n,nab->ab", w, g[near]) for k, g in G.items()}

    def ring_eta(r_c, th_rel, psi_deg, m=20e-6, L=4e-3, n=4):
        H = {k: np.zeros_like(g[0]) for k, g in G.items()}
        g_cam = r_c * (cam(r_c + 1e-4) - cam(r_c - 1e-4)) / 2e-4
        psi = math.radians(psi_deg) * np.sign(math.atan(g_cam))       # 깃 선 쪽으로 기울임
        s = (np.arange(n) - (n - 1) / 2) * L / n
        for k6 in range(6):
            th_c = cam(r_c) + math.radians(60.0 * k6 + th_rel)
            er = np.array([math.cos(th_c), math.sin(th_c)]); et = np.array([-math.sin(th_c), math.cos(th_c)])
            ax = math.cos(psi) * er + math.sin(psi) * et
            for si in s:
                x, y = r_c * er + si * ax
                for kk, v in point_G(x, y).items():
                    H[kk] += -(m / n) * v
        return {k: float(np.mean(np.linalg.eigvalsh(h))) for k, h in H.items()}

    rows = [dict(psi=psi, rc_mm=rc, theta_deg=a, **ring_eta(rc * 1e-3, a, psi))
            for psi in PSI for rc in RC for a in np.arange(0.0, 60.01, 1.0)]
    pd.DataFrame(rows).to_csv(SCAN, index=False)
    print(f"훑기 끝 → {SCAN} ({time.time() - t0:.0f}s)", flush=True)


if "--rerun" in sys.argv or not os.path.exists(SCAN):
    fem_scan()
S = pd.read_csv(SCAN)
F = json.load(open(os.path.join(GT, "out", "m_angle_fit.json")))
M120, INC, KEYS = F["M120"], F["INC"], F["keys"]
dfp = lambda e: 100 * (math.sqrt(1 + e) - 1)


def eta(psi, rc, th_fem):
    g = S[np.isclose(S.psi, psi)]
    rcs = np.sort(g.rc_mm.unique())
    rc = float(np.clip(rc, rcs[0], rcs[-1]))
    i = int(np.clip(np.searchsorted(rcs, rc) - 1, 0, len(rcs) - 2)); w = (rc - rcs[i]) / (rcs[i + 1] - rcs[i])
    out = {}
    for k in KEYS:
        v = [np.interp(th_fem % 60.0, *g[np.isclose(g.rc_mm, rr)].sort_values("theta_deg")[["theta_deg", k]].to_numpy().T)
             for rr in (rcs[i], rcs[i + 1])]
        out[k] = (1 - w) * v[0] + w * v[1]
    return out


def preds(psi, a120, a240, eps=0.0):
    """a120·a240: [(측정 각 깃 k에서[°], r[mm])]. 측정 각 = 60 − FEM 각(28번 거울상). eps를 측정 각에서 뺀다."""
    e1 = {k: 0.0 for k in KEYS}; e2 = {k: 0.0 for k in KEYS}
    for acc, pcs in ((e1, a120), (e2, a240)):
        for a, r in pcs:
            e = eta(psi, r, 60.0 - (a - eps))
            for k in KEYS:
                acc[k] += e[k] / len(pcs)
    p1 = {k: dfp(e1[k]) for k in KEYS}
    pi = {k: 100 * (math.sqrt(1 + e1[k] + e2[k]) / math.sqrt(1 + e1[k]) - 1) for k in KEYS}
    return p1, pi


def chi(M, P):
    y = np.array([M[k] for k in KEYS]); p = np.array([P[k] for k in KEYS])
    s = np.sqrt(0.07 ** 2 + (0.1 * np.abs(y)) ** 2); w = 1 / s ** 2
    c = float((w * (y - p)).sum() / w.sum())
    return float((w * (y - p - c) ** 2).sum())


T = pd.read_csv(os.path.join(GT, "m_piece_positions.csv"))
near_k = T[T.ang_from_k < T.bay_width / 2]; near_k1 = T[T.ang_from_k >= T.bay_width / 2]
a120 = [(float(g.ang_from_k.mean()), float(g.r_mm.mean())) for _, g in near_k.groupby("bay")]       # 두 사진 평균(칸마다)
a240 = [(60.0 - float(r.ang_to_k1), float(r.r_mm)) for _, r in near_k1.iterrows()]
# 11-20(59번 §3): 편 사진 좌표가 국부적으로 ±15–20 % 늘고 줄어 칸 폭이 50–75°로 잡힌다(조각 길이와 칸 폭 상관 0.94–0.96).
# 각을 그 칸의 폭으로 60°에 맞추면 두 사진이 rms 1.5°로 맞는다 → 이 '칸 폭 환산' 자리를 사진 자리로 쓴다.
sc = 60.0 / T.bay_width
T = T.assign(from_s=T.ang_from_k * sc, to_s=T.ang_to_k1 * sc)
near_k = T[T.ang_from_k < T.bay_width / 2]; near_k1 = T[T.ang_from_k >= T.bay_width / 2]
a120s = [(float(g.from_s.mean()), float(g.r_mm.mean())) for _, g in near_k.groupby("bay")]
a240s = [(60.0 - float(r.to_s), float(r.r_mm)) for _, r in near_k1.iterrows()]
res = dict(positions=dict(M120=a120, M240=a240, M120_scaled=a120s, M240_scaled=a240s), cases={})
for psi in PSI:
    for lab, (A, B) in (("template", ([(20.0, 34.3)] * 6, [(40.0, 34.3)] * 6)), ("photo", (a120, a240)),
                        ("photo_scaled", (a120s, a240s))):
        p1, pi = preds(psi, A, B)
        res["cases"][f"{lab}|{psi}"] = dict(chi_M120=chi(M120, p1), chi_inc=chi(INC, pi), pred_M120=p1, pred_inc=pi)
# 자리 불확도: 모든 조각의 각 ±1°·반지름 ±0.5 mm 공통 이동(그린 선·사진 펴기의 공통 오차)의 9점 포락
D_TH, D_R = 1.0, 0.5
env = {k: [] for k in ("M120", "inc")}
for dth in (-D_TH, 0.0, D_TH):
    for dr in (-D_R, 0.0, D_R):
        p1, pi = preds(25.0, [(a + dth, r + dr) for a, r in a120s], [(a + dth, r + dr) for a, r in a240s])
        env["M120"].append(p1); env["inc"].append(pi)
res["photo_scaled_env"] = dict(d_theta_deg=D_TH, d_r_mm=D_R, **{
    f"{s}_{b}": {k: float(f([e[k] for e in env[s]])) for k in KEYS} for s in env for b, f in (("lo", np.min), ("hi", np.max))})


def chi_pos(M, P, lo, hi):
    """측정 불확도에 자리 불확도(포락 반폭)를 제곱합으로 더한 χ²(공통 오프셋 포함)."""
    y = np.array([M[k] for k in KEYS]); p = np.array([P[k] for k in KEYS])
    half = 0.5 * np.abs(np.array([hi[k] for k in KEYS]) - np.array([lo[k] for k in KEYS]))
    s = np.sqrt(0.07 ** 2 + (0.1 * np.abs(y)) ** 2 + half ** 2); w = 1 / s ** 2
    c = float((w * (y - p)).sum() / w.sum())
    return float((w * (y - p - c) ** 2).sum())


E_, P_ = res["photo_scaled_env"], res["cases"]["photo_scaled|25.0"]
res["photo_scaled_env"]["chi_M120_pos"] = chi_pos(M120, P_["pred_M120"], E_["M120_lo"], E_["M120_hi"])
res["photo_scaled_env"]["chi_inc_pos"] = chi_pos(INC, P_["pred_inc"], E_["inc_lo"], E_["inc_hi"])
eps_scan = []
for eps in np.arange(-4.0, 4.01, 0.5):
    p1, pi = preds(25.0, a120, a240, eps)
    eps_scan.append((float(eps), chi(M120, p1), chi(INC, pi)))
e_best = min(eps_scan, key=lambda t: t[1] + t[2])
p1, pi = preds(25.0, a120, a240, e_best[0])
res["eps"] = dict(scan=eps_scan, best=e_best[0], chi_M120=e_best[1], chi_inc=e_best[2], pred_M120=p1, pred_inc=pi)
json.dump(res, open(os.path.join(GT, "out", "m_piece_fit.json"), "w"), indent=1)

print(f"사진 자리(그린 깃 선 기준): M120 {np.mean([a for a, _ in a120]):.1f}°(r {np.mean([r for _, r in a120]):.2f}), "
      f"M240 {np.mean([a for a, _ in a240]):.1f}°(r {np.mean([r for _, r in a240]):.2f}) — 도안 20°·40°, r 34.3")
print(f"칸 폭 환산 자리: M120 {np.mean([a for a, _ in a120s]):.1f}°, M240 {np.mean([a for a, _ in a240s]):.1f}°")
for psi in PSI:
    t_, p_, s_ = res["cases"][f"template|{psi}"], res["cases"][f"photo|{psi}"], res["cases"][f"photo_scaled|{psi}"]
    print(f"ψ {psi:4.1f}°: χ²(11 모드) 도안 자리 {t_['chi_M120']:.0f} + {t_['chi_inc']:.0f} | 사진 자리 {p_['chi_M120']:.0f} + {p_['chi_inc']:.0f}"
          f" | 칸 폭 환산 {s_['chi_M120']:.0f} + {s_['chi_inc']:.0f}")
print(f"칸 폭 환산 + 자리 불확도(±{D_TH}°, ±{D_R} mm): χ² {E_['chi_M120_pos']:.0f} + {E_['chi_inc_pos']:.0f}")
print("ε 훑기(ψ 25°): " + ", ".join(f"{e:+.1f}° {a:.0f}+{b:.0f}" for e, a, b in eps_scan))
print(f"\n최선 ε {e_best[0]:+.1f}°(그린 선 → 실제 깃): χ² {e_best[1]:.0f} + {e_best[2]:.0f}")
t_ = res["cases"]["template|25.0"]
print("모드 | M120 측정 / FEM 도안 → FEM 사진+ε (비) | M240 증분 측정 / FEM 도안 → FEM 사진+ε (비)")
for k in KEYS:
    print(f"{k:>5} | {M120[k]:+.2f} / {t_['pred_M120'][k]:+.2f} → {p1[k]:+.2f} ({M120[k] / p1[k]:.2f}) | "
          f"{INC[k]:+.2f} / {t_['pred_inc'][k]:+.2f} → {pi[k]:+.2f} ({INC[k] / pi[k]:.2f})")
print(f"→ {os.path.join(GT, 'out', 'm_piece_fit.json')}")
