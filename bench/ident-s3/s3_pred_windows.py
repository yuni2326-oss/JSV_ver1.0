"""창·테이프 단계 예측(42번 채택, 10-31) — G3 자유-자유 FEM(32×16), 짝은 측정 성분별.

상태(측정 깃 번호 규약; 측정 깃 v ↔ FEM 섹터 (−(v − 1)) mod 6, 28번 거울상):
  L1a·L1  깃 3–4 사이 B 창 하나(30°, ξ0.8–1.0) 0.03·0.06 mm = FEM 섹터 3 + 30°(2142 Hz 성분의 배)
  L6      깃 사이 B 창 여섯(0.06 mm). R(깃 위 창 여섯까지 = B 고리 0.06 mm)·C1·C2는 s3_pred_states.csv의 B3·C1·C2와 같다
  T-a     깃 2 위 테두리(ξ0.95) 30 mg                 = FEM 섹터 5 + 0°
  T-b     깃 3에서 깃 2 쪽 14° 테두리 30 mg            = FEM 섹터 4 + 14.1°
  T-c     깃 3–4 사이 테두리 10 mg                    = FEM 섹터 3 + 30°
  M75     테이프 고리 r 33.2·35.4 mm, 칸마다 깃 번호 방향 1/3 자리(측정 +20° = FEM 섹터 + 40°) 한 조각씩, 조각 6.25 mg(12조각)
  M150    M75 + 2/3 자리(측정 +40° = FEM 섹터 + 20°), 24조각 150 mg
  M120    (11-12, 실제로 붙인 것) M75 자리에 깃 선과 나란한 띠 20 mg 6개 — 같은 각의 두 점(r 33.2·35.4)을 잇는다
  M240    M120 + M150 자리(측정 +40°)에 띠 6개
h2 짝(2134/2142)은 측정 방향(10-30 ξ0.6 깃 6곳 θ0: 2134 117.2°, 2142 73.1°; s3_orient.py와 같은 규약, 90° 주기)을 FEM 짝
공간의 단위벡터 e로 옮겨, 1차는 eᵀHe, 재해석은 형상 상관으로 고른 손상 모드의 값을 낸다. 다른 짝은 성분 방향을 모르므로
쌍 평균과 반 갈라짐(half_split)을 낸다.
창: 1차(±5 % 중심차분, 0.02 mm 기준 → 깊이 비례) + 재해석(앞판 윗면·전체 꼭짓점 형상 부분공간 짝짓기).
테이프: 질량점 1차 Δλ/λ = −m Σ w|φ(x)|²(앞판 윗면 꼭짓점, 발자국 안 평균). 8° 띠 계산(s3_local)과 대조를 찍는다.
출력: fe-nominal/s3_pred_windows.csv — state, label, h, deg, f_Hz, member(2134·2142·mean·single), df_pct_lin,
      df_pct_res, half_split_pct, removed_mg(FEM, 붙인 질량은 음수), mac
사용: PYTHONPATH=/home/super/project-paper3 python s3_pred_windows.py   (약 25분)
      --tapes-only: 창(L1a·L1·L6)은 있는 표 값을 그대로 두고 테이프 상태만 다시 계산(약 7분)"""
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from impeller_fingerprint import geometry as geo
from impeller_fingerprint.rev1 import n1_mesh as nm
from impeller_fingerprint.rev1 import n6_freefree as n6

OUT = Path(__file__).resolve().parents[1] / "fe-nominal" / "s3_pred_windows.csv"
assert geo.ASBUILT.gap_in is None, "G3(간격 일정)로 돌린다 — PAPER3_GEOM을 비울 것"
PROBE = 0.05
THETA0 = {"2134": 117.2, "2142": 73.1}        # 10-30 ξ0.6 깃 6곳 공동 적합(28번)
TAPE_FOOT = {30.0: 2.0e-3, 10.0: 1.25e-3, 6.25: 1.0e-3, 5.0: 1.0e-3}   # 조각 질량[mg] → 발자국 반폭[m](납 테이프 0.13 mm 두께 어림; 5 mg = 20 mg 띠의 한 칸)
TAPES_ONLY = "--tapes-only" in sys.argv

t0 = time.time()
ff = n6.FFSpec(base=nm.FitSpec(n_r=32, n_bay=16, wrap_deg=geo.ASBUILT.wrap_deg), n_hub=6, n_z_collar=4,
               collar_h=geo.ASBUILT.collar_h, clamp_r=None)
fm = n6.build(ff)
km = n6.assemble(fm)
md, _ = n6.solve(km, 80, n_rigid=ff.n_rigid)
obs = n6.classify(fm, md)
print(f"건전 {md.freqs[0]:.1f}–{md.freqs[-1]:.1f} Hz, 관측량 {len(obs)} ({time.time() - t0:.0f}s)", flush=True)

fs = ff.base
X = fm.coors
th_v = np.arctan2(X[:, 1], X[:, 0])
xi_v = (fm.r_v - ff.a) / ff.extent
z_in, z_top = ff.z_front
front_v = (X[:, 2] >= z_in - 1e-12) & (X[:, 2] <= z_top + 1e-12) & (fm.r_v >= ff.a - 1e-12)
top_v = np.isclose(X[:, 2], z_top, atol=1e-9) & (fm.r_v >= ff.a + ff.collar_t - 1e-9)
gi = fm.info["grid_idx"]
PER = fs.n_vc + fs.n_bay
loc, sec = gi[:, 1] % PER, gi[:, 1] // PER
bot = np.isclose(X[:, 2], 0.0, atol=1e-9)
cam = lambda r: float(fs.camber(np.array([r]))[0])
wrap = lambda a: (a + math.pi) % (2 * math.pi) - math.pi
r_of = lambda xi: ff.a + xi * ff.extent
H2 = [o for o in obs if o.label == "h2:m2n0"][0]


# ---------------------------------------------------------------- 측정 짝 성분 → FEM 짝 공간 벡터
def back_point(k, xi):
    r_t = r_of(xi)
    W = []
    for l_ in (0, 1):
        c = np.nonzero(bot & (loc == l_) & (sec == k))[0]
        c = c[np.argsort(fm.r_v[c])]
        i = int(np.searchsorted(fm.r_v[c], r_t))
        lo_, hi_ = c[i - 1], c[i]
        t = (r_t - fm.r_v[lo_]) / (fm.r_v[hi_] - fm.r_v[lo_])
        W += [(lo_, 0.5 * (1 - t)), (hi_, 0.5 * t)]
    return W


V2 = md.vshape[H2.members]                                    # (2, 3·nv)
P6 = np.array([[sum(w * V2[m, 3 * n + 2] for n, w in back_point((-v) % 6, 0.6)) for m in range(2)] for v in range(6)])
TH6 = np.radians(60.0 * np.arange(6))
betas = np.radians(np.arange(0.0, 180.0, 0.25))
th0_of = np.array([(-np.degrees(np.angle(((P6 @ [math.cos(b), math.sin(b)]) * np.exp(-2j * TH6)).sum())) / 2) % 180
                   for b in betas])
E = {}
for name, t in THETA0.items():
    j = int(np.argmin(np.abs((4 * (th0_of - t) + 180) % 360 - 180)))
    E[name] = np.array([math.cos(betas[j]), math.sin(betas[j])])
ang = math.degrees(math.acos(abs(float(E["2134"] @ E["2142"]))))
print(f"측정 성분 → FEM 짝 벡터: 2134 β {math.degrees(math.atan2(*E['2134'][::-1])):.2f}°, "
      f"2142 β {math.degrees(math.atan2(*E['2142'][::-1])):.2f}° (사잇각 {ang:.1f}°, 직교면 90°)", flush=True)


def pair_rows(state, H, removed_mg, res=None, mac=None):
    """H: 관측량별 2×2(짝) 또는 1×1(홑) 섭동 행렬(η, 1차). res: 재해석 {label: η 또는 {member: η}}."""
    rows = []
    for o in obs:
        h = H[o.label]
        base = dict(state=state, label=o.label, h=o.h, deg=o.deg, f_Hz=o.f_Hz, removed_mg=removed_mg,
                    mac=(mac or {}).get(o.label, np.nan))
        dfp = lambda e: 100 * (math.sqrt(1 + e) - 1) if np.isfinite(e) else np.nan
        if o.deg == 1:
            rows.append(dict(base, member="single", df_pct_lin=dfp(h[0, 0]),
                             df_pct_res=dfp(res[o.label]) if res else np.nan, half_split_pct=0.0))
            continue
        w = np.linalg.eigvalsh(h)
        hs = 25 * (w[1] - w[0])                     # 반 갈라짐: η 차의 반 → Δf/f는 다시 반
        if o.label == "h2:m2n0":
            for name, e in E.items():
                rr = res[o.label][name] if res else np.nan
                rows.append(dict(base, member=name, df_pct_lin=dfp(float(e @ h @ e)), df_pct_res=dfp(rr), half_split_pct=0.0))
        else:
            rr = res[o.label]["mean"] if res else np.nan
            hr = res[o.label]["half"] if res else np.nan
            rows.append(dict(base, member="mean", df_pct_lin=dfp(0.5 * np.trace(h)), df_pct_res=dfp(rr),
                             half_split_pct=hs if not res else 100 * hr))
    return rows


def first_order(dK, dM):
    H = {}
    for o in obs:
        Ph = md.phi[:, o.members]
        lam = float(np.mean(md.lam[o.members]))
        h = Ph.T @ (dK @ Ph) / lam - Ph.T @ (dM @ Ph)
        H[o.label] = 0.5 * (h + h.T)
    return H


def window_p(centers_deg_rel, k_sectors, depth, xlo=0.8, xhi=1.0, width=30.0):
    r_c = r_of(0.5 * (xlo + xhi))
    p = np.zeros(len(X))
    for k in k_sectors:
        for rel in centers_deg_rel:
            th_c = cam(r_c) + math.radians(60.0 * k + rel)
            sel = front_v & (xi_v >= xlo - 1e-9) & (xi_v <= xhi + 1e-9) & (np.abs(wrap(th_v - th_c)) <= math.radians(width / 2) + 1e-9)
            p = np.where(sel, depth / fs.t_front, p)
    return p


def lin_window(p):
    kp = n6.assemble(fm, n6.morph_groove(fm, p * PROBE))
    kn = n6.assemble(fm, n6.morph_groove(fm, -p * PROBE))
    dK = (kp.K - kn.K) * (0.5 / PROBE)
    dM = (kp.M - kn.M) * (0.5 / PROBE)
    tx = np.zeros(dM.shape[0]); tx[0::3] = 1.0
    return first_order(dK, dM), -float(tx @ (dM @ tx)) * 1e6


def resolved(p):
    """깎은 형상으로 다시 풀고, 건전 관측량마다 형상 부분공간에 가장 많이 든 손상 모드를 고른다."""
    kd = n6.assemble(fm, n6.morph_groove(fm, p))
    mdd, _ = n6.solve(kd, md.phi.shape[1] + 6, n_rigid=ff.n_rigid)
    Vd = mdd.vshape / np.linalg.norm(mdd.vshape, axis=1, keepdims=True)
    used, res, mac = set(), {}, {}
    for o in sorted(obs, key=lambda o: o.f_Hz):
        Q, _ = np.linalg.qr(md.vshape[o.members].T)
        score = np.sum((Vd @ Q) ** 2, axis=1)
        score[list(used)] = -1
        cand = [k for k in np.argsort(-score)[:o.deg] if score[k] >= 0.8]
        lam0 = float(np.mean(md.lam[o.members]))
        if len(cand) < o.deg:
            res[o.label] = np.nan if o.deg == 1 else ({"2134": np.nan, "2142": np.nan} if o.label == "h2:m2n0"
                                                      else {"mean": np.nan, "half": np.nan})
            mac[o.label] = np.nan
            continue
        used.update(cand)
        mac[o.label] = float(np.min(score[cand]))
        eta = [float(mdd.lam[k] / lam0 - 1) for k in cand]
        if o.deg == 1:
            res[o.label] = eta[0]
        elif o.label == "h2:m2n0":
            S = {n: (V2.T @ e) / np.linalg.norm(V2.T @ e) for n, e in E.items()}
            c0 = abs(float(Vd[cand[0]] @ S["2134"])) + abs(float(Vd[cand[1]] @ S["2142"]))
            c1 = abs(float(Vd[cand[1]] @ S["2134"])) + abs(float(Vd[cand[0]] @ S["2142"]))
            a_, b_ = (0, 1) if c0 >= c1 else (1, 0)
            res[o.label] = {"2134": eta[a_], "2142": eta[b_]}
        else:
            res[o.label] = {"mean": float(np.mean(eta)), "half": 0.25 * abs(eta[1] - eta[0])}
    return res, mac, ff.base.rho * (km.volume - kd.volume) * 1e6


def tape_H(points):
    """points: [(r, θ_FEM, 질량 kg)] → 관측량별 질량점 1차 섭동 행렬(η)."""
    H = {o.label: np.zeros((o.deg, o.deg)) for o in obs}
    idx_top = np.nonzero(top_v)[0]
    for r0, th0, m in points:
        hw = TAPE_FOOT[round(m * 1e6, 2)]
        d_r = fm.r_v[idx_top] - r0
        d_t = r0 * wrap(th_v[idx_top] - th0)
        sel = idx_top[(np.abs(d_r) <= hw) & (np.abs(d_t) <= hw)]
        if sel.size == 0:
            sel = idx_top[[int(np.argmin(d_r ** 2 + d_t ** 2))]]
        for o in obs:
            U = md.vshape[o.members].reshape(o.deg, -1, 3)[:, sel, :]          # (deg, n, 3)
            G = np.einsum("anc,bnc->ab", U, U) / sel.size
            H[o.label] += -m * G
    return H


rows = []
# ---------------------------------------------------------------- 창(1차 + 재해석)
r_B = r_of(0.9)
k_L1 = [3]
WINDOWS = (("L1a", k_L1, 0.03e-3), ("L1", k_L1, 0.06e-3), ("L6", range(6), 0.06e-3))
if TAPES_ONLY:
    old = pd.read_csv(OUT)
    rows = old[old.state.isin([w[0] for w in WINDOWS])].to_dict("records")
for state, ks, depth in (() if TAPES_ONLY else WINDOWS):
    p_unit = window_p([30.0], ks, 0.02e-3)
    H, dm = lin_window(p_unit)
    H = {k_: v * (depth / 0.02e-3) for k_, v in H.items()}
    res, mac, dm_res = resolved(window_p([30.0], ks, depth))
    rows += pair_rows(state, H, dm_res, res, mac)
    print(f"{state}: 1차 기준 {dm * depth / 0.02e-3:.1f} mg, 재해석 {dm_res:.1f} mg ({time.time() - t0:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)

# ---------------------------------------------------------------- 테이프(질량점 1차)
r_T = r_of(0.95)
th_sec = lambda k, rel, r: cam(r) + math.radians(60.0 * k + rel)
TAPES = {
    "T-a": [(r_T, th_sec(5, 0.0, r_T), 30e-6)],
    "T-b": [(r_T, th_sec(4, 14.1, r_T), 30e-6)],
    "T-c": [(r_T, th_sec(3, 30.0, r_T), 10e-6)],
    "M75": [(r, th_sec(k, 40.0, r), 6.25e-6) for r in (33.2e-3, 35.4e-3) for k in range(6)],
}
TAPES["M150"] = TAPES["M75"] + [(r, th_sec(k, 20.0, r), 6.25e-6) for r in (33.2e-3, 35.4e-3) for k in range(6)]


def strip(k, rel, m=20e-6, length=6.4e-3, n=4, r_c=34.3e-3):
    """깃 선과 나란한 띠(11-12: 작은 조각은 붙이기 어렵다 — 사용자). 같은 각의 두 점(r 33.2·35.4)을 잇고 중심은 r 34.3.
    20 mg = 폭 2 mm·길이 6.4 mm(면밀도 1.56 mg/mm² 어림) → 띠 방향 1.6 mm 간격의 2 × 2 mm 칸 n개(m/n씩)."""
    g = r_c * (cam(r_c + 1e-4) - cam(r_c - 1e-4)) / 2e-4          # 깃 선의 r·dθ/dr → 띠 방향의 반지름 성분 cos ψ
    cpsi = 1.0 / math.sqrt(1.0 + g * g)
    s = (np.arange(n) - (n - 1) / 2) * length / n
    return [(r_c + si * cpsi, th_sec(k, rel, r_c + si * cpsi), m / n) for si in s]


TAPES["M120"] = [p for k in range(6) for p in strip(k, 40.0)]
TAPES["M240"] = TAPES["M120"] + [p for k in range(6) for p in strip(k, 20.0)]
print(f"띠: 깃 선 기울기(반지름 방향에서) {math.degrees(math.acos(abs(TAPES['M120'][1][0] - TAPES['M120'][0][0]) / 1.6e-3)):.1f}°, "
      f"칸 중심 r {', '.join(f'{p[0] * 1e3:.2f}' for p in TAPES['M120'][:4])} mm", flush=True)
for state, pts in TAPES.items():
    H = tape_H(pts)
    rows += pair_rows(state, H, -1e6 * sum(m for *_, m in pts))
    print(f"{state}: 붙인 질량 {1e6 * sum(m for *_, m in pts):.2f} mg ({time.time() - t0:.0f}s)", flush=True)
# 대조: 8° 띠 계산(s3_local R, 10 mg당 c: 깃 위 −0.053, 깃 사이 −0.122 %)과 질량점(10 mg)의 배 성분
for rel, ref in ((0.0, -0.0530), (30.0, -0.1223)):
    h = tape_H([(r_T, th_sec(0, rel, r_T), 10e-6)])["h2:m2n0"]
    print(f"대조 질량점 10 mg @{rel:.0f}°: 배 성분 {50 * np.linalg.eigvalsh(h)[0]:+.4f} % (8° 띠 {ref:+.4f} %)", flush=True)
pd.DataFrame(rows).to_csv(OUT, index=False)
print(f"→ {OUT} ({time.time() - t0:.0f}s)")
