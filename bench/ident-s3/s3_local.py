"""국부 손상 시험 사전 계산(42번) — G3 자유-자유 FEM(32×16)에서 앞판 바깥면 국부 포켓(부채꼴)의 1차 응답.

360° 홈은 축대칭이라 짝(축퇴 쌍)의 평균만 움직인다. 국부 포켓은 짝을 가른다. 그래서 짝은 자취 평균만이 아니라
2×2 섭동 행렬 H = Φᵀ(δK − λδM)Φ/λ 전체를 낸다(Φ = 건전 짝 두 성분, 질량 정규).
  - 강성 몫 H_K = Φᵀ δK Φ/λ, 질량 몫 H_M = −Φᵀ δM Φ (깎으면 δM < 0 → H_M > 0)
  - 기저 (c, s): H_M의 고유벡터 — c = 포켓 자리 운동에너지가 큰 성분(배), s = 작은 성분(마디 쪽)
  - 실물 짝은 이미 0.40 % 갈라져 방향이 고정돼 있다. 포켓 중심을 한 성분의 배에 맞추면 그 성분이 H_cc, 다른 성분이 H_ss만큼 움직인다.
질량 더하기(테이프)는 같은 자리 포켓의 질량 몫에 −(붙인 질량/깎은 질량)을 곱해 얻는다(얇은 판, 면외 운동 주도).
η는 고유값 상대 변화(Δf/f ≈ η/2). 깊이 0.02 mm, 1차(깊이에 선형).

과제(task)마다 따로 돌린다(병렬 실행용): B(ξ0.8–1.0) · C(ξ0.2–0.4) · MR(ξ0.5–0.7 + 테두리 테이프 자리 ξ0.9–1.0)
  · KEY:<곳> = 핵심 경우만, 빠진 질량(실측 41.3186 g − FEM)을 그곳에 비구조 질량으로 더한 모델(41번 규약, 곳 = all·vanes 등)
출력: fe-nominal/s3_local/local_<task>.csv(':'는 '_')  — 줄 = (경우, 관측량)
사용: PYTHONPATH=/home/super/project-paper3 python s3_local.py <task>"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from impeller_fingerprint import geometry as geo
from impeller_fingerprint.rev1 import n1_assembly as n1
from impeller_fingerprint.rev1 import n1_mesh as nm
from impeller_fingerprint.rev1 import n6_freefree as n6

TASK = sys.argv[1] if len(sys.argv) > 1 else "B"
SET, _, WHERE = TASK.partition(":")          # "KEY:all" = 핵심 경우만, 빠진 질량을 그곳(41번과 같은 규약)에 더한 모델
OUT = Path(__file__).resolve().parents[1] / "fe-nominal" / "s3_local"
OUT.mkdir(parents=True, exist_ok=True)
assert geo.ASBUILT.gap_in is None, "G3(간격 일정)로 돌린다 — PAPER3_GEOM을 비울 것"
DEPTH = 0.02e-3
PROBE = 0.05
#: 과제 → [(띠 이름, ξ 아래, ξ 위, 각폭 목록[°], 깃 기준 위치 목록[°])]. 위치 0° = 깃 위, 30° = 베이 가운데.
#: 각폭 360°는 같은 띠의 축대칭 고리(대조).
CASES = {
    "B": [("B", 0.8, 1.0, (20.0, 30.0, 60.0), (0.0, 15.0, 30.0, 45.0)), ("B", 0.8, 1.0, (360.0,), (0.0,))],
    "C": [("C", 0.2, 0.4, (20.0, 30.0, 60.0), (0.0, 15.0, 30.0, 45.0)), ("C", 0.2, 0.4, (360.0,), (0.0,))],
    "MR": [("M", 0.5, 0.7, (30.0, 60.0), (0.0, 30.0)), ("M", 0.5, 0.7, (360.0,), (0.0,)),
           ("R", 0.9, 1.0, (8.0, 16.0), (0.0, 15.0, 30.0, 45.0)), ("R", 0.9, 1.0, (360.0,), (0.0,))],
    "KEY": [("B", 0.8, 1.0, (30.0, 360.0), (0.0, 15.0, 30.0, 45.0)), ("C", 0.2, 0.4, (30.0,), (0.0, 30.0)),
            ("C", 0.2, 0.4, (360.0,), (0.0,)), ("R", 0.9, 1.0, (8.0,), (0.0, 15.0, 30.0, 45.0))],
}[SET]

t0 = time.time()
ff = n6.FFSpec(base=nm.FitSpec(n_r=32, n_bay=16, wrap_deg=geo.ASBUILT.wrap_deg), n_hub=6, n_z_collar=4,
               collar_h=geo.ASBUILT.collar_h, clamp_r=None)
fm = n6.build(ff)
km = n6.assemble(fm)
ndof = km.K.shape[0]
tx = np.zeros(ndof); tx[0::3] = 1.0
M_MEAS = geo.ASBUILT.mass_g * 1e-3
tag = {"where": WHERE or "nominal", "alpha": 0.0}
if WHERE:                                    # 41번 s3_km_split.py mass:<곳>과 같은 규약: M' = M + α·M_곳
    masks = n6.region_masks(fm)
    cen = fm.coors[fm.conn].mean(axis=1)
    rc = np.hypot(cen[:, 0], cen[:, 1])
    cm = {"all": np.ones(len(fm.conn), bool), "front": masks["front"], "back": masks["back_main"] | masks["hub"],
          "hub": masks["hub"], "vanes": masks["vanes"], "collar": masks["collar"], "rim": rc >= ff.b - 1.0e-3}[WHERE]
    kr = n6.assemble(fm, cell_mask=cm)
    m_fem = float(tx @ (km.M @ tx))
    alpha = (M_MEAS - m_fem) / float(tx @ (kr.M @ tx))
    km = n1.KM(km.K, (km.M + alpha * kr.M).tocsc(), km.make_full, km.n_vertex, km.volume)
    tag["alpha"] = alpha
    print(f"[{TASK}] 빠진 질량 {(M_MEAS - m_fem) * 1e3:.3f} g을 {WHERE}에 더함 (α {alpha:.3f})", flush=True)
md, _ = n6.solve(km, 80, n_rigid=ff.n_rigid)
obs = n6.classify(fm, md)
print(f"[{TASK}] {ndof} DOF, 건전 {md.freqs[0]:.1f}–{md.freqs[-1]:.1f} Hz, 관측량 {len(obs)} ({time.time() - t0:.0f}s)",
      flush=True)

th_v = np.arctan2(fm.coors[:, 1], fm.coors[:, 0])
xi_v = (fm.r_v - ff.a) / ff.extent
z_in, z_top = ff.z_front
front_v = (fm.coors[:, 2] >= z_in - 1e-12) & (fm.coors[:, 2] <= z_top + 1e-12) & (fm.r_v >= ff.a - 1e-12)


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def pocket_p(xlo, xhi, width_deg, rel_deg):
    """부채꼴 포켓 깊이비(절점). 중심각 = 띠 가운데 반지름에서 깃 0 캠버 + rel."""
    r_c = ff.a + 0.5 * (xlo + xhi) * ff.extent
    th_c = float(ff.base.camber(np.array([r_c]))[0]) + math.radians(rel_deg)
    inb = front_v & (xi_v >= xlo - 1e-9) & (xi_v <= xhi + 1e-9)
    if width_deg < 359.0:
        inb &= np.abs(wrap(th_v - th_c)) <= 0.5 * math.radians(width_deg) + 1e-9
    return np.where(inb, DEPTH / ff.base.t_front, 0.0), th_c, r_c


def one_case(band, xlo, xhi, wdeg, rel):
    p, th_c, r_c = pocket_p(xlo, xhi, wdeg, rel)
    kp = n6.assemble(fm, n6.morph_groove(fm, p * PROBE))
    kn = n6.assemble(fm, n6.morph_groove(fm, -p * PROBE))
    dK = (kp.K - kn.K) * (0.5 / PROBE)
    dM = (kp.M - kn.M) * (0.5 / PROBE)
    dm_mg = -float(tx @ (dM @ tx)) * 1e6
    cam_lo, cam_hi = (float(v) for v in ff.base.camber(np.array([ff.a + xlo * ff.extent, ff.a + xhi * ff.extent])))
    rows = []
    for o in obs:
        Ph = md.phi[:, o.members]
        lam = float(np.mean(md.lam[o.members]))
        HK = Ph.T @ (dK @ Ph) / lam
        HM = -(Ph.T @ (dM @ Ph))
        HK = 0.5 * (HK + HK.T)
        HM = 0.5 * (HM + HM.T)
        r = {"task": TASK, **tag, "band": band, "xi_lo": xlo, "xi_hi": xhi, "width_deg": wdeg, "rel_deg": rel,
             "theta_c_deg": math.degrees(th_c), "r_c_mm": r_c * 1e3,
             "camber_span_deg": math.degrees(abs(cam_hi - cam_lo)), "dm_mg": dm_mg,
             "label": o.label, "h": o.h, "deg": o.deg, "f_Hz": o.f_Hz, "resolvable": o.resolvable,
             "front_share": np.nan}
        if o.deg == 1:
            r.update(eta_K=float(HK[0, 0]), eta_M=float(HM[0, 0]), eta=float(HK[0, 0] + HM[0, 0]))
        else:
            H = HK + HM
            wM, VM = np.linalg.eigh(HM)
            c, s = VM[:, 1], VM[:, 0]           # c = 질량 몫이 큰 성분(포켓 자리 배)
            wH = np.linalg.eigvalsh(H)
            wK = np.linalg.eigvalsh(HK)
            r.update(eta_K=0.5 * float(np.trace(HK)), eta_M=0.5 * float(np.trace(HM)), eta=0.5 * float(np.trace(H)),
                     K_cc=float(c @ HK @ c), K_ss=float(s @ HK @ s), K_cs=float(c @ HK @ s),
                     M_cc=float(wM[1]), M_ss=float(wM[0]),
                     H_cc=float(c @ H @ c), H_ss=float(s @ H @ s), H_cs=float(c @ H @ s),
                     eig_lo=float(wH[0]), eig_hi=float(wH[1]), split=float(wH[1] - wH[0]),
                     K_eig_lo=float(wK[0]), K_eig_hi=float(wK[1]))
        rows.append(r)
    return rows


rows = []
for band, xlo, xhi, widths, rels in CASES:
    for wdeg in widths:
        for rel in rels:
            rows += one_case(band, xlo, xhi, wdeg, rel)
            print(f"[{TASK}] {band} ξ{xlo}–{xhi} {wdeg:.0f}° @{rel:.0f}° ({time.time() - t0:.0f}s)", flush=True)
            pd.DataFrame(rows).to_csv(OUT / f"local_{TASK.replace(':', '_')}.csv", index=False)
json.dump({"task": TASK, "seconds": time.time() - t0}, open(OUT / f"local_{TASK.replace(':', '_')}.done.json", "w"))
print(f"[{TASK}] 끝 ({time.time() - t0:.0f}s)", flush=True)
