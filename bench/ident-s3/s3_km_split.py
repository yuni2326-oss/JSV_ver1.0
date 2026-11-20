"""흠집(홈) 시험 사전 계산(41번, 전략 문서 반영) — G3 자유-자유 FEM(32×16) 1차 섭동을 강성 몫·질량 몫으로 나눈다.

과제(task)마다 따로 돌린다(병렬 실행용):
  nominal     건전 G3 그대로. (1) 홈 띠 ξ0.3·0.6·0.85·0.9(폭 0.2·span) 0.02 mm의 강성 몫 η_K·질량 몫 η_M,
              (2) 앞판 절점 고리 하나씩 0.02 mm 깎는 반지름 지도(η_K·η_M 밀도, 1/mm).
  nominal_extra  건전 G3, 테두리 쪽 좁은 띠 rim2(ξ0.9–1.0)·rim1(ξ0.95–1.0).
  mass:<곳>   FEM과 실측 질량 차(41.3186 g − FEM)를 비구조 질량으로 그곳에 더해(M' = M + α·M_곳) 건전 모드를 다시 풀고,
              같은 홈(띠 전부 + rim2·rim1, 0.02 mm, 구조 재료만 깎임)의 η_K·η_M과 반지름 지도를 낸다.
              곳 = all(전체 균일)·front(앞판)·back(뒤판+허브)·hub(허브)·vanes(깃)·collar(흡입구 테)·rim(바깥 1 mm 고리)
η는 고유값 상대 변화(Δf/f ≈ η/2). 쌍은 자취 평균(쌍 평균 이동).
출력: fe-nominal/s3_km/ nominal.csv·ring_map.csv·nominal_extra.csv·mass_<곳>.csv·ring_map_<곳>.csv.
사용: PYTHONPATH=/home/super/project-paper3 python s3_km_split.py <task>   (과제 하나 약 6–15분)"""
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

TASK = sys.argv[1] if len(sys.argv) > 1 else "nominal"
OUT = Path(__file__).resolve().parents[1] / "fe-nominal" / "s3_km"
OUT.mkdir(parents=True, exist_ok=True)
assert geo.ASBUILT.gap_in is None, "G3(간격 일정)로 돌린다 — PAPER3_GEOM을 비울 것"
M_MEAS = geo.ASBUILT.mass_g * 1e-3          # 2번 시편 분석저울 [kg]
DEPTH = 0.02e-3
PROBE = 0.05
BANDS = (0.3, 0.6, 0.85, 0.9)
#: 이름 → (중심 ξ, 폭/span). rim2 = ξ0.9–1.0(2.16 mm), rim1 = ξ0.95–1.0(1.08 mm) — 테두리 쪽 좁은 띠
EXTRA = {"rim2": (0.95, 0.1), "rim1": (0.975, 0.05)}

t0 = time.time()
ff = n6.FFSpec(base=nm.FitSpec(n_r=32, n_bay=16, wrap_deg=geo.ASBUILT.wrap_deg), n_hub=6, n_z_collar=4,
               collar_h=geo.ASBUILT.collar_h, clamp_r=None)
fm = n6.build(ff)
km = n6.assemble(fm)
ndof = km.K.shape[0]
tx = np.zeros(ndof); tx[0::3] = 1.0          # x 병진(벡터장 자유도는 절점마다 x·y·z 순)
m_fem = float(tx @ (km.M @ tx))
print(f"[{TASK}] 조립 {ndof} DOF, FEM 질량 {m_fem * 1e3:.3f} g (부피×ρ {ff.base.rho * km.volume * 1e3:.3f} g) "
      f"({time.time() - t0:.0f}s)", flush=True)


def split(md, obs, p):
    """깊이비 p(절점)의 1차 η_K·η_M(중심차분 ±PROBE)."""
    kp = n6.assemble(fm, n6.morph_groove(fm, p * PROBE))
    kn = n6.assemble(fm, n6.morph_groove(fm, -p * PROBE))
    dK = (kp.K - kn.K) * (0.5 / PROBE)
    dM = (kp.M - kn.M) * (0.5 / PROBE)
    eK = n1.first_order_eta(md, obs, dK, dM * 0.0)
    eM = n1.first_order_eta(md, obs, dK * 0.0, dM)
    dm = float(tx @ (dM @ tx))
    return eK, eM, dm


def rows_for(md, obs, tag, p, extra=None):
    eK, eM, dm = split(md, obs, p)
    out = []
    for j, o in enumerate(obs):
        r = {"task": TASK, "groove": tag, "label": o.label, "h": o.h, "deg": o.deg, "f_Hz": o.f_Hz,
             "resolvable": o.resolvable, "eta_K": eK[j], "eta_M": eM[j], "eta": eK[j] + eM[j],
             "dm_mg": -dm * 1e6}
        r.update(extra or {})
        out.append(r)
    return out


def ring_map(md, obs, extra=None):
    """반지름 지도: 앞판 주 고리 절점(테 바깥 a + t_c ~ b) 하나씩 — 모자 함수 깊이(넓이 = 깊이 × Δr, 끝 절점은 절반)."""
    rn = np.linspace(ff.a + ff.collar_t, ff.b, ff.base.n_r)
    dr = rn[1] - rn[0]
    zin, _ = ff.z_front
    front = (fm.coors[:, 2] >= zin - 1e-12) & (fm.r_v >= ff.a - 1e-12)
    ring = []
    for i, ri in enumerate(rn):
        sel = front & np.isclose(fm.r_v, ri, atol=1e-8)
        p = np.where(sel, DEPTH / ff.base.t_front, 0.0)
        w = dr * (0.5 if i in (0, len(rn) - 1) else 1.0)
        ring += rows_for(md, obs, f"node{i}", p, {"r_mm": ri * 1e3, "xi": (ri - ff.a) / ff.extent, "w_mm": w * 1e3,
                                                  **(extra or {})})
        if i % 8 == 0:
            print(f"[{TASK}] 고리 {i + 1}/{len(rn)} ({time.time() - t0:.0f}s)", flush=True)
    return ring


def extra_bands(md, obs, extra=None):
    rows = []
    for nm_, (xc, wf) in EXTRA.items():
        rows += rows_for(md, obs, nm_, n6.groove_depth_ratio(fm.r_v, ff, xc, DEPTH, wf * ff.extent), extra)
    return rows


if TASK == "nominal":
    md, _ = n6.solve(km, 80, n_rigid=ff.n_rigid)
    obs = n6.classify(fm, md)
    print(f"[{TASK}] 건전 {md.freqs[0]:.1f}–{md.freqs[-1]:.1f} Hz, 관측량 {len(obs)} ({time.time() - t0:.0f}s)", flush=True)
    rows = []
    for xi in BANDS:
        rows += rows_for(md, obs, f"xi{xi}", n6.groove_depth_ratio(fm.r_v, ff, xi, DEPTH))
        print(f"[{TASK}] 띠 ξ{xi} ({time.time() - t0:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "nominal.csv", index=False)
    pd.DataFrame(ring_map(md, obs)).to_csv(OUT / "ring_map.csv", index=False)
elif TASK == "nominal_extra":
    md, _ = n6.solve(km, 80, n_rigid=ff.n_rigid)
    obs = n6.classify(fm, md)
    pd.DataFrame(extra_bands(md, obs)).to_csv(OUT / "nominal_extra.csv", index=False)
else:
    where = TASK.split(":", 1)[1]
    masks = n6.region_masks(fm)
    cen = fm.coors[fm.conn].mean(axis=1)
    rc = np.hypot(cen[:, 0], cen[:, 1])
    sel = {"all": np.ones(len(fm.conn), bool), "front": masks["front"], "back": masks["back_main"] | masks["hub"],
           "hub": masks["hub"], "vanes": masks["vanes"], "collar": masks["collar"],
           "rim": rc >= ff.b - 1.0e-3}[where]
    kr = n6.assemble(fm, cell_mask=sel)
    m_r = float(tx @ (kr.M @ tx))
    alpha = (M_MEAS - m_fem) / m_r
    kmm = n1.KM(km.K, (km.M + alpha * kr.M).tocsc(), km.make_full, km.n_vertex, km.volume)
    print(f"[{TASK}] 구역 질량 {m_r * 1e3:.3f} g, 더한 질량 {(M_MEAS - m_fem) * 1e3:.3f} g (α {alpha:.3f}), "
          f"합 {float(tx @ (kmm.M @ tx)) * 1e3:.3f} g ({time.time() - t0:.0f}s)", flush=True)
    md, _ = n6.solve(kmm, 80, n_rigid=ff.n_rigid)
    obs = n6.classify(fm, md)
    print(f"[{TASK}] 건전 {md.freqs[0]:.1f}–{md.freqs[-1]:.1f} Hz, 관측량 {len(obs)} ({time.time() - t0:.0f}s)", flush=True)
    tag = {"where": where, "m_region_g": m_r * 1e3, "alpha": alpha}
    rows = []
    for xi in BANDS:
        rows += rows_for(md, obs, f"xi{xi}", n6.groove_depth_ratio(fm.r_v, ff, xi, DEPTH), tag)
    rows += extra_bands(md, obs, tag)
    pd.DataFrame(rows).to_csv(OUT / f"mass_{where}.csv", index=False)
    print(f"[{TASK}] 띠 끝 ({time.time() - t0:.0f}s)", flush=True)
    pd.DataFrame(ring_map(md, obs, tag)).to_csv(OUT / f"ring_map_{where}.csv", index=False)
json.dump({"task": TASK, "seconds": time.time() - t0}, open(OUT / f"{TASK.replace(':', '_')}.done.json", "w"))
print(f"[{TASK}] 끝 ({time.time() - t0:.0f}s)", flush=True)
