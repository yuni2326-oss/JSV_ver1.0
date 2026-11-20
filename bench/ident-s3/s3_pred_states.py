"""흠집(홈) 시험 상태별 예측 — G3 자유-자유 FEM(32×16)에서 상태마다 홈을 실제로 깎아 다시 푼다(재해석).

상태: B1·B2·B3 = 바깥 홈(ξ0.9, r 32.1–36.5 mm) 0.02·0.04·0.06 mm,
      C1·C2 = B3 위에 안쪽 홈(ξ0.3, r 19.2–23.5 mm) 0.02·0.04 mm를 더한 것(두 홈을 함께 깎아 푼다 — 35번의 '증분 합' 근사를 확인).
각 상태에서 1차(건전 상태 감도) 값도 함께 낸다(두 홈이면 1차 합).
출력: fe-nominal/s3_pred_states.csv — state, label, h, f_Hz(건전), df_pct_res, mac, df_pct_lin, f_state_Hz
사용: PYTHONPATH=/home/super/project-paper3 python s3_pred_states.py [n_r n_bay]   (기본 32 16, 약 40분)"""
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

n_r, n_bay = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (32, 16)
OUT = Path(__file__).resolve().parents[1] / "fe-nominal" / ("s3_pred_states.csv" if (n_r, n_bay) == (32, 16)
                                                            else f"s3_pred_states_{n_r}x{n_bay}.csv")
assert geo.ASBUILT.gap_in is None, "G3(간격 일정)로 돌린다 — PAPER3_GEOM을 비울 것"
STATES = [("B1", [(0.9, 0.02)]), ("B2", [(0.9, 0.04)]), ("B3", [(0.9, 0.06)]),
          ("C1", [(0.9, 0.06), (0.3, 0.02)]), ("C2", [(0.9, 0.06), (0.3, 0.04)])]

t0 = time.time()
ff = n6.FFSpec(base=nm.FitSpec(n_r=n_r, n_bay=n_bay, wrap_deg=geo.ASBUILT.wrap_deg), n_hub=6, n_z_collar=4,
               collar_h=geo.ASBUILT.collar_h, clamp_r=None)
fm = n6.build(ff)
km = n6.assemble(fm)
md, rigid = n6.solve(km, 80, n_rigid=ff.n_rigid)
obs = n6.classify(fm, md)
print(f"건전 {md.ndof} DOF, 탄성 {md.freqs[0]:.1f}–{md.freqs[-1]:.1f} Hz, 관측량 {len(obs)}, 질량 "
      f"{ff.base.rho * km.volume * 1e3:.3f} g ({time.time() - t0:.0f}s)", flush=True)

lin = {}
for xi, dep in sorted({g for _, gs in STATES for g in gs}):
    lin[(xi, dep)] = n6.groove_first_order(fm, md, obs, xi, dep * 1e-3)
    print(f"1차 ξ{xi} {dep} mm ({time.time() - t0:.0f}s)", flush=True)

rows = []
for state, grooves in STATES:
    ps = [n6.groove_depth_ratio(fm.r_v, ff, xi, dep * 1e-3) for xi, dep in grooves]
    assert not np.any(np.sum([q > 0 for q in ps], axis=0) > 1), "두 홈이 겹친다"
    p = np.sum(ps, axis=0)
    kd = n6.assemble(fm, n6.morph_groove(fm, p))
    mdd, _ = n6.solve(kd, md.phi.shape[1] + 6, n_rigid=ff.n_rigid)
    eta, mac = n1.track(md, mdd, obs)
    eta_lin = sum(lin[g] for g in grooves)
    dm = ff.base.rho * (km.volume - kd.volume) * 1e6
    for j, o in enumerate(obs):
        df = 100 * (math.sqrt(1 + eta[j]) - 1) if np.isfinite(eta[j]) else np.nan
        rows.append({"state": state, "grooves": " + ".join(f"ξ{xi} {dep} mm" for xi, dep in grooves),
                     "removed_mg": dm, "label": o.label, "h": o.h, "f_Hz": o.f_Hz, "resolvable": o.resolvable,
                     "df_pct_res": df, "mac": float(mac[j]),
                     "df_pct_lin": 100 * (math.sqrt(1 + eta_lin[j]) - 1),
                     "f_state_Hz": o.f_Hz * (1 + df / 100) if np.isfinite(df) else np.nan})
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"{state} 재해석 끝: 깎은 질량 {dm:.1f} mg ({time.time() - t0:.0f}s)", flush=True)
print(f"→ {OUT}")
