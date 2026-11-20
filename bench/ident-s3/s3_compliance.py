"""41번 — 앞판을 공구로 누를 때 얼마나 휘나: G3 FEM(32×16)에서 허브를 고정하고 앞판 바깥면에 축방향 힘 1 N을
지름 약 2 mm 조각에 고르게 걸어 정적 처짐을 푼다. 홈 띠 B(r 34.3 mm)·C(r 21.4 mm)·테두리(r b)에서,
깃 바로 위와 깃 사이 한가운데 두 곳씩.
사용: PYTHONPATH=/home/super/project-paper3 python s3_compliance.py → fe-nominal/s3_km/compliance.csv"""
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.linalg import splu

from impeller_fingerprint import geometry as geo
from impeller_fingerprint.rev1 import n1_mesh as nm
from impeller_fingerprint.rev1 import n6_freefree as n6

OUT = Path(__file__).resolve().parents[1] / "fe-nominal" / "s3_km"
OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time()
ff = n6.FFSpec(base=nm.FitSpec(n_r=32, n_bay=16, wrap_deg=geo.ASBUILT.wrap_deg), n_hub=6, n_z_collar=4,
               collar_h=geo.ASBUILT.collar_h, clamp_r=geo.ASBUILT.r_bore + 2.0e-3)
fm = n6.build(ff)
km = n6.assemble(fm)
nv = km.n_vertex
zin, ztop = ff.z_front
top = np.nonzero(np.isclose(fm.coors[:, 2], ztop, atol=1e-9) & (fm.r_v >= ff.a + ff.collar_t - 1e-9))[0]
th = np.arctan2(fm.coors[:, 1], fm.coors[:, 0])
vane = fm.info["cell_kind"] == 1
cen = fm.coors[fm.conn].mean(axis=1)
vr, vth = np.hypot(cen[vane, 0], cen[vane, 1]), np.arctan2(cen[vane, 1], cen[vane, 0])
lu = splu(km.K.tocsc(), permc_spec="MMD_AT_PLUS_A", options={"SymmetricMode": True})
probe = km.make_full(np.arange(km.K.shape[0], dtype=float) + 1.0)[: nv * 3]   # 꼭짓점 자유도 → 축소 번호 + 1(고정은 0)
print(f"조립·분해 {km.K.shape[0]} DOF, 허브 고정 r ≤ {ff.clamp_r * 1e3:.2f} mm ({time.time() - t0:.0f}s)", flush=True)


def gap_to_vane(i):
    near = np.abs(vr - fm.r_v[i]) < 0.8e-3
    d = np.angle(np.exp(1j * (vth[near] - th[i])))
    return float(np.min(np.abs(d))) * fm.r_v[i]          # 가장 가까운 깃까지 원주 거리 [m]


rows = []
for tag, r_t in (("B", 34.3e-3), ("C", 21.36e-3), ("rim", ff.b)):
    cand = top[np.abs(fm.r_v[top] - r_t) < 0.4e-3]
    gaps = np.array([gap_to_vane(i) for i in cand])
    for where, i0 in (("깃 위", cand[np.argmin(gaps)]), ("깃 사이", cand[np.argmax(gaps)])):
        patch = top[np.hypot(*(fm.coors[top, :2] - fm.coors[i0, :2]).T) <= 1.0e-3]
        f = np.zeros(nv * 3)
        f[patch * 3 + 2] = -1.0 / len(patch)              # 축방향(앞판을 누르는 쪽) 1 N
        red = np.zeros(km.K.shape[0])                      # 꼭짓점 하중만 쓴다
        ok = probe > 0
        red[(probe[ok] - 1).astype(int)] = f[ok]
        u = lu.solve(red)
        uf = km.make_full(u)[: nv * 3].reshape(-1, 3)
        w = float(-uf[patch, 2].mean())
        rows.append({"band": tag, "r_mm": fm.r_v[i0] * 1e3, "where": where, "gap_to_vane_mm": gap_to_vane(i0) * 1e3,
                     "patch_nodes": len(patch), "w_um_per_N": w * 1e6})
        print(f"{tag} r {fm.r_v[i0] * 1e3:.2f} mm {where}(깃까지 {gap_to_vane(i0) * 1e3:.1f} mm): "
              f"{w * 1e6:.1f} µm/N ({time.time() - t0:.0f}s)", flush=True)
pd.DataFrame(rows).to_csv(OUT / "compliance.csv", index=False)
print(f"→ {OUT / 'compliance.csv'}")
