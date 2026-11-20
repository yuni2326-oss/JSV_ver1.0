"""2번 시편 치수(10-30 저녁 실측)로 자유-자유 FE — 앞판 두께만 미측정이라 0.375 / 0.58 mm 두 경우.
공통: 외경 72.93(b 36.465), 보어 12.62(r 6.31), 흡입구 테 바깥지름 30.82·벽 0.52(a = 15.41 − 0.52 = 14.89), 테 높이 5.0(사진 추정),
뒤판 0.58, 깃 1.0, 간격 3.05(출구 끝 2.95–3.00, 안쪽 더 넓음), wrap 62°, r_vin 15, 32×16.
사용: s2e_fe.py <t_front_mm> [n_r n_bay]"""
import sys, time, json
sys.path.insert(0, "/home/super/project-paper3")
import numpy as np
import pandas as pd
from impeller_fingerprint.rev1 import n1_mesh as nm, n6_freefree as n6

tf = float(sys.argv[1]) * 1e-3
n_r, n_bay = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (32, 16)
OUT = "/tmp/claude-1000/-home-super-project/efb54ce3-bcf9-473d-9dc8-eb835c653517/scratchpad/impact1022/out/fe_s2/"
base = nm.FitSpec(n_r=n_r, n_bay=n_bay, a=0.01489, b=0.036465, t_front=tf, t_back=0.00058, channel=0.00305,
                  t_vane=0.0010, wrap_deg=62.0, r_vane_in=0.015)
ff = n6.FFSpec(base=base, r_bore=0.00631, collar_t=0.00052, collar_h=0.0050)
t0 = time.time()
fm = n6.build(ff)
km = n6.assemble(fm)
md, rigid = n6.solve(km, 60, n_rigid=6)
obs = n6.classify(fm, md)
# 구역 질량
masks = n6.region_masks(fm)
X, conn = fm.coors, fm.conn
g = np.array([-1, 1]) / np.sqrt(3)
xi = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)
gp = np.array([[p, q, r] for p in g for q in g for r in g])
dN = np.stack([xi[None, :, k] * np.prod(1 + gp[:, None, [m for m in range(3) if m != k]] * xi[None, :, [m for m in range(3) if m != k]], axis=2) / 8
               for k in range(3)], axis=2)
vol = np.abs(np.linalg.det(np.einsum("gnk,cnj->cgkj", dN, X[conn]))).sum(1)
rm = {k: float(base.rho * vol[m].sum() * 1e3) for k, m in masks.items()}
tag = f"tf{int(round(tf * 1e6))}_{n_r}x{n_bay}"
rows = [{"label": o.label, "h": o.h, "f_Hz": o.f_Hz, "m_dom": o.m_dom, "n_rad": o.n_rad,
         "members": "|".join(str(i) for i in o.members)} for o in obs]
np.savez_compressed(OUT + f"shapes_{tag}.npz", coors=fm.coors, conn=fm.conn, freqs=md.freqs, vshape=md.vshape.astype(np.float32),
                    grid_idx=fm.info["grid_idx"], cell_kind=fm.info["cell_kind"], cell_hub=fm.info["cell_hub"])
pd.DataFrame(rows).to_csv(OUT + f"modes_{tag}.csv", index=False)
meta = {"t_front_mm": tf * 1e3, "mass_g": base.rho * km.volume * 1e3, "region_mass_g": rm, "ndof": md.ndof,
        "rigid_Hz": rigid.tolist(), "seconds": time.time() - t0}
json.dump(meta, open(OUT + f"meta_{tag}.json", "w"), indent=1)
print(tag, json.dumps(meta, ensure_ascii=False))
print(pd.DataFrame(rows).head(30).to_string())
