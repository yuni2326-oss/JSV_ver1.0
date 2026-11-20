"""2번 시편 치수 FE(앞판 0.40, 32×16)를 기준으로 치수 하나씩 바꾼 민감도 실행 — 치수 측정 오차로 측정 주파수를 맞출 수 있는지.
사용: s2g_sens.py <tag> [tf=mm] [tb=mm] [tv=mm] [ch=mm] [colh=mm] [colt=mm]
기준: 외경 72.93(b 36.465), 보어 12.62, 테 바깥지름 30.82·벽 0.52(a 14.89)·높이 5.0, 앞판 0.40, 뒤판 0.58, 깃 1.0, 간격 3.05, wrap 62°, r_vin 15."""
import sys, time, json
sys.path.insert(0, "/home/super/project-paper3")
import numpy as np
import pandas as pd
from impeller_fingerprint.rev1 import n1_mesh as nm, n6_freefree as n6

tag = sys.argv[1]
p = {"tf": 0.40, "tb": 0.58, "tv": 1.0, "ch": 3.05, "colh": 5.0, "colt": 0.52}
for kv in sys.argv[2:]:
    k, v = kv.split("=")
    assert k in p, k
    p[k] = float(v)
OUT = "/tmp/claude-1000/-home-super-project/efb54ce3-bcf9-473d-9dc8-eb835c653517/scratchpad/impact1022/out/fe_s2/"
a = 0.01541 - p["colt"] * 1e-3      # 테 바깥지름 30.82 고정, 벽을 바꾸면 안지름이 바뀐다
base = nm.FitSpec(n_r=32, n_bay=16, a=a, b=0.036465, t_front=p["tf"] * 1e-3, t_back=p["tb"] * 1e-3, channel=p["ch"] * 1e-3,
                  t_vane=p["tv"] * 1e-3, wrap_deg=62.0, r_vane_in=0.015)
ff = n6.FFSpec(base=base, r_bore=0.00631, collar_t=p["colt"] * 1e-3, collar_h=p["colh"] * 1e-3)
t0 = time.time()
fm = n6.build(ff)
km = n6.assemble(fm)
md, rigid = n6.solve(km, 60, n_rigid=6)
obs = n6.classify(fm, md)
rows = [{"label": o.label, "h": o.h, "f_Hz": o.f_Hz, "m_dom": o.m_dom, "n_rad": o.n_rad,
         "members": "|".join(str(i) for i in o.members)} for o in obs]
np.savez_compressed(OUT + f"shapes_{tag}.npz", coors=fm.coors, conn=fm.conn, freqs=md.freqs, vshape=md.vshape.astype(np.float32),
                    grid_idx=fm.info["grid_idx"], cell_kind=fm.info["cell_kind"], cell_hub=fm.info["cell_hub"])
pd.DataFrame(rows).to_csv(OUT + f"modes_{tag}.csv", index=False)
meta = {"params_mm": p, "mass_g": base.rho * km.volume * 1e3, "ndof": md.ndof, "rigid_Hz": rigid.tolist(), "seconds": time.time() - t0}
json.dump(meta, open(OUT + f"meta_{tag}.json", "w"), indent=1)
print(tag, json.dumps(meta, ensure_ascii=False))
