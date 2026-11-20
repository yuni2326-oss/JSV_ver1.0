"""s2g_fit.py가 찾은 치수·질량 배치를 FE로 직접 확인 — 비구조 질량은 구역 질량행렬을 키워 넣는다(M + Σ_r x_r M_r).
사용: s2g_confirm.py <tag> [tf= tb= tv= ch= colh= colt= (mm)] [front= back_main= hub= vanes= collar= (g, 더할 질량)] [uniform=1]
uniform=1 이면 41.3186 g까지 모자란 질량을 밀도 비례로(모든 구역 같은 비율) 넣는다."""
import sys, time, json
sys.path.insert(0, "/home/super/project-paper3")
import numpy as np
import pandas as pd
from impeller_fingerprint.rev1 import n1_mesh as nm, n6_freefree as n6

tag = sys.argv[1]
p = {"tf": 0.40, "tb": 0.58, "tv": 1.0, "ch": 3.05, "colh": 5.0, "colt": 0.52}
add = {"front": 0.0, "back_main": 0.0, "hub": 0.0, "vanes": 0.0, "collar": 0.0}
uniform = False
for kv in sys.argv[2:]:
    k, v = kv.split("=")
    if k in p:
        p[k] = float(v)
    elif k in add:
        add[k] = float(v)
    elif k == "uniform":
        uniform = bool(int(v))
    else:
        raise KeyError(k)
OUT = "/tmp/claude-1000/-home-super-project/efb54ce3-bcf9-473d-9dc8-eb835c653517/scratchpad/impact1022/out/fe_s2/"
a = 0.01541 - p["colt"] * 1e-3
base = nm.FitSpec(n_r=32, n_bay=16, a=a, b=0.036465, t_front=p["tf"] * 1e-3, t_back=p["tb"] * 1e-3, channel=p["ch"] * 1e-3,
                  t_vane=p["tv"] * 1e-3, wrap_deg=62.0, r_vane_in=0.015)
ff = n6.FFSpec(base=base, r_bore=0.00631, collar_t=p["colt"] * 1e-3, collar_h=p["colh"] * 1e-3)
t0 = time.time()
fm = n6.build(ff)
km = n6.assemble(fm)
m_model = base.rho * km.volume * 1e3
ck, hub = fm.info["cell_kind"], fm.info["cell_hub"]
REG = {"front": ck == 2, "back_main": (ck == 0) & ~hub, "hub": (ck == 0) & hub, "vanes": ck == 1, "collar": ck == 3}
M = km.M.copy()
x_used = {}
if uniform:
    x = (41.3186 - m_model) / m_model
    M = M * (1 + x)
    x_used = {k: x for k in REG}
else:
    ones = np.ones(km.M.shape[0])
    for k, dm in add.items():
        if dm <= 0:
            continue
        kr = n6.assemble(fm, cell_mask=REG[k])
        # 구역 질량[g] = 병진 강체 모드 한 방향의 질량: uᵀ M_r u, u = x방향 단위 이동(자유도 순서 x,y,z 교대)
        ux = np.zeros(km.M.shape[0]); ux[0::3] = 1.0
        m_r = float(ux @ (kr.M @ ux)) * 1e3
        x_used[k] = dm / m_r
        M = M + x_used[k] * kr.M
km2 = n1_KM = type(km)(km.K, M.tocsc(), km.make_full, km.n_vertex, km.volume)
md, rigid = n6.solve(km2, 60, n_rigid=6)
obs = n6.classify(fm, md)
rows = [{"label": o.label, "h": o.h, "f_Hz": o.f_Hz, "m_dom": o.m_dom, "n_rad": o.n_rad,
         "members": "|".join(str(i) for i in o.members)} for o in obs]
np.savez_compressed(OUT + f"shapes_{tag}.npz", coors=fm.coors, conn=fm.conn, freqs=md.freqs, vshape=md.vshape.astype(np.float32),
                    grid_idx=fm.info["grid_idx"], cell_kind=fm.info["cell_kind"], cell_hub=fm.info["cell_hub"])
pd.DataFrame(rows).to_csv(OUT + f"modes_{tag}.csv", index=False)
ux = np.zeros(km.M.shape[0]); ux[0::3] = 1.0
meta = {"params_mm": p, "added_g": add, "uniform": uniform, "x_used": x_used, "mass_model_g": m_model,
        "mass_total_g": float(ux @ (M @ ux)) * 1e3, "ndof": md.ndof, "rigid_Hz": rigid.tolist(), "seconds": time.time() - t0}
json.dump(meta, open(OUT + f"meta_{tag}.json", "w"), indent=1)
print(tag, json.dumps(meta, ensure_ascii=False))
