"""깃 높이(판 사이 간격) 테이퍼 점검 (사용자 가설 10-31) — 출구 끝 3.00 mm에서 안쪽으로 3.19 mm까지 점점 넓어진다.

근거: 판 두께는 표준(0.40/0.60)으로 일정한데, 2번 시편 전체 두께가 출구 끝 4.00 mm, 깊숙한 곳 4.19 mm였다
(1번 시편도 4.07 → 4.12 → 4.23 mm로 같은 경향).

모형: 뒤판은 평판 그대로 두고 간격을 c(r) = c_out + (c_in − c_out)·clip((b − r)/(b − r_in), 0, 1)로 만든다.
깃(채널) 절점은 높이 비례로 늘리고, 앞판·흡입구 테 절점은 Δc(r)만큼 올린다 → 앞판이 얕은 원뿔(경사 약 0.5°)이 된다.
격자 위상은 그대로라 대칭 판정·모드 분류·손상 절점 선택은 평판 좌표로 하고, K·M만 테이퍼 좌표로 조립한다
(`assemble`을 감싸 넘어오는 좌표에 테이퍼를 얹는다). 감육 깊이비는 그 반경의 간격으로 두 면판 식을 푼다.

사용:
  s2m_taper.py ff <tag> [ci=3.19] [co=3.00] [rin=14.89] [n_r=32] [n_bay=16] [side=0]  자유-자유(E2-I) 모드·질량 → out/fe_s2/
  side: 기울어지는 판 — 0 = 앞판(뒤판 평판, 기본), 1 = 뒤판(앞판 평판), 0.5 = 두 판이 반씩
  s2m_taper.py k6 <tag> [ci=3.19] [clamp=9.21] [n_r=24] [n_bay=12] [step=0.04]  설치 상태 커널(허브 고정, n6 kernels와 같은 격자)
  s2m_taper.py k1 <tag> [ci=3.19] [n_r=24] [n_bay=12] [step=0.02]               레일 지지 조립체 커널(안쪽 테 고정, n1 kernels와 같은 격자)
ci = c_in = c_out이면 테이퍼 없는 기준과 같다(검산용)."""
import sys, time, json
sys.path.insert(0, "/home/super/project-paper3")
import numpy as np
import pandas as pd
from impeller_fingerprint.rev1 import n1_mesh as nm, n6_freefree as n6, n1_assembly as n1, g1_section as g1

SCR = "/tmp/claude-1000/-home-super-project/efb54ce3-bcf9-473d-9dc8-eb835c653517/scratchpad/impact1022/out/"
D_FF = SCR + "fe_s2/"
D_K = SCR + "taper/"

mode, tag = sys.argv[1], sys.argv[2]
p = {"ci": 3.19, "co": 3.00, "rin": None, "n_r": None, "n_bay": None, "clamp": 9.21, "step": None, "side": 0.0}
for kv in sys.argv[3:]:
    k, v = kv.split("=")
    assert k in p, k
    p[k] = float(v)
n_r = int(p["n_r"] or (32 if mode == "ff" else 24))
n_bay = int(p["n_bay"] or (16 if mode == "ff" else 12))
step = p["step"] or (0.04 if mode == "k6" else 0.02)


def make_taper(fs, grid_idx, c_in, c_out, r_in, side=0.0):
    """→ (taper(X), c_of_r). 채널 절점은 높이 비율 j/n_zc, 앞판·테 절점은 1, 뒤판(허브 포함) 절점은 0에서 side를 빼
    Δc(r)에 곱해 z에 더한다(side 0 = 앞판이 원뿔, 1 = 뒤판이 원뿔). 간격 분포는 side와 무관하다."""
    nzs, nzc = fs.n_z_shroud, fs.n_z_channel
    iz = grid_idx[:, 2]
    frac = np.where(iz <= nzs, 0.0, np.where(iz <= nzs + nzc, (iz - nzs) / nzc, 1.0))

    def dc(r):
        return (c_in - c_out) * np.clip((fs.b - np.asarray(r, float)) / (fs.b - r_in), 0.0, 1.0)

    def taper(X):
        Y = X.copy()
        Y[:, 2] += (frac - side) * dc(np.hypot(X[:, 0], X[:, 1]))
        return Y

    return taper, (lambda r: c_out + float(dc(r)))


def depth_local(dk, r, tf, tb, c_of_r):
    """d_K → 깊이비 p를 그 반경의 간격으로(두 면판 식, 링별)."""
    out = np.zeros_like(dk)
    rr = np.round(r, 9)
    for ru in np.unique(rr[dk > 0]):
        sel = (rr == ru) & (dk > 0)
        out[sel] = g1.depth_from_dk(dk[sel], tf, tb, c_of_r(ru))
    return out


def patch_n6(taper, c_of_r):
    orig = n6.assemble

    def assemble_t(fm, coors=None, cell_mask=None, order=2):
        return orig(fm, taper(fm.coors if coors is None else coors), cell_mask, order)

    def gdr(fm, xi, s_bar, w=n1.W_GAUSS):
        fs = fm.spec.base
        r = fm.r_v
        dk = np.clip(n1.dK_profile(r, xi, s_bar, w, fs.a, fs.b), 0.0, None)
        return np.where(r >= fs.a - 1e-12, depth_local(dk, r, fs.t_front, fs.t_back, c_of_r), 0.0)

    n6.assemble, n6.gaussian_depth_ratio = assemble_t, gdr


def patch_n1(taper, c_of_r):
    orig = n1.assemble

    def assemble_t(am, coors=None, order=2):
        return orig(am, taper(am.coors if coors is None else coors), order)

    def dpa(am, r, xi_d, s_bar, w):
        dk = n1.dK_profile(r, xi_d, s_bar, w, am.spec.a, am.spec.b)
        return depth_local(dk, r, am.t_f, am.t_o, c_of_r)

    n1.assemble, n1.depth_profile_am = assemble_t, dpa


t0 = time.time()
ci, co = p["ci"] * 1e-3, p["co"] * 1e-3
if mode == "ff":
    a = 0.01541 - 0.00052
    base = nm.FitSpec(n_r=n_r, n_bay=n_bay, a=a, b=0.036465, t_front=0.40e-3, t_back=0.60e-3, channel=co,
                      t_vane=1.0e-3, wrap_deg=62.0, r_vane_in=0.015)
    ff = n6.FFSpec(base=base, r_bore=0.00631, collar_t=0.52e-3, collar_h=5.0e-3)
    fm = n6.build(ff)
    r_in = base.a if p["rin"] is None else p["rin"] * 1e-3
    taper, c_of_r = make_taper(base, fm.info["grid_idx"], ci, co, r_in, p["side"])
    patch_n6(taper, c_of_r)
    km = n6.assemble(fm)
    md, rigid = n6.solve(km, 60, n_rigid=6)
    obs = n6.classify(fm, md)
    rows = [{"label": o.label, "h": o.h, "f_Hz": o.f_Hz, "m_dom": o.m_dom, "n_rad": o.n_rad,
             "members": "|".join(str(i) for i in o.members)} for o in obs]
    np.savez_compressed(D_FF + f"shapes_{tag}.npz", coors=taper(fm.coors), conn=fm.conn, freqs=md.freqs,
                        vshape=md.vshape.astype(np.float32), grid_idx=fm.info["grid_idx"],
                        cell_kind=fm.info["cell_kind"], cell_hub=fm.info["cell_hub"])
    pd.DataFrame(rows).to_csv(D_FF + f"modes_{tag}.csv", index=False)
    meta = {"params_mm": {"tf": 0.40, "tb": 0.60, "tv": 1.0, "c_out": p["co"], "c_in": p["ci"], "r_in": r_in * 1e3,
                          "colh": 5.0, "colt": 0.52, "side": p["side"]},
            "taper": "linear c(r); side 0 = back plate flat, front plate + collar lifted; side 1 = front flat, back plate + hub lowered",
            "mass_g": base.rho * km.volume * 1e3, "ndof": md.ndof, "rigid_Hz": rigid.tolist(), "seconds": time.time() - t0}
    json.dump(meta, open(D_FF + f"meta_{tag}.json", "w"), indent=1)
    print(tag, json.dumps(meta, ensure_ascii=False))
    sys.exit(0)

import os
os.makedirs(D_K, exist_ok=True)
xi_grid = np.round(np.arange(0.0 if mode == "k6" else 0.02, (1.0 if mode == "k6" else 0.98) + 1e-9, step), 4)
if mode == "k6":
    base = nm.FitSpec(n_r=n_r, n_bay=n_bay, wrap_deg=62.0)          # G3 정본(ASBUILT)
    ff = n6.FFSpec(base=base, n_hub=6, n_z_collar=4, collar_h=5.0e-3, clamp_r=p["clamp"] * 1e-3)
    fm = n6.build(ff)
    r_in = base.a if p["rin"] is None else p["rin"] * 1e-3
    taper, c_of_r = make_taper(base, fm.info["grid_idx"], ci, co, r_in)
    patch_n6(taper, c_of_r)
    km = n6.assemble(fm)
    md, _ = n6.solve(km, 80, n_rigid=0)
    obs = n6.classify(fm, md)
    print(f"[k6] {tag} ndof={md.ndof} 탄성 {md.freqs[0]:.1f}–{md.freqs[-1]:.1f} Hz, 관측량 {len(obs)}, "
          f"질량 {base.rho * km.volume * 1e3:.3f} g ({time.time() - t0:.0f}s)", flush=True)
    ks = n6.kernel_scan(fm, md, obs, xi_grid, s_probe=1e-3, verbose=True)
else:
    spec = n1.fit_spec(n_r=n_r, n_bay=n_bay, n_vc=1, n_z_shroud=2, n_z_channel=2, wrap_deg=62.0, support="inner_edge")
    am = n1.build(spec)
    r_in = spec.a if p["rin"] is None else p["rin"] * 1e-3
    taper, c_of_r = make_taper(spec, am.info["grid_idx"], ci, co, r_in)
    patch_n1(taper, c_of_r)
    km0 = n1.assemble(am)
    md = n1.solve(km0, 80)
    obs = n1.classify(am, md)
    print(f"[k1] {tag} ndof={md.ndof} 관측량 {len(obs)}, 질량 {spec.rho * km0.volume * 1e3:.3f} g "
          f"({time.time() - t0:.0f}s)", flush=True)
    ks = n1.kernel_scan(am, md, obs, xi_grid, s_probe=1e-3, km0=km0, verbose=True)
pd.DataFrame([{"label": o.label, "h": o.h, "deg": o.deg, "f_Hz": o.f_Hz, "m_dom": o.m_dom, "n_rad": o.n_rad,
               "resolvable": o.resolvable} for o in obs]).to_csv(D_K + f"registry_{tag}.csv", index=False)
pd.DataFrame([{"xi": xi, "label": o.label, "obs": j, "k": ks.k[i, j], "dk": ks.dk[i, j]}
              for i, xi in enumerate(ks.xi) for j, o in enumerate(obs)]).to_csv(D_K + f"kernels_{tag}.csv", index=False)
json.dump({"mode": mode, "params": p, "n_r": n_r, "n_bay": n_bay, "step": step, "ndof": md.ndof,
           "seconds": time.time() - t0}, open(D_K + f"meta_{tag}.json", "w"), indent=1)
print(f"[{mode}] {tag} 끝 ({time.time() - t0:.0f}s)")
