"""N1 드라이버 — 격자 사다리 · 건전 관측량 등록 · 조립체 커널 스캔 · 모드선택 표.

    python -m impeller_fingerprint.rev1.n1_run ladder
    python -m impeller_fingerprint.rev1.n1_run kernels --n-r 18 --n-theta 72 --n-modes 80
    python -m impeller_fingerprint.rev1.n1_run select  [--key 18x72z2w60]
    python -m impeller_fingerprint.rev1.n1_run nonlin  --xi 0.5 --s-bar 0.05

산출(PAPER3_OUT/data/paper3): rev1_n1_mesh_ladder.csv, rev1_n1_registry_<key>.csv,
rev1_n1_kernels_<key>.csv, rev1_n1_selection_<key>.csv, rev1_n1_nonlin_<key>.csv
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .. import forward as fwd
from .. import geometry as geo
from .. import kernels as ker
from .. import modeselect as msel
from .. import noise as noi
from . import n1_assembly as n1
from . import out_csv

EXTENT = geo.DISK.b - geo.DISK.a


def _git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, cwd=Path(__file__).resolve().parents[2]).stdout.strip()
    except Exception:
        return "?"


_GEOM0 = n1.nm.FitSpec()          # 정본(10-16 실측) — 이것과 다르면 키에 기하 꼬리표를 붙인다
_GEOM_FIELDS = ("t_front", "t_back", "channel", "t_vane", "r_vane_in", "a", "b")


def _geom_tag(spec) -> str:
    """정본과 다른 기하면 `_gF1000B1000C4100V1000R1700A1540O3656` 꼴(µm, 반경은 10 µm).

    08-13 기하(`FitSpec.geom0813()`)는 꼬리표가 붙는다. r4.0까지의 옛 산출물 파일명(꼬리표 없음)은
    옛 출력 루트(`docs/_generated`)에만 있다.
    """
    if not hasattr(spec, "tf"):
        return ""
    if all(getattr(spec, f) == getattr(_GEOM0, f) for f in _GEOM_FIELDS):
        return ""
    return (f"_gF{spec.tf * 1e6:.0f}B{spec.tb * 1e6:.0f}C{spec.channel * 1e6:.0f}"
            f"V{spec.t_vane * 1e6:.0f}R{spec.r_vane_in * 1e5:.0f}A{spec.a * 1e5:.0f}"
            f"O{spec.b * 1e5:.0f}")


def _key(spec) -> str:
    if hasattr(spec, "n_bay"):
        sup = "" if spec.support == "inner_edge" else f"_{spec.support}"
        return (f"fit{spec.n_r}x{spec.n_bay}v{spec.n_vc}z{spec.n_z_shroud}"
                f"c{spec.n_z_channel}w{int(round(spec.wrap_deg))}{sup}{_geom_tag(spec)}")
    return (f"{spec.n_r}x{spec.n_theta}z{spec.n_z_shroud}"
            f"c{spec.n_z_channel}w{int(round(spec.wrap_deg))}")


def _healthy(spec, n_modes):
    am = n1.build(spec)
    t0 = time.time()
    km0 = n1.assemble(am)
    md = n1.solve(km0, n_modes)
    obs = n1.classify(am, md)
    print(f"[n1] {_key(spec)} ndof={md.ndof} 조립+고유해 {time.time() - t0:.0f}s, "
          f"관측량 {len(obs)}개 (분해불가 {sum(not o.resolvable for o in obs)})", flush=True)
    return am, km0, md, obs


def _registry_rows(spec, md, obs):
    rows = []
    for j, o in enumerate(obs):
        rows.append({"key": _key(spec), "obs": j, "label": o.label, "h": o.h, "deg": o.deg,
                     "f_Hz": o.f_Hz, "m_dom": o.m_dom, "n_rad": o.n_rad,
                     "vane_share": o.vane_share, "sigma_factor": o.sigma_factor,
                     "resolvable": o.resolvable, "cluster": o.cluster,
                     "members": "|".join(str(i) for i in o.members),
                     "ndof": md.ndof})
    return rows


# ---------------------------------------------------------------- ladder
LADDER = [dict(n_r=16, n_bay=8), dict(n_r=24, n_bay=12), dict(n_r=32, n_bay=16),
          dict(n_r=24, n_bay=12, n_vc=2), dict(n_r=24, n_bay=12, n_z_shroud=3, n_z_channel=3),
          # G1: 실측 형상의 얇은 앞판(0.375 mm) 패널 모드는 면내 해상도에 민감해 한 단 더 둔다
          dict(n_r=40, n_bay=20)]


def cmd_ladder(args):
    rows = []
    done = set()
    if args.resume and out_csv("n1_mesh_ladder").exists():
        prev = pd.read_csv(out_csv("n1_mesh_ladder"))
        rows = prev.to_dict("records")
        done = set(prev.key.unique())
    for kw in LADDER:
        spec = n1.fit_spec(wrap_deg=args.wrap, **kw)
        if _key(spec) in done:
            print(f"[n1] {_key(spec)} 이미 있음 — 건너뜀", flush=True)
            continue
        am, km0, md, obs = _healthy(spec, args.n_modes)
        ks = n1.kernel_scan(am, md, obs, args.xi, s_probe=1e-3, km0=km0)
        for j, o in enumerate(obs):
            row = {"key": _key(spec), "n_r": spec.n_r, "n_theta": spec.n_theta,
                   "n_bay": spec.n_bay, "n_vc": spec.n_vc,
                   "n_z_shroud": spec.n_z_shroud, "n_z_channel": spec.n_z_channel,
                   "ndof": md.ndof, "label": o.label, "h": o.h, "deg": o.deg,
                   "f_Hz": o.f_Hz, "resolvable": o.resolvable, "vane_share": o.vane_share}
            for i, xi in enumerate(ks.xi):
                row[f"k_xi{xi:.2f}"] = ks.k[i, j]
                row[f"dk_xi{xi:.2f}"] = ks.dk[i, j]
            rows.append(row)
        pd.DataFrame(rows).to_csv(out_csv("n1_mesh_ladder"), index=False)
    print(f"[n1] → {out_csv('n1_mesh_ladder')}")


# ---------------------------------------------------------------- kernels
_GEOM_ARGS = {"t_front": "t_front", "t_back": "t_back", "t_vane": "t_vane", "channel": "channel",
              "r_vin": "r_vane_in", "a": "a", "b": "b"}


def _geom_kw(args) -> dict:
    """기하 덮어쓰기 인자(mm) → FitSpec 키워드(m). 주지 않은 값은 08-13 기본을 쓴다."""
    out = {}
    for arg, field_ in _GEOM_ARGS.items():
        v = getattr(args, arg, None)
        if v is not None:
            out[field_] = v * 1e-3
    return out


def cmd_kernels(args):
    spec = n1.fit_spec(n_r=args.n_r, n_bay=args.n_bay, n_vc=args.n_vc, n_z_shroud=args.n_z,
                       n_z_channel=args.n_zc, wrap_deg=args.wrap, support=args.support,
                       **_geom_kw(args))
    am, km0, md, obs = _healthy(spec, args.n_modes)
    key = _key(spec)
    reg = pd.DataFrame(_registry_rows(spec, md, obs))
    # 구역 에너지 분율(R2-Q3): 베인 0 뿌리창(스팬 0–25 %, b5와 같은 정의) / 베인 0 전체 / 전면 슈라우드
    c = am.coors[am.conn].mean(axis=1)
    zc = c[:, 2]
    masks = {"vane0_root": n1.vane_root_mask(am, 0, (0.0, 0.25)),
             "vane0_all": n1.vane_root_mask(am, 0, (0.0, 1.0)),
             "front_shroud": zc > spec.tb + spec.channel}
    for name, mk in masks.items():
        ef, kf = n1.energy_fractions(md, obs, n1.assemble_region(am, mk))
        reg[f"Efrac_{name}"] = ef
        reg[f"Tfrac_{name}"] = kf
        reg[f"ncell_{name}"] = int(mk.sum())
    reg.to_csv(out_csv(f"n1_registry_{key}"), index=False)
    xi_grid = np.round(np.arange(args.xi_lo, args.xi_hi + 1e-9, args.xi_step), 4)
    t0 = time.time()
    ks = n1.kernel_scan(am, md, obs, xi_grid, s_probe=1e-3, km0=km0, verbose=True)
    rows = []
    for i, xi in enumerate(ks.xi):
        for j, o in enumerate(obs):
            rows.append({"key": key, "xi": xi, "label": o.label, "obs": j,
                         "k": ks.k[i, j], "dk": ks.dk[i, j]})
    pd.DataFrame(rows).to_csv(out_csv(f"n1_kernels_{key}"), index=False)
    meta = {"key": key, "git": _git_head(), "ndof": md.ndof, "n_modes": args.n_modes,
            "a": spec.a, "b": spec.b, "t_front": spec.tf, "t_back": spec.tb, "channel": spec.channel, "channel_in": spec.channel_in,
            "t_vane": spec.t_vane, "wrap_deg": spec.wrap_deg, "r_vane_in": spec.r_vane_in,
            "s_probe": 1e-3, "dxi": 0.01, "w": n1.W_GAUSS, "support": "inner-edge clamp r=a",
            "damage": "front-shroud outer-face thinning, exact two-flange d_K→p",
            "rho": spec.rho, "seconds_scan": time.time() - t0}
    out_csv(f"n1_kernels_{key}").with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(f"[n1] → {out_csv(f'n1_kernels_{key}')} ({time.time() - t0:.0f}s)")


# ---------------------------------------------------------------- selection
RAIL_FAMILY_POOLS = msel.FAMILY_POOLS      # 계열 정의(G2 결정 D3)


def _rail_pool():
    return [ker.mode_kernel(geo.DISK, m=m, n=n) for m in range(5) for n in (0, 1)]


def _rail_rows(theta, w, c, laws=("exact", "sandwich_asbuilt")):
    """레일 대조(정본 Table 3a 규약: 1차 질량포함 야코비안)."""
    pool = _rail_pool()
    out = []
    for law in laws:
        for tag, sel in RAIL_FAMILY_POOLS:
            meas = [k for k in pool if sel(k)]
            default = [k for k in pool if k.n == 0 and k.m <= 3]
            m_def = msel.subset_metrics(default, geo.DISK, theta, w,
                                        noi.sigma_y_for_modes(default, c), mass=law)
            best = msel.d_optimal_subset(meas, geo.DISK, theta, w, float(c), k=4, mass=law)
            m_all = msel.subset_metrics(meas, geo.DISK, theta, w,
                                        noi.sigma_y_for_modes(meas, c), mass=law)
            lab2k = {k.label: k for k in pool}
            for name, mm, labels in (("default", m_def, [k.label for k in default]),
                                     ("d_optimal_4", best["metrics"], best["labels"]),
                                     ("all", m_all, [k.label for k in meas])):
                sub = [lab2k[l] for l in labels]
                sig = np.sqrt(np.diag(noi.sigma_y_for_modes(sub, c)))
                cm = common_mode_crlb(np.asarray(mm["J"]), sig)
                out.append({"model": f"rail_{law}", "pool": tag, "set": name,
                            "labels": "|".join(labels), "n_obs": len(labels),
                            "crlb_xi_mm": mm["crlb_xi_mm"], "crlb_s_pp": mm["crlb_s_pp"],
                            "det_F": mm["det_F"], "cond2": mm["cond2"], **cm})
    return out


def common_mode_crlb(J: np.ndarray, sig: np.ndarray, extent: float = EXTENT) -> dict:
    """공통모드 nuisance(모든 η에 같은 오프셋 — 균일 E 드리프트·온도)를 미지수로 둔 CRLB.

    θ = (ξ, S̄, e), ∂η/∂e = 1. Schur 보수 F_eff = F_θθ − F_θe F_ee⁻¹ F_eθ의 역 대각.
    관측량이 3개 미만이면 정의되지 않는다(NaN).
    """
    if J.shape[0] < 3:
        return {"crlb_xi_mm_cm": np.nan, "crlb_s_pp_cm": np.nan}
    Ja = np.column_stack([J, np.ones(J.shape[0])])
    Jw = Ja / sig[:, None]
    F = Jw.T @ Jw
    try:
        Fi = np.linalg.inv(F)
    except np.linalg.LinAlgError:
        return {"crlb_xi_mm_cm": np.inf, "crlb_s_pp_cm": np.inf}
    return {"crlb_xi_mm_cm": float(np.sqrt(Fi[0, 0]) * extent * 1e3),
            "crlb_s_pp_cm": float(np.sqrt(Fi[1, 1]) * 100.0)}


def _asm_J(kdf, labels, xi, s_bar):
    """선형(1차) 조립체 야코비안 at (ξ, S̄): J = [S̄·∂k/∂ξ, k]."""
    sub = kdf[np.isclose(kdf.xi, xi)].set_index("label")
    k = sub.loc[labels, "k"].to_numpy()
    dk = sub.loc[labels, "dk"].to_numpy()
    return np.column_stack([s_bar * dk, k])


def cmd_select(args):
    key, pre = args.key, args.prefix
    reg = pd.read_csv(out_csv(f"{pre}_registry_{key}"))
    kdf = pd.read_csv(out_csv(f"{pre}_kernels_{key}"))
    meta_p = out_csv(f"{pre}_kernels_{key}").with_suffix(".json")
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    # 조립체 ξ는 그 스펙의 a·b로 정의된다 — 옛 메타(a·b 없음)는 레일 반경을 쓴 08-13 실행이다
    ext_asm = (meta["b"] - meta["a"]) if "a" in meta else EXTENT
    c = args.c
    rows = []
    for xi in args.xi:
        for s_bar in args.s_bar:
            theta = (xi, s_bar)
            for r in _rail_rows(theta, n1.W_GAUSS, c):
                rows.append({"xi": xi, "s_bar": s_bar, "c": c, **r})
            res_reg = reg[reg.resolvable].sort_values("f_Hz")
            fam = res_reg.label.str.extract(r"m(\d+)n(\d+)").astype(int)
            pools = {
                # 레일 Table 3a 20 kHz 풀(n=0, m=0–4)의 구조족 대응
                "family_n0_m0-4": res_reg[(fam[1] == 0) & (fam[0] <= 4)],
                # 레일 25.6 kHz 풀(+ n=1, m=0–2)의 구조족 대응
                "family_n0_m0-4+n1_m0-2": res_reg[((fam[1] == 0) & (fam[0] <= 4))
                                                  | ((fam[1] == 1) & (fam[0] <= 2))],
            }
            for f_cut in args.f_cut:
                pools[f"f<={f_cut / 1e3:.1f}kHz"] = res_reg[res_reg.f_Hz <= f_cut]
            for pool_name, pool in pools.items():
                pool = pool.sort_values("f_Hz")
                labels = pool.label.tolist()
                sig = (pool.sigma_factor.to_numpy() * c)
                if len(labels) < 2:
                    continue
                J = _asm_J(kdf, labels, xi, s_bar)
                # 기본 대응집합: 각 h의 최저 n=0 계열(h0:m0n0, h1:m1n0, h2:m2n0, h3:m3n0 최저)
                default = []
                for want in ("h0:m0n0", "h1:m1n0", "h2:m2n0", "h3:m3n0"):
                    cand = pool[pool.label.str.startswith(want)]
                    if len(cand):
                        default.append(cand.label.iloc[0])
                sets = {}
                if len(default) >= 2:
                    idx = [labels.index(l) for l in default]
                    sets["default_analog"] = idx
                sets["all_resolvable"] = list(range(len(labels)))
                # D-최적 4 (제약 없음) / h당 1개 제약
                best = None
                best_c6 = None
                k_opt = min(4, len(labels))
                n_comb = math.comb(len(labels), k_opt)
                if n_comb > 200000:
                    print(f"[n1] {pool_name}: 조합 {n_comb}개 — D-최적 생략", flush=True)
                else:
                    for comb in itertools.combinations(range(len(labels)), k_opt):
                        mm = n1.fisher_metrics(J[list(comb)], sig[list(comb)], ext_asm)
                        if best is None or mm["det_F"] > best[0]:
                            best = (mm["det_F"], comb)
                        hs = pool.h.to_numpy()[list(comb)]
                        if len(set(hs)) == len(hs) and (best_c6 is None
                                                        or mm["det_F"] > best_c6[0]):
                            best_c6 = (mm["det_F"], comb)
                    sets["d_optimal_4"] = list(best[1])
                    if best_c6:
                        sets["d_optimal_4_one_per_h"] = list(best_c6[1])
                for name, idx in sets.items():
                    mm = n1.fisher_metrics(J[idx], sig[idx], ext_asm)
                    cm = common_mode_crlb(J[idx], sig[idx], ext_asm)
                    rows.append({"xi": xi, "s_bar": s_bar, "c": c, "model": f"assembly_{key}",
                                 "pool": pool_name, "set": name,
                                 "labels": "|".join(labels[i] for i in idx),
                                 "n_obs": len(idx), "crlb_xi_mm": mm["crlb_xi_mm"],
                                 "crlb_s_pp": mm["crlb_s_pp"], "det_F": mm["det_F"],
                                 "cond2": mm["cond2"], **cm})
    df = pd.DataFrame(rows)
    df.to_csv(out_csv(f"{pre}_selection_{key}"), index=False)
    print(f"[n1] → {out_csv(f'{pre}_selection_{key}')} ({len(df)} rows)")


# ---------------------------------------------------------------- nonlinear check
def cmd_nonlin(args):
    spec = n1.fit_spec(n_r=args.n_r, n_bay=args.n_bay, n_vc=args.n_vc, n_z_shroud=args.n_z,
                       n_z_channel=args.n_zc, wrap_deg=args.wrap, support=args.support,
                       **_geom_kw(args))
    key = _key(spec)
    am, km0, md, obs = _healthy(spec, args.n_modes)
    rows = []
    for xi in args.xi:
        for s_bar in args.s_bar:
            hx, hs = args.dxi, args.ds_rel * s_bar
            t0 = time.time()
            e0, mac0 = n1.resolved_eta(am, md, obs, xi, s_bar)
            ep, _ = n1.resolved_eta(am, md, obs, min(xi + hx, 1.0), s_bar)
            em, _ = n1.resolved_eta(am, md, obs, max(xi - hx, 0.0), s_bar)
            sp_, _ = n1.resolved_eta(am, md, obs, xi, s_bar + hs)
            sm_, _ = n1.resolved_eta(am, md, obs, xi, s_bar - hs)
            for j, o in enumerate(obs):
                rows.append({"key": key, "xi": xi, "s_bar": s_bar, "label": o.label,
                             "eta": e0[j], "mac": mac0[j],
                             "J_xi": (ep[j] - em[j]) / (min(xi + hx, 1) - max(xi - hx, 0)),
                             "J_s": (sp_[j] - sm_[j]) / (2 * hs)})
            print(f"[n1] nonlin ξ={xi} S̄={s_bar}: {time.time() - t0:.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(out_csv(f"n1_nonlin_{key}"), index=False)
    print(f"[n1] → {out_csv(f'n1_nonlin_{key}')}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="n1_run")
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("ladder")
    sp.add_argument("--n-modes", type=int, default=40)
    sp.add_argument("--wrap", type=float, default=geo.ASBUILT.wrap_deg)
    sp.add_argument("--xi", type=float, nargs="+", default=[0.5, 0.8])
    sp.add_argument("--resume", action="store_true",
                    help="이미 CSV에 있는 격자 키는 건너뛰고 행을 이어 붙인다")
    sp.set_defaults(func=cmd_ladder)
    for name, fn in (("kernels", cmd_kernels), ("nonlin", cmd_nonlin)):
        sp = sub.add_parser(name)
        sp.add_argument("--n-r", type=int, default=24)
        sp.add_argument("--n-bay", type=int, default=12)
        sp.add_argument("--n-vc", type=int, default=1)
        sp.add_argument("--n-z", type=int, default=2)
        sp.add_argument("--n-zc", type=int, default=2)
        sp.add_argument("--wrap", type=float, default=geo.ASBUILT.wrap_deg)
        sp.add_argument("--support", default="inner_edge", choices=["inner_edge", "back_sheet"])
        sp.add_argument("--n-modes", type=int, default=80)
        # 기하 덮어쓰기(mm) — 주지 않으면 정본(geometry.ASBUILT, G3). G1(10-16 실측): --t-front 0.375
        # --t-back 0.59 --channel 3.18 --t-vane 1.2 --a 14.955 --b 36.515 (선택: --r-vin 15 --wrap 62)
        for arg in _GEOM_ARGS:
            sp.add_argument(f"--{arg.replace('_', '-')}", type=float, default=None)
        if name == "kernels":
            sp.add_argument("--xi-lo", type=float, default=0.02)
            sp.add_argument("--xi-hi", type=float, default=0.98)
            sp.add_argument("--xi-step", type=float, default=0.02)
        else:
            sp.add_argument("--xi", type=float, nargs="+", default=[0.5])
            sp.add_argument("--s-bar", type=float, nargs="+", default=[0.05])
            sp.add_argument("--dxi", type=float, default=0.01)
            sp.add_argument("--ds-rel", type=float, default=0.1)
        sp.set_defaults(func=fn)
    sp = sub.add_parser("select")
    sp.add_argument("--key", default="fit32x16v1z2c2w62")
    sp.add_argument("--prefix", default="n1", help="n1(N1 조립체) 또는 n6(N6 허브 고정·자유-자유)")
    sp.add_argument("--c", type=float, default=1e-3)
    sp.add_argument("--xi", type=float, nargs="+", default=[0.2, 0.5, 0.8, 0.94])
    sp.add_argument("--s-bar", type=float, nargs="+", default=[0.01, 0.05])
    sp.add_argument("--f-cut", type=float, nargs="+", default=[1.2e4, 1.6e4, 2.0e4, 2.6e4])
    sp.set_defaults(func=cmd_select)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
