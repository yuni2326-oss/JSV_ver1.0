"""산출물 생성 엔트리 — `python -m impeller_fingerprint.cli <item> [...]`.

출력: $PAPER3_OUT/data/paper3/*.csv|npz, $PAPER3_OUT/figures/paper3/*.png
(기본 PAPER3_OUT = 이 리포의 docs/_generated. 워크트리에서 실행할 때는 본 리포를 가리키도록
환경변수로 지정한다 — 생성물은 git에 올리지 않으므로 위치만 맞추면 된다.)

그림 라벨은 논문(JSV)용이라 영문으로 쓴다.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import crack_shear as cs
from . import degenerate as deg
from . import figures as figs_mod
from . import forward as fwd
from . import geometry as geo
from . import geometry as geo_mod
from . import identifiability as idf
from . import kernels as ker
from . import modeselect as ms
from . import montecarlo as mc
from . import noise as noi
from . import references as refs
from . import severity as sev
from . import validity as val

OUT_ROOT = Path(os.environ.get(
    "PAPER3_OUT", Path(__file__).resolve().parents[1] / "docs" / "_generated"))
DATA = OUT_ROOT / "data" / "paper3"
FIGS = OUT_ROOT / "figures" / "paper3"

PLATE = geo.DISK
VANE = geo.VANE
MODES = [(0, 0), (1, 0), (2, 0), (3, 0)]
W_GAUSS = 0.003
SIGMA_RELS = (1e-4, 3e-4, 1e-3, 3e-3)


def _ensure_dirs():
    DATA.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)


def _save(df: pd.DataFrame, name: str):
    _ensure_dirs()
    path = DATA / name
    df.to_csv(path, index=False)
    print(f"[saved] {path}  ({len(df)} rows)")
    return path


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _pool(n_grid=2001):
    return [ker.mode_kernel(PLATE, m=m, n=n, n_grid=n_grid) for m, n in MODES]


# ---------------------------------------------------------------- A1
def cmd_a1(args):
    """파일럿(12×5, 이동량의 3 %)을 (ξ_d, S̄_D) 좌표로 재기술 — 부록 B 라벨링용."""
    from . import estimator as est
    pool = _pool()
    S_pilot, w_pilot, n_real = 0.0012, 0.003, 12
    s_bar = sev.s_bar_from_S(S_pilot, PLATE.extent)
    rows = []
    for frac in (0.2, 0.35, 0.5, 0.65, 0.8):
        y0 = fwd.eta_bar_linear(pool, frac, s_bar, w_pilot, PLATE)
        rng = np.random.default_rng(int(frac * 1000))
        errs_xi, errs_s, hits = [], [], 0
        for _ in range(n_real):
            y = y0 * (1.0 + 0.03 * rng.standard_normal(len(y0)))   # 옛 규약
            sigma = np.diag((0.03 * np.abs(y0)) ** 2)
            out = est.fit(y, pool, PLATE, sigma, w=w_pilot)
            errs_xi.append((out["xi_d"] - frac) * PLATE.extent * 1e3)
            errs_s.append((out["s_bar"] - s_bar) * 100.0)
            hits += int(out["boundary_hit"])
        rows.append({"xi_d": frac,
                     "r_d_mm": sev.xi_to_r(frac, PLATE.a, PLATE.b) * 1e3,
                     "S_mm": S_pilot * 1e3, "s_bar_pct": s_bar * 100.0,
                     "w_mm": w_pilot * 1e3, "n_real": n_real,
                     "abs_err_xi_mm_median": float(np.median(np.abs(errs_xi))),
                     "abs_err_s_pp_median": float(np.median(np.abs(errs_s))),
                     "boundary_hit_prob": hits / n_real,
                     "note": "PILOT: noise = 3% of shift (superseded convention)"})
    _save(pd.DataFrame(rows), "a1_severity_reparam.csv")


# ---------------------------------------------------------------- A2
def cmd_a2(args):
    pool = _pool()
    xi_grid = np.linspace(0.05, 0.95, args.n_xi)
    s_grid = np.geomspace(0.001, 0.30, args.n_s)
    M = val.e_pert_map(PLATE, pool, MODES, xi_grid, s_grid, W_GAUSS, mass=args.mass)
    Mabs = val.e_pert_map(PLATE, pool, MODES, xi_grid, s_grid, W_GAUSS, absolute=True, mass=args.mass)
    _ensure_dirs()
    name = "a2_epert_map_mass.npz" if args.mass else "a2_epert_map.npz"
    np.savez(DATA / name, xi=xi_grid, s=s_grid, rel=M, abs=Mabs,
             modes=np.array(MODES))
    print(f"[saved] {DATA / name}")

    rows = []
    for c in SIGMA_RELS:
        thr = val.validity_contour(PLATE, pool, MODES, xi_grid, s_grid, W_GAUSS, c,
                                   mass=args.mass, abs_map=Mabs)
        for xi, t in zip(xi_grid, thr):
            rows.append({"sigma_rel": c, "xi_d": xi, "s_bar_threshold": t})
    _save(pd.DataFrame(rows), "a2_validity_contours_mass.csv" if args.mass else "a2_validity_contours.csv")

    # ------------------------------------------------------------------ 그림
    # **그림은 npz 소비자로 분리했다**(F54): 이전에는 이 자리에 pcolormesh가 그대로 있어서
    # 그림 규약을 바꾸려면 위의 맵·등고선까지 재실행해야 했고, 그것이 상대맵 재스케일을
    # 미뤄 온 유일한 이유였다(설계서 §11.15 '미해소로 남긴 것'). 지금은 저장된 npz만으로
    # `figures.fig_a2_epert`가 그리므로 맵을 건드리지 않고 그림만 다시 낼 수 있다.
    from . import figures as figs
    figs.fig_a2_epert(DATA / name,
                      FIGS / ("fig_a2_epert_mass.png" if args.mass else "fig_a2_epert.png"),
                      coupling=args.mass)


# ---------------------------------------------------------------- A3
def cmd_a3(args):
    pool = _pool()
    xi_grid = np.linspace(0.05, 0.95, args.n_xi)
    s_grid = np.geomspace(0.002, 0.20, args.n_s)
    _ensure_dirs()
    store, rows = {}, []
    for c in SIGMA_RELS:
        sigma = noi.sigma_y_for_modes(pool, c)
        maps = idf.metric_maps(pool, PLATE, xi_grid, s_grid, W_GAUSS, sigma,
                               mass=args.mass)
        for key, arr in maps.items():
            if key in ("xi_grid", "s_grid"):
                continue
            store[f"{key}_c{c:g}"] = arr
        for xi in (0.2, 0.5, 0.8, 0.95):
            for s in (0.01, 0.05, 0.15):
                m = idf.metrics(pool, PLATE, (xi, s), W_GAUSS, sigma, mass=args.mass)
                rows.append({"sigma_rel": c, "xi_d": xi, "s_bar": s,
                             **{k: m[k] for k in ("sigma_min", "sigma_max", "cond2",
                                                  "det_F", "tr_Finv", "corr",
                                                  "crlb_xi_mm", "crlb_s_pp")},
                             # 결합비를 CSV에 남긴다 — 파일명 접미사는 내용의 증거가
                             # 아니다(설계서 F20 교훈). a4·a5·b1은 이미 이 열이 있었다.
                             "mass_coupling": str(args.mass)})
    name = ("a3_identifiability_mass.npz" if args.mass
            else "a3_identifiability.npz")
    np.savez(DATA / name, xi=xi_grid, s=s_grid, **store)
    print(f"[saved] {DATA / name}")
    _save(pd.DataFrame(rows), "a3_summary_mass.csv" if args.mass else "a3_summary.csv")

    plt = _plt()
    sigma = noi.sigma_y_for_modes(pool, 1e-3)
    maps = idf.metric_maps(pool, PLATE, xi_grid, s_grid, W_GAUSS, sigma,
                           mass=args.mass)
    keys = [("crlb_xi_mm", "CRLB $\\xi_d$ [mm]", True),
            ("crlb_s_pp", "CRLB $\\bar S_D$ [%p]", True),
            ("sigma_min", r"$\sigma_{min}(J_w)$", True),
            ("cond2", r"cond$_2(J_w)$", False)]
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.4), sharey=True)
    for ax, (key, label, logc) in zip(axes, keys):
        from matplotlib.colors import LogNorm
        arr = maps[key].T
        pc = ax.pcolormesh(xi_grid, s_grid * 100, arr, shading="auto", cmap="magma",
                           norm=LogNorm() if logc else None)
        ax.set_yscale("log")
        ax.set_xlabel(r"$\xi_d$")
        ax.set_title(label)
        fig.colorbar(pc, ax=ax)
    axes[0].set_ylabel(r"$\bar S_D$ [%]")
    # suptitle 제거(2026-08-24): 맵 종류·결합비·σ_f/f·물리한계는 **캡션이 말한다**.
    # 그림에 캡션을 다시 찍지 않는다(같은 규약을 fig_recovery에 2026-08-15 적용).
    fig.tight_layout()
    fig_name = ("fig_a3_identifiability_mass.png" if args.mass
                else "fig_a3_identifiability.png")
    figs_mod.save_canon(fig, FIGS / fig_name, dpi=200)   # PNG+PDF(제출 벡터)
    print(f"[saved] {FIGS / fig_name}")


# ---------------------------------------------------------------- A4
def cmd_a4(args):
    mass = getattr(args, "mass", None)
    pool = _pool()
    sigma = noi.sigma_y_for_modes(pool, 1e-3)
    xi_grid = np.linspace(0.02, 0.98, 481)
    s_grid = np.geomspace(0.002, 0.30, 241)
    rows, grids = [], {}
    for xi_t in (0.2, 0.5, 0.8):
        for s_t in (0.01, 0.05):
            y = (fwd.eta_bar_linear_mass(pool, xi_t, s_t, W_GAUSS, PLATE,
                                         coupling=mass) if mass
                 else fwd.eta_bar_linear(pool, xi_t, s_t, W_GAUSS, PLATE))
            prof = idf.profile_likelihood(y, pool, PLATE, sigma, W_GAUSS, xi_grid,
                                          mass=mass)
            lo, hi = idf.profile_interval(xi_grid, prof)
            chi2 = idf.objective_grid(y, pool, PLATE, sigma, W_GAUSS, xi_grid[::4],
                                      s_grid[::4], mass=mass)
            grids[f"chi2_xi{xi_t}_s{s_t}"] = chi2
            rows.append({"xi_true": xi_t, "s_bar_true": s_t,
                         "prof_lo": lo, "prof_hi": hi,
                         "prof_halfwidth_mm": 0.5 * (hi - lo) * PLATE.extent * 1e3,
                         "crlb_xi_mm": idf.metrics(pool, PLATE, (xi_t, s_t), W_GAUSS,
                                                   sigma,
                                                   mass=(mass or False))["crlb_xi_mm"],
                         "local_minima_in_grid": idf.local_minima_count(chi2),
                         "mass_coupling": str(mass)})
    _ensure_dirs()
    np.savez(DATA / ("a4_grids_mass.npz" if mass else "a4_grids.npz"),
             xi=xi_grid[::4], s=s_grid[::4], **grids)
    print(f"[saved] {DATA / ('a4_grids_mass.npz' if mass else 'a4_grids.npz')}")
    _save(pd.DataFrame(rows), "a4_profiles_mass.csv" if mass else "a4_profiles.csv")


# ---------------------------------------------------------------- A5
def cmd_a5(args):
    """모드셋 비교 — **측정가능 대역이 입력**이다(실측 두께 반영 후 결론이 대역에 걸림, F59).

    레일 t = 2 t_f = 2.0 mm에서 n=1 족은 22.3/23.0/24.9 kHz다. 설계서 §4-6이 동결한 DAQ 대역
    20 kHz에서는 **전부 측정 불가**이므로 후보 풀이 n=0 다섯 모드로 줄고 D-최적 집합이 바뀐다.
    대역이 25.6 kHz(51.2 kS/s)면 n=1이 되살아나므로, 두 대역을 모두 내고 `f_max_kHz` 열로 구분한다.

    `--pool-ms`/`--tag`는 **순환대칭 감도**(A13/F74)를 위한 것이다: 레일은 축대칭이라 m=4가
    독립 doublet이지만 실물은 C6이므로 m=4는 m=2와 같은 조화지수(h=2)에 접혀 독립 정보가
    아니다. 생산 산출(기본값)은 건드리지 않고 `--pool-ms 0 1 2 3 --tag c6`로 별도 파일을 낸다.
    """
    mass = getattr(args, "mass", None)
    pool_ms = tuple(getattr(args, "pool_ms", None) or (0, 1, 2, 3, 4))
    tag = getattr(args, "tag", "") or ""
    suffix = f"_{tag}" if tag else ""
    pool = ker.mode_pool(PLATE, ms=pool_ms, ns=(0, 1), n_grid=2001)
    _save(pd.DataFrame(ms.pool_table(pool)), f"a5_mode_pool{suffix}.csv")
    _save(pd.DataFrame(ms.collinearity_report(pool)),
          f"a5_collinearity{suffix}.csv")
    rows = []
    for f_max in args.f_max:
        for xi in (0.2, 0.5, 0.8, 0.95):
            for s in (0.01, 0.05):
                for c in SIGMA_RELS:
                    for r in ms.compare_mode_sets(PLATE, pool, (xi, s), W_GAUSS, c,
                                                  f_max=f_max, mass=mass):
                        rows.append({"xi_d": xi, "s_bar": s, "sigma_rel": c,
                                     "f_max_kHz": f_max / 1e3,
                                     **{k: v for k, v in r.items() if k != "labels"},
                                     "labels": "|".join(r["labels"]),
                                     "pool_ms": "|".join(str(m) for m in pool_ms),
                                     "mass_coupling": str(mass)})
    _save(pd.DataFrame(rows),
          (f"a5_modeselect_mass{suffix}.csv" if mass
           else f"a5_modeselect{suffix}.csv"))


# ---------------------------------------------------------------- A6
def cmd_a6(args):
    rows = []
    for m in (1, 2, 3, 4):
        for xi in (0.2, 0.5, 0.8):
            r_c = sev.xi_to_r(xi, PLATE.a, PLATE.b)
            for dth_deg in (10, 20, 30, 45, 60, 90):
                for depth in (0.1, 0.25, 0.5):
                    r1 = max(PLATE.a, r_c - 0.003)
                    r2 = min(PLATE.b, r1 + 0.006)
                    p = deg.Pocket(r1, r2, 0.4, np.deg2rad(dth_deg), depth)
                    o = deg.observables(PLATE, m, p, n_grid=2001, n_r=2001)
                    rows.append({"m": m, "xi_d": xi, "dtheta_deg": dth_deg,
                                 "depth_frac": depth,
                                 "r1_mm": r1 * 1e3, "r2_mm": r2 * 1e3,
                                 "eta_bar": o["eta_bar"],
                                 "delta_eta": o["delta_eta"],
                                 "ratio_split_over_mean": o["delta_eta"] / abs(o["eta_bar"]),
                                 "B_signed": o["B_signed"],
                                 "B_sign": "+" if o["B_signed"] > 0 else "-",
                                 "psi_lower_deg": np.rad2deg(o["psi_lower"]),
                                 "theta0_hat_deg": np.rad2deg(o["theta0_hat"]),
                                 "s_bar_radial_pct": o["severity_s_bar_radial"] * 100,
                                 "s_bar_azavg_pct": o["severity_s_bar"] * 100})
    _save(pd.DataFrame(rows), "a6_degenerate.csv")


# ---------------------------------------------------------------- A7
def cmd_a7(args):
    beam = cs.TimoBeam(L=VANE.L, h=VANE.h, b=2 * VANE.h, E=VANE.E, rho=VANE.rho,
                       nu=VANE.nu)
    a_bars = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    # 규약을 **둘 다** 낸다. 생산 규약은 `tada`(A11/F42가 2D 탄성으로 판정)이고
    # `dimarogonas`는 정본 Table 1·논문1과의 연속성 확인용 기준선으로 유지한다.
    rows = []
    for conv_name in ("tada", "dimarogonas"):
        for shear in (False, True):
            for coup in ((0.0,) if not shear else (0.0, 0.5, 0.9)):
                for r in cs.signature(beam, a_bars, xc_over_L=0.2, n_modes=3,
                                      n_elem=300, coupling=coup, shear_flex=shear,
                                      convention=conv_name):
                    rows.append({"arm": ("modeI" if not shear
                                         else f"modeI+II_c{coup:g}"),
                                 "h_over_L": VANE.h / VANE.L, **r})
        for hl in (0.10, 0.20):
            b2 = cs.TimoBeam(L=VANE.L, h=hl * VANE.L, b=2 * hl * VANE.L, E=VANE.E,
                             rho=VANE.rho, nu=VANE.nu)
            for shear in (False, True):
                for r in cs.signature(b2, a_bars, xc_over_L=0.2, n_modes=3,
                                      n_elem=300, shear_flex=shear,
                                      convention=conv_name):
                    rows.append({"arm": ("modeI" if not shear else "modeI+II_c0"),
                                 "h_over_L": hl, **r})
    _save(pd.DataFrame(rows), "a7_crack_shear.csv")

    conv = [{"a_bar": ab, "convention_default": "tada",
             **{k: v for k, v in cs.compliance(ab, beam.h, beam.b, beam.E,
                                               beam.nu).items()},
             "c_MM_dimarogonas": cs.compliance_dimarogonas(ab, beam.h, beam.b,
                                                           beam.E),
             "c_MM_tada_over_dimarogonas": (
                 cs.compliance(ab, beam.h, beam.b, beam.E, beam.nu)["c_MM"]
                 / cs.compliance_dimarogonas(ab, beam.h, beam.b, beam.E))}
            for ab in a_bars]
    _save(pd.DataFrame(conv), "a7_compliance_conventions.csv")


# ---------------------------------------------------------------- A11
#: A11 격자 사다리 — (nx_left, nx_right, nz_below, nz_above) = ref × (8, 32, 4, 4).
#: 균열선 x_c와 균열선단 y_tip에 절점을 정확히 놓고 그쪽으로 등비 등급화하므로
#: 균열깊이가 격자에 스냅되지 않는다(B2의 0.125 단위 양자화와 대비).
A11_REFS = (1, 2, 3, 4)
A11_REF_WORK = 3
A11_BIAS = 12.0
#: 정본 Table 1의 깊이 + B2(3D 노치)가 실제로 실현한 깊이 {0.25, 0.5, 0.625}.
A11_A_BARS = (0.1, 0.25, 0.3, 0.5, 0.6, 0.625)
A11_WIDTHS_MM = (0.0, 0.1, 0.25, 0.5, 1.0)


def _a11_grid(ref: int) -> dict:
    return {"nx_left": 8 * ref, "nx_right": 32 * ref, "nz_below": 4 * ref,
            "nz_above": 4 * ref, "bias": A11_BIAS}


def _a11_pair(c2, a_bar, plane, ref, kerf_width=0.0):
    """(건전, 손상) 한 쌍을 **같은 절점배치**에서 풀어 주파수비의 이산화오차를 상쇄한다."""
    kw = _a11_grid(ref)
    co0, cn0, _ = c2.slit_mesh(VANE.L, VANE.h, a_bar, crack=False, **kw)
    f0, _, nd0 = c2.flap_modes(co0, cn0, VANE.E, VANE.nu, VANE.rho, plane=plane)
    co, cn, info = c2.slit_mesh(VANE.L, VANE.h, a_bar, kerf_width=kerf_width, **kw)
    f, kinds, nd = c2.flap_modes(co, cn, VANE.E, VANE.nu, VANE.rho, plane=plane)
    return f0, f, info, nd0, nd, kinds


# ---------------------------------------------------------------- A12
def cmd_a12(args):
    """확정된 참고문헌 표를 CSV로 내고, 정본 md에 붙일 References 블록을 stdout에 찍는다.

    계산이 아니라 **검증 결과의 산출물화**다(설계서 M4). `impeller_fingerprint.references`가
    단일 정본이고 `tests/test_references.py`가 정본 md의 본문 인용번호와 1:1 대조한다.
    """
    _ensure_dirs()
    path = refs.write_csv(DATA / "a12_references.csv")
    n_ver = sum(1 for r in refs.REFERENCES if r.verified)
    print(f"[saved] {path}  ({len(refs.REFERENCES)} rows, verified {n_ver}"
          f"/{len(refs.REFERENCES)})")
    if args.block:
        print()
        print(refs.markdown_block())


def cmd_a11(args):
    """2D 평면탄성 + **폭 0 균열** — 정본 §3.6-iv의 남은 arm.

    B2(3D)는 요소제거라 최소 커프 폭이 0.25 mm였고 등가깊이가 물리깊이보다 13–70 % 깊었다
    (설계서 F10). 즉 3D는 **노치**를 풀었고 폭 0 균열은 어느 arm도 풀지 않았다. 2D는 균열선
    절점을 이중화해 폭이 정확히 0인 슬릿을 만들 수 있으므로 **회전스프링이 이상화하는
    날카로운 균열 극한**을 직접 검정한다.
    """
    from . import crack2d as c2
    beam = cs.TimoBeam(L=VANE.L, h=VANE.h, b=2 * VANE.h, E=VANE.E, rho=VANE.rho,
                       nu=VANE.nu)
    planes = tuple(args.planes)

    # (A) 메시수렴 — 이동량 기준. 균열선단은 응력 특이성이 있어 수렴이 느리다.
    conv = []
    for plane in planes:
        for ab in args.conv_a_bars:
            prev = None
            for ref in A11_REFS:
                f0, f, info, nd0, nd, _ = _a11_pair(c2, ab, plane, ref)
                r = f[:3] / f0[:3]
                row = {"plane": plane, "a_bar": ab, "ref": ref,
                       "n_elem": info["n_elem"], "ndof": nd,
                       "tip_elem_over_h": info["tip_elem_over_h"],
                       "f1_healthy_Hz": f0[0], "f1_cracked_Hz": f[0],
                       "ratio_f1": r[0], "ratio_f2": r[1], "ratio_f3": r[2],
                       "shift_f1_pct": 100 * (1 - r[0]),
                       "shift_f2_pct": 100 * abs(r[1] - 1),
                       "shift_f3_pct": 100 * (1 - r[2])}
                for k in ("shift_f1_pct", "shift_f2_pct", "shift_f3_pct"):
                    row[f"d_{k}_rel"] = (abs(row[k] - prev[k]) / prev[k]
                                        if prev and prev[k] > 0 else np.nan)
                conv.append(row)
                prev = row
                print(f"[a11] conv {plane} ab={ab} ref={ref} ndof={nd} "
                      f"r1={r[0]:.5f} shift2={row['shift_f2_pct']:.4f}%")
    conv = pd.DataFrame(conv)
    _save(conv, "a11_convergence.csv")
    worst = conv[conv.ref == A11_REFS[-1]][["d_shift_f1_pct_rel",
                                            "d_shift_f2_pct_rel",
                                            "d_shift_f3_pct_rel"]].max().max()
    print(f"[a11] 최종 격자단계 이동량 상대변화 최대 = {100*worst:.2f} % (기준 5 %)")

    # (B) 주 표 — 작업격자에서 ā 스윕 + 등가 회전유연도 역산
    rows = []
    for plane in planes:
        for ab in A11_A_BARS:
            f0, f, info, nd0, nd, kinds = _a11_pair(c2, ab, plane, args.ref)
            r = f[:3] / f0[:3]
            c_2d = c2.invert_c_theta(beam, float(r[0]))
            # **f₁을 맞춘 회전스프링**의 고차모드 예측 — 2D의 mode-2 초과분 중
            # "등가 균열이 더 깊다"로 설명되는 부분과 "힌지 이상화 자체가 틀렸다"로
            # 남는 부분을 분리한다(폭→0 극한이 회전스프링에 수렴하는가의 정확한 형태).
            rh = c2.beam_ratios_from_cmm(beam, c_2d, n_elem=300, n_modes=3)
            cdim = cs.compliance_dimarogonas(ab, beam.h, beam.b, beam.E)
            ctad = cs.compliance(ab, beam.h, beam.b, beam.E, beam.nu,
                                 convention="tada")["c_MM"]
            rows.append({
                "plane": plane, "a_bar": ab, "ref": args.ref, "ndof": nd,
                "f1_healthy_Hz": f0[0], "f2_healthy_Hz": f0[1],
                "f3_healthy_Hz": f0[2],
                "f1_cracked_Hz": f[0], "f2_cracked_Hz": f[1], "f3_cracked_Hz": f[2],
                "ratio_f1": r[0], "ratio_f2": r[1], "ratio_f3": r[2],
                "shift_f2_pct": 100 * abs(r[1] - 1),
                "ratio_f2_hinge_matched": float(rh[1]),
                "ratio_f3_hinge_matched": float(rh[2]),
                "shift_f2_pct_hinge_matched": 100 * abs(float(rh[1]) - 1),
                "shift_f2_2d_over_hinge": (abs(r[1] - 1) / abs(rh[1] - 1)
                                           if abs(rh[1] - 1) > 0 else np.nan),
                "c_theta_2d": c_2d,
                "c_theta_dimarogonas": cdim, "c_theta_tada": ctad,
                "nd_c_theta_2d": c2.dimensionless_c_theta(c_2d, beam),
                "nd_c_theta_dimarogonas": c2.dimensionless_c_theta(cdim, beam),
                "nd_c_theta_tada": c2.dimensionless_c_theta(ctad, beam),
                "c2d_over_dimarogonas": c_2d / cdim,
                "c2d_over_tada": c_2d / ctad,
                "a_eq_dimarogonas": _a11_a_from_c_theta(beam, c_2d, "dimarogonas"),
                "a_eq_tada": _a11_a_from_c_theta(beam, c_2d, "tada"),
                "flap_kinds": ",".join(kinds[:4])})
            rows[-1]["a_eq_dim_over_physical"] = rows[-1]["a_eq_dimarogonas"] / ab
            rows[-1]["a_eq_tada_over_physical"] = rows[-1]["a_eq_tada"] / ab
            print(f"[a11] {plane} ab={ab} r1={r[0]:.5f} shift2={rows[-1]['shift_f2_pct']:.4f}% "
                  f"c2d/dim={c_2d/cdim:.3f} c2d/tada={c_2d/ctad:.3f}")
    main_df = pd.DataFrame(rows)
    _save(main_df, "a11_crack2d.csv")

    # (C) 폭 → 0 극한 — **같은 2D 모델 안에서** 유한폭 노치(요소제거)와 폭 0 슬릿을 비교한다.
    #     B2의 "노치 등가깊이가 13–70 % 깊다"(F10)가 순수 폭 효과인지 판정한다.
    wrows = []
    for depth in args.width_depths:
        for wmm in A11_WIDTHS_MM:
            f0, f, info, nd0, nd, _ = _a11_pair(c2, depth, args.width_plane,
                                                args.ref, kerf_width=wmm * 1e-3)
            r = f[:3] / f0[:3]
            c_2d = c2.invert_c_theta(beam, float(r[0]))
            a_eq = _a11_a_equiv(beam, float(r[0]), convention="tada")
            a_eq_dim = _a11_a_equiv(beam, float(r[0]), convention="dimarogonas")
            a_tad = _a11_a_from_c_theta(beam, c_2d, "tada")
            wrows.append({"plane": args.width_plane, "a_bar": depth,
                          "kerf_width_mm": wmm, "n_elem": info["n_elem"],
                          "ndof": nd, "ratio_f1": r[0], "ratio_f2": r[1],
                          "ratio_f3": r[2], "shift_f2_pct": 100 * abs(r[1] - 1),
                          "c_theta_2d": c_2d, "a_bar_equivalent": a_eq,
                          "a_bar_equiv_over_physical": a_eq / depth,
                          "a_bar_equiv_dimarogonas": a_eq_dim,
                          "a_bar_equiv_dim_over_physical": a_eq_dim / depth,
                          "a_eq_tada": a_tad,
                          "a_eq_tada_over_physical": a_tad / depth})
            print(f"[a11] width {wmm:.2f}mm depth={depth} r1={r[0]:.5f} f2r={r[1]:.6f} "
                  f"shift2={wrows[-1]['shift_f2_pct']:.4f}% a_eq/a={a_eq/depth:.3f} "
                  f"a_eq_tada/a={a_tad/depth:.3f}")
    _save(pd.DataFrame(wrows), "a11_width_limit.csv")

    # (D) 다섯 arm 대조표
    _save(_a11_arms(beam, main_df), "a11_arm_comparison.csv")

    # (E) 정본 Table 1 규약 대조 + 격차의 2인자 분해 — 표를 CSV에서 조판하기 위한 정본 소스.
    _save(_a11_table1(beam, main_df), "a11_table1_conventions.csv")
    _a11_figure(main_df, pd.DataFrame(wrows), conv)


#: 정본 Table 1의 균열깊이.
A11_TABLE1_A_BARS = (0.1, 0.3, 0.5, 0.6)


def _a11_table1(beam, main_df: pd.DataFrame) -> pd.DataFrame:
    """정본 Table 1을 세 규약으로 나란히 낸다 + 스프링↔2D 격차의 2인자 분해.

    **왜 논문3 안에서 다시 계산하는가.** Table 1의 값은 논문1 `crack_beam`의 해석 전달행렬에서
    나오는데 그 함수는 ā에서 Dimarogonas κ를 내부에서 만들고 논문1 코드는 수정 금지다.
    `cs.exact_eb_ratios`가 같은 8×8 특성행렬식을 c_θ를 직접 받아 푸는 논문3 소유 경로다
    (회귀검정이 Dimarogonas c_θ에서 논문1 함수와 1e-9 일치를 고정).

    **분해**(관측량마다 다르다 — 이것이 A11의 핵심 소득):
      conv_factor  = 강하(스프링·Tada) / 강하(스프링·Dimarogonas)   — 규약 편향
      hinge_factor = 강하(2D 폭 0)      / 강하(스프링·Tada)          — 점 힌지 이상화
    f₁에서는 hinge_factor ≈ 1.0(격차가 거의 전부 규약)이고, 곡률-null인 mode 2에서는
    두 인자가 대등하다 — 힌지 환원은 균열이 크게 움직이는 모드에서 정확하고 거의
    움직이지 않는 모드에서 부정확하다.
    """
    from . import eb_reference as cb
    ds = main_df[main_df.plane == "stress"].set_index("a_bar")
    dn = main_df[main_df.plane == "strain"].set_index("a_bar")
    rows = []
    for ab in A11_TABLE1_A_BARS:
        cdim = cs.compliance_dimarogonas(ab, beam.h, beam.b, beam.E)
        ctad = cs.compliance(ab, beam.h, beam.b, beam.E, beam.nu,
                             convention="tada")["c_MM"]
        rdim = cs.exact_eb_ratios(beam, cdim, n_modes=3)
        rtad = cs.exact_eb_ratios(beam, ctad, n_modes=3)
        r2s = np.array([ds.loc[ab, f"ratio_f{i}"] for i in (1, 2, 3)])
        r2n = np.array([dn.loc[ab, f"ratio_f{i}"] for i in (1, 2, 3)])
        row = {"a_bar": ab, "J": cb.flexibility_J(ab),
               "d0_equiv_Lc_h": cb.crack_knockdown(ab, beam.h, beam.L, 0.2).d0,
               "nd_c_theta_dimarogonas": c2_dimensionless(cdim, beam),
               "nd_c_theta_tada": c2_dimensionless(ctad, beam),
               "c_tada_over_dimarogonas": ctad / cdim}
        for tag, r in (("spring_dim", rdim), ("spring_tada", rtad),
                       ("2d_stress", r2s), ("2d_strain", r2n)):
            for i in range(3):
                row[f"ratio_f{i+1}_{tag}"] = float(r[i])
        for i, nm in ((0, "f1"), (1, "f2"), (2, "f3")):
            sd = abs(1 - rdim[i]) * 100
            st = abs(1 - rtad[i]) * 100
            s2 = abs(1 - r2s[i]) * 100
            row[f"shift_{nm}_pct_spring_dim"] = sd
            row[f"shift_{nm}_pct_spring_tada"] = st
            row[f"shift_{nm}_pct_2d_stress"] = s2
            row[f"conv_factor_{nm}"] = st / sd if sd > 0 else np.nan
            row[f"hinge_factor_{nm}"] = s2 / st if st > 0 else np.nan
            row[f"total_factor_{nm}"] = s2 / sd if sd > 0 else np.nan
        row["plane_rel_diff_shift_f1_pct"] = 100 * (abs(1 - r2n[0]) / abs(1 - r2s[0]) - 1)
        row["plane_rel_diff_shift_f2_pct"] = 100 * (abs(1 - r2n[1]) / abs(1 - r2s[1]) - 1)
        # 경쟁 모델(매끄러운 가우시안 강성장)을 **2D의 f₁ 강하**에 정합시킨다. 정본 §4.1의
        # 판별력 진술("mode 2가 0.1 % 대 2 %")은 정합 기준이 바뀌면 값이 바뀌므로 다시 낸다.
        gd, dmax = _a11_matched_gaussian(100 * (1 - r2s[0]))
        row["gauss_dmax_matched_2d"] = dmax
        for i, nm in ((0, "f1"), (1, "f2"), (2, "f3")):
            row[f"gauss_shift_{nm}_pct"] = gd[i]
        row["gauss_ratio_m2_over_m1_pct"] = 100 * abs(gd[1]) / abs(gd[0])
        row["crack2d_ratio_m2_over_m1_pct"] = (100 * abs(1 - r2s[1])
                                               / abs(1 - r2s[0]))
        rows.append(row)
        print(f"[a11] table1 ab={ab} f1: dim={row['shift_f1_pct_spring_dim']:.3f} "
              f"tada={row['shift_f1_pct_spring_tada']:.3f} 2D={row['shift_f1_pct_2d_stress']:.3f} "
              f"(conv ×{row['conv_factor_f1']:.3f}, hinge ×{row['hinge_factor_f1']:.3f}) | "
              f"m2 conv ×{row['conv_factor_f2']:.3f}, hinge ×{row['hinge_factor_f2']:.3f}")
    return pd.DataFrame(rows)


def c2_dimensionless(c_theta: float, beam) -> float:
    from . import crack2d as c2
    return c2.dimensionless_c_theta(c_theta, beam)


#: 경쟁 손상장(매끄러운 가우시안)의 폭 — 정본 §4.1·Fig 1(b)와 동일 규약(x̃ 단위).
A11_GAUSS_WIDTH = 0.12


def _a11_matched_gaussian(target_drop1_pct: float, xc_over_L: float = 0.2):
    """f₁ 강하를 맞춘 매끄러운 가우시안 강성장의 (Δf₁,Δf₂,Δf₃) [%]와 d_max.

    d_max에 단조이므로 brentq로 푼다(격자 스캔보다 빠르고 정확). Ritz 재해는 논문1
    `classical_ritz`를 **읽기전용**으로 쓴다.
    """
    from scipy.optimize import brentq

    from .eb_reference import solve_ritz
    geo = dict(L=VANE.L, h=VANE.h, E=VANE.E, rho=VANE.rho)
    fan = geo_mod.Beam(**geo, nu=VANE.nu).eb_frequencies(3)

    def drops(dmax):
        def dfun(xt, dm=dmax):
            return dm * np.exp(-((xt - xc_over_L) ** 2) / (A11_GAUSS_WIDTH ** 2))
        r = solve_ritz(**geo, n_modes=3, damage=dfun, n_trial=7)
        return [100 * (1 - r[k]["f"] / fan[k]) for k in range(3)]

    # 하한은 얕은 균열(ā = 0.1, 강하 0.43 %)도 브래킷하도록 충분히 작게 잡는다.
    lo, hi = 1e-3, 0.99
    if (drops(lo)[0] - target_drop1_pct) * (drops(hi)[0] - target_drop1_pct) >= 0:
        return [float("nan")] * 3, float("nan")
    dmax = float(brentq(lambda d: drops(d)[0] - target_drop1_pct, lo, hi, xtol=1e-5))
    return drops(dmax), dmax


def _a11_a_equiv(beam, ratio_f1: float, convention: str = "tada") -> float:
    """f₁ 강하와 일치하는 등가 균열깊이 ā_eq (B2와 동일 정의 — 규약을 **명시**한다).

    규약을 인자로 받는 이유: `cs.compliance`의 기본값이 2026-08-10에 dimarogonas → tada로
    바뀌었으므로(F42), 기본값에 의존하면 산출물의 의미가 조용히 바뀐다(설계서 F20의 교훈).
    """
    from scipy.optimize import brentq

    def g(ab):
        return cs.signature(beam, [ab], xc_over_L=0.2, n_modes=1, n_elem=200,
                            shear_flex=False, convention=convention
                            )[0]["ratio_f1"] - ratio_f1

    lo, hi = 0.02, 0.90
    if g(lo) * g(hi) >= 0:
        return float("nan")
    return float(brentq(g, lo, hi, xtol=1e-4))


def _a11_a_from_c_theta(beam, c_target: float, convention: str) -> float:
    """c_θ = c_target을 주는 균열깊이 — **규약별** 등가깊이.

    B2·F10의 ā_eq는 Dimarogonas 규약으로 정의됐다. 같은 c_θ를 Tada 규약으로 되읽으면
    등가깊이가 달라지므로, "노치가 더 깊게 작용한다"의 얼마가 폭 효과이고 얼마가
    **핸드북 규약차**(F7)인지 분리할 수 있다.
    """
    from scipy.optimize import brentq

    def c_of(ab):
        if convention == "dimarogonas":
            return cs.compliance_dimarogonas(ab, beam.h, beam.b, beam.E)
        return cs.compliance(ab, beam.h, beam.b, beam.E, beam.nu,
                             convention="tada")["c_MM"]

    if not np.isfinite(c_target):
        return float("nan")
    lo, hi = 0.02, 0.95
    if (c_of(lo) - c_target) * (c_of(hi) - c_target) >= 0:
        return float("nan")
    return float(brentq(lambda ab: c_of(ab) - c_target, lo, hi, xtol=1e-5))


def _a11_arms(beam, main_df: pd.DataFrame) -> pd.DataFrame:
    """(a) 정확 전달행렬 회전스프링 — Dimarogonas 규약(정본 Table 1)과 Tada 규약(생산 규약)
    (b) A7 Timoshenko+Mode-I/II (c) B2 3D 노치 (d) 2D 폭 0 슬릿."""
    from . import eb_reference as cb
    rows = []
    f0 = np.array(cb.cracked_cantilever_frequencies(VANE.L, VANE.h, VANE.E,
                                                    VANE.rho, 0.0, 0.2, n_modes=3))
    for ab in A11_A_BARS:
        f = np.array(cb.cracked_cantilever_frequencies(VANE.L, VANE.h, VANE.E,
                                                       VANE.rho, ab, 0.2, n_modes=3))
        rows.append({"arm": "a_exact_spring_EB", "convention": "dimarogonas",
                     "a_bar": ab, "width_mm": 0.0, **_a11_ratio_cols(f / f0)})
        ctad = cs.compliance(ab, beam.h, beam.b, beam.E, beam.nu,
                             convention="tada")["c_MM"]
        rows.append({"arm": "a_exact_spring_EB", "convention": "tada",
                     "a_bar": ab, "width_mm": 0.0,
                     **_a11_ratio_cols(cs.exact_eb_ratios(beam, ctad, n_modes=3))})
    # 규약을 **둘 다** 낸다: tada = 생산 규약(F42), dimarogonas = 정본 Table 1·F6·F44의
    # 기준선(하위호환). 둘을 나란히 두어야 규약 인자와 힌지 인자를 분리해 인용할 수 있다.
    for shear, coup, tag in ((False, 0.0, "b_timoshenko_modeI"),
                             (True, 0.0, "b_timoshenko_modeI+II"),
                             (True, 0.9, "b_timoshenko_modeI+II_c0.9")):
        for conv in ("tada", "dimarogonas"):
            for r in cs.signature(beam, A11_A_BARS, xc_over_L=0.2, n_modes=3,
                                  n_elem=300, coupling=coup, shear_flex=shear,
                                  convention=conv):
                rows.append({"arm": tag, "convention": conv, "a_bar": r["a_bar"],
                             "width_mm": 0.0,
                             **_a11_ratio_cols(np.array([r["ratio_f1"], r["ratio_f2"],
                                                         r["ratio_f3"]]))})
    b2 = DATA / "b2_vane3d.csv"
    if b2.exists():
        d = pd.read_csv(b2)
        for _, s in d.iterrows():
            # 스팬방향 폭이 여러 개면 arm 이름으로 구분한다 — as-built 4.1 mm와 옛 규약 2h가
            # 같은 (깊이, 커프폭) 행으로 겹치면 표에서 둘을 분간할 수 없다.
            vw = float(s.get("vane_width_mm", 2e3 * VANE.h))
            rows.append({"arm": f"c_3d_notch_w{vw:.1f}", "convention": "fem",
                         "a_bar": s["depth_frac_actual"],
                         "width_mm": s["kerf_width_actual_mm"],
                         "vane_width_mm": vw,
                         **_a11_ratio_cols(np.array([s["ratio_f1"], s["ratio_f2"],
                                                     s["ratio_f3"]]))})
    else:
        print("[a11] 경고: b2_vane3d.csv 없음 — 3D arm 열이 비었다")
    for _, s in main_df.iterrows():
        rows.append({"arm": f"d_2d_{s['plane']}_slit", "convention": "fem",
                     "a_bar": s["a_bar"], "width_mm": 0.0,
                     **_a11_ratio_cols(np.array([s["ratio_f1"], s["ratio_f2"],
                                                 s["ratio_f3"]]))})
    return pd.DataFrame(rows).sort_values(["a_bar", "arm", "convention"])


def _a11_ratio_cols(r) -> dict:
    return {"ratio_f1": float(r[0]), "ratio_f2": float(r[1]), "ratio_f3": float(r[2]),
            "shift_f1_pct": 100 * (1 - float(r[0])),
            "shift_f2_pct": 100 * abs(float(r[1]) - 1),
            "shift_f3_pct": 100 * (1 - float(r[2]))}


def _a11_figure(main_df, wdf, conv):
    plt = _plt()
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0))
    ps = main_df[main_df.plane == "stress"].sort_values("a_bar")
    ax[0].plot(ps.a_bar, 100 * (1 - ps.ratio_f1), "o-", label="2D slit, mode 1")
    ax[0].plot(ps.a_bar, ps.shift_f2_pct, "s-", label="2D slit, mode 2")
    ax[0].set_yscale("log"); ax[0].set_xlabel(r"$\bar a = a/h$")
    ax[0].set_ylabel(r"$|\Delta f_m/f_m|$ [%]"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[0].set_title("(a)")
    for plane, mk in (("stress", "o-"), ("strain", "^--")):
        d = main_df[main_df.plane == plane].sort_values("a_bar")
        if d.empty:
            continue
        ax[1].plot(d.a_bar, d.c2d_over_tada, mk, label=f"/ Tada ({plane})")
        ax[1].plot(d.a_bar, d.c2d_over_dimarogonas, mk,
                   label=f"/ Dimarogonas ({plane})")
    ax[1].axhline(1.0, color="k", lw=.8)
    ax[1].set_xlabel(r"$\bar a$"); ax[1].set_ylabel(r"$c_\theta^{2D}/c_\theta^{handbook}$")
    ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)
    ax[1].set_title("(b)")
    for depth, g in wdf.groupby("a_bar"):
        g = g.sort_values("kerf_width_mm")
        ln, = ax[2].plot(g.kerf_width_mm, g.a_bar_equiv_over_physical, "o-",
                         label=rf"$\bar a$ = {depth} (Dimarogonas)")
        ax[2].plot(g.kerf_width_mm, g.a_eq_tada_over_physical, "s--",
                   color=ln.get_color(), label=rf"$\bar a$ = {depth} (Tada)")
    ax[2].axhline(1.0, color="k", lw=.8)
    ax[2].set_xlabel("kerf width [mm]")
    ax[2].set_ylabel(r"$\bar a_{eq}/\bar a_{phys}$")
    ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)
    ax[2].set_title("(c)")
    fig.tight_layout()
    _ensure_dirs()
    path = FIGS / "fig_a11_crack2d.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print(f"[saved] {path}")


# ---------------------------------------------------------------- A8
def cmd_a8(args):
    from . import eb_reference as fl
    rows = []
    for navmi in (0.5, 0.65, 0.8):
        for b_over_h in (2.0, 4.0, 8.0):
            beta = fl.beta_beam(b_over_h, VANE.rho, navmi=navmi)
            rows.append({"component": "vane", "navmi": navmi, "b_over_h": b_over_h,
                         "beta": beta, "f_wet_over_f_dry": fl.wet_ratio(beta),
                         "drop_pct": 100 * (1 - fl.wet_ratio(beta)),
                         "note": "strip theory + NAVMI; E1 measurement pending"})
    _save(pd.DataFrame(rows), "a8_wet_correction.csv")


# ---------------------------------------------------------------- B1
def cmd_b1(args):
    xi_list = np.linspace(0.05, 0.95, args.n_loc)
    s_list = (0.01, 0.05, 0.15)
    rows = mc.run_production(PLATE, MODES, xi_list, s_list, SIGMA_RELS,
                             w=W_GAUSS, n_real=args.n_real,
                             n_workers=args.workers, n_grid=1001,
                             n_real_requested=5000,
                             mass=(args.mass or None))
    _save(pd.DataFrame(rows),
          "b1_mc_summary_mass.csv" if args.mass else "b1_mc_summary.csv")


# ---------------------------------------------------------------- B2
def cmd_b2(args):
    """3D 베인 + EDM 커프: 실명 예측의 3D 탄성 검정 + 노치→c_θ 등가(E2 (b)(c)(d) 사전계산).

    **스팬방향 폭(`--vane-widths`)**: 실측으로 베인은 두께 t_f = 1.0 mm의 판이고 그 폭은
    유로폭 b₂ = **4.1 mm**(두 슈라우드 사이 높이)다. 따라서 as-built 쿠폰은 30 × 4.1 × 1.0 mm다.
    옛 산출물은 폭을 `2h`로 가정했으므로(= 2.0 mm) 차원성 항(F46)의 연속성 확인용으로 함께 낸다.
    """
    from scipy.optimize import brentq

    from . import rail3d as r3
    beam = cs.TimoBeam(L=VANE.L, h=VANE.h, b=2 * VANE.h, E=VANE.E, rho=VANE.rho,
                       nu=VANE.nu)
    nx, ny, nz = args.nx, args.ny, args.nz
    rows = []
    for w_vane in args.vane_widths:
        _b2_one_width(args, r3, brentq, beam, w_vane, nx, ny, nz, rows)
    _save(pd.DataFrame(rows), "b2_vane3d.csv")


def _b2_one_width(args, r3, brentq, beam, w_vane, nx, ny, nz, rows):
    """한 스팬방향 폭에 대한 B2 스윕(건전 1회 + 깊이×커프폭)."""
    c0, cn0, _ = r3.vane_mesh(L=VANE.L, w=w_vane, h=VANE.h, nx=nx, ny=ny, nz=nz)
    res0 = r3.solve_modes(c0, cn0, VANE.E, VANE.nu, VANE.rho, r3.clamp_root(),
                          n_modes=10, order=2, keep_shapes=True)
    kinds0 = r3.beam_mode_kinds(res0)
    flap0 = [i for i, k in enumerate(kinds0) if k == "flap"]
    f0 = res0.freqs[flap0]                      # **형상으로 고른** 면외 굽힘 계열
    print(f"[b2] w={w_vane*1e3:.2f}mm healthy 3D all={np.round(res0.freqs[:6], 1)} "
          f"kinds={kinds0[:6]}")
    print(f"[b2] flapwise: {np.round(f0[:3], 1)} (ndof={res0.ndof})")

    for depth in args.depths:
        for width in args.widths:
            c, cnn, info = r3.vane_mesh(L=VANE.L, w=w_vane, h=VANE.h, nx=nx, ny=ny,
                                        nz=nz, kerf={"xc_over_L": 0.2,
                                                     "width": width,
                                                     "depth_frac": depth})
            res = r3.solve_modes(c, cnn, VANE.E, VANE.nu, VANE.rho, r3.clamp_root(),
                                 n_modes=10, order=2, keep_shapes=True)
            flap = [i for i, k in enumerate(r3.beam_mode_kinds(res)) if k == "flap"]
            f = res.freqs[flap]
            # 등가 균열깊이: **무차원 주파수비**로 맞춘다(모델 간 건전주파수 차이를 상쇄).
            # 3D 노치의 f1 강하와 같은 강하를 주는 이상균열 깊이 ā_eq → E2 (b)(c)(d) 단계.
            target = f[0] / f0[0]

            # 등가깊이는 **규약에 따라 다르다**(F42·F43). 생산값은 `tada`이고
            # `dimarogonas`는 F10 이전 값과의 연속성 확인용으로 함께 낸다.
            def ratio_model(ab: float, conv: str) -> float:
                return cs.signature(beam, [ab], xc_over_L=0.2, n_modes=1,
                                    n_elem=200, shear_flex=False,
                                    convention=conv)[0]["ratio_f1"]

            eq = {}
            note = ""
            for conv in ("tada", "dimarogonas"):
                a_eq, c_eq = float("nan"), float("nan")
                try:
                    lo_ab, hi_ab = 0.02, 0.90
                    if ((ratio_model(lo_ab, conv) - target)
                            * (ratio_model(hi_ab, conv) - target) < 0):
                        a_eq = brentq(lambda ab: ratio_model(ab, conv) - target,
                                      lo_ab, hi_ab, xtol=1e-4)
                        c_eq = (cs.compliance_dimarogonas(a_eq, beam.h, beam.b, beam.E)
                                if conv == "dimarogonas" else
                                cs.compliance(a_eq, beam.h, beam.b, beam.E, beam.nu,
                                              convention="tada")["c_MM"])
                    else:
                        note = "no bracket: notch beyond crack-model range"
                except Exception as exc:                  # pragma: no cover
                    note = f"solve failed: {exc}"
                eq[conv] = (a_eq, c_eq)
            (a_eq, c_eq) = eq["tada"]
            (a_eq_dim, c_eq_dim) = eq["dimarogonas"]
            rows.append({"vane_width_mm": w_vane * 1e3,
                         "vane_width_over_h": w_vane / VANE.h,
                         "depth_frac_requested": depth,
                         "depth_frac_actual": info["depth_actual"],
                         "kerf_width_requested_mm": width * 1e3,
                         "kerf_width_actual_mm": info["width_actual"] * 1e3,
                         "f1_flap_Hz": f[0], "f2_flap_Hz": f[1],
                         "f3_flap_Hz": f[2] if len(f) > 2 else np.nan,
                         "ratio_f1": f[0] / f0[0], "ratio_f2": f[1] / f0[1],
                         "ratio_f3": (f[2] / f0[2]) if len(f) > 2 and len(f0) > 2 else np.nan,
                         "abs_shift_f2_pct": 100 * abs(f[1] / f0[1] - 1),
                         "convention": "tada",
                         "c_MM_equivalent": c_eq,
                         "a_bar_equivalent": a_eq,
                         "a_bar_equiv_over_physical": (a_eq / info["depth_actual"]
                                                       if np.isfinite(a_eq) else np.nan),
                         "c_MM_equivalent_dimarogonas": c_eq_dim,
                         "a_bar_equivalent_dimarogonas": a_eq_dim,
                         "a_bar_equiv_dim_over_physical": (
                             a_eq_dim / info["depth_actual"]
                             if np.isfinite(a_eq_dim) else np.nan),
                         "note": note, "ndof": res.ndof})
            print(f"[b2] w={w_vane*1e3:.2f}mm depth={depth} width={width*1e3:.2f}mm -> "
                  f"f1r={rows[-1]['ratio_f1']:.4f} f2r={rows[-1]['ratio_f2']:.5f} "
                  f"a_eq={a_eq:.3f}")


# ---------------------------------------------------------------- B3
#: B3 포켓의 방위중심 — nθ ∈ {24,48,72,96}에서 Δθ ∈ {15°,30°,60°}가 **셀면에 정확히**
#: 놓이도록 22.5°(=π/8)로 잡는다. 옛 값 0.4 rad(=22.918°)는 어느 격자에서도 셀면이 아니어서
#: 실현 포켓이 0.4° 어긋났다(형상보존 규약, 설계서 F11′).
B3_THETA0 = np.pi / 8.0


def cmd_b3(args):
    """3D 환형판 + 포켓: 분리·배향 ground truth와 FEM 수준 e_pert(설계서 §3.6-i·-iii).

    **형상보존 격자 규약(F11′)**: 포켓 반경경계를 ξ_c ± `--half-xi`(기본 0.1)에 두면
    경계가 ξ의 0.1 배수에 놓이므로 nr ∈ {10,20,30}에서 정확히 셀면에 떨어진다.
    nθ는 Δθ의 정수배, nz는 깊이비의 정수배여야 한다. 실현 형상이 요청과 다르면
    `shape_exact=False`로 기록한다 — 격자마다 형상이 달라지면 메시수렴 비교가 무효다.

    **m 매칭은 전부 형상 기반**(`rail3d.match_order`). 주파수 근접 폴백은 쓰지 않는다.
    포켓이 축대칭을 강하게 깨 m=0 지배 모드가 없으면 그 셀의 m=0을 **버린다**(F21).
    """
    from . import rail3d as r3

    nr, nt, nz = args.nr, args.ntheta, args.nz
    pmin = args.purity_min
    c0, cn0, _ = r3.disk_mesh(a=PLATE.a, b=PLATE.b, t=PLATE.t, nr=nr, ntheta=nt,
                              nz=nz)
    res0 = r3.solve_modes(c0, cn0, PLATE.E, PLATE.nu, PLATE.rho,
                          r3.clamp_inner_rim(PLATE.a), n_modes=12, order=2,
                          keep_shapes=True)
    lam0 = (2 * np.pi * res0.freqs) ** 2
    ord0, _, pur0 = r3.azimuthal_orders(res0)
    print(f"[b3] healthy 3D: {np.round(res0.freqs[:8], 1)} m={ord0[:8]} "
          f"(ndof={res0.ndof}, purity≥{pur0.min():.2f})")
    h0 = {m: r3.match_order(res0, m, n_take=(1 if m == 0 else 2), purity_min=pmin)
          for m in (0, 1, 2, 3)}
    for m in (0, 1, 2, 3):
        if not h0[m]["matched"]:
            raise SystemExit(f"[b3] 건전 격자에서 m={m} 형상매칭 실패 — 격자를 의심할 것")
    idx_m0 = h0[0]["idx"][0]
    pair_idx = {m: tuple(h0[m]["idx"]) for m in (1, 2, 3)}

    rows = []
    for xi in args.xi:
        xi1, xi2 = xi - args.half_xi, xi + args.half_xi
        r1 = sev.xi_to_r(xi1, PLATE.a, PLATE.b)
        r2 = sev.xi_to_r(xi2, PLATE.a, PLATE.b)
        for dth_deg in args.dtheta:
            for depth in args.depths:
                pk = {"r1": r1, "r2": r2, "theta0": B3_THETA0,
                      "dtheta": np.deg2rad(dth_deg), "depth_frac": depth}
                c, cnn, info = r3.disk_mesh(a=PLATE.a, b=PLATE.b, t=PLATE.t, nr=nr,
                                            ntheta=nt, nz=nz, pocket=pk)
                res = r3.solve_modes(c, cnn, PLATE.E, PLATE.nu, PLATE.rho,
                                     r3.clamp_inner_rim(PLATE.a), n_modes=12,
                                     order=2, keep_shapes=True)
                lam = (2 * np.pi * res.freqs) ** 2
                # 진단용 방위차수 투영(순도)과 **매칭 기준인 subspace MAC**을 함께 낸다.
                _, psid, _ = r3.azimuthal_orders(res)
                md = {m: r3.match_order(res, m, n_take=(1 if m == 0 else 2),
                                        purity_min=pmin) for m in (0, 1, 2, 3)}
                mac = {m: r3.subspace_mac_match(res0, h0[m]["idx"], res,
                                                n_take=(1 if m == 0 else 2),
                                                mac_min=args.mac_min)
                       for m in (0, 1, 2, 3)}
                obs, orient = {}, {}
                for m, (j1, j2) in pair_idx.items():
                    if not mac[m]["matched"]:
                        continue
                    k1, k2 = mac[m]["idx"]                # MAC으로 찾은 손상 후 쌍
                    l0 = 0.5 * (lam0[j1] + lam0[j2])
                    obs[m] = (0.5 * (lam[k1] + lam[k2]) / l0 - 1.0,
                              (lam[k2] - lam[k1]) / l0)
                    orient[m] = (float(psid[k1]), float(psid[k2]), md[m]["purity"][0])
                # --- m=0: 형상(MAC) 기반. 미달이면 짝짓지 않고 버린다(F21).
                m0_ok = mac[0]["matched"]
                eta_m0 = (lam[mac[0]["idx"][0]] / lam0[idx_m0] - 1.0
                          if m0_ok else float("nan"))

                # 이론(섭동 H^(m))과 대조 — **실현된** 형상으로.
                pocket = deg.Pocket(info["r1_actual"], info["r2_actual"],
                                    info["theta0_actual"], info["dtheta_actual"],
                                    info["depth_actual"])
                theory = {m: deg.observables(PLATE, m, pocket, n_grid=2001, n_r=2001)
                          for m in pair_idx}
                s_true = (1 - (1 - info["depth_actual"]) ** 3) * \
                    (info["r2_actual"] - info["r1_actual"]) / PLATE.extent * \
                    (info["dtheta_actual"] / (2 * np.pi))
                row = {"xi_d_true": xi, "dtheta_deg": dth_deg,
                       "depth_frac": info["depth_actual"],
                       "s_bar_true_pct": s_true * 100,
                       "eta_m0_3d": eta_m0, "m0_matched": m0_ok,
                       "m0_mac": mac[0]["mac"][0],
                       "m0_purity": md[0]["purity"][0],
                       "m0_is_dominant": md[0]["is_dominant"][0],
                       "nr": nr, "ntheta": nt, "nz": nz, "ndof": res.ndof,
                       "shape_exact": info["shape_exact"],
                       "r1_mm": info["r1_actual"] * 1e3,
                       "r2_mm": info["r2_actual"] * 1e3,
                       "theta0_deg": np.rad2deg(info["theta0_actual"]),
                       "dtheta_actual_deg": np.rad2deg(info["dtheta_actual"]),
                       "snap_r1_mm": info["snap_r1_mm"],
                       "snap_r2_mm": info["snap_r2_mm"]}
                for m in (1, 2, 3):
                    row[f"m{m}_matched"] = mac[m]["matched"]
                    row[f"m{m}_mac"] = min(mac[m]["mac"])
                    row[f"m{m}_purity"] = min(md[m]["purity"])
                    row[f"m{m}_order_agrees_mac"] = (tuple(mac[m]["idx"])
                                                     == tuple(md[m]["idx"]))
                    if m not in obs:
                        continue
                    row[f"eta_bar_3d_m{m}"] = obs[m][0]
                    row[f"eta_bar_theory_m{m}"] = theory[m]["eta_bar"]
                    row[f"delta_eta_3d_m{m}"] = abs(obs[m][1])
                    row[f"delta_eta_theory_m{m}"] = theory[m]["delta_eta"]
                    # FEM 수준 e_pert (설계서 §3.6-iii): |η̄^lin − η̄^3D|.
                    # A2의 e_pert(Kirchhoff 안의 선형화 오차)와 **다른 양**이다 —
                    # 여기에는 모델형식(Kirchhoff↔3D 탄성) 차이와 이산화가 함께 들어간다.
                    row[f"epert_fem_abs_m{m}"] = abs(theory[m]["eta_bar"] - obs[m][0])
                    row[f"epert_fem_rel_m{m}"] = (abs(theory[m]["eta_bar"] - obs[m][0])
                                                  / max(abs(obs[m][0]), 1e-300))
                    row[f"B_sign_theory_m{m}"] = "+" if theory[m]["B_signed"] > 0 else "-"
                    # 3D의 부호 판정: 낮은 쪽 짝의 배향이 손상 방위와 정렬이면 B̄<0
                    psi_lo = orient[m][0]
                    period = np.pi / m
                    d = abs(((psi_lo - info["theta0_actual"]) % period))
                    aligned = min(d, period - d) < 0.3 * period
                    row[f"B_sign_3d_m{m}"] = "-" if aligned else "+"
                    row[f"psi_lower_3d_m{m}_deg"] = np.rad2deg(psi_lo)
                    row[f"sign_agrees_m{m}"] = (row[f"B_sign_3d_m{m}"]
                                                == row[f"B_sign_theory_m{m}"])
                rows.append(row)
                em2 = obs.get(2, (float("nan"),))[0]
                print(f"[b3] xi={xi} dθ={dth_deg}° depth={info['depth_actual']:.2f} "
                      f"exact={info['shape_exact']} -> η̄_m2 3D {em2:.3e} vs 이론 "
                      f"{theory[2]['eta_bar']:.3e} | MAC "
                      f"{[round(min(mac[m]['mac']), 3) for m in (0, 1, 2, 3)]} "
                      f"| m0 {'OK' if m0_ok else 'DROP'} η̄_m0={eta_m0:.3e}")
    _save(pd.DataFrame(rows), "b3_disk3d.csv")


# ---------------------------------------------------------------- B4
def cmd_b4(args):
    """메시수렴 — 이동량 기준(정본 §3.6): 건전·손상 f 각 <0.1–0.2 %, 상대이동 <5 %."""
    from . import rail3d as r3
    rows = []
    # 반경 경계도 격자에 맞춰야 하므로 nr의 공약 배수 위치를 쓴다(a+0.3L, a+0.6L).
    pk = {"r1": PLATE.a + 0.3 * PLATE.extent, "r2": PLATE.a + 0.6 * PLATE.extent,
          "theta0": 0.0, "dtheta": np.deg2rad(30), "depth_frac": 0.5}
    for nr, nt, nz in args.grids:
        c0, cn0, _ = r3.disk_mesh(a=PLATE.a, b=PLATE.b, t=PLATE.t, nr=nr,
                                  ntheta=nt, nz=nz)
        r0 = r3.solve_modes(c0, cn0, PLATE.E, PLATE.nu, PLATE.rho,
                            r3.clamp_inner_rim(PLATE.a), n_modes=8, order=2)
        c1, cn1, info = r3.disk_mesh(a=PLATE.a, b=PLATE.b, t=PLATE.t, nr=nr,
                                     ntheta=nt, nz=nz, pocket=pk)
        r1_ = r3.solve_modes(c1, cn1, PLATE.E, PLATE.nu, PLATE.rho,
                             r3.clamp_inner_rim(PLATE.a), n_modes=8, order=2)
        lam0 = (2 * np.pi * r0.freqs) ** 2
        lam = (2 * np.pi * r1_.freqs) ** 2
        l0 = 0.5 * (lam0[3] + lam0[4])
        rows.append({"nr": nr, "ntheta": nt, "nz": nz, "ndof": r0.ndof,
                     "f_m0_healthy": r0.freqs[0], "f_m2_healthy": r0.freqs[3],
                     "eta_bar_m2": 0.5 * (lam[3] + lam[4]) / l0 - 1.0,
                     "delta_eta_m2": abs(lam[4] - lam[3]) / l0,
                     "shape_exact": info["shape_exact"],
                     "snap_r1_mm": info["snap_r1_mm"],
                     "snap_r2_mm": info["snap_r2_mm"],
                     "dtheta_actual_deg": np.rad2deg(info["dtheta_actual"]),
                     "depth_actual": info["depth_actual"],
                     "r1_actual_mm": info["r1_actual"] * 1e3,
                     "r2_actual_mm": info["r2_actual"] * 1e3,
                     "geometry_key": (f"{np.rad2deg(info['dtheta_actual']):.2f}deg_"
                                      f"d{info['depth_actual']:.3f}_"
                                      f"r{info['r1_actual']*1e3:.2f}-{info['r2_actual']*1e3:.2f}")})
        print(f"[b4] {nr}x{nt}x{nz}: ndof={r0.ndof} f_m2={r0.freqs[3]:.1f} "
              f"η̄={rows[-1]['eta_bar_m2']:.4e} Δη={rows[-1]['delta_eta_m2']:.4e}")
    _save(pd.DataFrame(rows), "b4_convergence.csv")


# ---------------------------------------------------------------- B3X
def cmd_b3x(args):
    """B3의 3D pair mean을 **생산 순방향 맵**(질량항·정확결합)으로 재역식별 — model-form penalty.

    `b3_disk3d.csv`의 역식별 열은 질량항 도입(F12) *이전*의 강성전용 추정기로 계산됐다.
    3D 해는 비싸므로 재계산하지 않고, 저장된 pair mean을 입력으로 추정기만 다시 돌린다.
    같은-모델 baseline(섭동맵이 스스로 만든 관측량을 역식별)과의 차이가 model-form penalty다.
    """
    from . import estimator as est
    src = DATA / "b3_disk3d.csv"
    if not src.exists():
        raise SystemExit(f"{src} 없음 — 먼저 `cli b3`를 실행할 것")
    d = pd.read_csv(src)
    pool = _pool()
    sigma = noi.sigma_y_for_modes([m for m, _ in MODES], args.sigma_rel)
    sigma3 = noi.sigma_y_for_modes([1, 2, 3], args.sigma_rel)          # 참고용(4모드 중 m>0 3개)
    rows = []
    for _, r in d.iterrows():
        y3d = np.array([r["eta_m0_3d"], r["eta_bar_3d_m1"],
                        r["eta_bar_3d_m2"], r["eta_bar_3d_m3"]])
        xi_t, s_t = float(r["xi_d_true"]), float(r["s_bar_true_pct"]) / 100.0
        # m=0은 포켓이 축대칭을 깨면 순수 m=0 모드가 없을 수 있어 `cmd_b3`가 셀을 버린다
        # (F21: 옛 코드의 **주파수 근접 폴백**은 §3.6이 금지한 매칭이었다). 버려진 셀에서는
        # 전체모드 적합을 내지 않는다 — 없는 관측량을 지어내지 않는다.
        m0_ok = bool(r.get("m0_matched", True)) and np.isfinite(y3d[0])
        # m=0 포함 적합 — MAC 미달로 버려진 차수는 넣지 않는다(없는 관측량을 지어내지 않는다).
        use_a = [i for i in (0, 1, 2, 3) if np.isfinite(y3d[i])]
        f3 = None
        if m0_ok and len(use_a) >= 3:
            f3 = est.fit(y3d[use_a], [pool[i] for i in use_a], PLATE,
                         noi.sigma_y_for_modes(use_a, args.sigma_rel), w=W_GAUSS,
                         n_starts=8, free_w=True, mass="exact")
        # m>0만 쓴 적합 — 정본이 인용하는 값이다.
        use = [i for i in (1, 2, 3) if np.isfinite(y3d[i])]
        if len(use) < 2:
            print(f"[b3x] xi={r['xi_d_true']} dθ={r['dtheta_deg']:.0f}° -> "
                  f"관측량 {len(use)}개뿐 — 2모수 역식별 불가, 셀 제외")
            continue
        pool_u = [pool[i] for i in use]
        sig_u = noi.sigma_y_for_modes(use, args.sigma_rel)
        # 관측량이 2개뿐이면 w를 풀 수 없다(3모수 > 2관측). 자유도를 함께 기록한다.
        free_w = len(use) >= 3
        n_par = 3 if free_w else 2
        f3m = est.fit(y3d[use], pool_u, PLATE, sig_u, w=W_GAUSS, n_starts=8,
                      free_w=free_w, mass="exact")
        # 같은-모델 baseline: 섭동맵이 스스로 만든 관측량(자기일관 데이터)을 역식별
        y_self = fwd.eta_bar_linear_mass(pool, xi_t, s_t, W_GAUSS, PLATE,
                                         coupling="exact")
        fs = est.fit(y_self, pool, PLATE, sigma, w=W_GAUSS, n_starts=8, free_w=True,
                     mass="exact")
        nan = float("nan")
        crlb3 = idf.metrics(pool_u, PLATE, (xi_t, s_t), W_GAUSS, sig_u,
                            mass="exact")["crlb_xi_mm"]
        err_mgt0 = (f3m["xi_d"] - xi_t) * PLATE.extent * 1e3
        rows.append({"xi_d_true": xi_t, "dtheta_deg": r["dtheta_deg"],
                     "depth_frac": r["depth_frac"], "s_bar_true_pct": r["s_bar_true_pct"],
                     "sigma_rel": args.sigma_rel, "mass_coupling": "exact",
                     "m0_matched": m0_ok,
                     "modes_used_all": "+".join(f"m{i}" for i in use_a),
                     "modes_used_mgt0": "+".join(f"m{i}" for i in use),
                     "n_obs_mgt0": len(use), "n_par_mgt0": n_par,
                     "dof_mgt0": len(use) - n_par,
                     "xi_hat_3d": f3["xi_d"] if f3 else nan,
                     "s_hat_3d_pct": f3["s_bar"] * 100 if f3 else nan,
                     "err_xi_mm_3d": ((f3["xi_d"] - xi_t) * PLATE.extent * 1e3
                                      if f3 else nan),
                     "err_s_pp_3d": (f3["s_bar"] * 100 - r["s_bar_true_pct"]
                                     if f3 else nan),
                     "w_hat_mm_3d": f3["w"] * 1e3 if f3 else nan,
                     "chi2_3d": f3["chi2"] if f3 else nan,
                     "boundary_hit_3d": f3["boundary_hit"] if f3 else nan,
                     "crlb_xi_mm_mgt0": crlb3,
                     "xi_hat_3d_mgt0": f3m["xi_d"],
                     "err_xi_mm_3d_mgt0": err_mgt0,
                     "penalty_over_crlb_mgt0": abs(err_mgt0) / crlb3,
                     "err_s_pp_3d_mgt0": f3m["s_bar"] * 100 - r["s_bar_true_pct"],
                     "chi2_3d_mgt0": f3m["chi2"],
                     "boundary_hit_3d_mgt0": f3m["boundary_hit"],
                     "err_xi_mm_samemodel": (fs["xi_d"] - xi_t) * PLATE.extent * 1e3,
                     "err_s_pp_samemodel": fs["s_bar"] * 100 - r["s_bar_true_pct"],
                     "chi2_samemodel": fs["chi2"],
                     "model_form_penalty_xi_mm": abs(err_mgt0)
                     - abs((fs["xi_d"] - xi_t) * PLATE.extent * 1e3)})
        print(f"[b3x] xi={xi_t} dθ={r['dtheta_deg']:.0f}° -> 3D역식별(m>0) "
              f"{err_mgt0:+.2f} mm = {abs(err_mgt0)/crlb3:.1f}×CRLB "
              f"(같은모델 {rows[-1]['err_xi_mm_samemodel']:+.2e} mm, "
              f"m0 {'포함가능' if m0_ok else '버림'})")
    _save(pd.DataFrame(rows), "b3_modelform_mass.csv")


# ---------------------------------------------------------------- A10
def cmd_a10(args):
    """샌드위치 vs 균일판 손상법칙 비교 (설계서 §5.3, 리뷰 지적 5).

    레일(균일두께 환형판)에서는 두께제거가 D ∝ t³ → d_K = 1−(1−p)³, d_M = p로 **정확**하다.
    실물 슈라우드는 면판 2장 샌드위치라 한쪽 면판을 깊이 p·t_f 가공하면
    D_eff ∝ t₁t₂s²/(t₁+t₂) → d_K = p/(2−p), d_M = p/2, 즉 **d_M = d_K/(1+d_K)** 다.
    두 법칙이 ξ*·e_pert·CRLB·E3 가공깊이 예측을 각각 얼마나 움직이는지 산출한다.

    **2026-08-13 실측 반영**: t_f = 1.0 mm, s = 5.1 mm이 확정돼 t_f/s = 0.196이다. 이는 얇은면판
    극한이 아니므로 위 닫힌형(선행차수)은 d_K를 19–39 % 과소평가한다. 정확 단면으로 계산한
    **as-built 법칙**(`sandwich_asbuilt`)을 세 번째 법칙으로 함께 낸다(설계서 F60).
    """
    from . import figures as figs
    pool = _pool()
    rows = []
    T_F, S_SEP = geo.T_SHEET, geo.FACE_SEPARATION
    LAWS = ("exact", "sandwich", "sandwich_asbuilt")

    # (0) as-built 단면 — 실측 t_f/s에서 선행차수·1차보정·정확값의 관계
    for p in (0.02, 0.05, 0.1, 0.2, 0.25, 0.3, 0.5):
        dk_ex = float(fwd.sandwich_dk_from_depth(p, T_F, S_SEP))
        dk_thin = p / (2.0 - p)
        dk_1st = dk_thin * (1.0 + 2.0 * (1.0 - p) * T_F / S_SEP)
        rows.append({"block": "asbuilt_section", "p_depth_frac": p,
                     "t_f_mm": T_F * 1e3, "s_mm": S_SEP * 1e3, "t_f_over_s": T_F / S_SEP,
                     "d_K_asbuilt": dk_ex, "d_K_thin_face": dk_thin,
                     "d_K_first_order": dk_1st,
                     "dK_asbuilt_over_thin": dk_ex / dk_thin,
                     "dK_asbuilt_over_first_order": dk_ex / dk_1st,
                     "d_M": p / 2.0, "zeta_asbuilt": (p / 2.0) / dk_ex,
                     "zeta_thin_face": (p / 2.0) / dk_thin})

    # (1) 법칙 자체 — 같은 깊이비 p에서, 그리고 같은 d_K에서
    for p in (0.02, 0.05, 0.1, 0.2, 0.25, 0.3, 0.5):
        dk_mono, dm_mono = 1.0 - (1.0 - p) ** 3, p
        dk_sw, dm_sw = p / (2.0 - p), p / 2.0
        dk_ab = float(fwd.sandwich_dk_from_depth(p, T_F, S_SEP))
        rows.append({"block": "law_vs_depth", "p_depth_frac": p,
                     "d_K_monolithic": dk_mono, "d_M_monolithic": dm_mono,
                     "zeta_monolithic": dm_mono / dk_mono,
                     "d_K_sandwich": dk_sw, "d_M_sandwich": dm_sw,
                     "zeta_sandwich": dm_sw / dk_sw,
                     "d_K_asbuilt": dk_ab, "d_M_asbuilt": dm_sw,
                     "zeta_asbuilt": dm_sw / dk_ab,
                     "dK_ratio_mono_over_sw": dk_mono / dk_sw,
                     "dK_ratio_mono_over_asbuilt": dk_mono / dk_ab})
    for dk in (0.01, 0.05, 0.1, 0.2, 0.5):
        a = float(fwd.mass_field_exact(np.array([dk]))[0])
        b = float(fwd.mass_field_sandwich(np.array([dk]))[0])
        c = float(fwd.mass_field_sandwich_asbuilt(np.array([dk]))[0])
        rows.append({"block": "coupling_at_fixed_dK", "d_K": dk,
                     "d_M_monolithic": a, "d_M_sandwich": b, "d_M_asbuilt": c,
                     "dM_ratio_sw_over_mono": b / a,
                     "dM_ratio_asbuilt_over_mono": c / a,
                     "dM_ratio_asbuilt_over_sw": c / b})

    # (2) 부호전환 반경 ξ*
    for law in LAWS:
        _, _, loci = figs.null_loci(pool, coupling=law)
        for (m, n), x in zip(MODES, loci):
            rows.append({"block": "sign_reversal", "law": law, "m": m, "n": n,
                         "xi_star": x,
                         "r_star_mm": (PLATE.a + x * PLATE.extent) * 1e3
                         if np.isfinite(x) else np.nan})

    # (3) 식별성(CRLB·cond₂) — 같은 S̄_D에서 두 법칙 비교
    sigma = noi.sigma_y_for_modes(pool, 1e-3)
    for xi in (0.2, 0.5, 0.8, 0.95):
        for s in (0.01, 0.05):
            out = {}
            for law in LAWS:
                m_ = idf.metrics(pool, PLATE, (xi, s), W_GAUSS, sigma, mass=law)
                out[law] = m_
                rows.append({"block": "identifiability", "law": law, "xi_d": xi,
                             "s_bar": s, "sigma_rel": 1e-3,
                             "cond2": m_["cond2"], "det_F": m_["det_F"],
                             "crlb_xi_mm": m_["crlb_xi_mm"],
                             "crlb_s_pp": m_["crlb_s_pp"]})
            rows.append({"block": "identifiability_ratio", "xi_d": xi, "s_bar": s,
                         "sigma_rel": 1e-3,
                         "crlb_xi_ratio_sw_over_mono": (out["sandwich"]["crlb_xi_mm"]
                                                        / out["exact"]["crlb_xi_mm"]),
                         "crlb_s_ratio_sw_over_mono": (out["sandwich"]["crlb_s_pp"]
                                                       / out["exact"]["crlb_s_pp"]),
                         "cond2_ratio_sw_over_mono": (out["sandwich"]["cond2"]
                                                      / out["exact"]["cond2"]),
                         "crlb_xi_ratio_asbuilt_over_mono": (
                             out["sandwich_asbuilt"]["crlb_xi_mm"]
                             / out["exact"]["crlb_xi_mm"]),
                         "crlb_s_ratio_asbuilt_over_mono": (
                             out["sandwich_asbuilt"]["crlb_s_pp"]
                             / out["exact"]["crlb_s_pp"]),
                         "cond2_ratio_asbuilt_over_mono": (
                             out["sandwich_asbuilt"]["cond2"] / out["exact"]["cond2"])})

    # (4) 선형화 유효성 — 절대오차/floor (정본 §3.4 규약)
    floor = 2.0 * 1e-3
    for xi in (0.2, 0.5, 0.8, 0.95):
        for s in (0.01, 0.05):
            for law in LAWS:
                e = val.e_pert_abs(PLATE, pool, MODES, xi, s, W_GAUSS, mass=law)
                rows.append({"block": "linearization", "law": law, "xi_d": xi,
                             "s_bar": s, "abs_err_over_floor_max": float(np.max(e)) / floor,
                             "abs_err_over_floor_m0": float(e[0]) / floor,
                             "worst_mode": int(MODES[int(np.argmax(e))][0])})

    # (5) E3 사전등록: 목표 국소 d_K를 내는 **절대 가공깊이**
    #     (레일 t = 2 t_f = 2.0 mm, 실측 면판 t_f = 1.0 mm)
    t_tot = PLATE.t
    t_face = T_F
    for dk in (0.05, 0.1, 0.15, 0.25):
        depth_mono = t_tot * (1.0 - (1.0 - dk) ** (1.0 / 3.0))
        depth_sw = t_face * 2.0 * dk / (1.0 + dk)     # p = 2d_K/(1+d_K), depth = p·t_f
        depth_ab = t_face * float(fwd.sandwich_depth_from_dk(dk, T_F, S_SEP))
        rows.append({"block": "machining_depth", "d_K_local_target": dk,
                     "depth_mm_monolithic": depth_mono * 1e3,
                     "depth_mm_sandwich": depth_sw * 1e3,
                     "depth_mm_asbuilt": depth_ab * 1e3,
                     "depth_ratio_sw_over_mono": depth_sw / depth_mono,
                     "depth_ratio_asbuilt_over_mono": depth_ab / depth_mono})

    _save(pd.DataFrame(rows), "a10_sandwich_law.csv")


# ---------------------------------------------------------------- B6
def _band_kirchhoff_cols(pool, r1a, r2a, p_act, xi_c, s_bar, n_trial, n_trial_hi):
    """밴드 앵커의 Kirchhoff 쪽(1차 섭동·정확재해)과 오차 3분해 열 — 3D 없이 재계산 가능."""
    eta_lin = fwd.eta_bar_linear_band(pool, r1a, r2a, p_act, PLATE, coupling="exact")
    eta_kex = fwd.eta_bar_exact_band(PLATE, MODES, r1a, r2a, p_act, coupling="exact",
                                     n_trial=n_trial)
    eta_hi = fwd.eta_bar_exact_band(PLATE, MODES, r1a, r2a, p_act, coupling="exact",
                                    n_trial=n_trial_hi)
    e_a2 = val.e_pert(PLATE, pool, MODES, xi_c, s_bar, W_GAUSS, mass="exact")
    return eta_lin, eta_kex, eta_hi, e_a2


def _decomp(eta_lin_i, eta_kex_i, e3d, eta_hi_i, e_a2_i, n_trial):
    return {
        "eta_lin": eta_lin_i, "eta_kirchhoff_exact": eta_kex_i, "eta_3d": e3d,
        # ---- 오차 3분해 (분모는 각 단계의 '더 정확한' 쪽)
        "e_lin_abs": abs(eta_lin_i - eta_kex_i),
        "e_lin_rel": abs(eta_lin_i - eta_kex_i) / max(abs(eta_kex_i), 1e-300),
        "e_modelform_abs": abs(eta_kex_i - e3d),
        "e_modelform_rel": abs(eta_kex_i - e3d) / max(abs(e3d), 1e-300),
        "e_total_abs": abs(eta_lin_i - e3d),
        "e_total_rel": abs(eta_lin_i - e3d) / max(abs(e3d), 1e-300),
        # 모델형식 불일치의 두 표현. **부호 있는 가법 δ가 기본**이다 — 곱셈 인자 κ는
        # η̄가 0을 지나는 반경(부호전환)에서 발산하거나 음수가 되어 쓸 수 없다(F39).
        "delta_modelform": e3d - eta_kex_i,
        "kappa_modelform": e3d / eta_kex_i if eta_kex_i != 0 else np.nan,
        # Ritz 시행함수 수렴(정확재해 쪽 이산화) — 절대값도 낸다(η̄→0 근처에서 상대는 무의미)
        "ritz_n_trial": n_trial,
        "ritz_abs_change": abs(eta_hi_i - eta_kex_i),
        "ritz_ntrial_rel_change": abs(eta_hi_i - eta_kex_i) / max(abs(eta_kex_i), 1e-300),
        # A2(가우시안·Kirchhoff 내부)와의 대조 — **다른 양**임을 데이터로 보이기 위함
        "e_pert_a2_gaussian_rel": e_a2_i,
    }


def _band_anchor_rows(res0, lam0, h0, r3, nr, nt, nz, xi_c, half, depth,
                      pool, n_trial_hi, n_trial=36):
    """축대칭 밴드 앵커 1개 — 3D / Kirchhoff 정확재해 / 1차 섭동을 같은 형상에서."""
    r1 = sev.xi_to_r(xi_c - half, PLATE.a, PLATE.b)
    r2 = sev.xi_to_r(xi_c + half, PLATE.a, PLATE.b)
    pk = {"r1": r1, "r2": r2, "theta0": B3_THETA0, "dtheta": 2 * np.pi,
          "depth_frac": depth}
    c, cnn, info = r3.disk_mesh(a=PLATE.a, b=PLATE.b, t=PLATE.t, nr=nr, ntheta=nt,
                                nz=nz, pocket=pk)
    res = r3.solve_modes(c, cnn, PLATE.E, PLATE.nu, PLATE.rho,
                         r3.clamp_inner_rim(PLATE.a), n_modes=12, order=2,
                         keep_shapes=True)
    lam = (2 * np.pi * res.freqs) ** 2
    p_act = info["depth_actual"]
    r1a, r2a = info["r1_actual"], info["r2_actual"]
    s_bar = fwd.band_s_bar(r1a, r2a, p_act, PLATE.extent)

    eta_lin, eta_kex, eta_kex_hi, e_a2_all = _band_kirchhoff_cols(
        pool, r1a, r2a, p_act, xi_c, s_bar, n_trial, n_trial_hi)
    rows, eta3d_all, kappa_all = [], [], []
    for i, (m, n) in enumerate(MODES):
        md = r3.match_order(res, m, n_take=(1 if m == 0 else 2))
        mc_ = r3.subspace_mac_match(res0, h0[m]["idx"], res,
                                    n_take=(1 if m == 0 else 2), mac_min=0.8)
        if not mc_["matched"]:
            rows.append({"xi_c": xi_c, "depth_frac": p_act, "m": m, "n": n,
                         "matched": False, "purity": min(md["purity"]),
                         "mac": min(mc_["mac"])})
            eta3d_all.append(np.nan)
            kappa_all.append(np.nan)
            continue
        ks = mc_["idx"]
        j = h0[m]["idx"]
        l0 = float(np.mean([lam0[x] for x in j]))
        e3d = float(np.mean([lam[x] for x in ks])) / l0 - 1.0
        eta3d_all.append(e3d)
        kappa_all.append(e3d / eta_kex[i] if eta_kex[i] != 0 else np.nan)
        rows.append({
            "xi_c": xi_c, "half_xi": half, "depth_frac": p_act,
            "r1_mm": r1a * 1e3, "r2_mm": r2a * 1e3,
            "s_bar_pct": s_bar * 100, "m": m, "n": n, "matched": True,
            "purity": min(md["purity"]), "mac": min(mc_["mac"]), "ndof": res.ndof,
            "nr": nr, "ntheta": nt, "nz": nz,
            "shape_exact": info["shape_exact"],
            **_decomp(eta_lin[i], eta_kex[i], e3d, eta_kex_hi[i], float(e_a2_all[i]),
                      n_trial),
        })
    return rows, np.array(eta3d_all), np.array(kappa_all), s_bar


# ---------------------------------------------------------------- B5
#: B5 형상 구성 — as-built 판두께·유로폭은 고정, **미확정**(베인 입구반경·wrap 각)만 스윕한다.
B5_CONFIGS = {
    "asbuilt":   dict(r_vane_in=0.0170, wrap_deg=90.0),
    "vane_in19": dict(r_vane_in=0.0190, wrap_deg=90.0),
    "wrap60":    dict(r_vane_in=0.0190, wrap_deg=60.0),
}


def cmd_b5(args):
    """§3.6-ii — 폐쇄형 조립체 3D 모달 × 베인 1매 감육 스윕 (F16의 산출 명령).

    **왜 이제 명령으로 만드나**: 2026-08-02의 F16 실행은 임시 스크립트였고 `b5_impeller_cad_sweep.csv`를
    만드는 명령이 리포에 없었다(F51과 같은 재현성 결함 — 파일이 있어도 명령이 없으면 재현 불가).
    실측 2치수로 유로폭이 1.65 → 4.1 mm가 됐으므로 어차피 전면 재실행이 필요해, 이번에 명령으로 고정한다.

    산출 열에는 논문이 인용하는 **파생량까지** 넣는다(F57: 정본 md 안에만 있는 사실은 검정 불가) —
    1차 축퇴쌍의 pair mean 강하, 쌍 분리(Hz와 f의 %), 베인 참여수.
    """
    from . import impeller_cad as icad
    rows = []
    for name in args.configs:
        cfg = B5_CONFIGS[name]
        base = None
        for frac in args.damage:
            spec = icad.ImpellerSpec(damage_vane=(0 if frac > 0 else -1),
                                     damage_frac=frac, **cfg)
            spec.check()
            coors, conn = icad.build_geometry(spec, mesh_size=args.mesh_size,
                                             workdir=args.workdir,
                                             tag=f"b5_{name}_{int(frac*100)}")
            res = icad.solve_modal(spec, coors, conn, n_modes=args.n_modes, order=2)
            part = icad.vane_localization(res, coors, spec)
            f = res.freqs
            if base is None:
                base = f.copy()
            pair_mean = 0.5 * (f[0] + f[1])
            pair_mean0 = 0.5 * (base[0] + base[1])
            split = f[1] - f[0]
            print(f"[b5] {name} damage={frac}: ndof={res.ndof} "
                  f"f={np.round(f[:5], 1)} pair_mean={pair_mean:.2f} "
                  f"({100*(pair_mean/pair_mean0-1):+.3f} %) split={split:.3f} Hz "
                  f"({100*split/pair_mean:.3f} % of f) part={np.round(part[:3], 3)}")
            for i in range(len(f)):
                rows.append({
                    "config": name, "damage_frac": frac, "mode": i,
                    "f_Hz": float(f[i]), "f_ratio": float(f[i] / base[i]),
                    "vane_participation": float(part[i]), "ndof": int(res.ndof),
                    "mesh_size_mm": args.mesh_size * 1e3,
                    "t_front_mm": spec.t_front * 1e3, "t_back_mm": spec.t_back * 1e3,
                    "t_vane_mm": spec.t_vane * 1e3,
                    "channel_gap_mm": spec.gap * 1e3,
                    "total_thickness_mm": spec.total_thickness * 1e3,
                    "r_vane_in_mm": spec.r_vane_in * 1e3, "wrap_deg": spec.wrap_deg,
                    # 논문 인용 파생량(1차 축퇴쌍) — 모든 행에 같은 값을 실어 조판을 단순화
                    "pair1_mean_Hz": float(pair_mean),
                    "pair1_mean_shift_pct": float(100 * (pair_mean / pair_mean0 - 1)),
                    "pair1_split_Hz": float(split),
                    "pair1_split_pct_of_f": float(100 * split / pair_mean),
                    "pair1_member_shift_lo_pct": float(100 * (f[0] / base[0] - 1)),
                    "pair1_member_shift_hi_pct": float(100 * (f[1] / base[1] - 1)),
                })
    _save(pd.DataFrame(rows), "b5_impeller_cad_sweep.csv")


def cmd_b6(args):
    """§3.6-iii — **FEM 수준 e_pert**와 그 분해(선형화 / Kirchhoff↔3D 모델형식 / 이산화).

    정본 §3.6-iii은 "3D 데이터에는 세 오차가 겹쳐 있어 분리 없이는 e_pert로 인용 불가"라며
    미실행으로 남겨 두었다. 분리 방법: 손상을 **축대칭 반경 밴드**(포켓의 Δθ→2π 극한)로
    두면 같은 형상을 세 경로로 모두 풀 수 있다 —

        η̄^lin      1차 섭동 (역식별이 쓰는 맵)
        η̄^K,exact  Kirchhoff 비섭동 정확재해(Rayleigh–Ritz 재해)
        η̄^3D       3D 솔리드(비섭동·모델형식 독립)

        e_lin       = |η̄^lin − η̄^K,exact|     ← A2와 **같은 종류**(Kirchhoff 내부 선형화)
        e_modelform = |η̄^K,exact − η̄^3D|      ← Kirchhoff↔3D 탄성 (+ 3D 이산화)
        e_pert^FEM  = |η̄^lin − η̄^3D|          ← 정본이 요구한 FEM 수준 양(둘의 합성)

    **A2와의 구분**: A2의 e_pert는 같은 Kirchhoff 안의 선형화 오차이고 손상형상은 가우시안
    축대칭이다. 여기의 e_pert^FEM은 모델형식 차이까지 포함하며 손상형상은 실제 재료제거다.
    같은 (ξ, S̄)에서 두 값을 나란히 낸다(`e_pert_a2_gaussian_rel` 열).
    """
    from . import rail3d as r3
    nr, nt, nz = args.nr, args.ntheta, args.nz
    pool = _pool()

    if args.recompute_kirchhoff:
        # 3D는 그대로 두고 Kirchhoff 쪽(1차 섭동·정확재해)만 다시 계산한다.
        # Ritz 시행함수 수렴이 부족했음을 나중에 발견했을 때 26분짜리 3D를 다시 돌리지 않는다.
        src = DATA / "b6_epert_fem.csv"
        if not src.exists():
            raise SystemExit(f"{src} 없음")
        d = pd.read_csv(src)
        out = []
        for (h_, x_, p_), g in d.groupby(["half_xi", "xi_c", "depth_frac"]):
            r1a, r2a = float(g.r1_mm.iloc[0]) / 1e3, float(g.r2_mm.iloc[0]) / 1e3
            s_bar = fwd.band_s_bar(r1a, r2a, p_, PLATE.extent)
            lin, kex, hi, a2 = _band_kirchhoff_cols(pool, r1a, r2a, p_, x_, s_bar,
                                                    args.n_trial, args.n_trial_hi)
            for _, row in g.iterrows():
                i = [m for m, _ in MODES].index(int(row["m"]))
                rec = row.to_dict()
                if bool(row["matched"]):
                    rec.update(_decomp(lin[i], kex[i], float(row["eta_3d"]), hi[i],
                                       float(a2[i]), args.n_trial))
                out.append(rec)
            print(f"[b6] recompute half={h_} ξ={x_} p={p_}: "
                  f"κ={np.round([r['kappa_modelform'] for r in out[-len(g):]], 3)}")
        _save(pd.DataFrame(out), "b6_epert_fem.csv")
        return

    c0, cn0, _ = r3.disk_mesh(a=PLATE.a, b=PLATE.b, t=PLATE.t, nr=nr, ntheta=nt, nz=nz)
    res0 = r3.solve_modes(c0, cn0, PLATE.E, PLATE.nu, PLATE.rho,
                          r3.clamp_inner_rim(PLATE.a), n_modes=12, order=2,
                          keep_shapes=True)
    lam0 = (2 * np.pi * res0.freqs) ** 2
    h0 = {m: r3.match_order(res0, m, n_take=(1 if m == 0 else 2)) for m, _ in MODES}
    print(f"[b6] healthy 3D {nr}x{nt}x{nz}: ndof={res0.ndof} "
          f"f={np.round(res0.freqs[:6], 1)}")

    rows = []
    # 폭(half_xi)과 깊이(p)를 **둘 다** 흔든다: 같은 S̄를 서로 다른 (폭, 깊이)로 만들 수 있어
    # κ가 S̄만의 함수인지(대리모델의 전제) 데이터로 검정할 수 있다.
    for half in args.half_xi:
        for xi_c in args.xi:
            for depth in args.depths:
                rs, eta3d, kappa, s_bar = _band_anchor_rows(
                    res0, lam0, h0, r3, nr, nt, nz, xi_c, half, depth,
                    pool, args.n_trial_hi, n_trial=args.n_trial)
                rows += rs
                print(f"[b6] half={half} ξ_c={xi_c} p={depth} S̄={s_bar*100:.2f}% -> "
                      f"κ={np.round(kappa, 3)} η̄_3D={np.round(eta3d, 5)}")
    out = pd.DataFrame(rows)
    prev = DATA / "b6_epert_fem.csv"
    if args.append and prev.exists():                # 여러 번에 나눠 돌릴 때 누적
        old = pd.read_csv(prev)
        key = ["xi_c", "half_xi", "depth_frac", "m", "nr", "ntheta", "nz"]
        out = (pd.concat([old, out], ignore_index=True)
               .drop_duplicates(subset=key, keep="last"))
    _save(out, "b6_epert_fem.csv")

    # 이산화(격자) 검정 — 형상보존 nr 사다리에서 첫 앵커 재계산
    if args.grid_check:
        grows = []
        for nr2 in args.grid_check:
            c0b, cn0b, _ = r3.disk_mesh(a=PLATE.a, b=PLATE.b, t=PLATE.t, nr=nr2,
                                        ntheta=nt, nz=nz)
            r0b = r3.solve_modes(c0b, cn0b, PLATE.E, PLATE.nu, PLATE.rho,
                                 r3.clamp_inner_rim(PLATE.a), n_modes=12, order=2,
                                 keep_shapes=True)
            l0b = (2 * np.pi * r0b.freqs) ** 2
            h0b = {m: r3.match_order(r0b, m, n_take=(1 if m == 0 else 2))
                   for m, _ in MODES}
            rs, _, _, _ = _band_anchor_rows(r0b, l0b, h0b, r3, nr2, nt, nz,
                                            args.xi[0], args.half_xi[-1],
                                            args.depths[-1], pool, args.n_trial_hi,
                                            n_trial=args.n_trial)
            grows += rs
            print(f"[b6] grid-check nr={nr2} ndof={r0b.ndof}")
        _save(pd.DataFrame(grows), "b6_grid_check.csv")


# ---------------------------------------------------------------- B7
def cmd_b7(args):
    """3D 레일 **대리모델** 몬테카를로 (설계서 §3.5의 "independent-FEM rail: ≥1,000 via surrogate").

    대리모델: `y_m = η̄^K,exact_m(밴드 ξ±half) + δ_m(ξ, S̄)`.
      · η̄^K,exact는 **비섭동** Kirchhoff 재해라 유한 심각도의 비선형성을 정확히 담고,
      · δ_m = η̄^3D − η̄^K,exact는 B6 앵커에서 측정한 **모델형식 불일치**다.
    곱셈 인자 κ = η̄^3D/η̄^K,exact는 쓰지 않는다 — 부호전환 반경에서 발산하고 실제로
    음수가 나온다(ξ_c=0.7·m=2에서 κ = −1.00/−1.41, F39). 가법 δ는 그곳에서도 유계다.
    진실은 3D 레일의 손상족(축대칭 밴드)이고 역식별은 생산 섭동맵(가우시안 파라미터화)이므로,
    회복통계에 노이즈뿐 아니라 **모델형식 + 형상 불일치 + 선형화**가 모두 들어간다.

    **범위 한정(중요).** 앵커를 반폭 2종 × 깊이 2종으로 만들어 "κ는 (ξ, S̄)만의 함수"라는
    전제를 검정한 결과 **거짓**이었다(F38): 같은 ξ·같은 S̄ 근처에서도 폭이 좁으면 κ가 크다.
    따라서 대리모델은 **한 가지 손상 발자국 폭**(`--half-ref`)으로 한정하고, 그 안에서만
    (ξ, 깊이=S̄) 보간을 쓴다. MC 심각도도 그 족의 앵커 범위 **안**으로 제한한다(외삽 금지).

    **검증.**
      V1 ξ-LOO (게이트): 앵커 ξ 하나를 빼고 나머지로 보간해 그 앵커의 3D를 예측한다.
      V2 폭 전이 (진단, 게이트 아님): 기준족 κ를 **다른 폭**의 앵커에 적용해 본다 —
         F38의 규모를 산출물에 남기기 위한 것이고, 대리모델은 이 전이를 주장하지 않는다.
    V1 실패 시 SystemExit로 **중단**한다. 억지로 숫자를 만들지 않는다.
    """
    src = DATA / "b6_epert_fem.csv"
    if not src.exists():
        raise SystemExit(f"{src} 없음 — 먼저 `cli b6`를 실행할 것")
    d = pd.read_csv(src)
    d = d[d["matched"] == True]                                       # noqa: E712
    ms = [m for m, _ in MODES]
    half = args.half_ref if args.half_ref else float(max(d["half_xi"].unique()))
    # S̄는 실현 기하에서 나오므로 ξ마다 부동소수 끝자리가 갈린다 → 격자 노드는 양자화한다
    # (안 하면 s 노드가 중복 생성돼 κ 격자에 빈 칸이 생긴다).
    d = d.copy()
    d["s_key"] = d["s_bar_pct"].round(2)      # 11.5625 vs 11.562500000000009 병합
    fam = d[np.isclose(d.half_xi, half)]
    other = d[~np.isclose(d.half_xi, half)]
    xi_nodes = [float(x) for x in sorted(fam["xi_c"].unique())]
    s_nodes = [float(s) for s in sorted(fam["s_key"].unique())]
    print(f"[b7] 기준 손상족 반폭 {half}: ξ {xi_nodes} × S̄ {np.round(s_nodes,2)} % "
          f"| 폭 전이 진단용 다른 폭 {sorted(float(h) for h in other['half_xi'].unique())}")

    VAL = "delta_modelform"          # 가법 불일치가 기본(F39)

    def kgrid(frame, xis, ss):
        """(n_modes, n_xi, n_s) δ 격자 — 빠진 칸은 NaN."""
        K = np.full((len(ms), len(xis), len(ss)), np.nan)
        for i, m in enumerate(ms):
            for j, x in enumerate(xis):
                for k, s in enumerate(ss):
                    sel = frame[(frame.m == m) & np.isclose(frame.xi_c, x)
                                & np.isclose(frame.s_key, s)]
                    if len(sel):
                        K[i, j, k] = float(sel[VAL].iloc[0])
        return K

    def fill(K, xis, ss):
        """결측은 ξ 방향 → S̄ 방향 순으로 보간해 메운다(외삽은 클램프)."""
        K = K.copy()
        for i in range(K.shape[0]):
            for k in range(K.shape[2]):
                col, ok = K[i, :, k], np.isfinite(K[i, :, k])
                if 0 < ok.sum() < len(xis) and ok.sum() >= 2:
                    K[i, :, k] = np.interp(xis, np.array(xis)[ok], col[ok])
            for j in range(K.shape[1]):
                row, ok = K[i, j, :], np.isfinite(K[i, j, :])
                if 0 < ok.sum() < len(ss) and ok.sum() >= 1:
                    K[i, j, :] = (np.interp(ss, np.array(ss)[ok], row[ok])
                                  if ok.sum() >= 2 else row[ok][0])
        return K

    K0 = kgrid(fam, xi_nodes, s_nodes)
    n_missing = int(np.isnan(K0).sum())
    K = fill(K0, xi_nodes, s_nodes)
    spec = {"xi": [float(x) for x in xi_nodes], "s_bar": [s / 100 for s in s_nodes],
            "delta": K.tolist(), "half_xi": half}

    def _predict_rows(check, frame, sp, drop_desc):
        out = []
        for _, g in frame.groupby(["half_xi", "xi_c", "depth_frac"]):
            h2, x2, p2 = (float(g["half_xi"].iloc[0]), float(g["xi_c"].iloc[0]),
                          float(g["depth_frac"].iloc[0]))
            s2 = float(g["s_bar_pct"].iloc[0])
            r1, r2 = float(g["r1_mm"].iloc[0]) / 1e3, float(g["r2_mm"].iloc[0]) / 1e3
            eta_kex = fwd.eta_bar_exact_band(PLATE, MODES, r1, r2, p2, coupling="exact")
            dh = mc.kappa_eval(sp, x2, s2 / 100, key="delta")
            for k, m in enumerate(ms):
                row = g[g.m == m]
                if row.empty:
                    continue
                a = float(row["eta_3d"].iloc[0])
                pred = eta_kex[k] + dh[k]
                out.append({"check": check, "held_out": drop_desc, "xi_c": x2,
                            "s_bar_pct": s2, "half_xi": h2, "depth_frac": p2, "m": m,
                            "delta_hat": dh[k],
                            "delta_actual": float(row["delta_modelform"].iloc[0]),
                            "kappa_actual": float(row["kappa_modelform"].iloc[0]),
                            "eta_pred": pred, "eta_3d": a,
                            "resid_abs": abs(pred - a),
                            "resid_rel": abs(pred - a) / max(abs(a), 1e-300),
                            "resid_over_floor": abs(pred - a) / (2 * 1e-3)})
        return out

    vrows = []
    # ---- V1: ξ 방향 LOO
    for xi_c in xi_nodes:
        keep = [x for x in xi_nodes if x != xi_c]
        if len(keep) < 2:
            continue
        sp = {"xi": [float(x) for x in keep], "s_bar": [s / 100 for s in s_nodes],
              "delta": fill(kgrid(fam[~np.isclose(fam.xi_c, xi_c)], keep, s_nodes),
                            keep, s_nodes).tolist(), "half_xi": half}
        vrows += _predict_rows("loo_xi", fam[np.isclose(fam.xi_c, xi_c)], sp,
                               f"xi={xi_c}")
    # ---- V2: 폭 전이 진단(게이트 아님) — 기준족 κ를 다른 폭 앵커에 그대로 적용
    if len(other):
        vrows += _predict_rows("width_transfer", other, spec,
                               f"half={sorted(other['half_xi'].unique())}")
    vdf = pd.DataFrame(vrows)
    vdf["delta_grid_missing_cells"] = n_missing
    _save(vdf, "b7_surrogate_validation.csv")

    stats = {}
    for chk in ("loo_xi", "width_transfer"):
        g = vdf[vdf.check == chk]
        if g.empty:
            continue
        stats[chk] = (float(g.resid_rel.median()), float(g.resid_rel.quantile(0.95)))
        print(f"[b7] {chk}: 상대잔차 중앙 {stats[chk][0]:.3f} / p95 {stats[chk][1]:.3f} "
              f"(n={len(g)}){'  [게이트]' if chk == 'loo_xi' else '  [진단]'}")
    if "loo_xi" not in stats:
        raise SystemExit("[b7] **중단**: ξ-LOO 검증을 할 앵커가 없다")
    if stats["loo_xi"][0] > args.loo_max:
        raise SystemExit(
            f"[b7] **중단**: ξ-LOO 상대잔차 중앙 {stats['loo_xi'][0]:.3f} > 허용 "
            f"{args.loo_max}. 앵커에서 보간할 수 있는 모델형식 인자가 아니다 — "
            "숫자를 만들지 않는다.")

    # ---- MC: 진실 = 대리모델(3D 레일 손상족), 역식별 = 생산 섭동맵
    xi_list = np.linspace(min(xi_nodes), max(xi_nodes), args.n_loc)
    s_lo, s_hi = min(s_nodes) / 100, max(s_nodes) / 100
    s_list = [s for s in args.s_bar if s_lo <= s <= s_hi]     # 앵커 범위 **안**만
    dropped = [s for s in args.s_bar if s not in s_list]
    if dropped:
        print(f"[b7] 앵커 심각도 범위({s_lo*100:.2f}–{s_hi*100:.2f} %) 밖이라 제외: "
              f"{[round(s*100,2) for s in dropped]} % — 외삽으로 통계를 내지 않는다")
    if not s_list:
        raise SystemExit("[b7] **중단**: 앵커 범위 안의 심각도가 없다")
    rows = mc.run_production(
        PLATE, MODES, xi_list, s_list, SIGMA_RELS, w=W_GAUSS,
        n_real=args.n_real, n_workers=args.workers, n_grid=1001,
        mass="exact", surrogate=spec, n_starts=args.n_starts, seed0=20260808)
    # 같은 셀의 자기일관(B1식) 기준선을 함께 내어 model-form 몫을 분리한다
    base = mc.run_production(
        PLATE, MODES, xi_list, s_list, SIGMA_RELS, w=W_GAUSS,
        n_real=args.n_real, n_workers=args.workers, n_grid=1001,
        mass="exact", n_starts=args.n_starts, seed0=20260808)
    for r in rows + base:
        r["surrogate_half_xi"] = half
        r["loo_xi_resid_rel_median"] = stats["loo_xi"][0]
        r["loo_xi_resid_rel_p95"] = stats["loo_xi"][1]
        r["width_transfer_resid_rel_median"] = stats.get("width_transfer", (np.nan,))[0]
    _save(pd.DataFrame(rows + base), "b7_mc_surrogate.csv")


# ---------------------------------------------------------------- A13 / Fig 1
#: A13 순환대칭 규명 구성. `arcN`은 베인 방위 점유 셀수를 키워 C6 변조를 **연속적으로**
#: 0까지 줄이는 스윕(arc18 = 방위 균일 웹 = 정확한 축대칭 대조군).
A13_CONFIGS: dict = {
    "c6_asbuilt": dict(n_vane=6),                       # 물리 t_vane(≈1셀) = 그림 형상
    "c6_web":     dict(n_vane=6, vane_mode="web"),      # 축대칭 대조군
    "c4":         dict(n_vane=4, vane_arc_cells=1),     # m=N/2=2가 singlet이 되는가
    "c12":        dict(n_vane=12, vane_arc_cells=1),    # m=3이 doublet으로 되돌아오는가
    **{f"c6_arc{k}": dict(n_vane=6, vane_arc_cells=k)
       for k in (1, 2, 6, 9, 12, 15, 17, 18)},
}


def _a13_solve(name: str, n_modes: int, wrap_deg: float):
    from . import impeller_hex as ihx
    spec = ihx.HexImpellerSpec(wrap_deg=wrap_deg, **A13_CONFIGS[name])
    coors, conn, minfo = ihx.mesh(spec)
    res, info = ihx.solve_free_free(spec, coors, conn, n_modes=n_modes,
                                    mesh_info=minfo)
    return spec, res, info, coors, conn


def cmd_a13(args):
    """A13 — **순환대칭 제약의 규명**(Fig 1이 드러낸 것). 정본 §3.2의 유효 m 범위 근거.

    베인 N매의 임펠러는 축대칭이 아니라 C_N이다. C_N의 기약표현은 순환조화지수
    h = 0…⌊N/2⌋이고, 공간 조화차수 m은 h ≡ ±m (mod N)로 **접힌다(aliasing)**:

        h = 0        1차원 표현 → 축퇴 **비보호**            m ∈ {0, N, 2N, …}
        0 < h < N/2  2차원 표현 → 축퇴 **보호**(doublet)     m ∈ {h, N−h, N+h, …}
        h = N/2      1차원 표현 → 축퇴 **비보호**(N 짝수)    m ∈ {N/2, 3N/2, …}

    N = 6이면 m=1·2는 보호된 doublet, **m=3 = N/2는 비보호**(축대칭에서 겹쳐 있던 두
    성분이 베인 때문에 갈라진다), m=4·5는 m=2·1과, m=6은 m=0과 같은 표현이다.

    판정은 **주파수를 쓰지 않는다**: 대칭연산 R(2π/N)에 대한 모드형의 질량정규 자기내적
    c = cos(2πh/N)에서 h를 읽는다(`impeller_hex.harmonic_indices`). 구조격자
    (`n_theta % n_vane == 0`)라 R이 절점 순열로 정확히 실현되고, 사면체 CAD와 달리 격자
    유래 미스튜닝이 m ≤ 5에 들어오지 않는다(F62가 철회한 분리 배수의 원인이 그것이었다).
    """
    from . import impeller_hex as ihx
    rows, summ = [], []
    for name in args.configs:
        spec, res, info, _, _ = _a13_solve(name, args.n_modes, args.wrap_deg)
        rows.extend({"config": name, **r}
                    for r in ihx.cyclic_symmetry_rows(spec, res, info,
                                                      m_max=args.m_max))
        s = ihx.splitting_summary(spec, res, info, m_max=args.m_max)
        summ.append({"config": name, "n_vane": spec.n_vane,
                     "vane_mode": spec.vane_mode,
                     "vane_arc_cells": (-1 if spec.vane_arc_cells is None
                                        else spec.vane_arc_cells),
                     "modulation_depth": spec.modulation_depth,
                     "n_dof": info["n_dof"], "n_rigid": info["n_rigid"],
                     "mass_fem_g": info["mass_fem_g"],
                     "mass_analytic_g": info["mass_analytic_g"],
                     "f_first_elastic_Hz": info["f_first_elastic_Hz"], **s})
        print(f"[a13] {name:11s} depth={spec.modulation_depth:.3f} "
              f"h={s['h_half']} split={s['split_hN2_rel']*100:8.4f} % of f "
              f"(overlap {s['partner_overlap']:.3f}, dψ {s['partner_dpsi_deg']:+.2f}°) "
              f"floor={s['floor_split_rel_max']*100:.2e} %")
    _save(pd.DataFrame(rows), "a13_cyclic_symmetry.csv")
    _save(pd.DataFrame(summ), "a13_splitting_vs_modulation.csv")


#: A13x 메시 사다리 — (n_r, n_theta, n_z_shroud, n_z_channel). **형상보존 규약(F11′·F78)**:
#: n_r은 18의, n_theta는 108의 정수배여야 기준 발자국이 셀면에 정확히 놓인다. z는 어떤
#: 층수에서도 물리 계면(0, t, t+b₂, 2t+b₂)이 절점면이므로 형상에 무관하다.
A13X_LEVELS: tuple = ((18, 108, 2, 2), (36, 216, 3, 3), (54, 324, 4, 4))
#: 발자국을 판정하는 기준격자 — 사다리 전체가 **같은 고체영역**을 쓴다는 뜻.
A13X_FOOTPRINT: tuple = (18, 108)


#: 사다리 한 단계를 식별하는 키 — 이 조합이 같으면 같은 계산이다.
A13X_KEY = ("config", "n_r", "n_theta", "n_z_shroud", "n_z_channel")


def _a13x_merge(rows: list) -> pd.DataFrame:
    """새 행을 **기존 CSV와 병합**한다 — 최촘 격자가 45분이라 덮어쓰기가 곧 재계산이다.

    같은 키(구성 + 격자)의 옛 행은 새 행으로 갈아치우고, 다른 키는 보존한다. 그래서
    `--configs c6_web`처럼 한 구성만 추가 실행해도 앞선 사다리가 남는다.
    """
    new = pd.DataFrame(rows)
    old_path = DATA / "a13x_mesh_ladder.csv"
    if old_path.exists():
        old = pd.read_csv(old_path)
        if set(A13X_KEY) <= set(old.columns):
            keys = set(map(tuple, new[list(A13X_KEY)].to_numpy().tolist()))
            keep = [t not in keys
                    for t in map(tuple, old[list(A13X_KEY)].to_numpy().tolist())]
            new = pd.concat([old[keep], new], ignore_index=True)
    return new.sort_values(["config", "n_dof"]).reset_index(drop=True)


def cmd_a13x(args):
    """A13x — m = N/2 분리량의 **메시 사다리**(§12⑪ 해제 시도). 형상을 고정한 채 정제한다.

    **왜 이것이 필요한가.** A13은 m = N/2 단일화 분리량을 40.4 % of f로 냈지만 메시
    의존성을 재지 않았고, 그래서 정본이 라이선스한 것은 "인공물 floor를 자릿수로
    압도한다"까지였다(§12⑪). 여기서는 그 수치의 이산화 오차를 측정한다.

    **형상보존이 이 사다리의 전제다(F11′의 교훈).** 물리 t_vane 판정은 셀 방위폭 r·dθ와
    비교하므로 n_theta를 올리면 베인 발자국 셀수가 바뀌어 **형상이 달라진다** — 그 상태로
    잰 '수렴'은 이산화가 아니라 기하 변화를 재는 것이고, F11이 3D 포켓 레일에서 정확히
    그 함정에 빠졌다. 그래서 발자국을 기준격자(18×108)에서 한 번 판정하고 정제격자는
    그것을 정수배로 세분만 한다(`impeller_hex.footprint_grid`). 불변량으로 발자국 방위
    점유율(`vane_area_frac`)을 각 단계에 함께 기록해 **형상이 같음을 수치로 증명**한다.

    **동시에 인공물 floor를 각 격자에서 낸다.** 분리량이 수렴하는 것만으로는 부족하다 —
    측정된 분리가 고유해의 수치 잡음이 아니라는 것은 대칭보호 doublet의 관측 분리(floor)와
    의 자릿수 차이로만 말할 수 있고, 그 floor는 격자마다 다시 재야 한다(F62).

    **판정.** 정본 §3.6의 기준대로 연속 두 격자 사이 분리량(% of f)의 상대변화 < 5 %.
    남는 것: 절대주파수는 1차 육면체 + lumped mass라 위로 편향돼 있고(총질량 대비 FEM
    질량비를 함께 기록한다 — 원환의 다각형 근사가 정제와 함께 수렴하는 것이 보인다),
    이 사다리는 **그 편향의 수렴이 아니라 분리비의 수렴**을 판정한다.
    """
    from . import impeller_hex as ihx
    rows = []
    for name in args.configs:
        prev = None
        for (nr, nth, nzs, nzc) in args.levels:
            cfg = dict(A13_CONFIGS[name])
            spec = ihx.HexImpellerSpec(
                wrap_deg=args.wrap_deg, n_r=nr, n_theta=nth, n_z_shroud=nzs,
                n_z_channel=nzc,
                **({} if (nr, nth) == tuple(args.footprint)
                   else dict(footprint_n_r=args.footprint[0],
                             footprint_n_theta=args.footprint[1])),
                **cfg)
            t0 = time.time()
            coors, conn, minfo = ihx.mesh(spec)
            res, info = ihx.solve_free_free(spec, coors, conn,
                                            n_modes=args.n_modes, mesh_info=minfo)
            s = ihx.splitting_summary(spec, res, info, m_max=args.m_max)
            split = 100.0 * s["split_hN2_rel"]              # % of f
            floor = 100.0 * s["floor_split_rel_max"]        # % of f
            row = {"config": name, "n_r": nr, "n_theta": nth,
                   "n_z_shroud": nzs, "n_z_channel": nzc,
                   "n_dof": info["n_dof"], "n_elem": info["n_elem"],
                   "n_modes": args.n_modes, "n_rigid": info["n_rigid"],
                   # 형상 불변량 — 사다리 전 단계에서 같아야 한다(F78)
                   "footprint_key": info["footprint_key"],
                   "vane_area_frac": info["vane_area_frac"],
                   "vane_cells_per_vane": info["vane_cells_per_vane"],
                   "mass_fem_g": info["mass_fem_g"],
                   "mass_analytic_g": info["mass_analytic_g"],
                   "mass_fem_over_analytic": (info["mass_fem_g"]
                                              / info["mass_analytic_g"]),
                   "f_first_elastic_Hz": info["f_first_elastic_Hz"],
                   "split_lo_Hz": s["split_hN2_lo_Hz"],
                   "split_hi_Hz": s["split_hN2_hi_Hz"],
                   "split_Hz": s["split_hN2_Hz"],
                   "split_pct_of_f": split,
                   "partner_overlap": s["partner_overlap"],
                   "partner_dpsi_deg": s["partner_dpsi_deg"],
                   "n_protected_pairs": s["n_protected_pairs"],
                   "floor_pct_of_f_max": floor,
                   "floor_pct_of_f_median": 100.0 * s["floor_split_rel_median"],
                   "split_over_floor_decades": (np.log10(abs(split) / floor)
                                                if floor > 0 and split == split
                                                else np.nan),
                   "d_split_rel_pct": (np.nan if prev is None else
                                       100.0 * abs(split - prev) / abs(prev)),
                   "seconds": time.time() - t0}
            rows.append(row)
            prev = split
            print(f"[a13x] {name:11s} {nr:3d}x{nth:4d}x({nzs},{nzc}) "
                  f"ndof={info['n_dof']:>8,d} f1el={info['f_first_elastic_Hz']:8.1f} "
                  f"split={split:8.4f} % of f "
                  f"(Δ vs prev {row['d_split_rel_pct']:6.2f} %) "
                  f"floor={floor:.2e} % → {row['split_over_floor_decades']:.1f} decades "
                  f"[{row['seconds']:.0f} s]", flush=True)
            # 단계마다 저장한다 — 마지막 격자가 메모리로 죽어도 사다리가 남아야 한다.
            _save(_a13x_merge(rows), "a13x_mesh_ladder.csv")
    d = _a13x_merge(rows)
    for name, g in d.groupby("config"):
        w = g.d_split_rel_pct.max()
        shape = sorted(set(g.vane_area_frac.round(12)))
        # 분리가 floor 수준인 구성(축대칭 대조군)에는 5 % 기준을 적용하지 않는다 —
        # 0에 대한 상대변화는 수치 잡음의 비이고 판정이 아니다.
        at_floor = bool((g.split_pct_of_f <= 10 * g.floor_pct_of_f_max).all())
        verdict = ("floor 수준(분리 없음) — 5 % 기준 비적용" if at_floor
                   else ("수렴" if w < 5 else "미수렴"))
        print(f"[a13x] {name}: 분리량 상대변화 최대 = {w:.2f} % (기준 5 %) → {verdict}; "
              f"형상 불변량 vane_area_frac = {shape}"
              f"{'' if len(shape) == 1 else '  ⚠ 형상이 단계마다 다르다'}")


def cmd_fig1(args):
    """Figure 1 — 임펠러 기하 + 모드형 3패널. npz를 내고 `figures`가 그것만 소비한다."""
    from . import figures as figs
    from . import impeller_hex as ihx
    spec, res, info, coors, conn = _a13_solve(args.config, args.n_modes,
                                              args.wrap_deg)
    h, c, deg = ihx.harmonic_indices(spec, res, info["grid_idx"], info["_M_diag"])
    orders, _, _ = ihx.r3.azimuthal_orders(res, m_max=args.m_max)
    nrig = info["n_rigid"]
    el = list(range(nrig, len(res.freqs)))
    h_int = np.rint(h).astype(int)
    # 패널 모드는 **형상으로** 고른다(주파수 순서 금지, 정본 §3.6).
    k_doublet = next(k for k in el if deg[k] == 2)                 # 첫 탄성 축퇴쌍
    k_half = next(k for k in el if h_int[k] == spec.n_vane // 2)   # m = N/2 단일
    panel = np.array([k_doublet, k_doublet, k_half], dtype=int)
    _ensure_dirs()
    npz = DATA / "fig1_impeller_modes.npz"
    np.savez_compressed(
        npz, coors=coors, conn=conn, freqs=res.freqs, shapes=res.full_shapes,
        panel_modes=panel, m_dom=orders, h_hat=h, c_rotation=c, degeneracy=deg,
        n_vane=spec.n_vane, n_rigid=nrig, wrap_deg=spec.wrap_deg,
        z_cut=spec.t_sheet + spec.channel - 1e-5,
        cut_sector_deg=np.array(args.cut_sector, dtype=float),
        t_sheet=spec.t_sheet, channel=spec.channel, a=spec.a, b=spec.b,
        n_theta=spec.n_theta, n_r=spec.n_r, n_dof=info["n_dof"],
        mass_fem_g=info["mass_fem_g"])
    print(f"[saved] {npz}")
    for k in panel[[0, 2]]:
        print(f"  panel mode {k}: f={res.freqs[k]:.1f} Hz  m={orders[k]}  "
              f"h={h_int[k]}  deg_protected={deg[k]}")
    figs.fig1_impeller_modes(npz, FIGS / "fig1_impeller_modes.png",
                             panel_dir=FIGS if args.panels else None)


def cmd_fig2(args):
    """Figure 2 — 균열 지문(§4.1). A11 산출 CSV만 소비하므로 재계산이 없다(수초).

    `cli a11`(2D·3D 팔 포함)이 이미 돌아 있어야 한다. 그림이 인용하는 판별비·정합
    가우시안은 전부 `a11_*.csv`에서 읽으며, 정본 캡션·본문과의 일치는
    `tests/test_fig2_signature.py`가 회귀검정한다(F77).
    """
    from . import figures as figs
    _ensure_dirs()
    figs.fig2_crack_signature(DATA, FIGS)


# ---------------------------------------------------------------- A14
#: 질량부하 한계를 평가할 예산 — floor(σ_f/f = 0.1 %)의 절반이 사전등록 기준값이다.
A14_BUDGETS = {"half_floor_0p05pct": 5.0e-4, "floor_0p1pct": 1.0e-3,
               "half_tight_0p015pct": 1.5e-4}
#: 대표 센서 질량 [kg] — 최경량 방수 가속도계급 / 일반 소형 / 케이블 포함 실효.
A14_SENSORS = {"0p2g": 0.2e-3, "0p5g": 0.5e-3, "2g": 2.0e-3}


def cmd_a14(args):
    """§5 E1 [FILL] 2 — 접촉센서 질량부하 한계를 **정확식**으로 확정한다.

    질량정규화 모드형이 있으므로 한계는 경험규칙이 아니라 δf/f = −½ m_a |φ(x)|²다
    (`massload` 모듈 도크스트링). 세 레일에서 |φ|²의 절점 최댓값을 구해 예산별 m_a 상한을 낸다.

      canonical  = 정본 b5 레일(as-built, 보어 클램프, mesh 1.2 mm, order 2) — 논문이 인용하는 레일
      free_free  = 육면체 조립체 자유-자유 — **E1의 실제 현수조건**이므로 교차확인용
      vane       = as-built 베인 쿠폰 30 × 4.1 × 1.0 mm — E2 시편

    `phi2_max_all`은 전 절점 최댓값(= antinode, 상계)이고 `phi2_max_surface`는 **실제로
    센서를 붙일 수 있는 외부면**(전면 슈라우드 상면 ∪ 외주면)에서의 최댓값이다. 폐쇄형
    임펠러의 내부 유로면은 접근 불가이므로 둘을 함께 보고해 여유를 드러낸다.
    """
    from . import massload as ml
    from . import rail3d as r3
    rows = []

    def emit(rail, k, f, shape, coors, mask, m_tot, ndof, extra):
        d_all = ml.mode_phi2_max(shape, coors)
        d_sur = ml.mode_phi2_max(shape, coors, mask=mask) if mask is not None else d_all
        row = {"rail": rail, "mode": k, "f_Hz": float(f), "mass_total_g": 1e3 * m_tot,
               "ndof": int(ndof), "phi2_max_all_per_kg": d_all["phi2_max"],
               "phi2_max_surface_per_kg": d_sur["phi2_max"],
               "m_eff_all_g": 1e3 * d_all["m_eff_kg"],
               "m_eff_surface_g": 1e3 * d_sur["m_eff_kg"],
               "antinode_r_mm": d_all["r_mm"], "antinode_z_mm": d_all["z_mm"],
               "phi2_max_all_times_mass": d_all["phi2_max"] * m_tot, **extra}
        for nm, b in A14_BUDGETS.items():
            row[f"m_limit_all_{nm}_mg"] = 1e6 * float(ml.mass_limit(d_all["phi2_max"], b))
            row[f"m_limit_surface_{nm}_mg"] = 1e6 * float(
                ml.mass_limit(d_sur["phi2_max"], b))
            row[f"m_limit_all_{nm}_pct_of_part"] = 100 * float(
                ml.mass_limit(d_all["phi2_max"], b)) / m_tot
        for nm, ma in A14_SENSORS.items():
            row[f"dff_pct_all_{nm}"] = 100 * abs(float(
                ml.df_f_point_mass(ma, d_all["phi2_max"])))
            row[f"dff_pct_surface_{nm}"] = 100 * abs(float(
                ml.df_f_point_mass(ma, d_sur["phi2_max"])))
        rows.append(row)

    # --- (1) 정본 b5 레일 -------------------------------------------------
    if "canonical" in args.rails:
        from . import impeller_cad as icad
        spec = icad.ImpellerSpec(damage_vane=-1, damage_frac=0.0,
                                 **B5_CONFIGS[args.b5_config])
        spec.check()
        coors, conn = icad.build_geometry(spec, mesh_size=args.mesh_size,
                                         workdir=args.workdir, tag="a14_healthy")
        res = icad.solve_modal(spec, coors, conn, n_modes=args.n_modes, order=2)
        m_tot = spec.rho * ml.mesh_volume(coors, conn)
        mask = ml.outer_surface_mask(res.field_coors, r_out=spec.r_out,
                                     z_top=res.field_coors[:, 2].max())
        print(f"[a14] canonical: ndof={res.ndof} mass={m_tot*1e3:.2f} g "
              f"f={np.round(res.freqs[:5], 1)} surface_nodes={int(mask.sum())}")
        for k in range(len(res.freqs)):
            emit("canonical_b5_clamped", k, res.freqs[k], res.full_shapes[k],
                 res.field_coors, mask, m_tot, res.ndof,
                 {"config": args.b5_config, "bc": "bore clamp",
                  "mesh_size_mm": args.mesh_size * 1e3})

    # --- (2) 자유-자유 조립체(E1 현수조건) --------------------------------
    if "free_free" in args.rails:
        from . import impeller_hex as ih
        hspec = ih.HexImpellerSpec()
        hspec.check()
        hres, hinfo = ih.solve_free_free(hspec, n_modes=args.n_modes + 6)
        m_tot = 1e-3 * hinfo["mass_fem_g"]
        # 질량정규화 직접검증 — 대각(lumped) 질량이므로 φᵀMφ를 벡터화로 계산한다
        Md = hinfo["_M_diag"]
        dev = max(abs(float((hres.shapes[:, k] ** 2 * Md).sum()) - 1.0)
                  for k in range(hres.shapes.shape[1]))
        print(f"[a14] free_free: mass={m_tot*1e3:.2f} g  max|φᵀMφ−1| = {dev:.2e}")
        nr = hinfo["n_rigid"]
        mask = ml.outer_surface_mask(hres.field_coors, r_out=hspec.b,
                                     z_top=hres.field_coors[:, 2].max())
        for k in range(nr, len(hres.freqs)):
            emit("free_free_hex", k - nr, hres.freqs[k], hres.full_shapes[k],
                 hres.field_coors, mask, m_tot, hres.ndof,
                 {"config": "hex_asbuilt", "bc": "free-free",
                  "mesh_size_mm": float("nan"), "mass_norm_dev": dev})

    # --- (3) 베인 쿠폰(E2 시편) ------------------------------------------
    if "vane" in args.rails:
        L, w, h = VANE.L, geo.B2_CHANNEL, VANE.h
        c, cn, _ = r3.vane_mesh(L=L, w=w, h=h, nx=60, ny=4, nz=8)
        res = r3.solve_modes(c, cn, VANE.E, VANE.nu, VANE.rho, r3.clamp_root(),
                             n_modes=args.n_modes, order=2, keep_shapes=True)
        kinds = r3.beam_mode_kinds(res)
        m_tot = VANE.rho * L * w * h
        anchor = 1.0 / (ml.EB_TIP_EFFECTIVE_MASS_RATIO * m_tot)
        print(f"[a14] vane: mass={m_tot*1e3:.3f} g  해석앵커 |φ(L)|²={anchor:.1f} kg⁻¹")
        for k in range(len(res.freqs)):
            emit("vane_coupon", k, res.freqs[k], res.full_shapes[k], res.field_coors,
                 None, m_tot, res.ndof,
                 {"config": "asbuilt_30x4.1x1.0", "bc": "root clamp",
                  "mesh_size_mm": float("nan"), "mode_kind": kinds[k],
                  "eb_tip_anchor_per_kg": anchor})

    _save(pd.DataFrame(rows), "a14_massload.csv")


# ---------------------------------------------------------------- 제출본
CANON_MD = (Path(__file__).resolve().parents[1] / "docs" / "paper3-jsv"
            / "2026-07-31-paperB-jsv-v2.1.md")


# ---------------------------------------------------------------- A19
#: R0 = a18의 직선 쿠폰. **이 격자·물성이 a18 산출을 정확히 재현한다**
#: (ndof 68040/58644, f₁ flap 886.9 → 265.6 Hz). a18은 만드는 명령이 리포에 없어
#: 재현이 불가능했다(F51과 같은 부류) — a19가 그 단을 다시 계산하며 함께 닫는다.
A19_COUPON = dict(L=0.030, w=0.0041, h=0.0010, nx=60, ny=4, nz=10)
A19_COUPON_MAT = dict(E=193e9, nu=0.29, rho=8000.0)
A19_KERF = {"xc_over_L": 0.125, "width": 0.0075, "depth_frac": 0.6}


def ladder_factors(shifts):
    """사다리 단계별 희석 인자 shifts[i−1]/shifts[i] (첫 단은 기준이라 nan).

    강하가 0(수치해상도 이하)이면 ∞ 대신 nan을 돌려준다 — 표에 ∞를 싣지 않는다.
    """
    out = [float("nan")]
    for prev, cur in zip(shifts[:-1], shifts[1:]):
        out.append(prev / cur if cur else float("nan"))
    return np.array(out)


def _a19_assembly_shift(mesh_mm: float, config: str):
    """조립체 pair mean 강하 [%] — **커밋된 b5 산출에서 읽는다**(재풀이 없음).

    사다리의 마지막 단은 이미 정본이 인용하는 값이므로 다시 풀면 숫자가 갈라질 위험만 있다.
    """
    for name in ("b5_impeller_cad_sweep.csv", "b5_mesh_refine_1p0mm.csv"):
        p = DATA / name
        if not p.exists():
            continue
        d = pd.read_csv(p)
        sel = d[(d["config"] == config) & np.isclose(d["mesh_size_mm"], mesh_mm)
                & np.isclose(d["damage_frac"], 0.6)]
        if len(sel):
            return float(sel["pair1_mean_shift_pct"].iloc[0]), name
    return float("nan"), ""


def cmd_a19(args):
    """§4.3-vii — **정합 기하** 대조. 65–89배를 기하·구속·조립체 인자로 분해한다.

    a18은 손상만 맞췄고 부품 쪽은 직선 30 mm 쿠폰의 뿌리 캔틸레버였다. 실측 형상의 베인은
    캠버 호길이가 44.6 mm이고 **양 슈라우드에 접합**돼 있어 그 비에는 기하와 경계조건이
    섞여 있다 — 정본이 "a geometry-matched control … is still outstanding"이라 적은 지점.
    한 단에 하나만 바꾸는 사다리로 분해한다:

        R0 직선 쿠폰(뿌리 클램프) → R1 as-built 베인(뿌리 클램프)
        → R2 같은 베인(슈라우드면 클램프) → R3 조립체 pair mean

    손상 이상화는 전 단계 동일(뿌리 1/4 스팬 × 두께 60 %)이고 R1·R2·R3의 기하·손상은
    **같은 코드**(`impeller_cad.profile_polygon`)에서 나온다 — 재구현하면 사다리의 전제가
    깨진다. 각 단에서 손상 창의 변형·운동에너지 분율도 내어, 서로 다른 구조에서 잰
    주파수강하를 1차 감도(§3.2의 γ^K·γ^M에 해당)로 통약한다.

    모드는 **형상으로** 고른다(정본 §3.6이 금지한 주파수 순서 매칭 회피): 직선 쿠폰은
    `rail3d.beam_mode_kinds`, 곡면 베인은 `impeller_cad.vane_mode_kinds`.
    """
    from dataclasses import replace

    from . import impeller_cad as icad
    from . import rail3d as r3

    span_hi, d_frac = args.damage_span, args.damage_frac
    dln_k = 1.0 - (1.0 - d_frac) ** 3        # 국소 굽힘강성 손실률 (∝ t³)
    dln_m = d_frac                           # 국소 면적질량 손실률 (∝ t)

    def _solve(coors, conn, mat, selector, mask=None):
        np.random.seed(20260824)             # 축퇴 부분공간 기저를 실행 간 고정(F102)
        return r3.solve_modes(coors, conn, mat["E"], mat["nu"], mat["rho"], selector,
                              n_modes=args.n_modes, order=2, keep_shapes=True,
                              region_mask=mask)

    def _pred_pct(uk, um):
        """1차 예측 강하 [%]: δλ/λ = −U_K·δlnK + U_M·δlnM, Δf/f = ½ δλ/λ."""
        return 50.0 * (-uk * dln_k + um * dln_m)

    # ---- R0: a18의 직선 쿠폰을 재계산(같은 격자·물성) -----------------------
    c0h, n0h, _ = r3.vane_mesh(**A19_COUPON)
    c0d, n0d, kinfo = r3.vane_mesh(kerf=A19_KERF, **A19_COUPON)
    m0 = c0h[n0h][:, :, 0].mean(axis=1) <= span_hi * A19_COUPON["L"]
    r0h = _solve(c0h, n0h, A19_COUPON_MAT, r3.clamp_root(), m0)
    r0d = _solve(c0d, n0d, A19_COUPON_MAT, r3.clamp_root())
    i0h = r3.beam_mode_kinds(r0h).index("flap")
    i0d = r3.beam_mode_kinds(r0d).index("flap")
    vol0 = icad.cell_volumes(c0h, n0h)
    row0 = {
        "rung": "R0", "model": "straight coupon 30x4.1x1.0 (hex)",
        "geometry": "straight, span 30.00 mm", "boundary": "root clamp",
        "observable": "own flapwise fundamental",
        "mesh_mm": A19_COUPON["L"] / A19_COUPON["nx"] * 1e3,
        "ndof_healthy": int(r0h.ndof), "ndof_damaged": int(r0d.ndof),
        "mode_index_healthy": int(i0h), "mode_index_damaged": int(i0d),
        "f_healthy_Hz": float(r0h.freqs[i0h]), "f_damaged_Hz": float(r0d.freqs[i0d]),
        "UK_window": float(r0h.region_energy_frac[i0h]),
        "UM_window": float(r0h.region_kinetic_frac[i0h]),
        "window_volume_mm3": float(vol0[m0].sum() * 1e9),
        "window_volume_frac": float(vol0[m0].sum() / vol0.sum()),
        "note": f"a18 rung; kerf {kinfo['width_actual']*1e3:.2f} mm x "
                f"{kinfo['depth_actual']:.2f} depth; rho 8000 (rail material)",
    }
    row0["shift_pct"] = 100 * (row0["f_damaged_Hz"] / row0["f_healthy_Hz"] - 1)
    row0["shift_pct_first_order"] = _pred_pct(row0["UK_window"], row0["UM_window"])
    print(f"[a19] R0 f1={row0['f_healthy_Hz']:.1f} → {row0['f_damaged_Hz']:.1f} Hz "
          f"({row0['shift_pct']:+.3f} %) U_K={row0['UK_window']:.4f} "
          f"U_M={row0['UM_window']:.4f}", flush=True)

    cfg = B5_CONFIGS[args.config]
    spec_h = icad.ImpellerSpec(**cfg)
    spec_d = replace(spec_h, damage_vane=0, damage_frac=d_frac,
                     damage_span=(0.0, span_hi))
    xs, ys = icad.vane_camber(spec_h, 0, n_pts=4001)
    arc_mm = float(np.hypot(np.diff(xs), np.diff(ys)).sum() * 1e3)
    mat = dict(E=spec_h.E, nu=spec_h.nu, rho=spec_h.rho)

    rows = []
    for ms in args.mesh_size:
        tag = f"a19_{int(round(ms * 1e4))}"
        ch, nh = icad.build_geometry(spec_h, mesh_size=ms, workdir=args.workdir,
                                     tag=f"{tag}_h", include_shrouds=False, vanes=(0,))
        cd, nd = icad.build_geometry(spec_d, mesh_size=ms, workdir=args.workdir,
                                     tag=f"{tag}_d", include_shrouds=False, vanes=(0,))
        mask = icad.damaged_cell_mask(spec_h, ch, nh, 0)
        vol = icad.cell_volumes(ch, nh)
        group = [dict(row0, ladder_mm=ms * 1e3, chain=True)]
        win = {"window_volume_mm3": float(vol[mask].sum() * 1e9),
               "window_volume_frac": float(vol[mask].sum() / vol.sum())}

        # ---- R1: 손상·경계는 그대로, **기하만** as-built로 ----------------------
        sel1 = icad.clamp_vane_root(spec_h, 0)
        r1h, r1d = _solve(ch, nh, mat, sel1, mask), _solve(cd, nd, mat, sel1)
        kh, _ = icad.vane_mode_kinds(spec_h, r1h, 0)
        kd, _ = icad.vane_mode_kinds(spec_d, r1d, 0)
        ih, idd = kh.index("flap"), kd.index("flap")
        row1 = {
            "rung": "R1", "model": "as-built isolated vane (tet, order 2)",
            "geometry": f"log-spiral camber, arc {arc_mm:.2f} mm",
            "boundary": "root clamp", "observable": "own flapwise fundamental",
            "mesh_mm": ms * 1e3, "ladder_mm": ms * 1e3, "chain": True,
            "ndof_healthy": int(r1h.ndof), "ndof_damaged": int(r1d.ndof),
            "mode_index_healthy": int(ih), "mode_index_damaged": int(idd),
            "f_healthy_Hz": float(r1h.freqs[ih]),
            "f_damaged_Hz": float(r1d.freqs[idd]),
            "UK_window": float(r1h.region_energy_frac[ih]),
            "UM_window": float(r1h.region_kinetic_frac[ih]),
            "f_next_ratio_healthy": float(r1h.freqs[ih + 1] / r1h.freqs[ih]),
            "note": f"config {args.config}; kinds {','.join(kh[:4])}", **win,
        }
        row1["shift_pct"] = 100 * (row1["f_damaged_Hz"] / row1["f_healthy_Hz"] - 1)
        row1["shift_pct_first_order"] = _pred_pct(row1["UK_window"],
                                                 row1["UM_window"])
        group.append(row1)
        print(f"[a19] R1 mesh={ms*1e3:.1f} mm f1={row1['f_healthy_Hz']:.1f} → "
              f"{row1['f_damaged_Hz']:.1f} Hz ({row1['shift_pct']:+.3f} %) "
              f"U_K={row1['UK_window']:.4f} U_M={row1['UM_window']:.4f} "
              f"f2/f1={row1['f_next_ratio_healthy']:.2f} kinds={kh[:4]}", flush=True)

        # ---- R2: 슈라우드를 **강체**로 둔 극한 — 사다리 인자가 아니라 진단 -------
        # 4.1 mm 스팬 스트립이 되어 f가 300배로 오르고 호방향 하모닉이 **밀집**한다.
        # 손상 쪽은 얇아진 구역에 국재화된 모드가 클러스터 아래로 떨어지므로, 기본모드끼리
        # 짝지으면 정본 §3.6이 금지한 **주파수 순서 매칭**이 된다(메시가 달라 MAC도 불가).
        sel2 = icad.clamp_vane_shroud_faces(spec_h)
        r2h, r2d = _solve(ch, nh, mat, sel2, mask), _solve(cd, nd, mat, sel2)
        spacing = float(100 * np.max(r2h.freqs[1:] / r2h.freqs[:-1] - 1))
        n_below = int((r2d.freqs < r2h.freqs[0]).sum())
        row2 = {
            "rung": "R2", "model": "as-built isolated vane (tet, order 2)",
            "geometry": f"log-spiral camber, arc {arc_mm:.2f} mm",
            "boundary": "both shroud faces clamped (rigid-shroud limit)",
            "observable": "mode cluster — not a single matched mode",
            "mesh_mm": ms * 1e3, "ladder_mm": ms * 1e3, "chain": False,
            "ndof_healthy": int(r2h.ndof), "ndof_damaged": int(r2d.ndof),
            "f_healthy_Hz": float(r2h.freqs[0]),
            "f_cluster_hi_Hz": float(r2h.freqs[-1]),
            "n_modes_in_cluster": int(len(r2h.freqs)),
            "cluster_max_spacing_pct": spacing,
            "n_damaged_modes_below_cluster": n_below,
            "f_damaged_lowest_Hz": float(r2d.freqs[0]),
            "UK_window": float(np.mean(r2h.region_energy_frac)),
            "UM_window": float(np.mean(r2h.region_kinetic_frac)),
            "shift_pct": float("nan"),
            "note": "diagnostic only — dense cluster + damage localization; "
                    "frequency-order matching is forbidden (§3.6) and MAC is "
                    "unavailable across the two gmsh meshes", **win,
        }
        group.append(row2)
        print(f"[a19] R2 mesh={ms*1e3:.1f} mm cluster "
              f"{row2['f_healthy_Hz']/1e3:.1f}–{row2['f_cluster_hi_Hz']/1e3:.1f} kHz "
              f"(max spacing {spacing:.2f} %), damaged modes below cluster: {n_below} "
              f"(lowest {row2['f_damaged_lowest_Hz']/1e3:.1f} kHz), "
              f"U_K(cluster mean)={row2['UK_window']:.4f}", flush=True)

        # ---- R3: 강하는 커밋된 b5에서, 에너지 분율만 건전 조립체 1회 해에서 -----
        shift3, src = _a19_assembly_shift(ms * 1e3, args.config)
        row3 = {
            "rung": "R3", "model": "welded assembly (tet, order 2)",
            "geometry": f"same vane in 6-vane assembly, arc {arc_mm:.2f} mm",
            "boundary": "bore clamp (shrouds carry load)",
            "observable": "global m=1 pair mean", "mesh_mm": ms * 1e3,
            "ladder_mm": ms * 1e3, "shift_pct": shift3, "chain": True,
            "note": f"shift read from {src}" if src else "b5 output missing",
        }
        if not args.skip_assembly:
            ca, na = icad.build_geometry(spec_h, mesh_size=ms, workdir=args.workdir,
                                         tag=f"{tag}_asm")
            mask_a = icad.damaged_cell_mask(spec_h, ca, na, 0)
            vol_a = icad.cell_volumes(ca, na)
            ra = icad.solve_modal(spec_h, ca, na, n_modes=args.n_modes, order=2,
                                  region_mask=mask_a)
            # 축퇴쌍은 부분공간 안의 기저가 임의다 — **자취**(두 멤버 평균)만 불변이다(F135).
            row3.update({
                "ndof_healthy": int(ra.ndof),
                "f_healthy_Hz": float(np.mean(ra.freqs[:2])),
                "UK_window": float(np.mean(ra.region_energy_frac[:2])),
                "UM_window": float(np.mean(ra.region_kinetic_frac[:2])),
                "UK_member_lo": float(ra.region_energy_frac[0]),
                "UK_member_hi": float(ra.region_energy_frac[1]),
                "window_volume_mm3": float(vol_a[mask_a].sum() * 1e9),
                "window_volume_frac": float(vol_a[mask_a].sum() / vol_a.sum()),
            })
            row3["f_damaged_Hz"] = row3["f_healthy_Hz"] * (1 + shift3 / 100)
            row3["shift_pct_first_order"] = _pred_pct(row3["UK_window"],
                                                      row3["UM_window"])
            print(f"[a19] R3 mesh={ms*1e3:.1f} mm pair mean={row3['f_healthy_Hz']:.1f} Hz "
                  f"shift={shift3:+.3f} % (b5) U_K={row3['UK_window']:.5f} "
                  f"U_M={row3['UM_window']:.5f} "
                  f"members {row3['UK_member_lo']:.5f}/{row3['UK_member_hi']:.5f}",
                  flush=True)
        group.append(row3)

        # 인자 사슬은 **관측량이 정의된 단**만 잇는다(R2는 진단이라 빠진다).
        chain = [g for g in group if g["chain"]]
        fac = ladder_factors([g["shift_pct"] for g in chain])
        for g, f in zip(chain, fac):
            g["factor_vs_prev"] = float(f)
            g["factor_cum_vs_R0"] = float(chain[0]["shift_pct"] / g["shift_pct"]
                                          if g["shift_pct"] else float("nan"))
        # 정합 기하 기준의 희석비 = R1 → R3 (정본이 인용해야 하는 값)
        matched = (chain[1]["shift_pct"] / chain[-1]["shift_pct"]
                   if chain[-1]["shift_pct"] else float("nan"))
        for g in group:
            g.setdefault("factor_vs_prev", float("nan"))
            g.setdefault("factor_cum_vs_R0", float("nan"))
            g["dilution_matched_geometry"] = float(matched)
            g["damage_frac"] = d_frac
            g["damage_span_hi"] = span_hi
            g["dlnK_local"] = dln_k
            g["dlnM_local"] = dln_m
        rows += group
        print(f"[a19] mesh={ms*1e3:.1f} mm 사슬 인자: "
              + " × ".join(f"{f:.2f}" for f in fac[1:])
              + f" = {chain[0]['shift_pct']/chain[-1]['shift_pct']:.1f} 배 "
              f"| 정합기하 희석(R1→R3) = {matched:.1f} 배", flush=True)

    _save(pd.DataFrame(rows), "a19_geometry_matched_control.csv")


def docx_converter() -> Path:
    """markdown → docx 변환기의 경로. `MD2DOCX`로 덮어쓴다.

    이 패키지는 markdown까지만 만들고 문서 변환은 **외부 스크립트**에 맡긴다 — 변환기는
    이 저장소에 없을 수 있으므로(코드·데이터 배포본) 경로를 박지 않고 환경변수를 먼저 본다.
    """
    env = os.environ.get("MD2DOCX")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[1] / "docs" / "_generated" / "md2docx.py"


def render_docx(out_md, docx_path: str) -> None:
    """외부 변환기로 docx를 렌더한다. 변환기가 없으면 **경로를 말하고** 멈춘다."""
    import subprocess
    import sys

    conv = docx_converter()
    if not conv.exists():
        raise SystemExit(
            f"markdown → docx 변환기가 없다: {conv}\n"
            "  이 저장소는 markdown만 만든다 — 변환기 경로를 `MD2DOCX`로 지정하거나 "
            "`--docx` 없이 돌려 md를 쓰라.")
    subprocess.run([sys.executable, str(conv), str(out_md), docx_path], check=True)


def cmd_datapackage(args):
    """제출 **데이터 패키지** zip — 정본 인용 목록에서 뽑아 묶고 MANIFEST를 넣는다.

    보충자료 서두가 "artifacts travel with this document as a separate data package, listed
    with byte counts and SHA-256 digests in its manifest"라고 약속하므로, 그 약속을 손이 아니라
    명령이 지키게 한다(a18·이전 패키지가 실제로 낡었다 — 설계서 F153).
    """
    from . import submission as sub
    res = sub.build_data_package(Path(args.canon).read_text(),
                                 args.data or DATA, args.figs or FIGS, args.out)
    print(f"[datapackage] {res['out']}  파일 {len(res['files'])}개")
    if res["missing"]:
        raise SystemExit(f"인용됐으나 디스크에 없는 산출물: {res['missing']}")


def cmd_submission(args):
    """정본 md → JSV 제출본 md(+docx). 그림을 실제로 박고 내부표시를 뗀다.

    정본은 손대지 않는다 — STATUS NOTE와 `[SW]`는 이력이므로 제출본에서만 사라진다.
    docx는 외부 markdown → docx 변환기로 렌더하며(`MD2DOCX`, 없으면 저장소 상대경로),
    제출본 md의 이미지 경로가 그 md 위치
    기준 상대경로여야 그림이 박힌다(`--fig-dir` 기본값이 그렇게 잡혀 있다).
    """
    from . import submission as sub
    out_md = Path(args.out_md)
    res = sub.build_file(args.canon, out_md, fig_dir=args.fig_dir)
    miss = [p for ps in res["figures_inserted"].values()
            for p in ps if not (out_md.parent / p).exists()]
    print(f"[submission] {res['out']}  STATUS NOTE 제거 {res['status_notes_removed']}건, "
          f"그림 {len(res['figures_inserted'])}개({res['n_figure_files']} 파일) 삽입")
    for num, ps in sorted(res["figures_inserted"].items()):
        print(f"    Fig. {num}: {', '.join(Path(p).name for p in ps)}")
    if miss:
        raise SystemExit(f"그림 파일 없음 — `cli figs`를 먼저 돌려라: {miss}")
    if args.docx:
        render_docx(out_md, args.docx)


def cmd_supplementary(args):
    """보충자료 md(+docx) 생성 — 정본이 §Data availability·A.6·B.4에서 약속한 표들.

    표는 산출 CSV에서 직접 조판하므로 재실행하면 문서가 따라 바뀐다(손으로 옮겨 적지 않는다).
    """
    from . import supplementary as supp
    out_md = Path(args.out_md)
    tables = supp.SUPP_TABLES_R36 if args.layout == "r36" else None
    res = supp.build(args.data or DATA, tables=tables)
    out_md.write_text(res["text"], encoding="utf-8")
    res["out"] = str(out_md)
    print(f"[supplementary] {res['out']}")
    for tag, fname, n in res["tables"]:
        print(f"    Table {tag}: {fname} ({n} rows)")
    if res["missing"]:
        raise SystemExit(f"산출물 없음 — 먼저 생성해야 한다: {res['missing']}")
    if args.docx:
        render_docx(out_md, args.docx)


def build_parser():
    """서브커맨드 파서. `--mass`는 어디서도 `store_true`가 아니어야 한다(설계서 F20 ①·M8) —
    `test_mass_term.TestCouplingConvention`이 이 파서를 직접 검사한다."""
    p = argparse.ArgumentParser(prog="impeller_fingerprint.cli")
    sub = p.add_subparsers(dest="item", required=True)
    sp = sub.add_parser("a5")
    sp.add_argument("--mass", nargs="?", const="exact", default=None)
    #: 20 kHz = 설계서 §4-6이 동결한 DAQ 대역, 25.6 kHz = 51.2 kS/s 채널의 대역(F59 감도).
    sp.add_argument("--f-max", type=float, nargs="+", default=[2.0e4, 2.56e4])
    sp.add_argument("--pool-ms", type=int, nargs="+", default=None,
                    help="후보 풀의 방위차수(기본 0 1 2 3 4). C6 감도용: 0 1 2 3")
    sp.add_argument("--tag", default="",
                    help="산출 파일명 접미(생산 산출을 덮지 않게 한다)")
    sp.set_defaults(func=cmd_a5)
    for name, fn in [("a1", cmd_a1), ("a6", cmd_a6), ("a7", cmd_a7),
                     ("a8", cmd_a8), ("a10", cmd_a10)]:
        sp = sub.add_parser(name)
        sp.set_defaults(func=fn)
    for name, fn in [("a2", cmd_a2), ("a3", cmd_a3)]:
        sp = sub.add_parser(name)
        sp.add_argument("--n-xi", type=int, default=41)
        sp.add_argument("--n-s", type=int, default=31)
        sp.add_argument("--mass", nargs="?", const="exact", default=None,
                        help="질량항 포함 맵(F12). 결합비: exact(기본) 또는 숫자. "
                             "store_true였을 때 args.mass=True가 coupling=float(True)=1.0으로 "
                             "새는 버그가 있었다(설계서 M7) — 반드시 결합비를 명시적으로 넘긴다.")
        sp.set_defaults(func=fn)
    sp = sub.add_parser("figs")
    sp.set_defaults(func=lambda a: __import__("impeller_fingerprint.figures",
                                              fromlist=["x"]).make_all(DATA, FIGS))
    sp = sub.add_parser("a4")
    sp.add_argument("--mass", nargs="?", const="exact", default=None)
    sp.set_defaults(func=cmd_a4)
    sp = sub.add_parser("b1")
    sp.add_argument("--n-real", type=int, default=5000)
    sp.add_argument("--n-loc", type=int, default=20)
    sp.add_argument("--workers", type=int, default=18)
    sp.add_argument("--mass", nargs="?", const="exact", default=None,
                    help="질량항 결합비: exact(기본) 또는 숫자")
    sp.set_defaults(func=cmd_b1)

    sp = sub.add_parser("a12")
    sp.add_argument("--block", action="store_true",
                    help="정본 md에 붙일 References 블록도 stdout에 출력")
    sp.set_defaults(func=cmd_a12)

    sp = sub.add_parser("a11")
    sp.add_argument("--ref", type=int, default=A11_REF_WORK,
                    help="작업격자 단계(수렴 사다리 A11_REFS 중)")
    sp.add_argument("--planes", nargs="+", default=["stress", "strain"],
                    help="평면응력(좁은 보) / 평면변형(넓은 판 스트립) — 둘 다 실행")
    sp.add_argument("--conv-a-bars", type=float, nargs="+", default=[0.3, 0.5, 0.6],
                    help="메시수렴 사다리를 돌릴 균열깊이")
    sp.add_argument("--width-depths", type=float, nargs="+",
                    default=[0.25, 0.5, 0.625],
                    help="폭→0 극한 스윕의 깊이(B2 3D가 실현한 깊이와 동일)")
    sp.add_argument("--width-plane", default="stress")
    sp.set_defaults(func=cmd_a11)

    sp = sub.add_parser("b2")
    # **정본 산출 기본값**(2026-08-11 수정). 이전 기본값(nx=120, widths 0.0003/0.0006)은
    # 커밋된 `b2_vane3d.csv`를 재현하지 못했다: 요소제거 손상이므로 실현 커프 폭은
    # in_kerf 열수 × L/nx로 **양자화**되고, x_c = 0.2L이 절점에 놓이므로 선택은 대칭
    # 짝수열이다. nx=240(L/nx = 0.125 mm)에서 요청 0.25 mm → 2열 = 0.25 mm,
    # 1.0 mm → 8열 = 1.0 mm로 정확히 실현된다. nx=120에서는 둘 다 0.5 mm로 붕괴했다.
    sp.add_argument("--nx", type=int, default=240)
    sp.add_argument("--ny", type=int, default=2)
    sp.add_argument("--nz", type=int, default=8)
    sp.add_argument("--depths", type=float, nargs="+", default=[0.25, 0.5, 0.625])
    sp.add_argument("--widths", type=float, nargs="+", default=[0.00025, 0.001])
    # 스팬방향 폭: as-built(유로폭 b₂ = 4.1 mm)를 먼저, 옛 규약 2h를 연속성 확인용으로.
    sp.add_argument("--vane-widths", type=float, nargs="+",
                    default=[geo.B2_CHANNEL, 2 * geo.VANE.h])
    sp.set_defaults(func=cmd_b2)

    sp = sub.add_parser("b3")
    # 형상보존 기본값(F11′): nr ∈ {10,20,30}에서 ξ_c±0.1 경계가 셀면, nθ=48은 Δθ 15/30/60°의
    # 정수배(7.5° 셀)이자 θ₀=22.5°가 셀면, nz=4는 깊이비 0.5의 정수배.
    sp.add_argument("--nr", type=int, default=20)
    sp.add_argument("--ntheta", type=int, default=48)
    sp.add_argument("--nz", type=int, default=4)
    sp.add_argument("--xi", type=float, nargs="+", default=[0.2, 0.5, 0.8])
    sp.add_argument("--dtheta", type=float, nargs="+", default=[15, 30])
    sp.add_argument("--depths", type=float, nargs="+", default=[0.5])
    sp.add_argument("--half-xi", type=float, default=0.1,
                    help="포켓 반경 반폭(ξ 단위). 0.1이면 경계가 ξ의 0.1 배수 → nr∈{10,20,30} 정합")
    sp.add_argument("--purity-min", type=float, default=0.5,
                    help="방위차수 투영의 진단용 순도 하한(F21)")
    sp.add_argument("--mac-min", type=float, default=0.8,
                    help="**매칭 수용기준** subspace MAC 하한(정본 §3.6·§5 E2와 동일한 0.8). "
                         "미달 셀은 해당 차수를 버린다 — 억지 짝짓기 금지")
    sp.set_defaults(func=cmd_b3)

    sp = sub.add_parser("b3x")
    sp.add_argument("--sigma-rel", type=float, default=3e-4)
    sp.set_defaults(func=cmd_b3x)

    sp = sub.add_parser("b4")
    sp.add_argument("--grids", type=int, nargs="+", action="append", default=None)
    sp.set_defaults(func=cmd_b4)

    sp = sub.add_parser("b5")
    sp.add_argument("--configs", nargs="+", default=list(B5_CONFIGS),
                    choices=list(B5_CONFIGS))
    sp.add_argument("--damage", type=float, nargs="+", default=[0.0, 0.3, 0.6])
    sp.add_argument("--n-modes", type=int, default=10)
    sp.add_argument("--mesh-size", type=float, default=0.0012)
    sp.add_argument("--workdir", default="/tmp")
    sp.set_defaults(func=cmd_b5)

    sp = sub.add_parser("a19")
    sp.add_argument("--config", default="asbuilt", choices=list(B5_CONFIGS))
    sp.add_argument("--mesh-size", type=float, nargs="+", default=[0.0012, 0.0010],
                   help="고립 베인·조립체 격자(조립체 사다리와 같은 값을 쓴다)")
    sp.add_argument("--damage-frac", type=float, default=0.6)
    sp.add_argument("--damage-span", type=float, default=0.25)
    sp.add_argument("--n-modes", type=int, default=8)
    sp.add_argument("--workdir", default="/tmp")
    sp.add_argument("--skip-assembly", action="store_true",
                   help="조립체 에너지 분율 계산 생략(강하는 커밋된 b5에서 읽는다)")
    sp.set_defaults(func=cmd_a19)

    sp = sub.add_parser("b6")
    sp.add_argument("--nr", type=int, default=20)
    sp.add_argument("--ntheta", type=int, default=48)
    sp.add_argument("--nz", type=int, default=4)
    sp.add_argument("--xi", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.8, 0.9])
    sp.add_argument("--half-xi", type=float, nargs="+", default=[0.1],
                    help="밴드 반폭(ξ). nr=20에서 0.05·0.1이 셀면에 정확히 놓인다")
    sp.add_argument("--depths", type=float, nargs="+", default=[0.25, 0.5])
    sp.add_argument("--append", action="store_true",
                    help="기존 b6_epert_fem.csv에 누적(같은 앵커는 새 값으로 대체)")
    sp.add_argument("--n-trial", type=int, default=36,
                    help="Kirchhoff 정확재해의 Ritz 시행함수 수(급격한 밴드는 36 필요)")
    sp.add_argument("--n-trial-hi", type=int, default=44,
                    help="Ritz 수렴검정용 상위 차수")
    sp.add_argument("--recompute-kirchhoff", action="store_true",
                    help="3D는 재사용하고 Kirchhoff 열만 재계산(수렴차수 상향 시)")
    sp.add_argument("--grid-check", type=int, nargs="*", default=None,
                    help="형상보존 nr 사다리(예: 10 30)로 첫 앵커를 재계산")
    sp.set_defaults(func=cmd_b6)

    sp = sub.add_parser("b7")
    sp.add_argument("--n-real", type=int, default=1000)
    sp.add_argument("--n-loc", type=int, default=15)
    sp.add_argument("--workers", type=int, default=18)
    sp.add_argument("--n-starts", type=int, default=4)
    sp.add_argument("--half-ref", type=float, default=None,
                    help="κ를 뽑을 기준 손상족의 반폭(기본: b6에 있는 최대값)")
    sp.add_argument("--s-bar", type=float, nargs="+",
                    default=[0.12, 0.135, 0.15],
                    help="MC 심각도. 기준족 앵커 범위 밖은 자동 제외된다(외삽 금지)")
    sp.add_argument("--loo-max", type=float, default=0.25,
                    help="대리모델 수용기준: 검증 상대잔차 중앙값 상한. 넘으면 중단한다")
    sp.set_defaults(func=cmd_b7)

    sp = sub.add_parser("a13")
    sp.add_argument("--configs", nargs="+", default=list(A13_CONFIGS),
                    choices=list(A13_CONFIGS))
    sp.add_argument("--n-modes", type=int, default=30)
    sp.add_argument("--m-max", type=int, default=12)
    sp.add_argument("--wrap-deg", type=float, default=60.0,
                    help="베인 wrap 각(**미확정** 스윕 파라미터, 정본 §3.6-ii)")
    sp.set_defaults(func=cmd_a13)

    sp = sub.add_parser("supplementary")
    sp.add_argument("--layout", default="default", choices=["default", "r36"],
                   help="r36 = 제출 계열(§5 삭제판) 번호: ρ 스윕이 S5, 정합기하가 S9")
    sp.add_argument("--data", default=None)
    sp.add_argument("--out-md", default=str(CANON_MD.parent
                                           / "2026-08-15-paperB-jsv-supplementary.md"))
    sp.add_argument("--docx", default=None)
    sp.set_defaults(func=cmd_supplementary)

    sp = sub.add_parser("datapackage")
    sp.add_argument("--canon", default=str(CANON_MD))
    sp.add_argument("--data", default=None)
    sp.add_argument("--figs", default=None)
    #: 제출 문서 세트와 같은 날짜 이름을 쓴다(submission의 `--out-md` 규약과 같다).
    sp.add_argument("--out", default=str(OUT_ROOT.parent / "_out"
                                        / "2026-08-16-paperB-jsv-supplementary-data.zip"))
    sp.set_defaults(func=cmd_datapackage)

    sp = sub.add_parser("submission")
    sp.add_argument("--canon", default=str(CANON_MD))
    sp.add_argument("--out-md", default=str(CANON_MD.parent
                                           / "2026-08-15-paperB-jsv-submission.md"))
    sp.add_argument("--fig-dir", default="../_generated/figures/paper3",
                    help="제출본 md 기준 **상대경로**여야 변환기가 그림을 찾는다")
    sp.add_argument("--docx", default=None, help="지정하면 docx도 렌더한다")
    sp.set_defaults(func=cmd_submission)

    sp = sub.add_parser("a14")
    sp.add_argument("--rails", nargs="+", default=["canonical", "free_free", "vane"],
                    choices=["canonical", "free_free", "vane"])
    sp.add_argument("--b5-config", default="asbuilt", choices=list(B5_CONFIGS))
    sp.add_argument("--mesh-size", type=float, default=0.0012,
                    help="정본 b5와 같은 격자여야 인용값이 같은 레일에서 나온다")
    sp.add_argument("--n-modes", type=int, default=10)
    sp.add_argument("--workdir", default="/tmp")
    sp.set_defaults(func=cmd_a14)

    sp = sub.add_parser("a13x")
    sp.add_argument("--configs", nargs="+", default=["c6_asbuilt"],
                    choices=list(A13_CONFIGS))
    sp.add_argument("--levels", type=int, nargs=4, action="append", default=None,
                    metavar=("N_R", "N_THETA", "N_Z_SHROUD", "N_Z_CHANNEL"),
                    help="사다리 한 단계(여러 번 지정). 기본 = A13X_LEVELS")
    sp.add_argument("--footprint", type=int, nargs=2, default=list(A13X_FOOTPRINT),
                    metavar=("N_R", "N_THETA"),
                    help="발자국 기준격자 — 사다리 전체가 같은 고체영역을 쓴다(F78)")
    sp.add_argument("--n-modes", type=int, default=20,
                    help="모든 단계에서 같아야 한다(floor 통계의 비교가능성)")
    sp.add_argument("--m-max", type=int, default=12)
    sp.add_argument("--wrap-deg", type=float, default=60.0)
    sp.set_defaults(func=cmd_a13x)

    sp = sub.add_parser("fig2")
    sp.set_defaults(func=cmd_fig2)

    sp = sub.add_parser("fig1")
    sp.add_argument("--config", default="c6_asbuilt", choices=list(A13_CONFIGS))
    sp.add_argument("--n-modes", type=int, default=20)
    sp.add_argument("--m-max", type=int, default=12)
    sp.add_argument("--wrap-deg", type=float, default=60.0)
    sp.add_argument("--cut-sector", type=float, nargs=2, default=[-105.0, 5.0],
                    help="전면 슈라우드를 잘라내는 방위 구간[deg] (베인 노출용)")
    sp.add_argument("--panels", action="store_true",
                    help="패널별 개별 png도 저장")
    sp.set_defaults(func=cmd_fig1)

    return p


def main(argv=None):
    p = build_parser()
    args = p.parse_args(argv)
    if getattr(args, "item", None) == "a13x" and not args.levels:
        args.levels = [list(l) for l in A13X_LEVELS]
    if getattr(args, "item", None) == "b4" and not args.grids:
        # 포켓(반경 [a+0.3L, a+0.6L], Δθ=30°, 깊이 0.5)을 **정확히** realize 하는 격자만
        # 쓴다(설계서 F11′): nr ∈ {10,20,30} → 0.3nr·0.6nr 정수라 반경경계가 셀면,
        # nθ ∈ {24,48,72} → 30°가 정수배, nz 짝수 → 깊이 0.5가 층 경계.
        # (구 기본값 (6,24,2)/(12,48,4)/(18,72,6)은 nr이 규약을 위반해 반경경계가
        #  격자마다 다른 셀면으로 스냅됐다 — F11이 오진한 '미수렴'의 원인.)
        args.grids = [(10, 24, 2), (20, 48, 4), (30, 72, 6)]
    args.func(args)


if __name__ == "__main__":
    main()
