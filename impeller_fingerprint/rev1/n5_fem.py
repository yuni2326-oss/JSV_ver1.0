"""N5·b7을 **FEM 정확 Kirchhoff**로 다시 낸다 — 원고 §4.3 수치를 한 솔버로 통일(리드 후속 지시).

배경(N4 §2·§9, F178): 정본 밴드 재해 `forward.eta_bar_exact_band`(르장드르 36, `kernels._basis`가
르장드르를 단항식 계수로 바꿔 평가)는 깊은 밴드에서 자릿수를 잃어 b6 앵커에서 FEM 대비 최대 26 floor
틀린다. b7 대리 진실과 N5의 밴드 정확맵·δ 학습이 모두 그 경로를 썼으므로 여기서 FEM으로 바꾼다.

**바꾸는 것**(나머지는 N5·b7 설계 그대로):
- 밴드 정확맵 표 η̄(ξ, p): N5와 같은 격자(ξ 0.01 × p 0.02, 꺾임점 ξ = 0.1·0.9)를 `n4_width.band_fem_eta`
  (반경 Hermite FEM, 밴드 경계 절점)로 채운다.
- 앵커 δ = η̄_3D − η̄_K,FEM(b6 16개 + N5 외부 3D 8개). 3D 값은 그대로 쓴다(재해석 없음).
- b7 대리 진실 y = η̄_K,FEM,band + δ̂_FEM(b7 보간 규칙). `montecarlo.surrogate_truth`를 **이 프로세스
  안에서만** FEM 진실로 바꿔 끼우고(`unittest.mock.patch.object`) 정본 `run_production`을 같은
  셀·시드·실현수로 부른다 — 생성·잡음·적합·통계 코드는 정본 그대로다. 정본 파일은 고치지 않는다.

N5의 함수(`heldout_noiseless`, `_mc_worker`, `wls_fit`, `delta_spec`, `b7_stats` …)는 **읽기 전용으로**
가져다 쓴다. 정본 표 경로를 하드코딩한 `heldout_mc`·`external_test`만 같은 논리로 여기서 다시 썼다.

산출: `rev1_n4_b7_fem.csv`, `rev1_n5_heldout_fem_{band_table.npz, anchors, noiseless, mc, external_test,
external_mc, summary.json}`. 실행: `python -m impeller_fingerprint.rev1.n5_fem all --workers 4`.
"""
from __future__ import annotations

import argparse
import json
import time
from multiprocessing import get_context
from unittest import mock

import numpy as np
import pandas as pd

from .. import montecarlo as mc
from . import DATA, out_csv
from . import n4_width as n4
from . import n5_heldout as n5

TABLE_NPZ = "rev1_n5_heldout_fem_band_table.npz"


# ============================================================ FEM 밴드 표
def _band_fem_row(args):
    xi, ps, half = args
    r1, r2 = n5.band_edges(xi, half)
    out = []
    for p in ps:
        if p <= 0.0:
            out.append(np.zeros(len(n5.MODES)))
        else:
            out.append(n4.band_fem_eta(r1, r2, 1.0 - (1.0 - p) ** 3))
    return np.array(out)


def build_band_table_fem(half: float = n5.HALF_REF, dxi: float = 0.01, dp: float = 0.02,
                         p_max: float = 0.80, workers: int = 4):
    """N5 `build_band_table`과 같은 격자·같은 조각 분할 — 값만 FEM."""
    xs = np.round(np.arange(0.0, 1.0 + 1e-9, dxi), 10)
    for b in (half, 1.0 - half):
        if not np.isclose(xs, b).any():
            raise ValueError(f"ξ 격자가 꺾임점 {b}를 포함하지 않는다")
    ps = np.round(np.arange(0.0, p_max + 1e-9, dp), 10)
    t0 = time.time()
    with get_context("fork").Pool(workers) as pl:
        rows = pl.map(_band_fem_row, [(float(x), ps, half) for x in xs], chunksize=1)
    vals = np.array(rows)                                  # (n_xi, n_p, n_modes)
    tab = n5.PiecewiseTable(xs, ps, vals, breaks=(half, 1.0 - half))
    meta = {"kind": "band_fem", "half": half, "fem_n_elem": n4.FEM_N, "dxi": dxi,
            "dp": dp, "p_max": p_max, "build_s": time.time() - t0}
    np.savez(DATA / TABLE_NPZ, xs=xs, ys=ps, vals=vals, meta=json.dumps(meta))
    return tab


def load_band_fem(half: float = n5.HALF_REF):
    z = np.load(DATA / TABLE_NPZ, allow_pickle=False)
    tab = n5.PiecewiseTable(z["xs"], z["ys"], z["vals"], breaks=(half, 1.0 - half))
    return n5.BandMap(tab, half), {"xs": z["xs"], "ys": z["ys"], "vals": z["vals"]}


def table_check(band, n: int = 200, seed: int = 7) -> dict:
    """표 보간 대 직접 FEM — 임의 (ξ, p)에서의 최대·중앙 오차(η̄ 단위와 0.1 % floor 단위)."""
    rng = np.random.default_rng(seed)
    fl = n5.floors(1e-3)
    errs = []
    for _ in range(n):
        xi, p = float(rng.uniform(0.0, 1.0)), float(rng.uniform(0.01, 0.78))
        r1, r2 = n5.band_edges(xi)
        direct = n4.band_fem_eta(r1, r2, 1.0 - (1.0 - p) ** 3)
        errs.append(np.abs(band.t(xi, p) - direct) / fl)
    e = np.array(errs)
    return {"n": n, "median_floor": float(np.median(e)), "max_floor": float(np.max(e))}


# ============================================================ 앵커(FEM δ)
def _k_fem(r1_mm: float, r2_mm: float, depth: float) -> np.ndarray:
    return n4.band_fem_eta(r1_mm / 1e3, r2_mm / 1e3, 1.0 - (1.0 - depth) ** 3)


def anchors_fem(half: float | None = n5.HALF_REF) -> pd.DataFrame:
    """N5 `load_anchors` 형식 — etak·delta 열을 FEM으로 교체(정본 값은 *_canon 열로 남긴다)."""
    a = n5.load_anchors(half)
    for idx, r in a.iterrows():
        kf = _k_fem(r.r1_mm, r.r2_mm, r.depth_frac)
        for i, m in enumerate(n5.MS):
            a.loc[idx, f"etak_canon_m{m}"] = r[f"etak_m{m}"]
            a.loc[idx, f"delta_canon_m{m}"] = r[f"delta_m{m}"]
            a.loc[idx, f"etak_m{m}"] = kf[i]
            a.loc[idx, f"delta_m{m}"] = (r[f"eta3d_m{m}"] - kf[i]
                                         if np.isfinite(r[f"eta3d_m{m}"]) else np.nan)
    return a


def ext_anchors_fem() -> pd.DataFrame:
    """N5 외부 3D 8구성 — 3D 값은 그대로, δ_true만 FEM 기준으로."""
    e = pd.read_csv(out_csv("n5_ext3d_anchors"))
    e = e[e["matched"] == True].copy()                          # noqa: E712
    e["s_key"] = e["s_bar_pct"].round(2)
    rows = []
    for (x, p), g in e.groupby(["xi_c", "depth_frac"]):
        kf = _k_fem(float(g.r1_mm.iloc[0]), float(g.r2_mm.iloc[0]), float(p))
        rec = {"xi_c": float(x), "depth_frac": float(p),
               "s_bar": float(g["s_bar_pct"].iloc[0]) / 100.0,
               "s_key": float(g["s_key"].iloc[0]), "n_modes_matched": int(len(g))}
        for i, m in enumerate(n5.MS):
            gm = g[g.m == m]
            e3 = float(gm["eta_3d"].iloc[0]) if len(gm) else np.nan
            rec[f"eta3d_m{m}"] = e3
            rec[f"delta_m{m}"] = e3 - kf[i] if np.isfinite(e3) else np.nan
            rec[f"delta_canon_m{m}"] = (float(gm["delta_modelform"].iloc[0]) if len(gm)
                                        else np.nan)
        rows.append(rec)
    return pd.DataFrame(rows)


# ============================================================ held-out MC(FEM 표)
def heldout_mc_fem(anchors: pd.DataFrame, band, band_npz: dict, c_list=(1e-3, 3e-4),
                   n_real: int = 300, workers: int = 4) -> pd.DataFrame:
    """N5 `heldout_mc`와 같은 논리·같은 시드 규칙 — 표만 FEM."""
    jobs = []
    for fo in n5.make_folds(anchors):
        spec = n5.delta_spec(fo["train"])
        sig_delta = n5.nested_loo_cov(fo["train"], band)
        for idx in fo["held"]:
            r = anchors.loc[idx]
            xi, s = float(r.xi_c), float(r.s_bar)
            hull = n5.inside_hull(fo["train"], xi, float(r.s_key))
            if not hull:
                continue
            y3d = np.array([r[f"eta3d_m{m}"] for m in n5.MS])
            key = {"fold": fo["fold"], "held_desc": fo["desc"], "xi_true": xi,
                   "s_true": s, "inside_hull": hull,
                   "sig_delta_diag": "|".join(f"{v:.3e}" for v in np.diag(sig_delta))}
            for ci, c in enumerate(c_list):
                seed = n5.SEED0 + 1000 * int(round(xi * 100)) + 10 * int(round(s * 1000)) + ci
                jobs.append((key, y3d, xi, s, c, seed, n_real, spec, sig_delta, n5.HALF_REF,
                             band_npz))
    with get_context("fork").Pool(workers) as pl:
        res = pl.map(n5._mc_worker, jobs, chunksize=1)
    return pd.DataFrame([r for rr in res for r in rr])


def external_test_fem(anchors: pd.DataFrame, band, band_npz: dict, c_list=(1e-3, 3e-4),
                      n_real: int = 300, workers: int = 4):
    """N5 `external_test`와 같은 논리·같은 시드 — 16개 앵커 전부로 훈련, 표·δ는 FEM."""
    ext = ext_anchors_fem()
    spec = n5.delta_spec(anchors)
    corr = n5.CorrectedMap(band, spec)
    lin = n5.LinearGaussMap()
    pool = n5.pool_kernels()
    sig = n5.sigma_y(1e-3)
    sig_delta = n5.nested_loo_cov(anchors, band)
    det, jobs = [], []
    for _, r in ext.iterrows():
        xi, s = float(r.xi_c), float(r.s_bar)
        y3d = np.array([r[f"eta3d_m{m}"] for m in n5.MS])
        if np.isnan(y3d).any():
            continue
        rec = {"xi_true": xi, "s_true": s, "depth_frac": float(r.depth_frac),
               "inside_hull": n5.inside_hull(anchors, xi, float(r.s_key))}
        e_unc, e_cor = y3d - band(xi, s), y3d - corr(xi, s)
        for c in c_list:
            fl = n5.floors(c)
            rec[f"fwd_unc_floor_max_c{c:g}"] = float(np.max(np.abs(e_unc) / fl))
            rec[f"fwd_cor_floor_max_c{c:g}"] = float(np.max(np.abs(e_cor) / fl))
        dh = mc.kappa_eval(spec, xi, s, key="delta")
        for i, m in enumerate(n5.MS):
            rec[f"delta_true_m{m}"] = float(r[f"delta_m{m}"])
            rec[f"delta_hat_m{m}"] = float(dh[i])
        fa = n5.est.fit(y3d, pool, n5.PLATE, sig, w=n5.W_GAUSS, mass="exact", n_starts=4)
        rec.update(n5._fit_cols("prod_lin_gauss", fa["xi_d"], fa["s_bar"], xi, s, lin, y3d,
                                sig, fa["boundary_hit"]))
        for tag, fmap in (("unc_band", band), ("cor_band", corr)):
            f = n5.wls_fit(y3d, fmap, sig, grid=True)
            rec.update(n5._fit_cols(tag, f["xi"], f["s"], xi, s, fmap, y3d, sig,
                                    f["boundary_xi"] or f["boundary_s"], f["penalty"]))
        det.append(rec)
        for ci, c in enumerate(c_list):
            seed = (n5.SEED0 + 500000 + 1000 * int(round(xi * 100))
                    + 10 * int(round(s * 1000)) + ci)
            key = {"fold": "external_3d", "held_desc": f"xi={xi:.2f},p={r.depth_frac}",
                   "xi_true": xi, "s_true": s, "inside_hull": rec["inside_hull"],
                   "sig_delta_diag": "|".join(f"{v:.3e}" for v in np.diag(sig_delta))}
            jobs.append((key, y3d, xi, s, c, seed, n_real, spec, sig_delta, n5.HALF_REF,
                         band_npz))
    with get_context("fork").Pool(workers) as pl:
        res = pl.map(n5._mc_worker, jobs, chunksize=1)
    return pd.DataFrame(det), pd.DataFrame([r for rr in res for r in rr])


# ============================================================ b7(FEM 진실)
def fem_surrogate_truth(plate, modes, xi_d: float, s_bar: float, spec: dict) -> np.ndarray:
    """`montecarlo.surrogate_truth`와 같은 서명·같은 기하 — Kirchhoff 밴드만 FEM으로."""
    r1, r2 = n5.band_edges(xi_d, spec["half_xi"])
    dk = s_bar * plate.extent / (r2 - r1)          # = band_depth_for_s_bar의 d_K
    return (n4.band_fem_eta(r1, r2, dk, modes=tuple(tuple(x) for x in modes))
            + mc.kappa_eval(spec, xi_d, s_bar, key="delta"))


def b7_spec_fem(anchors: pd.DataFrame) -> dict:
    """b7 격자 규칙(전 앵커, `delta_spec`)으로 FEM δ 사양."""
    return n5.delta_spec(anchors)


def rerun_b7_fem(spec: dict, n_real: int = 1000, workers: int = 4) -> pd.DataFrame:
    """정본 `run_production`을 같은 셀·시드(seed0 = 20260808)·실현수·n_starts = 4로 —
    진실 함수만 이 프로세스(와 fork된 워커) 안에서 FEM으로 바꿔 끼운다."""
    with mock.patch.object(mc, "surrogate_truth", fem_surrogate_truth):
        rows = mc.run_production(n5.PLATE, n5.MODES, n5.B7_XI, list(n5.B7_S),
                                 n5.SIGMA_RELS, w=n5.W_GAUSS, n_real=n_real,
                                 n_workers=workers, n_grid=1001, mass="exact",
                                 surrogate=spec, n_starts=4, seed0=20260808)
    base = mc.run_production(n5.PLATE, n5.MODES, n5.B7_XI, list(n5.B7_S), n5.SIGMA_RELS,
                             w=n5.W_GAUSS, n_real=n_real, n_workers=workers, n_grid=1001,
                             mass="exact", n_starts=4, seed0=20260808)
    d = pd.DataFrame(rows + base)
    d["kirchhoff_solver"] = "radial_hermite_fem_n100"
    return d


# ============================================================ 요약
def summarize_fem(ho: pd.DataFrame, hmc: pd.DataFrame, xdet: pd.DataFrame,
                  xmc: pd.DataFrame, b7: pd.DataFrame | None) -> dict:
    out = n5.summarize(None, ho, hmc)
    g = ho[(ho.fold == "leave_one_anchor") & ho.inside_hull]
    out["ho_interp_n"] = int(len(g))
    # 보정 후 |e_ξ| > 0.1 mm 사례와 χ² 판정(dof 2, 95 % = 5.991) — 앵커 껍질 안 + 외부
    cases = []
    for src, df in (("anchor_loo_interp", g), ("external", xdet)):
        for _, r in df.iterrows():
            if abs(r["cor_band_bias_xi_mm"]) > 0.1:
                cases.append({"source": src, "xi": float(r.xi_true), "s": float(r.s_true),
                              "e_xi_mm": float(r["cor_band_bias_xi_mm"]),
                              "e_s_pp": float(r["cor_band_bias_s_pp"]),
                              "chi2": float(r["cor_band_chi2_c1e3"]),
                              "flagged_95": bool(r["cor_band_chi2_c1e3"] > 5.991)})
    out["corrected_gt_0p1mm_cases"] = cases
    out["corrected_gt_0p1mm_n"] = len(cases)
    out["corrected_gt_0p1mm_flagged"] = int(sum(c["flagged_95"] for c in cases))
    for tag in ("prod_lin_gauss", "unc_band", "cor_band"):
        out[f"ext_{tag}_abs_err_xi_median_mm"] = float(xdet[f"{tag}_bias_xi_mm"].abs().median())
        out[f"ext_{tag}_abs_err_xi_max_mm"] = float(xdet[f"{tag}_bias_xi_mm"].abs().max())
        out[f"ext_{tag}_abs_err_s_median_pp"] = float(xdet[f"{tag}_bias_s_pp"].abs().median())
        out[f"ext_{tag}_abs_err_s_max_pp"] = float(xdet[f"{tag}_bias_s_pp"].abs().max())
    for (c, mp), gg in xmc.groupby(["c", "map"]):
        k = f"extmc_c{c:g}_{mp}"
        out[k + "_rmse_xi_median_mm"] = float(gg["rmse_xi_mm"].median())
        out[k + "_coverage95_xi_median"] = float(gg["coverage95_xi"].median())
        out[k + "_coverage95_s_median"] = float(gg["coverage95_s"].median())
    if b7 is not None:
        out["b7_fem"] = n5.b7_stats(b7)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("item", choices=["table", "b7", "heldout", "external", "summary", "all"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n-real", type=int, default=300)
    ap.add_argument("--n-real-b7", type=int, default=1000)
    args = ap.parse_args(argv)
    items = ["table", "b7", "heldout", "external", "summary"] if args.item == "all" else [args.item]
    for it in items:
        t0 = time.time()
        if it == "table":
            build_band_table_fem(workers=args.workers)
            band, _ = load_band_fem()
            chk = table_check(band)
            json.dump(chk, open(out_csv("n5_heldout_fem_table_check").with_suffix(".json"), "w"),
                      indent=1)
            print("[n5_fem] table check", chk, flush=True)
        elif it == "b7":
            spec = b7_spec_fem(anchors_fem())
            rerun_b7_fem(spec, n_real=args.n_real_b7, workers=args.workers).to_csv(
                out_csv("n4_b7_fem"), index=False)
        elif it == "heldout":
            band, npz = load_band_fem()
            anc = anchors_fem()
            anc.to_csv(out_csv("n5_heldout_fem_anchors"), index=False)
            n5.heldout_noiseless(anc, band).to_csv(out_csv("n5_heldout_fem_noiseless"),
                                                   index=False)
            heldout_mc_fem(anc, band, npz, n_real=args.n_real, workers=args.workers).to_csv(
                out_csv("n5_heldout_fem_mc"), index=False)
        elif it == "external":
            band, npz = load_band_fem()
            det, xm = external_test_fem(anchors_fem(), band, npz, n_real=args.n_real,
                                        workers=args.workers)
            det.to_csv(out_csv("n5_heldout_fem_external_test"), index=False)
            xm.to_csv(out_csv("n5_heldout_fem_external_mc"), index=False)
        elif it == "summary":
            rd = lambda n: pd.read_csv(out_csv(n))
            b7p = out_csv("n4_b7_fem")
            s = summarize_fem(rd("n5_heldout_fem_noiseless"), rd("n5_heldout_fem_mc"),
                              rd("n5_heldout_fem_external_test"),
                              rd("n5_heldout_fem_external_mc"),
                              pd.read_csv(b7p) if b7p.exists() else None)
            with open(out_csv("n5_heldout_fem_summary").with_suffix(".json"), "w") as fh:
                json.dump(s, fh, indent=1, default=float)
            print(json.dumps({k: v for k, v in s.items() if not isinstance(v, list)},
                             indent=1, default=float)[:4000], flush=True)
        print(f"[n5_fem] {it} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
