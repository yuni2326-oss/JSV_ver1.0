"""N0 — 생산 MC(B1)의 물리 허용영역 감사 (R1-P5 보조, 받은 계획서 v2 §6 N0 마지막 항목들).

B1(`cli b1 --mass exact`)과 **같은 셀·시드·적합**을 다시 돌려 B1 요약통계를 비트 단위로 재현하는지 확인하고,
다음을 센다.

- 최종 추정 S̄̂가 물리 상한을 넘은 적합 수. 상한은 peak d_K = 1 ↔ S̄_D = w√π/(b − a) = 25.13 %(w = 3 mm).
- 탐색 중 상한 위에서 평가된 시행점(목적함수 호출) 수와, 그런 시행점을 한 번이라도 거친 적합의 수.
- 경계 접촉(ξ̂ ∈ {0, 1} 또는 S̄̂ ∈ {0, 1})과 함께 세 상태(성공·경계포화·물리 무효)로 나눈 개수.

수치 처리(정본 그대로, 수정하지 않음): 1차 맵의 질량항은 d_K를 [0, 1 − 10⁻¹²]로 잘라 d_M → 1에서
포화시키고 강성항은 자르지 않는다(`forward.mass_field_exact`). 상한 위 시행점은 이 연장으로 평가된다.

    PYTHONPATH=. python -m impeller_fingerprint.rev1.n0_admissibility --workers 16   (저장소 루트에서; 출력 루트는 PAPER3_OUT)
"""
from __future__ import annotations

import argparse
import math
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from .. import estimator as est
from .. import forward as fwd
from .. import geometry as geo
from .. import identifiability as idf
from .. import kernels as ker
from .. import montecarlo as mc
from .. import noise as noi
from . import out_csv

PLATE = geo.DISK
MODES = [(0, 0), (1, 0), (2, 0), (3, 0)]
W_GAUSS = 0.003
SIGMA_RELS = (1e-4, 3e-4, 1e-3, 3e-3)
S_LIST = (0.01, 0.05, 0.15)
SEED0 = 20260801


def s_phys_max(w: float = W_GAUSS, plate=PLATE) -> float:
    """peak d_K = 1에 대응하는 명목 심각도 상한 w√π/(b − a)."""
    return w * math.sqrt(math.pi) / plate.extent


_CALLS = {"n": 0, "above": 0}


def _counting(orig, limit):
    def wrapped(pool, xi_d, s_bar, w, plate, coupling=fwd.MASS_COUPLING):
        _CALLS["n"] += 1
        if s_bar > limit:
            _CALLS["above"] += 1
        return orig(pool, xi_d, s_bar, w, plate, coupling=coupling)
    return wrapped


def audit_cell(cell: mc.Cell, n_grid: int = 1001, mass: str = "exact") -> dict:
    """`montecarlo.run_cell`(B1 경로)와 같은 계산 + 허용영역 계수."""
    limit = s_phys_max()
    orig = fwd.eta_bar_linear_mass
    fwd.eta_bar_linear_mass = _counting(orig, limit)
    try:
        pool = [ker.mode_kernel(PLATE, m=m, n=n, n_grid=n_grid) for m, n in MODES]
        sigma = noi.sigma_y_for_modes(pool, cell.sigma_rel, rho=0.0)
        y0 = orig(pool, cell.xi_d, cell.s_bar, W_GAUSS, PLATE,
                  coupling=fwd.resolve_coupling(mass))
        rng = np.random.default_rng(cell.seed)
        eps = noi.sample(sigma, rng, size=cell.n_real)
        met = idf.metrics(pool, PLATE, (cell.xi_d, cell.s_bar), W_GAUSS, sigma, mass=mass)
        xis = np.empty(cell.n_real)
        sbs = np.empty(cell.n_real)
        trial_above = np.zeros(cell.n_real, dtype=int)
        trial_n = np.zeros(cell.n_real, dtype=int)
        hit = np.zeros(cell.n_real, dtype=bool)
        for k in range(cell.n_real):
            _CALLS["n"] = _CALLS["above"] = 0
            out = est.fit(y0 + eps[k], pool, PLATE, sigma, w=W_GAUSS, mass=mass)
            xis[k], sbs[k] = out["xi_d"], out["s_bar"]
            hit[k] = bool(out["boundary_hit"])
            trial_n[k], trial_above[k] = _CALLS["n"], _CALLS["above"]
    finally:
        fwd.eta_bar_linear_mass = orig

    err_xi_mm = (xis - cell.xi_d) * PLATE.extent * 1e3
    final_invalid = sbs > limit
    sat = hit                     # 정본 `estimator.fit`의 경계 판정(허용오차 1e-6)
    return {"xi_d": cell.xi_d, "s_bar": cell.s_bar, "sigma_rel": cell.sigma_rel,
            "n_real": cell.n_real, "seed": cell.seed,
            "abs_err_xi_mm_median": float(np.median(np.abs(err_xi_mm))),
            "bias_xi_mm": float(np.mean(err_xi_mm)),
            "std_xi_mm": float(np.std(xis, ddof=1) * PLATE.extent * 1e3),
            "crlb_xi_mm": met["crlb_xi_mm"],
            "boundary_hit_prob": float(hit.mean()),
            "s_hat_max": float(sbs.max()), "s_hat_p999": float(np.quantile(sbs, 0.999)),
            "s_phys_max": limit,
            "n_final_invalid": int(final_invalid.sum()),
            "n_saturated": int((sat & ~final_invalid).sum()),
            "n_success": int((~sat & ~final_invalid).sum()),
            "n_fits_trial_above": int((trial_above > 0).sum()),
            "trial_evals": int(trial_n.sum()), "trial_evals_above": int(trial_above.sum())}


def _cells(n_real: int = 5000, n_loc: int = 20) -> list[mc.Cell]:
    cells, k = [], 0
    for xi in np.linspace(0.05, 0.95, n_loc):
        for s in S_LIST:
            for c in SIGMA_RELS:
                cells.append(mc.Cell(float(xi), float(s), float(c), int(n_real), SEED0 + k))
                k += 1
    return cells


def compare_with_b1(df: pd.DataFrame, b1: pd.DataFrame) -> pd.DataFrame:
    """B1 요약(`b1_mc_summary_mass.csv`)과 같은 셀끼리 대조 — 재현이면 차이 0."""
    key = ["xi_d", "s_bar", "sigma_rel"]
    m = df.merge(b1, on=key, suffixes=("", "_b1"))
    for col in ("abs_err_xi_mm_median", "bias_xi_mm", "std_xi_mm", "boundary_hit_prob"):
        m[f"d_{col}"] = (m[col] - m[f"{col}_b1"]).abs()
    return m


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--n-real", type=int, default=5000)
    args = ap.parse_args(argv)
    cells = _cells(args.n_real)
    t0 = time.time()
    with Pool(args.workers) as p:
        rows = p.map(audit_cell, cells)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv("n0_admissibility_b1"), index=False)
    from . import DATA
    b1 = pd.read_csv(DATA / "b1_mc_summary_mass.csv")
    cmp_ = compare_with_b1(df, b1)
    cmp_.to_csv(out_csv("n0_admissibility_b1_vs_b1"), index=False)
    print(f"[n0] {len(df)} cells, {df.n_real.sum()} fits, {time.time() - t0:.0f}s")
    print(f"[n0] final S̄̂ > {s_phys_max():.4f}: {df.n_final_invalid.sum()} fits; "
          f"max S̄̂ = {df.s_hat_max.max():.4f}")
    print(f"[n0] fits with a trial point above the limit: {df.n_fits_trial_above.sum()} "
          f"({df.trial_evals_above.sum()} of {df.trial_evals.sum()} evaluations)")
    print(f"[n0] states: success {df.n_success.sum()}, saturated {df.n_saturated.sum()}, "
          f"invalid {df.n_final_invalid.sum()}")
    print("[n0] max |Δ| vs B1:", {c: float(cmp_[c].max()) for c in cmp_.columns
                                  if c.startswith("d_")})


if __name__ == "__main__":
    main()


def invalid_effect(cell: mc.Cell, n_grid: int = 1001, mass: str = "exact") -> dict:
    """무효 적합(S̄̂ > 상한)을 넣을 때와 뺄 때의 셀 통계 차 — 무효가 있는 셀에만 쓴다."""
    limit = s_phys_max()
    pool = [ker.mode_kernel(PLATE, m=m, n=n, n_grid=n_grid) for m, n in MODES]
    sigma = noi.sigma_y_for_modes(pool, cell.sigma_rel, rho=0.0)
    y0 = fwd.eta_bar_linear_mass(pool, cell.xi_d, cell.s_bar, W_GAUSS, PLATE,
                                 coupling=fwd.resolve_coupling(mass))
    rng = np.random.default_rng(cell.seed)
    eps = noi.sample(sigma, rng, size=cell.n_real)
    met = idf.metrics(pool, PLATE, (cell.xi_d, cell.s_bar), W_GAUSS, sigma, mass=mass)
    crlb_xi = met["crlb_xi_mm"] / (PLATE.extent * 1e3)
    xis = np.empty(cell.n_real)
    sbs = np.empty(cell.n_real)
    for k in range(cell.n_real):
        out = est.fit(y0 + eps[k], pool, PLATE, sigma, w=W_GAUSS, mass=mass)
        xis[k], sbs[k] = out["xi_d"], out["s_bar"]
    keep = sbs <= limit

    def stats(m):
        e = (xis[m] - cell.xi_d) * PLATE.extent * 1e3
        return {"abs_err_xi_mm_median": float(np.median(np.abs(e))),
                "bias_xi_mm": float(np.mean(e)),
                "std_xi_mm": float(np.std(xis[m], ddof=1) * PLATE.extent * 1e3),
                "coverage95_xi": float(np.mean(np.abs(xis[m] - cell.xi_d) <= 1.96 * crlb_xi)),
                "bias_s_pp": float(np.mean((sbs[m] - cell.s_bar) * 100.0))}

    a, b = stats(np.ones_like(keep)), stats(keep)
    row = {"xi_d": cell.xi_d, "s_bar": cell.s_bar, "sigma_rel": cell.sigma_rel,
           "n_invalid": int((~keep).sum())}
    for k in a:
        row[f"{k}_all"], row[f"{k}_valid"] = a[k], b[k]
        row[f"d_{k}"] = b[k] - a[k]
    invalid = np.nonzero(~keep)[0]
    row["invalid_xi_hat"] = ";".join(f"{xis[i]:.4f}" for i in invalid)
    row["invalid_s_hat"] = ";".join(f"{sbs[i]:.4f}" for i in invalid)
    return row


def run_invalid_effect(workers: int = 8) -> pd.DataFrame:
    from . import DATA
    df = pd.read_csv(DATA / "rev1_n0_admissibility_b1.csv")
    bad = df[df.n_final_invalid > 0]
    cells = [mc.Cell(float(r.xi_d), float(r.s_bar), float(r.sigma_rel), int(r.n_real), int(r.seed))
             for r in bad.itertuples()]
    with Pool(min(workers, len(cells))) as p:
        rows = p.map(invalid_effect, cells)
    out = pd.DataFrame(rows)
    out.to_csv(out_csv("n0_admissibility_invalid_effect"), index=False)
    return out
