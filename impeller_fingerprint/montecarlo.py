"""B1 — 생산 몬테카를로 (정본 §3.5·§4.2 대체).

파일럿(12실현 × 5위치, i.i.d. 3 % of shift)을 대체한다. 셀 = 위치 × 심각도 × 노이즈수준,
각 셀에서 N회 실현을 적합해 다음을 보고한다(정본이 요구한 항목 그대로).

  median / IQR / 90·95 % 분위수 (위치오차 mm, 심각도오차 %p)
  경계접촉확률(ξ̂이 0 또는 1에 붙는 비율)
  커버리지(CRLB 기반 95 % 구간이 진실을 포함하는 비율)
  CRLB 대비 경험표준편차 비
  (ξ, S̄) 결합오차 공분산 → 오차타원 축·각도

축소가 있으면 **결과에 명시**한다(`n_real_requested` vs `n_real`): 조용한 축소 금지.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from multiprocessing import Pool

import numpy as np

from . import estimator as est
from . import forward as fwd
from . import identifiability as idf
from . import kernels as ker
from . import noise as noi


@dataclass(frozen=True)
class Cell:
    xi_d: float
    s_bar: float
    sigma_rel: float
    n_real: int
    seed: int


def _ellipse(cov: np.ndarray) -> dict:
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    return {"ellipse_major": float(math.sqrt(max(vals[0], 0.0))),
            "ellipse_minor": float(math.sqrt(max(vals[1], 0.0))),
            "ellipse_angle_deg": float(math.degrees(math.atan2(vecs[1, 0], vecs[0, 0])))}


def kappa_eval(spec: dict, xi_d: float, s_bar: float,
               key: str = "kappa") -> np.ndarray:
    """모델형식 불일치의 앵커 격자 값 — (ξ, S̄) 이중선형 보간(밖은 상수 클램프).

    `spec[key]`는 1D 형식 `(n_modes, n_xi)`(S̄ 무의존) 또는 2D 형식
    `(n_modes, n_xi, n_s)`를 받는다. 외삽하지 않고 **클램프**하는 이유는, 앵커가 없는
    영역에서 매끄러움을 가정하면 그것이 곧 지어낸 숫자가 되기 때문이다.
    """
    xs = np.asarray(spec["xi"], dtype=float)
    K = np.asarray(spec[key], dtype=float)
    if K.ndim == 2:
        return np.array([np.interp(xi_d, xs, K[i]) for i in range(K.shape[0])])
    ss = np.asarray(spec["s_bar"], dtype=float)
    out = np.empty(K.shape[0])
    for i in range(K.shape[0]):
        row = np.array([np.interp(s_bar, ss, K[i, j]) for j in range(len(xs))])
        out[i] = float(np.interp(xi_d, xs, row))
    return out


def surrogate_truth(plate, modes, xi_d: float, s_bar: float, spec: dict) -> np.ndarray:
    """**3D 레일 대리모델** 관측량 (설계서 §3.5의 surrogate MC, B7).

        y_m = η̄^{K,exact}_m(밴드 ξ_d ± half, 목표 S̄) + δ_m(ξ_d, S̄)    ← 기본(가법)
        y_m = κ_m(ξ_d, S̄) · η̄^{K,exact}_m(...)                        ← 곱셈(대안)

    η̄^{K,exact}는 비섭동 Kirchhoff 재해(유한 심각도의 비선형성을 정확히 포함)이고
    δ_m(또는 κ_m)는 3D 앵커에서 측정한 **모델형식 불일치**(B6)다.

    **왜 가법이 기본인가**(설계서 F39): 질량항 때문에 η̄는 모드별 부호전환 반경에서 0을
    지난다. 그 근처에서 곱셈 인자 κ = η̄^3D/η̄^{K,exact}는 발산하고, 두 모델의 부호전환
    반경이 조금만 달라도 **κ가 음수**가 된다(실측: ξ_c=0.7·m=2에서 κ = −1.00, −1.41).
    가법 δ는 그 지점에서도 유계이므로 보간·전이가 가능하다.

    앵커와 **같은 손상족**(같은 반폭의 축대칭 밴드)에만 쓰므로 형상 외삽은 없다 —
    보간은 (ξ, S̄) 두 방향뿐이고 그 오차를 B7이 LOO로 검증해 함께 보고한다.
    """
    r1 = plate.a + (xi_d - spec["half_xi"]) * plate.extent
    r2 = plate.a + (xi_d + spec["half_xi"]) * plate.extent
    r1, r2 = max(r1, plate.a), min(r2, plate.b)
    p = fwd.band_depth_for_s_bar(s_bar, r1, r2, plate.extent)
    eta = fwd.eta_bar_exact_band(plate, modes, r1, r2, p, coupling="exact")
    if "delta" in spec:
        return eta + kappa_eval(spec, xi_d, s_bar, key="delta")
    return eta * kappa_eval(spec, xi_d, s_bar, key="kappa")


def run_cell(cell: Cell, plate, modes, w: float, n_grid: int = 1001,
             rho: float = 0.0, mass=None, surrogate=None, n_starts: int = 1,
             free_w: bool = False) -> dict:
    """한 셀 실행 — 워커에서 커널을 재생성한다(피클 비용 회피).

    `surrogate`가 주어지면 진실 관측량을 **3D 레일 대리모델**로 만든다(B7). 역식별 맵은
    그대로 생산 섭동맵이므로 회복오차에 모델형식·형상불일치·선형화가 함께 들어간다.
    `surrogate=None`이면 B1과 **비트단위 동일**한 자기일관 진실(inverse-crime 기준선)이다.
    """
    pool = [ker.mode_kernel(plate, m=m, n=n, n_grid=n_grid) for m, n in modes]
    sigma = noi.sigma_y_for_modes(pool, cell.sigma_rel, rho=rho)
    if surrogate is not None:
        y0 = surrogate_truth(plate, modes, cell.xi_d, cell.s_bar, surrogate)
    else:
        y0 = (fwd.eta_bar_linear_mass(pool, cell.xi_d, cell.s_bar, w, plate,
                                      coupling=fwd.resolve_coupling(mass))
              if mass is not None else fwd.eta_bar_linear(pool, cell.xi_d, cell.s_bar, w, plate))
    rng = np.random.default_rng(cell.seed)
    eps = noi.sample(sigma, rng, size=cell.n_real)

    met = idf.metrics(pool, plate, (cell.xi_d, cell.s_bar), w, sigma,
                      mass=(mass if mass is not None else False))
    crlb_xi = met["crlb_xi_mm"] / (plate.extent * 1e3)      # ξ 단위
    crlb_s = met["crlb_s_pp"] / 100.0                       # S̄ 단위

    xis = np.empty(cell.n_real)
    sbs = np.empty(cell.n_real)
    hits = 0
    for k in range(cell.n_real):
        out = est.fit(y0 + eps[k], pool, plate, sigma, w=w,
                      mass=(mass if mass is not None else False),
                      n_starts=n_starts, free_w=free_w)
        xis[k], sbs[k] = out["xi_d"], out["s_bar"]
        hits += int(out["boundary_hit"])

    err_xi_mm = (xis - cell.xi_d) * plate.extent * 1e3
    err_s_pp = (sbs - cell.s_bar) * 100.0
    cov = np.cov(np.stack([xis - cell.xi_d, sbs - cell.s_bar]), ddof=1)

    cover_xi = float(np.mean(np.abs(xis - cell.xi_d) <= 1.96 * crlb_xi))
    cover_s = float(np.mean(np.abs(sbs - cell.s_bar) <= 1.96 * crlb_s))

    def q(a, p):
        return float(np.quantile(a, p))

    return {**asdict(cell),
            "abs_err_xi_mm_median": float(np.median(np.abs(err_xi_mm))),
            "abs_err_xi_mm_iqr": q(np.abs(err_xi_mm), 0.75) - q(np.abs(err_xi_mm), 0.25),
            "abs_err_xi_mm_p90": q(np.abs(err_xi_mm), 0.90),
            "abs_err_xi_mm_p95": q(np.abs(err_xi_mm), 0.95),
            "abs_err_s_pp_median": float(np.median(np.abs(err_s_pp))),
            "abs_err_s_pp_p95": q(np.abs(err_s_pp), 0.95),
            "bias_xi_mm": float(np.mean(err_xi_mm)),
            "bias_s_pp": float(np.mean(err_s_pp)),
            "std_xi_mm": float(np.std(xis, ddof=1) * plate.extent * 1e3),
            "std_s_pp": float(np.std(sbs, ddof=1) * 100.0),
            "crlb_xi_mm": met["crlb_xi_mm"],
            "crlb_s_pp": met["crlb_s_pp"],
            "ratio_std_over_crlb_xi": float(np.std(xis, ddof=1) / crlb_xi),
            "ratio_std_over_crlb_s": float(np.std(sbs, ddof=1) / crlb_s),
            "boundary_hit_prob": hits / cell.n_real,
            "coverage95_xi": cover_xi,
            "coverage95_s": cover_s,
            "cond2": met["cond2"], "mass_coupling": str(mass),
            "n_starts": n_starts, "free_w": bool(free_w),
            "truth_model": ("3d_rail_surrogate" if surrogate is not None
                            else "self_consistent_linear"),
            "corr_xi_s": float(cov[0, 1] / math.sqrt(cov[0, 0] * cov[1, 1])),
            **_ellipse(cov)}


def _worker(payload):
    cell, plate, modes, w, n_grid, rho, mass, surrogate, n_starts, free_w = payload
    return run_cell(cell, plate, modes, w, n_grid=n_grid, rho=rho, mass=mass,
                    surrogate=surrogate, n_starts=n_starts, free_w=free_w)


def run_production(plate, modes, xi_list, s_list, sigma_rels, *, w: float,
                   n_real: int = 5000, n_workers: int = 18, n_grid: int = 1001,
                   rho: float = 0.0, seed0: int = 20260801,
                   n_real_requested: int | None = None, mass=None,
                   surrogate=None, n_starts: int = 1,
                   free_w: bool = False) -> list[dict]:
    """전체 셀 실행(병렬). 반환 행에 요청 N과 실제 N을 함께 남긴다."""
    cells = []
    k = 0
    for xi in xi_list:
        for s in s_list:
            for c in sigma_rels:
                cells.append(Cell(float(xi), float(s), float(c), int(n_real),
                                  seed0 + k))
                k += 1
    payloads = [(c, plate, list(modes), w, n_grid, rho, mass, surrogate, n_starts,
                 free_w) for c in cells]
    if n_workers > 1:
        with Pool(n_workers) as p:
            rows = p.map(_worker, payloads)
    else:
        rows = [_worker(pl) for pl in payloads]
    req = n_real_requested if n_real_requested is not None else n_real
    for r in rows:
        r["n_real_requested"] = req
        r["reduced"] = bool(req != n_real)
    return rows
