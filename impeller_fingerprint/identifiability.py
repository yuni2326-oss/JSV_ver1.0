"""식별성 — 조건수 단독을 넘어선 추정이론 지표 (정본 §3.5·§4.3).

리뷰의 요지: 조건수는 민감도 타원의 **종횡비**만 주고 정보의 **크기**를 주지 않는다(cond₂가 같은
두 야코비안이 σ_min에서 100배 차이날 수 있다). 따라서 다음을 (ξ_d, S̄_D) **2D 맵**으로 보고한다.

    J_w = Σ_y^{−1/2} J,  σ_min(J_w), σ_max(J_w), cond₂(J_w)
    F   = Jᵀ Σ_y^{−1} J,  det F (D-최적 정보), tr F⁻¹ (A-최적 불확실도), 상관 ρ_{ξS}
    CRLB: √diag(F⁻¹) → 위치는 mm, 심각도는 %p
    프로파일우도 χ²_p(ξ) = min_{S̄} χ², 밀집격자 목적함수면 χ²(ξ, S̄)

좁은 손상 선형화에서 η̄_m ≈ −S̄_D γ_m(ξ_d)이므로 야코비안의 위치열이 S̄_D에 비례한다 →
**작은 손상은 본질적으로 위치추정이 어렵다**. 이 구조적 사실이 2D 맵을 요구하는 이유이며,
`test_smaller_severity_is_harder_to_locate`가 이를 코드에서 고정한다.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.optimize import minimize_scalar

from . import forward as fwd
from . import noise as noi

_METRIC_KEYS = ("sigma_min", "sigma_max", "cond2", "det_F", "tr_Finv",
                "corr", "crlb_xi_mm", "crlb_s_pp")


def whitened_jacobian(pool: Sequence, plate, theta: tuple[float, float], w: float,
                      sigma: np.ndarray) -> np.ndarray:
    J = fwd.jacobian_linear(pool, theta[0], theta[1], w, plate)
    return noi.whitening(sigma) @ J


def metrics(pool: Sequence, plate, theta: tuple[float, float], w: float,
            sigma: np.ndarray, mass=False) -> dict:
    """단일 (ξ_d, S̄_D)에서의 식별성 지표 묶음(선형 커널 야코비안 사용).

    mass=True면 질량항 포함 맵(설계서 F12)의 야코비안을 쓴다.
    """
    J = (fwd.jacobian_linear_mass(pool, theta[0], theta[1], w, plate,
                                  coupling=fwd.resolve_coupling(mass))
         if mass else fwd.jacobian_linear(pool, theta[0], theta[1], w, plate))
    return metrics_from_J(J, sigma, plate.extent)


def metrics_from_J(J: np.ndarray, sigma: np.ndarray, extent: float) -> dict:
    """임의의 야코비안 J(∂관측/∂(ξ_d, S̄_D))에 대한 식별성 지표.

    관측벡터가 pair mean 외의 양(분리 Δη 등)을 포함할 때도 같은 규약으로 보고하기 위한 진입점.
    """
    J = np.asarray(J, dtype=float)
    Wh = noi.whitening(sigma)
    Jw = Wh @ J
    sv = np.linalg.svd(Jw, compute_uv=False)
    F = Jw.T @ Jw                      # = Jᵀ Σ⁻¹ J
    out = {"J": J, "Jw": Jw, "F": F,
           "sigma_min": float(sv[-1]), "sigma_max": float(sv[0]),
           "cond2": float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf"),
           "det_F": float(np.linalg.det(F))}
    if out["det_F"] <= 0 or not np.isfinite(out["det_F"]):
        out.update({"tr_Finv": float("inf"), "corr": float("nan"),
                    "crlb_xi_mm": float("inf"), "crlb_s_pp": float("inf")})
        return out
    Finv = np.linalg.inv(F)
    out["tr_Finv"] = float(np.trace(Finv))
    out["corr"] = float(Finv[0, 1] / np.sqrt(Finv[0, 0] * Finv[1, 1]))
    out["crlb_xi_mm"] = float(np.sqrt(Finv[0, 0]) * extent * 1e3)
    out["crlb_s_pp"] = float(np.sqrt(Finv[1, 1]) * 100.0)
    return out


def metric_maps(pool: Sequence, plate, xi_grid: np.ndarray, s_grid: np.ndarray,
                w: float, sigma: np.ndarray, mass: bool = False) -> dict:
    """(n_xi, n_s) 2D 맵 묶음 — 정본 §4.3(i)의 맵 패널."""
    xi_grid = np.asarray(xi_grid, dtype=float)
    s_grid = np.asarray(s_grid, dtype=float)
    maps = {k: np.empty((xi_grid.size, s_grid.size)) for k in _METRIC_KEYS}
    for i, xi in enumerate(xi_grid):
        for j, s in enumerate(s_grid):
            m = metrics(pool, plate, (float(xi), float(s)), w, sigma, mass=mass)
            for k in _METRIC_KEYS:
                maps[k][i, j] = m[k]
    maps["xi_grid"] = xi_grid
    maps["s_grid"] = s_grid
    return maps


def _eta(pool, xi, s_bar, w, plate, mass=None) -> np.ndarray:
    """순방향 pair mean — `mass`가 주어지면 질량항 포함 맵(설계서 F12)을 쓴다."""
    if mass:
        return fwd.eta_bar_linear_mass(
            pool, xi, s_bar, w, plate, coupling=fwd.resolve_coupling(mass))
    return fwd.eta_bar_linear(pool, xi, s_bar, w, plate)


def _chi2_at(y, pool, plate, sigma_inv, w, xi, s_bar, mass=None) -> float:
    eta = _eta(pool, xi, s_bar, w, plate, mass=mass)
    return noi.chi2(eta - np.asarray(y, dtype=float), sigma_inv)


def profile_likelihood(y: np.ndarray, pool: Sequence, plate, sigma: np.ndarray,
                       w: float, xi_grid: np.ndarray,
                       s_bounds: tuple[float, float] = (0.0, 1.0),
                       mass=None) -> np.ndarray:
    """χ²_p(ξ) = min_{S̄} χ²(ξ, S̄).

    **강성전용(mass=None)**: 선형모델 η̄ = S̄·g(ξ)에서 χ²(ξ,S̄)는 S̄의 이차식이므로 최적
    S̄* = (gᵀΣ⁻¹y)/(gᵀΣ⁻¹g) (경계 클리핑 적용).

    **질량항(mass 지정)**: 정확결합 d_M = 1−(1−d_K)^{1/3}에서 맵이 S̄에 **비선형**이라
    위 해석해가 성립하지 않는다(설계서 M7/A4). 따라서 각 ξ에서 `minimize_scalar`
    (bounded Brent)로 S̄를 수치 최소화한다.
    """
    Wh = noi.whitening(sigma)
    sigma_inv = Wh.T @ Wh
    y = np.asarray(y, dtype=float)
    out = np.empty(len(xi_grid))
    for i, xi in enumerate(np.asarray(xi_grid, dtype=float)):
        xi = float(xi)
        if mass:
            res = minimize_scalar(
                lambda s: _chi2_at(y, pool, plate, sigma_inv, w, xi, float(s),
                                   mass=mass),
                bounds=s_bounds, method="bounded",
                options={"xatol": 1e-10, "maxiter": 500})
            s_star = float(np.clip(res.x, s_bounds[0], s_bounds[1]))
        else:
            g = fwd.eta_bar_linear(pool, xi, 1.0, w, plate)      # S̄=1 기준벡터
            denom = float(g @ sigma_inv @ g)
            s_star = float(g @ sigma_inv @ y) / denom if denom > 0 else 0.0
            s_star = float(np.clip(s_star, s_bounds[0], s_bounds[1]))
        out[i] = _chi2_at(y, pool, plate, sigma_inv, w, xi, s_star, mass=mass)
    return out


def profile_interval(xi_grid: np.ndarray, prof: np.ndarray,
                     delta_chi2: float = 3.84) -> tuple[float, float]:
    """프로파일 신뢰구간 [ξ_lo, ξ_hi] — χ²_p ≤ min + Δχ² 영역의 양 끝."""
    xi_grid = np.asarray(xi_grid, dtype=float)
    prof = np.asarray(prof, dtype=float)
    inside = np.nonzero(prof <= prof.min() + delta_chi2)[0]
    return float(xi_grid[inside[0]]), float(xi_grid[inside[-1]])


def objective_grid(y: np.ndarray, pool: Sequence, plate, sigma: np.ndarray, w: float,
                   xi_grid: np.ndarray, s_grid: np.ndarray, mass=None) -> np.ndarray:
    """밀집격자 목적함수면 χ²(ξ_d, S̄_D) — 다중시작 논증을 대체하는 정본 §4.3(iii).

    `mass`가 주어지면 질량항 포함 순방향으로 평가한다(설계서 M7/A4).
    """
    Wh = noi.whitening(sigma)
    sigma_inv = Wh.T @ Wh
    xi_grid = np.asarray(xi_grid, dtype=float)
    s_grid = np.asarray(s_grid, dtype=float)
    out = np.empty((xi_grid.size, s_grid.size))
    for i, xi in enumerate(xi_grid):
        for j, s in enumerate(s_grid):
            out[i, j] = _chi2_at(y, pool, plate, sigma_inv, w, float(xi), float(s),
                                 mass=mass)
    return out


def local_minima_count(chi2_grid: np.ndarray) -> int:
    """격자면의 국소최소 개수(8-이웃) — "경쟁 basin이 관측되었는가"의 수치판.

    정본 §3.3 language rule에 맞춰, 이 수는 *평가된 격자 안에서 관측된* 국소최소의 수일 뿐이며
    유일성의 증명이 아니다.
    """
    g = np.asarray(chi2_grid, dtype=float)
    pad = np.pad(g, 1, mode="constant", constant_values=np.inf)
    count = 0
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            block = pad[i:i + 3, j:j + 3]
            if g[i, j] <= block.min():
                count += 1
    return count
