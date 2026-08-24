"""가중 비선형최소제곱 추정기 — 정본 §3.3의 역맵.

    θ̂ = argmin ‖Σ_y^{−1/2}(y − η̄(θ))‖²,   θ = (ξ_d, S̄_D),  ξ_d ∈ [0,1], S̄_D ≥ 0

- 순방향은 기본 선형섭동. `exact=True`면 비섭동 정확재해로 교체(정본 §3.4: e_pert가 선형영역을
  벗어난 심각도에서 같은 추정기 안에서 순방향만 바꾼다).
- 가우시안 폭 w는 기본 고정(nuisance). `free_w=True`면 느슨한 경계로 함께 추정한다.
- 다중시작은 ξ_d를 전 구간에 흩뿌린다. **표현 규칙**(정본 §3.3 language rule): 결과 dict의
  `n_starts_tried`·`boundary_hit`을 그대로 보고하고, "유일해임이 증명됨" 같은 문장은 쓰지 않는다.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.optimize import least_squares

from . import forward as fwd
from . import noise as noi

_S_MAX = 1.0          # S̄_D 상한(=100 % 평균 강성손실) — 물리적 상한
_W_BOUNDS = (0.0015, 0.008)


def _residual_fn(y, pool, plate, whiten, w, exact, modes, mass=False):
    y = np.asarray(y, dtype=float)

    def resid(p):
        xi, s_bar = float(p[0]), float(p[1])
        ww = float(p[2]) if len(p) > 2 else w
        if exact:
            eta = fwd.eta_bar_exact(plate, modes, xi, s_bar, ww)
        elif mass:
            eta = fwd.eta_bar_linear_mass(pool, xi, s_bar, ww, plate,
                                          coupling=fwd.resolve_coupling(mass))
        else:
            eta = fwd.eta_bar_linear(pool, xi, s_bar, ww, plate)
        return whiten @ (eta - y)

    return resid


def fit(y: np.ndarray, pool: Sequence, plate, sigma: np.ndarray, w: float,
        *, exact: bool = False, modes: Sequence[tuple[int, int]] | None = None,
        n_starts: int = 1, free_w: bool = False, seed: int = 0,
        s_bar_max: float = _S_MAX, mass=False) -> dict:
    """가중 NLS 적합. 반환 dict: xi_d, s_bar, w, chi2, resid_rms, boundary_hit, n_starts_tried."""
    if exact and modes is None:
        raise ValueError("exact=True requires modes=[(m,n),...]")
    whiten = noi.whitening(sigma)
    sigma_inv = whiten.T @ whiten
    resid = _residual_fn(y, pool, plate, whiten, w, exact, modes, mass=mass)

    lo = [0.0, 0.0]
    hi = [1.0, s_bar_max]
    if free_w:
        lo.append(_W_BOUNDS[0])
        hi.append(_W_BOUNDS[1])

    if n_starts <= 1:
        starts = [[0.5, 0.02] + ([w] if free_w else [])]
    else:
        rng = np.random.default_rng(seed)
        centers = np.linspace(0.05, 0.95, n_starts)
        starts = [[float(np.clip(c + 0.02 * rng.standard_normal(), 0.0, 1.0)), 0.02]
                  + ([w] if free_w else []) for c in centers]

    best = None
    for p0 in starts:
        sol = least_squares(resid, p0, bounds=(lo, hi))
        xi, s_bar = float(sol.x[0]), float(sol.x[1])
        ww = float(sol.x[2]) if free_w else w
        r = resid(sol.x)
        c2 = float(r @ r)
        if best is None or c2 < best["chi2"]:
            best = {"xi_d": xi, "s_bar": s_bar, "w": ww, "chi2": c2,
                    "resid_rms": float(np.sqrt(np.mean(r ** 2)))}
    tol = 1e-6
    best["boundary_hit"] = bool(best["xi_d"] < tol or best["xi_d"] > 1.0 - tol
                                or best["s_bar"] < tol)
    best["n_starts_tried"] = len(starts)
    best["exact_forward"] = bool(exact)
    best["mass_term"] = bool(mass)
    best["sigma_inv"] = sigma_inv
    return best


def chi2_of(y: np.ndarray, eta_model: np.ndarray, sigma: np.ndarray) -> float:
    """사전지정 후보모델의 χ²(정본 §5 E2의 모델비교용)."""
    whiten = noi.whitening(sigma)
    return noi.chi2(np.asarray(eta_model) - np.asarray(y), whiten.T @ whiten)
