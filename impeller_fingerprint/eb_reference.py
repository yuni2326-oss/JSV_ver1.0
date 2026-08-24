"""EB 캔틸레버 기준해 — 회전스프링 균열 전달행렬 · (1−d) 가중 Ritz · 등가구간 · 습윤비.

**왜 여기 있는가**: A11(Table 1·arm 비교·수렴·폭한계)과 A8(습윤보정)이 논문1
`impeller_pinn`(`crack_beam`·`beam_modes`·`classical_ritz`·`fluid_loading`)을 읽기전용으로
불러 쓰고 있었다. 그 결과 **코드·데이터만 배포한 트리에서 두 산출물을 재생성할 수 없었다**
(설계서 F162). 여기 담은 것은 전부 표준 정식화이고 논문1의 기여가 아니므로, 폴더 규약이
허용하는 **재구현 + 교차검증**으로 자립시킨다 — 두 구현의 일치는
`tests/test_eb_reference.py`가 rel 1e-12로 고정한다(논문1이 없으면 그 검정만 skip).

정식화
  * 균열 = 회전스프링. 무차원 균열 파라미터 κ = 5.346 (h/L) J(ā)이고 J는 문헌 다항식
    (`crack_shear.flexibility_J`, 정본 [3]). 균열단면에서 처짐·모멘트·전단은 연속이고
    기울기만 Δφ′ = (EI/k_θ)φ″만큼 튄다 ⇒ 8×8 특성행렬식의 근 s = βL.
  * 등가 저강성 구간: 폭 L_c = (Lc_over_h)·h 안에서 ∫(1/(1−d) − 1)dx/EI = c_θ가 되도록
    d₀ = 1 − 1/(1 + 5.346 (h/L_c) J). (1−d) 엔진에 그대로 넣을 수 있는 손상장이 된다.
  * (1−d) 가중 Ritz: 단항 시행함수 ψ_k = x̃^{k+1}, K = ∫(1−d)ψ″ᵢψ″ⱼ, M = ∫ψᵢψⱼ.
  * 습윤: 부가질량비 β = Γ(π/4)(ρ_f/ρ_s)(b/h), f_wet/f_dry = 1/√(1+β).

주파수 스케일은 전부 ω_b = (h/√12)√(E/ρ)/L²이며 폭 b는 상쇄된다(단위폭 2D와 같다).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import brentq

from .crack_shear import flexibility_J


def kappa(a_bar: float, h: float, L: float) -> float:
    """무차원 균열 파라미터 κ = 5.346 (h/L) J(ā) — 회전스프링 slope-jump 항의 계수."""
    return 5.346 * (h / L) * flexibility_J(a_bar)


def _charmat(s: float, xc: float, kap: float) -> float:
    """무차원 파수 s = βL, 균열위치 ξ_c, 균열파라미터 κ에서의 8×8 특성행렬식."""
    def f0(t): return np.array([math.cosh(t), math.sinh(t), math.cos(t), math.sin(t)])
    def f1(t): return np.array([math.sinh(t), math.cosh(t), -math.sin(t), math.cos(t)])
    def f2(t): return np.array([math.cosh(t), math.sinh(t), -math.cos(t), -math.sin(t)])
    def f3(t): return np.array([math.sinh(t), math.cosh(t), math.sin(t), -math.cos(t)])

    z = np.zeros(4)
    tc, tL = s * xc, s
    M = np.empty((8, 8))
    M[0] = np.concatenate([[1, 0, 1, 0], z])                       # W₁(0) = 0
    M[1] = np.concatenate([[0, 1, 0, 1], z])                       # W₁′(0) = 0
    M[2] = np.concatenate([z, f2(tL)])                             # 자유단 모멘트
    M[3] = np.concatenate([z, f3(tL)])                             # 자유단 전단
    M[4] = np.concatenate([f0(tc), -f0(tc)])                       # 처짐 연속
    M[5] = np.concatenate([f2(tc), -f2(tc)])                       # 모멘트 연속
    M[6] = np.concatenate([f3(tc), -f3(tc)])                       # 전단 연속
    M[7] = np.concatenate([-f1(tc) - kap * s * f2(tc), f1(tc)])    # 기울기 불연속
    return float(np.linalg.det(M))


def cracked_cantilever_frequencies(L: float, h: float, E: float, rho: float,
                                   a_bar: float, xc_over_L: float, n_modes: int = 3,
                                   s_max: float = 11.0, n_scan: int = 2200) -> list[float]:
    """회전스프링 균열 캔틸레버 고유주파수[Hz] n_modes개(해석 전달행렬). ā = 0이면 건전."""
    kap = kappa(a_bar, h, L)
    xc = float(xc_over_L)
    ss = np.linspace(0.3, s_max, n_scan)
    dv = np.array([_charmat(s, xc, kap) for s in ss])
    roots: list[float] = []
    for i in range(len(ss) - 1):
        if dv[i] == 0.0:
            roots.append(ss[i])
        elif dv[i] * dv[i + 1] < 0:
            roots.append(brentq(_charmat, ss[i], ss[i + 1], args=(xc, kap), xtol=1e-10))
        if len(roots) >= n_modes:
            break
    coef = (h / math.sqrt(12.0)) * math.sqrt(E / rho) / (L ** 2)
    return [(s ** 2) * coef / (2 * math.pi) for s in roots[:n_modes]]


def crack_knockdown(a_bar: float, h: float, L: float, xc_over_L: float,
                    Lc_over_h: float = 1.0):
    """균열과 같은 국소 유연도를 갖는 **등가 저강성 구간**의 손상장 d(x̃) 콜러블.

    폭 L_c = (Lc_over_h)·h, 깊이 d₀ = 1 − 1/(1 + 5.346 (h/L_c) J(ā)). 진단용으로
    `.d0`·`.width_xt` 속성을 함께 단다(정규화 x̃ = x/L).
    """
    Lc = Lc_over_h * h
    d0 = 1.0 - 1.0 / (1.0 + 5.346 * (h / Lc) * flexibility_J(a_bar))
    xc = float(xc_over_L)
    half = (Lc / L) / 2.0

    def d(xt):
        xt = np.asarray(xt, dtype=float)
        return np.where(np.abs(xt - xc) <= half, d0, 0.0)

    d.d0 = d0
    d.width_xt = Lc / L
    return d


def solve_ritz(L: float, h: float, E: float, rho: float, n_modes: int = 3,
               damage=None, n_trial: int = 7, n_grid: int = 4001) -> list[dict]:
    """(1−d) 가중 캔틸레버 Ritz — 주파수[Hz]·무차원 고유값·내부 절점수.

    `damage`는 x̃(ndarray) → d(x̃) 콜러블이거나 None(건전)이다.
    """
    x = np.linspace(0.0, 1.0, n_grid)
    w = 1.0 - (np.asarray(damage(x), dtype=float) if damage is not None else 0.0)
    ks = np.arange(1, n_trial + 1)
    psi = np.stack([x ** (k + 1) for k in ks])                     # ψ_k
    d2 = np.stack([(k + 1) * k * x ** (k - 1) for k in ks])        # ψ_k″
    K = np.array([[np.trapezoid(w * d2[i] * d2[j], x) for j in range(n_trial)]
                  for i in range(n_trial)])
    M = np.array([[np.trapezoid(psi[i] * psi[j], x) for j in range(n_trial)]
                  for i in range(n_trial)])
    Lam, C = eigh(K, M)
    omega_b = (h / math.sqrt(12.0)) * math.sqrt(E / rho) / (L ** 2)
    out = []
    for n in range(min(n_modes, n_trial)):
        lam = float(Lam[n])
        f = math.sqrt(max(lam, 0.0)) * omega_b / (2 * math.pi)
        shape = C[:, n] @ psi
        s = shape[x > 0.03]                     # 고정단 근방 이산화 아티팩트 제외
        nodes = int(np.sum(np.diff(np.sign(s)) != 0))
        out.append({"f": f, "Lambda": lam, "nodes": nodes})
    return out


def beta_beam(b_over_h: float, rho_s: float, rho_f: float = 1000.0,
              navmi: float = 0.65) -> float:
    """캔틸레버 판 부가질량비 β = Γ (π/4)(ρ_f/ρ_s)(b/h) — Γ는 NAVMI 계수."""
    return navmi * (math.pi / 4.0) * (rho_f / rho_s) * b_over_h


def wet_ratio(beta: float) -> float:
    """습윤/건조 주파수비 f_wet/f_dry = 1/√(1+β)."""
    return 1.0 / math.sqrt(1.0 + beta)
