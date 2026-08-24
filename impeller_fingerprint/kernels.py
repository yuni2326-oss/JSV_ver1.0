"""민감도 커널 γ_{m,n}(r)과 환형판 고유주파수 — 섭동 역맵(P 레일)의 공급원.

정식화(논문1 `classical_annular_plate`와 동일 — 교차검증 T6): 절점직경 m 모드
W(r,θ)=R(r)cos mθ, 곡률 κ_rr=R'', κ_θθ=R'/r−m²R/r², κ_rθ=m(R'/r−R/r²),
굽힘에너지밀도 e=(κ_rr+κ_θθ)²−2(1−ν)(κ_rr κ_θθ−κ_rθ²).
Rayleigh 몫 λ=ω²=(D/ρh)·∫(1−d) e r dr/∫R² r dr. 시행함수 ψ_k(ξ)=ξ^{k+1}(내경 클램프 자동만족),
r=a+(b−a)ξ.

**커널 정의**: γ_{m,n}(r) = e_n·r / ∫e_n·r dr  (∫γ dr = 1).
설계서 §4의 정의 동결에 따라 관측량은 **고유값 기반**
    η̄_{m,n} = δλ/λ = −∫ γ_{m,n}(r) d(r) dr,      Δf/f = ½ η̄
이고, ½은 주파수 표현에만 붙는다(정본 §3.3의 계수 불일치 M1 해소).

여기서 `damage`는 **물리 반경 r의 함수** d(r)=δD/D를 받는다(논문1의 solve_annular_plate는
정규화 ξ의 함수를 받는다 — 어댑터는 호출부에서 처리).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.linalg import eigh

Damage = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class ModeKernel:
    """절점직경 m·반경차수 n 모드의 주파수와 민감도 커널."""

    m: int
    n: int
    f: float           # [Hz]
    Lambda: float      # 무차원 Rayleigh 고유값 (λ = (D/ρh)·Lambda)
    r: np.ndarray      # [m]
    gamma: np.ndarray  # 굽힘(강성) 커널 γ^K, ∫γ dr = 1
    gamma_mass: np.ndarray | None = None   # 질량 커널 γ^M = R²r/∫R²r dr, ∫γ^M dr = 1
    R: np.ndarray | None = None      # 반경 모드형 R(r) (임의 스케일)
    dR: np.ndarray | None = None     # R'(r)
    d2R: np.ndarray | None = None    # R''(r)

    @property
    def label(self) -> str:
        return f"m{self.m}n{self.n}"


def _basis(a: float, b: float, n_trial: int, n_grid: int,
           basis: str = "monomial"):
    """(r, ψ, ψ'(r), ψ''(r)) — 내경 클램프(ψ=ψ'=0 at ξ=0)를 자동 만족하는 시행함수.

    `basis="monomial"`(**기본, 생산 규약**): ψ_k(ξ)=ξ^{k+1}, k=1…n_trial.
    `basis="legendre"`: ψ_k(ξ)=ξ²·L̃_{k−1}(ξ) (L̃ = [0,1] 위 이동 르장드르).
    단항 기저는 차수가 오르면 Gram 행렬이 힐베르트형이 되어 n_trial ≳ 13에서 Cholesky가
    실패한다 — 급격한 밴드 손상의 **정확재해 수렴검정**에는 르장드르 기저를 쓴다.
    두 기저는 같은 공간(ξ² × 다항식)을 span하므로 수렴한 고유값은 일치해야 하며,
    테스트가 그것을 검정한다.
    """
    L = b - a
    xi = np.linspace(0.0, 1.0, n_grid)
    r = a + L * xi
    if basis == "monomial":
        ks = np.arange(1, n_trial + 1)
        psi = np.stack([xi ** (k + 1) for k in ks])
        dps = np.stack([(k + 1) * xi ** k for k in ks]) / L
        d2p = np.stack([(k + 1) * k * xi ** (k - 1) for k in ks]) / L ** 2
    elif basis == "legendre":
        from numpy.polynomial import polynomial as nppoly
        from numpy.polynomial import legendre as npleg
        t = np.array([-1.0, 2.0])                       # t(ξ) = 2ξ − 1
        x2 = np.array([0.0, 0.0, 1.0])                  # ξ²
        polys = []
        for j in range(n_trial):
            e = np.zeros(j + 1)
            e[j] = 1.0
            pj = npleg.leg2poly(e)                      # L_j(t)의 t-다항 계수
            polys.append(nppoly.polymul(_compose(pj, t), x2))
        psi = np.stack([nppoly.polyval(xi, c) for c in polys])
        dps = np.stack([nppoly.polyval(xi, nppoly.polyder(c, 1)) for c in polys]) / L
        d2p = np.stack([nppoly.polyval(xi, nppoly.polyder(c, 2)) for c in polys]) / L ** 2
    else:
        raise ValueError(f"unknown basis: {basis}")
    return r, psi, dps, d2p


def _compose(outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
    """다항식 합성 outer(inner(x)) — 계수배열(낮은 차수부터)."""
    from numpy.polynomial import polynomial as nppoly
    out = np.zeros(1)
    for c in outer[::-1]:
        out = nppoly.polyadd(nppoly.polymul(out, inner), np.array([c]))
    return out


def _curvatures(m: int, r: np.ndarray, psi, dps, d2p):
    krr = d2p
    kth = dps / r - (m ** 2) * psi / r ** 2
    krt = m * (dps / r - psi / r ** 2)
    return krr, kth, krt


def _matrices(m: int, nu: float, r: np.ndarray, psi, dps, d2p,
              weight: np.ndarray | float = 1.0,
              weight_m: np.ndarray | float = 1.0):
    """(K, M) — K는 (1−d) 가중 굽힘강성, M은 질량(둘 다 D, ρh 인자 제외)."""
    krr, kth, krt = _curvatures(m, r, psi, dps, d2p)
    A = krr + kth
    n_trial = psi.shape[0]
    K = np.empty((n_trial, n_trial))
    M = np.empty((n_trial, n_trial))
    for i in range(n_trial):
        for j in range(i, n_trial):
            q = (A[i] * A[j]
                 - (1 - nu) * (krr[i] * kth[j] + kth[i] * krr[j])
                 + 2 * (1 - nu) * krt[i] * krt[j])
            K[i, j] = K[j, i] = np.trapezoid(weight * q * r, r)
            M[i, j] = M[j, i] = np.trapezoid(weight_m * psi[i] * psi[j] * r, r)
    return K, M


def mode_kernel_props(a: float, b: float, D: float, rhoh: float, nu: float,
                      m: int, n: int = 0, n_trial: int = 8,
                      n_grid: int = 2001) -> ModeKernel:
    """건전 환형판의 (m, n) 모드 주파수와 민감도 커널."""
    r, psi, dps, d2p = _basis(a, b, n_trial, n_grid)
    K, M = _matrices(m, nu, r, psi, dps, d2p)
    Lam, C = eigh(K, M)
    if n >= n_trial:
        raise ValueError(f"n={n} >= n_trial={n_trial}")
    lam = float(Lam[n])
    c = C[:, n]
    R, Rr, Rrr = c @ psi, c @ dps, c @ d2p
    kr = Rrr
    kt = Rr / r - (m ** 2) * R / r ** 2
    kx = m * (Rr / r - R / r ** 2)
    e = (kr + kt) ** 2 - 2 * (1 - nu) * (kr * kt - kx ** 2)
    g = e * r
    g = g / np.trapezoid(g, r)
    gm = R ** 2 * r
    gm = gm / np.trapezoid(gm, r)
    f = math.sqrt(max((D / rhoh) * lam, 0.0)) / (2 * math.pi)
    return ModeKernel(m=m, n=n, f=f, Lambda=lam, r=r, gamma=g, gamma_mass=gm,
                      R=R, dR=Rr, d2R=Rrr)


def mode_kernel(plate, m: int, n: int = 0, n_trial: int = 8,
                n_grid: int = 2001) -> ModeKernel:
    """`geometry.Plate`/`Sandwich`용 편의 래퍼."""
    return mode_kernel_props(plate.a, plate.b, plate.D, plate.rhoh, plate.nu,
                             m=m, n=n, n_trial=n_trial, n_grid=n_grid)


def solve_frequencies_props(a: float, b: float, D: float, rhoh: float, nu: float,
                            m: int, n_modes: int = 1, damage: Damage | None = None,
                            n_trial: int = 8, n_grid: int = 4001) -> np.ndarray:
    """손상장 d(r)(=δD/D, **물리 반경 함수**)을 반영한 비섭동 정확재해 주파수[Hz]."""
    r, psi, dps, d2p = _basis(a, b, n_trial, n_grid)
    w = 1.0 - (np.asarray(damage(r), dtype=float) if damage is not None else 0.0)
    K, M = _matrices(m, nu, r, psi, dps, d2p, weight=w)
    Lam = eigh(K, M, eigvals_only=True)
    out = [math.sqrt(max((D / rhoh) * float(lam), 0.0)) / (2 * math.pi)
           for lam in Lam[:n_modes]]
    return np.array(out)


def solve_frequencies(plate, m: int, n_modes: int = 1, damage: Damage | None = None,
                      n_trial: int = 8, n_grid: int = 4001) -> np.ndarray:
    return solve_frequencies_props(plate.a, plate.b, plate.D, plate.rhoh, plate.nu,
                                   m=m, n_modes=n_modes, damage=damage,
                                   n_trial=n_trial, n_grid=n_grid)


def solve_eigenvalues_props(a: float, b: float, nu: float, m: int, n_modes: int = 1,
                            damage: Damage | None = None, n_trial: int = 8,
                            n_grid: int = 4001,
                            damage_mass: Damage | None = None,
                            precondition: bool = False,
                            basis: str = "monomial") -> np.ndarray:
    """무차원 Rayleigh 고유값 Λ — η̄ 계산에 쓰면 D·ρh가 소거된다.

    `damage`는 강성손실 d_K(r), `damage_mass`는 **질량손실 d_M(r)**. 재료제거는 둘을 동시에
    바꾸므로(설계서 M7) 질량손상을 빼면 정확재해가 실제 손상을 표현하지 못한다.

    `precondition`: 단항 시행함수 ψ_k = ξ^{k+1}는 차수가 오르면 거의 선형종속이 되어
    `eigh`의 Cholesky가 n_trial ≳ 11에서 실패한다(실측: n_trial=12에서 LinAlgError).
    True면 ψ_k를 M의 대각으로 정규화해(고유값 불변) n_trial ~20까지 풀 수 있다.
    **기본값은 False** — 기존 산출물의 비트단위 재현성을 깨지 않기 위해서다(설계서 §11.8 교훈).
    """
    r, psi, dps, d2p = _basis(a, b, n_trial, n_grid, basis=basis)
    w = 1.0 - (np.asarray(damage(r), dtype=float) if damage is not None else 0.0)
    wm = 1.0 - (np.asarray(damage_mass(r), dtype=float) if damage_mass is not None else 0.0)
    K, M = _matrices(m, nu, r, psi, dps, d2p, weight=w, weight_m=wm)
    if precondition:
        s = 1.0 / np.sqrt(np.diag(M))
        K = K * s[:, None] * s[None, :]
        M = M * s[:, None] * s[None, :]
    return eigh(K, M, eigvals_only=True)[:n_modes]


def mode_pool(plate, ms=(0, 1, 2, 3, 4), ns=(0, 1), n_trial: int = 8,
              n_grid: int = 2001, f_max: float | None = None) -> list[ModeKernel]:
    """후보 모드 풀(설계서 §4-6). f_max를 주면 측정가능 대역으로 걸러낸다."""
    pool = [mode_kernel(plate, m=m, n=n, n_trial=n_trial, n_grid=n_grid)
            for m in ms for n in ns]
    if f_max is not None:
        pool = [k for k in pool if k.f <= f_max]
    return sorted(pool, key=lambda k: k.f)
