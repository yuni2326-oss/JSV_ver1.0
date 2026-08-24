"""축퇴쌍 섭동이론 H^(m) — 정본 §3.2, 논문의 spine.

축대칭 판의 절점직경 m(>0) 고유값 λ_m은 이중축퇴이고 고유공간은 span{φ_c, φ_s},
φ_c = R_m(r)cos mθ, φ_s = R_m(r)sin mθ (질량정규화). 방위 θ₀에 국소화된 손상은

    H^(m)_{ij} = ∫∫ [ δD(r,θ)·q_{ij}(r,θ) − λ_m·δ(ρh)(r,θ)·φ_i φ_j ] r dr dθ

를 통해 작용한다(가공포켓은 강성과 질량을 **동시에** 제거한다). 관측량 3종:

    pair mean   η̄_m   = tr H/(2λ_m)        — θ₀에 무관(반경 역문제의 적격성 근거)
    splitting   Δη_m  = (λ₊−λ₋)/λ_m = 2|B|/λ_m — 방위 2m차 푸리에 성분에 비례
    orientation ψ_m                          — 분리 고유벡터가 손상 방위에 정렬

**해석 구조**(이 모듈이 수치적으로 검증한다): 곡률의 각의존이 분리되므로
q_cc = U cos²+V sin², q_ss = U sin²+V cos², q_cs = (U−V)cos·sin 이고
(U = A²−2(1−ν)BC, V = 2(1−ν)T², A=R''+R'/r−m²R/r², B=R'', C=R'/r−m²R/r², T=m(R'/r−R/r²)),
질량항은 (U_M,V_M)=(R²,0)이다. 포켓 [r₁,r₂]×[θ₀±Δθ/2]에 대한 각적분은
∫cos² = I₀+I₁cos2mθ₀, ∫sin² = I₀−I₁cos2mθ₀, ∫cos·sin = I₁ sin2mθ₀,
I₀ = Δθ/2, **I₁ = sin(mΔθ)/(2m)** 이므로

    H = [[Ā + B̄ cos2mθ₀, B̄ sin2mθ₀], [B̄ sin2mθ₀, Ā − B̄ cos2mθ₀]]
    Ā = −β_D D (R_U+R_V) I₀ + λ β_M ρh R_M I₀,   B̄ = −β_D D (R_U−R_V) I₁ + λ β_M ρh R_M I₁

→ 고유값 Ā±|B̄|, **tr H = 2Ā (θ₀ 무관)**, 고유벡터 (cos mθ₀, sin mθ₀) ⇒ ψ_m = θ₀ (mod π/m),
그리고 **Δθ = π/m에서 I₁ = 0 → 분리 null**. 두 계산경로(닫힌형 `pair_matrix`,
2D 수치적분 `pair_matrix_quadrature`)를 모두 제공해 서로 검정한다(테스트).

방위까지 넣은 완전 결합역산(z = [η̄; Δη; ψ])은 정본이 스스로 sequel로 선언했으므로 여기서는
**관측량 산출**까지만 한다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import kernels as ker


@dataclass(frozen=True)
class Pocket:
    """가공포켓 형상 — 재료제거(강성·질량 동시).

    r1, r2: 반경 구간 [m] / theta0: 중심 방위 [rad] / dtheta: 각폭 [rad]
    depth_frac: 제거깊이/판두께 ∈ (0,1)
    """

    r1: float
    r2: float
    theta0: float
    dtheta: float
    depth_frac: float

    @property
    def beta_D(self) -> float:
        """굽힘강성 손실률 1−(1−p)³ (t³ 의존)."""
        return 1.0 - (1.0 - self.depth_frac) ** 3

    @property
    def beta_M(self) -> float:
        """면적질량 손실률 = 깊이비."""
        return float(self.depth_frac)

    @property
    def radial_width(self) -> float:
        return self.r2 - self.r1


def _normalized_shape(plate, m: int, n: int = 0, n_grid: int = 4001):
    """질량정규화된 반경 모드형과 λ_m 반환.

    정규화: ∫∫ ρh φ² r dr dθ = 1. m>0에서 ∫cos²(mθ)dθ = π, m=0에서는 2π.
    """
    k = ker.mode_kernel(plate, m=m, n=n, n_grid=n_grid)
    ang = math.pi if m > 0 else 2.0 * math.pi
    norm = plate.rhoh * ang * float(np.trapezoid(k.R ** 2 * k.r, k.r))
    scale = 1.0 / math.sqrt(norm)
    lam = (plate.D / plate.rhoh) * k.Lambda
    return k.r, k.R * scale, k.dR * scale, k.d2R * scale, lam


def _UV(plate, m: int, r, R, dR, d2R):
    """(U, V, R²) 반경 함수 — q_cc/q_ss/q_cs의 반경 인자."""
    nu = plate.nu
    A = d2R + dR / r - (m ** 2) * R / r ** 2
    B = d2R
    C = dR / r - (m ** 2) * R / r ** 2
    T = m * (dR / r - R / r ** 2)
    U = A ** 2 - 2.0 * (1.0 - nu) * B * C
    V = 2.0 * (1.0 - nu) * T ** 2
    return U, V, R ** 2


def _pocket_radial_integrals(r, U, V, RM, pocket: Pocket, n_r: int = 4001):
    """포켓 반경구간에서의 ∫U r dr, ∫V r dr, ∫R² r dr (선형보간 후 적분)."""
    rr = np.linspace(pocket.r1, pocket.r2, n_r)
    Ui = np.interp(rr, r, U)
    Vi = np.interp(rr, r, V)
    Mi = np.interp(rr, r, RM)
    return (float(np.trapezoid(Ui * rr, rr)),
            float(np.trapezoid(Vi * rr, rr)),
            float(np.trapezoid(Mi * rr, rr)))


def pair_coefficients(plate, m: int, pocket: Pocket, mass_term: bool = True,
                      n_grid: int = 4001, n_r: int = 4001) -> dict:
    """닫힌형 계수 (Ā, B̄, λ_m)과 그 구성요소.

    **B̄의 부호는 보편적이지 않다** — 굽힘항 R_U, 비틀림항 R_V, 질량항 R_M의 균형으로 정해진다.
    부호가 양이면 손상 위에 *절점선*을 갖는 쪽이 더 떨어지고, 음이면 *antinode*를 갖는 쪽이
    더 떨어진다. 따라서 측정된 배향에서 손상 방위를 복원할 때 모델의 sign(B̄) 예측이 필요하다
    (정본 §3.2의 "orientation locking"에 붙어야 하는 단서).
    """
    if m < 1:
        raise ValueError("H^(m) framework applies to m >= 1 (m=0 is nondegenerate)")
    r, R, dR, d2R, lam = _normalized_shape(plate, m, n_grid=n_grid)
    U, V, RM = _UV(plate, m, r, R, dR, d2R)
    Ru, Rv, Rm = _pocket_radial_integrals(r, U, V, RM, pocket, n_r=n_r)

    I0 = 0.5 * pocket.dtheta
    I1 = math.sin(m * pocket.dtheta) / (2.0 * m)
    kD = -pocket.beta_D * plate.D
    kM = (lam * pocket.beta_M * plate.rhoh) if mass_term else 0.0

    A_bar = kD * (Ru + Rv) * I0 + kM * Rm * I0
    B_bar = kD * (Ru - Rv) * I1 + kM * Rm * I1
    return {"A_bar": A_bar, "B_bar": B_bar, "lambda_m": lam,
            "R_U": Ru, "R_V": Rv, "R_M": Rm, "I0": I0, "I1": I1,
            "stiff_B": kD * (Ru - Rv) * I1, "mass_B": kM * Rm * I1}


def pair_matrix(plate, m: int, pocket: Pocket, mass_term: bool = True,
                n_grid: int = 4001, n_r: int = 4001) -> np.ndarray:
    """2×2 H^(m) — 해석 각적분 경로(닫힌형). m ≥ 1."""
    co = pair_coefficients(plate, m, pocket, mass_term=mass_term,
                           n_grid=n_grid, n_r=n_r)
    A_bar, B_bar = co["A_bar"], co["B_bar"]
    c = math.cos(2.0 * m * pocket.theta0)
    s = math.sin(2.0 * m * pocket.theta0)
    return np.array([[A_bar + B_bar * c, B_bar * s],
                     [B_bar * s, A_bar - B_bar * c]])


def pair_matrix_quadrature(plate, m: int, pocket: Pocket, mass_term: bool = True,
                           n_grid: int = 4001, n_r: int = 2001,
                           n_theta: int = 2001) -> np.ndarray:
    """2×2 H^(m) — (r,θ) 2D 수치적분 경로(닫힌형의 독립 검증용)."""
    if m < 1:
        raise ValueError("H^(m) framework applies to m >= 1")
    r, R, dR, d2R, lam = _normalized_shape(plate, m, n_grid=n_grid)
    U, V, RM = _UV(plate, m, r, R, dR, d2R)

    rr = np.linspace(pocket.r1, pocket.r2, n_r)
    Ui = np.interp(rr, r, U)[:, None]
    Vi = np.interp(rr, r, V)[:, None]
    Mi = np.interp(rr, r, RM)[:, None]
    th = np.linspace(pocket.theta0 - 0.5 * pocket.dtheta,
                     pocket.theta0 + 0.5 * pocket.dtheta, n_theta)[None, :]
    co, si = np.cos(m * th), np.sin(m * th)

    kD = -pocket.beta_D * plate.D
    kM = (lam * pocket.beta_M * plate.rhoh) if mass_term else 0.0
    integ = {
        (0, 0): kD * (Ui * co ** 2 + Vi * si ** 2) + kM * Mi * co ** 2,
        (1, 1): kD * (Ui * si ** 2 + Vi * co ** 2) + kM * Mi * si ** 2,
        (0, 1): kD * (Ui - Vi) * co * si + kM * Mi * co * si,
    }
    H = np.empty((2, 2))
    for (i, j), f in integ.items():
        val = float(np.trapezoid(np.trapezoid(f * rr[:, None], rr, axis=0),
                                 th.ravel()))
        H[i, j] = H[j, i] = val
    return H


def observables(plate, m: int, pocket: Pocket, mass_term: bool = True,
                n_grid: int = 4001, n_r: int = 4001) -> dict:
    """정본 §3.2의 관측량 3종 + 심각도 환산.

    반환 주요 키
      eta_bar     pair mean η̄_m = tr H/(2λ)  — θ₀ 무관
      delta_eta   분리 Δη_m = 2|B̄|/λ ≥ 0
      psi_lower   **관측 가능한 배향** = 낮은 쪽 짝의 antinode 방위 [rad] (mod π/m)
      theta0_hat  모델의 sign(B̄)를 써서 psi_lower에서 복원한 손상 방위 (mod π/m)
      B_signed    부호 있는 B̄ (양 = 절점선-위-손상 쪽이 더 떨어짐)
      eta_plus/eta_minus, lambda_m, H, severity_s_bar_radial, severity_s_bar
    """
    co = pair_coefficients(plate, m, pocket, mass_term=mass_term,
                           n_grid=n_grid, n_r=n_r)
    H = pair_matrix(plate, m, pocket, mass_term=mass_term, n_grid=n_grid, n_r=n_r)
    lam, A_bar, B_bar = co["lambda_m"], co["A_bar"], co["B_bar"]
    vals, vecs = np.linalg.eigh(H)                     # 오름차순
    eta_lo, eta_hi = float(vals[0]) / lam, float(vals[1]) / lam
    period = math.pi / m

    v = vecs[:, 0]                                     # 낮은 쪽 짝
    psi_lower = (math.atan2(float(v[1]), float(v[0])) / m) % period
    # 낮은 쪽이 antinode-위-손상인 경우는 B̄<0. B̄>0이면 절점선-위-손상이 낮은 쪽이므로
    # 손상 방위는 반주기 이동해 있다.
    theta0_hat = (psi_lower if B_bar < 0 else psi_lower + 0.5 * period) % period

    from . import severity as sev
    s_rad = sev.pocket_depth_to_severity(pocket.depth_frac, pocket.radial_width,
                                         plate.extent)
    return {"eta_bar": A_bar / lam,
            "delta_eta": 2.0 * abs(B_bar) / lam,
            "psi_lower": psi_lower,
            "theta0_hat": theta0_hat,
            "B_signed": B_bar,
            "eta_plus": eta_hi,
            "eta_minus": eta_lo,
            "lambda_m": lam,
            "H": H,
            "stiff_B": co["stiff_B"],
            "mass_B": co["mass_B"],
            "R_U": co["R_U"],
            "R_V": co["R_V"],
            "severity_s_bar_radial": s_rad,
            "severity_s_bar": s_rad * pocket.dtheta / (2.0 * math.pi)}
