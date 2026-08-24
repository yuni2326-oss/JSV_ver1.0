"""forward·validity 테스트 — 선형섭동 ↔ 비섭동 정확재해, e_pert 유효성 맵.

T4(설계서 §9): 소 S̄_D에서 섭동 ≈ 정확재해(e_pert → 0), 대 S̄_D에서 단조 증가.
정의 동결(§4): η̄ = δλ/λ = −∫γ d dr,  Δf/f = ½η̄.
"""
import numpy as np
import pytest

from impeller_fingerprint import forward as fwd
from impeller_fingerprint import geometry as geo
from impeller_fingerprint import kernels as ker
from impeller_fingerprint import severity as sev
from impeller_fingerprint import validity as val

PLATE = geo.DISK
MODES = [(0, 0), (1, 0), (2, 0), (3, 0)]
W = 0.003


@pytest.fixture(scope="module")
def pool():
    return [ker.mode_kernel(PLATE, m=m, n=n) for m, n in MODES]


class TestLinearForward:
    def test_damage_gives_negative_eta(self, pool):
        eta = fwd.eta_bar_linear(pool, xi_d=0.5, s_bar=0.05, w=W, plate=PLATE)
        assert np.all(eta < 0)

    def test_eta_is_twice_relative_frequency_shift(self, pool):
        eta = fwd.eta_bar_linear(pool, xi_d=0.4, s_bar=0.03, w=W, plate=PLATE)
        df = fwd.rel_freq_shift_linear(pool, xi_d=0.4, s_bar=0.03, w=W, plate=PLATE)
        assert np.allclose(eta, 2.0 * df, rtol=1e-14)

    def test_linear_in_severity(self, pool):
        e1 = fwd.eta_bar_linear(pool, xi_d=0.4, s_bar=0.01, w=W, plate=PLATE)
        e3 = fwd.eta_bar_linear(pool, xi_d=0.4, s_bar=0.03, w=W, plate=PLATE)
        assert np.allclose(e3, 3.0 * e1, rtol=1e-12)

    def test_matches_impeller_pinn_forward_shifts(self, pool):
        """논문1 `inverse_damage.forward_shifts`(Δf/f 규약)와 수치 일치 — 파일럿 재현 근거(T7)."""
        inv = pytest.importorskip("impeller_pinn.inverse_damage")
        r = pool[0].r
        gammas = [k.gamma for k in pool]
        r_d = sev.xi_to_r(0.35, PLATE.a, PLATE.b)
        S = sev.S_from_s_bar(0.0567, PLATE.extent)
        ref = inv.forward_shifts(r, gammas, r_d, S, W)
        mine = fwd.rel_freq_shift_linear(pool, xi_d=0.35, s_bar=0.0567, w=W, plate=PLATE)
        assert np.allclose(mine, ref, rtol=1e-12)


class TestAnalyticJacobian:
    def test_jacobian_matches_finite_difference(self, pool):
        xi, s_bar = 0.42, 0.04
        J = fwd.jacobian_linear(pool, xi_d=xi, s_bar=s_bar, w=W, plate=PLATE)
        h_xi, h_s = 1e-6, 1e-8
        fd_xi = (fwd.eta_bar_linear(pool, xi + h_xi, s_bar, W, PLATE)
                 - fwd.eta_bar_linear(pool, xi - h_xi, s_bar, W, PLATE)) / (2 * h_xi)
        fd_s = (fwd.eta_bar_linear(pool, xi, s_bar + h_s, W, PLATE)
                - fwd.eta_bar_linear(pool, xi, s_bar - h_s, W, PLATE)) / (2 * h_s)
        assert np.allclose(J[:, 0], fd_xi, rtol=1e-5)
        assert np.allclose(J[:, 1], fd_s, rtol=1e-8)

    def test_severity_column_is_eta_over_severity(self, pool):
        """선형성 → ∂η̄/∂S̄ = η̄/S̄ (정확)."""
        xi, s_bar = 0.6, 0.02
        J = fwd.jacobian_linear(pool, xi_d=xi, s_bar=s_bar, w=W, plate=PLATE)
        eta = fwd.eta_bar_linear(pool, xi, s_bar, W, PLATE)
        assert np.allclose(J[:, 1], eta / s_bar, rtol=1e-12)


class TestExactForward:
    def test_exact_eta_negative_and_scale_invariant(self):
        """η̄는 무차원 Λ 비로 계산 → D·ρh(두께·물성)에 무관."""
        eta1 = fwd.eta_bar_exact(PLATE, MODES, xi_d=0.5, s_bar=0.05, w=W)
        thick = geo.Plate(a=PLATE.a, b=PLATE.b, t=3 * PLATE.t, E=2 * PLATE.E,
                          rho=PLATE.rho, nu=PLATE.nu)
        eta2 = fwd.eta_bar_exact(thick, MODES, xi_d=0.5, s_bar=0.05, w=W)
        assert np.all(eta1 < 0)
        assert np.allclose(eta1, eta2, rtol=1e-10)

    def test_zero_severity_gives_zero(self):
        eta = fwd.eta_bar_exact(PLATE, MODES, xi_d=0.5, s_bar=0.0, w=W)
        assert np.allclose(eta, 0.0, atol=1e-12)

    def test_T4_perturbation_error_is_second_order(self, pool):
        """절대 섭동오차가 S̄²에 비례(=2차 항) → 상대오차 e_pert는 S̄에 1차로 감소.

        임계값이 아니라 **수렴차수**를 검정한다(문턱 없는 검정). 부호도 함께 고정:
        Rayleigh 상계 성질상 정확재해의 고유값 강하가 항상 선형예측보다 크다.
        """
        s1, s2 = 0.02, 0.01
        a1 = val.e_pert_abs(PLATE, pool, MODES, xi_d=0.5, s_bar=s1, w=W)
        a2 = val.e_pert_abs(PLATE, pool, MODES, xi_d=0.5, s_bar=s2, w=W)
        assert np.allclose(a1 / a2, 4.0, rtol=0.1), a1 / a2
        r1 = val.e_pert(PLATE, pool, MODES, xi_d=0.5, s_bar=s1, w=W)
        r2 = val.e_pert(PLATE, pool, MODES, xi_d=0.5, s_bar=s2, w=W)
        assert np.allclose(r1 / r2, 2.0, rtol=0.1), r1 / r2
        # 부호: |정확| > |선형| (정확해가 더 많이 떨어진다)
        lin = fwd.eta_bar_linear(pool, xi_d=0.5, s_bar=s1, w=W, plate=PLATE)
        ex = fwd.eta_bar_exact(PLATE, MODES, xi_d=0.5, s_bar=s1, w=W)
        assert np.all(np.abs(ex) > np.abs(lin))

    def test_T4_error_magnitude_at_one_percent_severity(self, pool):
        """관측된 규모 고정(회귀): S̄_D=1 %에서 e_pert는 m=0에서 최대 ~1.3 %."""
        rel = val.e_pert(PLATE, pool, MODES, xi_d=0.5, s_bar=0.01, w=W)
        assert rel.max() < 0.02, rel
        assert rel[0] == pytest.approx(0.0129, abs=0.002)
        assert rel[0] > rel[-1]          # 저차 모드가 더 취약

    def test_T4_error_grows_with_severity(self, pool):
        errs = [val.e_pert(PLATE, pool, MODES, xi_d=0.5, s_bar=s, w=W).max()
                for s in (0.01, 0.05, 0.15, 0.30)]
        assert all(b > a for a, b in zip(errs, errs[1:])), errs


class TestValidityMap:
    def test_map_shape_and_positivity(self, pool):
        xi_grid = np.linspace(0.2, 0.8, 4)
        s_grid = np.array([0.01, 0.05, 0.2])
        M = val.e_pert_map(PLATE, pool, MODES, xi_grid, s_grid, w=W)
        assert M.shape == (len(MODES), len(xi_grid), len(s_grid))
        assert np.all(M >= 0.0)
        assert np.all(np.isfinite(M))

    def test_map_matches_pointwise(self, pool):
        xi_grid = np.array([0.35])
        s_grid = np.array([0.08])
        M = val.e_pert_map(PLATE, pool, MODES, xi_grid, s_grid, w=W)
        point = val.e_pert(PLATE, pool, MODES, xi_d=0.35, s_bar=0.08, w=W)
        assert np.allclose(M[:, 0, 0], point, rtol=1e-12)

    def test_noise_floor_contour_monotone_in_noise(self, pool):
        """측정floor 교차: 노이즈가 커지면 '섭동오차가 무의미해지는' 영역이 넓어진다."""
        xi_grid = np.linspace(0.2, 0.8, 5)
        s_grid = np.array([0.02, 0.1])
        frac_small = val.fraction_below_floor(PLATE, pool, MODES, xi_grid, s_grid,
                                              w=W, sigma_rel=1e-4)
        frac_big = val.fraction_below_floor(PLATE, pool, MODES, xi_grid, s_grid,
                                            w=W, sigma_rel=3e-3)
        assert frac_big >= frac_small
