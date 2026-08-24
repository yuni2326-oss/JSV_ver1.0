"""kernels 테스트 — 민감도 커널 γ_{m,n}(r)과 고유주파수.

T3(설계서 §9): γ_m은 판 두께에 무관 → §5.2의 "레일 균일판으로 결론이 이전된다" 정당화의 근거.
T6: 논문1 `classical_annular_plate`(다른 패키지, 같은 정식화)와 교차검증.
추가: 정본 §4.3이 주장하는 커널 기하(γ₀≈γ₁ 근사공선, m=3의 외측 rim 우세)를 테스트로 고정.
"""
import numpy as np
import pytest

from impeller_fingerprint import geometry as geo
from impeller_fingerprint import kernels as ker


class TestNormalizationAndOrdering:
    def test_gamma_integrates_to_one(self):
        for m in range(5):
            for n in (0, 1):
                k = ker.mode_kernel(geo.DISK, m=m, n=n)
                assert np.trapezoid(k.gamma, k.r) == pytest.approx(1.0, rel=1e-6)

    def test_radial_order_raises_frequency(self):
        for m in range(4):
            f0 = ker.mode_kernel(geo.DISK, m=m, n=0).f
            f1 = ker.mode_kernel(geo.DISK, m=m, n=1).f
            assert f1 > f0

    def test_frequencies_positive_and_plausible(self):
        f = [ker.mode_kernel(geo.DISK, m=m, n=0).f for m in range(4)]
        assert all(500.0 < x < 2.0e4 for x in f), f

    def test_sandwich_frequency_ratio_scales_as_sqrt_D_over_rhoh(self):
        """레일 균일판 → 샌드위치 절대주파수 사상은 √(D/ρh) 비율(설계서 §5.2)."""
        p, s = geo.DISK, geo.SANDWICH
        ratio = np.sqrt((s.D / s.rhoh) / (p.D / p.rhoh))
        f_plate = ker.mode_kernel(p, m=2, n=0).f
        f_sand = ker.mode_kernel_props(s.a, s.b, s.D, s.rhoh, s.nu, m=2, n=0).f
        assert f_sand / f_plate == pytest.approx(ratio, rel=1e-9)


class TestT3ThicknessInvariance:
    def test_gamma_invariant_under_thickness(self):
        thin = geo.DISK
        thick = geo.Plate(a=thin.a, b=thin.b, t=2 * thin.t, E=thin.E,
                          rho=thin.rho, nu=thin.nu)
        for m in range(4):
            g1 = ker.mode_kernel(thin, m=m, n=0).gamma
            g2 = ker.mode_kernel(thick, m=m, n=0).gamma
            assert np.max(np.abs(g1 - g2)) < 1e-8 * np.max(np.abs(g1))

    def test_gamma_invariant_under_modulus(self):
        base = geo.DISK
        stiff = geo.Plate(a=base.a, b=base.b, t=base.t, E=3 * base.E,
                          rho=base.rho, nu=base.nu)
        g1 = ker.mode_kernel(base, m=1, n=0).gamma
        g2 = ker.mode_kernel(stiff, m=1, n=0).gamma
        assert np.max(np.abs(g1 - g2)) < 1e-8 * np.max(np.abs(g1))


class TestT6CrossPackage:
    """논문1 impeller_pinn과의 교차검증(읽기전용 import)."""

    def test_gamma_matches_impeller_pinn(self):
        cap = pytest.importorskip("impeller_pinn.classical_annular_plate")
        p = geo.DISK
        for m in range(4):
            r_ref, g_ref = cap.gamma_curve(p.a, p.b, p.D, p.rhoh, p.nu, m,
                                           n_trial=8, n_grid=1001)
            k = ker.mode_kernel(p, m=m, n=0, n_trial=8, n_grid=1001)
            assert np.allclose(k.r, r_ref, rtol=0, atol=1e-15)
            assert np.max(np.abs(k.gamma - g_ref)) < 1e-6 * np.max(np.abs(g_ref))

    def test_frequency_matches_impeller_pinn(self):
        cap = pytest.importorskip("impeller_pinn.classical_annular_plate")
        p = geo.DISK
        for m in range(4):
            f_ref = cap.solve_annular_plate(p.a, p.b, p.D, p.rhoh, p.nu, m,
                                            n_modes=1, n_trial=8, n_grid=4001)[0]["f"]
            f = ker.mode_kernel(p, m=m, n=0, n_trial=8, n_grid=4001).f
            assert f == pytest.approx(f_ref, rel=1e-9)


class TestPaperKernelGeometryClaims:
    def test_gamma0_gamma1_near_collinear(self):
        """정본 §4.3: γ₀ ≈ γ₁ 근사공선성이 위치·심각도 분리를 어렵게 만든다."""
        g0 = ker.mode_kernel(geo.DISK, m=0, n=0).gamma
        g1 = ker.mode_kernel(geo.DISK, m=1, n=0).gamma
        cos = float(g0 @ g1 / (np.linalg.norm(g0) * np.linalg.norm(g1)))
        assert cos > 0.99, cos

    def test_m3_dominates_outer_rim(self):
        """정본 §4.3: m=3이 외측 rim을 지배한다."""
        p = geo.DISK
        k0 = ker.mode_kernel(p, m=0, n=0)
        k3 = ker.mode_kernel(p, m=3, n=0)
        xi = (k0.r - p.a) / p.extent
        outer = xi > 0.7
        mass0 = np.trapezoid(k0.gamma[outer], k0.r[outer])
        mass3 = np.trapezoid(k3.gamma[outer], k3.r[outer])
        assert mass3 > mass0


class TestDamagedSolve:
    def test_damage_lowers_frequency(self):
        p = geo.DISK
        f0 = ker.solve_frequencies(p, m=1, n_modes=1)[0]
        d = lambda r: 0.05 * np.exp(-((r - 0.5 * (p.a + p.b)) / 0.003) ** 2)
        f1 = ker.solve_frequencies(p, m=1, n_modes=1, damage=d)[0]
        assert f1 < f0

    def test_zero_damage_equals_healthy(self):
        p = geo.DISK
        f0 = ker.solve_frequencies(p, m=2, n_modes=2)
        f1 = ker.solve_frequencies(p, m=2, n_modes=2, damage=lambda r: np.zeros_like(r))
        assert np.allclose(f0, f1, rtol=1e-12)
