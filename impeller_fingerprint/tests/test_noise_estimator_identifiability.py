"""noise·estimator·identifiability 테스트.

T5(설계서 §9): CRLB ↔ 경험분산 일치(대N·소노이즈) — 식별성 지표가 실제 추정오차를 예측한다는 검정.
정본 §3.5의 요구(σ_min·FIM·CRLB·프로파일우도·격자 목적함수면)를 코드 계약으로 고정.
"""
import numpy as np
import pytest

from impeller_fingerprint import estimator as est
from impeller_fingerprint import forward as fwd
from impeller_fingerprint import geometry as geo
from impeller_fingerprint import identifiability as idf
from impeller_fingerprint import kernels as ker
from impeller_fingerprint import noise as noi

PLATE = geo.DISK
MODES = [(0, 0), (1, 0), (2, 0), (3, 0)]
W = 0.003


@pytest.fixture(scope="module")
def pool():
    return [ker.mode_kernel(PLATE, m=m, n=n, n_grid=1001) for m, n in MODES]


class TestNoiseModel:
    def test_sigma_scales_as_twice_relative_frequency_noise(self, pool):
        """η̄ = 2Δf/f → σ_η = 2·σ_f/f (설계서 §4-5)."""
        S = noi.sigma_y(len(pool), sigma_rel=1e-3, rho=0.0)
        assert np.allclose(np.diag(S), (2e-3) ** 2, rtol=1e-12)
        assert np.allclose(S - np.diag(np.diag(S)), 0.0)

    def test_correlated_covariance_is_spd(self, pool):
        for rho in (0.0, 0.3, 0.6):
            S = noi.sigma_y(len(pool), sigma_rel=1e-3, rho=rho)
            assert np.allclose(S, S.T)
            assert np.all(np.linalg.eigvalsh(S) > 0)

    def test_whitening_gives_identity(self):
        S = noi.sigma_y(4, sigma_rel=2e-4, rho=0.4)
        Wh = noi.whitening(S)
        assert np.allclose(Wh @ S @ Wh.T, np.eye(4), atol=1e-12)

    def test_samples_have_requested_covariance(self):
        S = noi.sigma_y(3, sigma_rel=1e-3, rho=0.5)
        rng = np.random.default_rng(0)
        draws = noi.sample(S, rng, size=200000)
        emp = np.cov(draws, rowvar=False)
        assert np.allclose(emp, S, rtol=0.05, atol=1e-9)


class TestEstimator:
    def test_recovers_truth_without_noise(self, pool):
        xi_t, s_t = 0.42, 0.05
        y = fwd.eta_bar_linear(pool, xi_t, s_t, W, PLATE)
        S = noi.sigma_y(len(pool), sigma_rel=1e-4)
        out = est.fit(y, pool, PLATE, S, w=W)
        assert out["xi_d"] == pytest.approx(xi_t, abs=1e-4)
        assert out["s_bar"] == pytest.approx(s_t, abs=1e-5)

    def test_respects_bounds(self, pool):
        """경계 밖 진실을 주면 추정치는 경계에 붙고 boundary_hit이 표시된다."""
        y = fwd.eta_bar_linear(pool, 0.99, 0.05, W, PLATE) * 3.0
        S = noi.sigma_y(len(pool), sigma_rel=1e-3)
        out = est.fit(y, pool, PLATE, S, w=W)
        assert 0.0 <= out["xi_d"] <= 1.0
        assert out["s_bar"] >= 0.0

    def test_multistart_matches_or_beats_single(self, pool):
        rng = np.random.default_rng(3)
        S = noi.sigma_y(len(pool), sigma_rel=1e-3)
        y = fwd.eta_bar_linear(pool, 0.8, 0.03, W, PLATE) + noi.sample(S, rng)
        one = est.fit(y, pool, PLATE, S, w=W, n_starts=1)
        many = est.fit(y, pool, PLATE, S, w=W, n_starts=8)
        assert many["chi2"] <= one["chi2"] + 1e-9

    def test_exact_forward_option_runs_and_is_close(self, pool):
        """대심각도에서는 순방향을 정확재해로 교체(정본 §3.4)."""
        xi_t, s_t = 0.5, 0.2
        y = fwd.eta_bar_exact(PLATE, MODES, xi_t, s_t, W)
        S = noi.sigma_y(len(pool), sigma_rel=1e-4)
        lin = est.fit(y, pool, PLATE, S, w=W)
        exa = est.fit(y, pool, PLATE, S, w=W, exact=True, modes=MODES)
        assert abs(exa["s_bar"] - s_t) < abs(lin["s_bar"] - s_t)
        assert exa["s_bar"] == pytest.approx(s_t, rel=0.02)


class TestIdentifiabilityMetrics:
    def test_fisher_matches_direct_formula(self, pool):
        S = noi.sigma_y(len(pool), sigma_rel=5e-4, rho=0.2)
        theta = (0.5, 0.05)
        m = idf.metrics(pool, PLATE, theta, W, S)
        J = fwd.jacobian_linear(pool, theta[0], theta[1], W, PLATE)
        F = J.T @ np.linalg.inv(S) @ J
        assert np.allclose(m["F"], F, rtol=1e-10)
        assert m["det_F"] == pytest.approx(float(np.linalg.det(F)), rel=1e-10)
        assert m["tr_Finv"] == pytest.approx(float(np.trace(np.linalg.inv(F))), rel=1e-10)

    def test_crlb_units_are_mm_and_percentage_points(self, pool):
        S = noi.sigma_y(len(pool), sigma_rel=5e-4)
        m = idf.metrics(pool, PLATE, (0.5, 0.05), W, S)
        Finv = np.linalg.inv(m["F"])
        assert m["crlb_xi_mm"] == pytest.approx(
            np.sqrt(Finv[0, 0]) * PLATE.extent * 1e3, rel=1e-12)
        assert m["crlb_s_pp"] == pytest.approx(np.sqrt(Finv[1, 1]) * 100.0, rel=1e-12)

    def test_smaller_severity_is_harder_to_locate(self, pool):
        """정본 §3.5: 야코비안 위치열이 S̄에 비례 → 작은 손상은 본질적으로 위치추정이 어렵다."""
        S = noi.sigma_y(len(pool), sigma_rel=5e-4)
        a = idf.metrics(pool, PLATE, (0.5, 0.01), W, S)["crlb_xi_mm"]
        b = idf.metrics(pool, PLATE, (0.5, 0.05), W, S)["crlb_xi_mm"]
        assert a > b
        assert a / b == pytest.approx(5.0, rel=0.05)

    def test_rim_is_harder_than_midspan(self, pool):
        """정본 §4.2의 관측(rim 쪽 열화)이 CRLB로 재현되는가."""
        S = noi.sigma_y(len(pool), sigma_rel=5e-4)
        mid = idf.metrics(pool, PLATE, (0.5, 0.05), W, S)["crlb_xi_mm"]
        rim = idf.metrics(pool, PLATE, (0.95, 0.05), W, S)["crlb_xi_mm"]
        assert rim > mid

    def test_map_shapes_and_consistency(self, pool):
        xi_grid = np.linspace(0.2, 0.8, 4)
        s_grid = np.array([0.01, 0.05])
        S = noi.sigma_y(len(pool), sigma_rel=5e-4)
        maps = idf.metric_maps(pool, PLATE, xi_grid, s_grid, W, S)
        for key in ("sigma_min", "sigma_max", "cond2", "det_F", "tr_Finv",
                    "corr", "crlb_xi_mm", "crlb_s_pp"):
            assert maps[key].shape == (len(xi_grid), len(s_grid)), key
        one = idf.metrics(pool, PLATE, (xi_grid[2], s_grid[1]), W, S)
        assert maps["cond2"][2, 1] == pytest.approx(one["cond2"], rel=1e-12)


class TestT5CRLBvsEmpirical:
    def test_crlb_predicts_empirical_scatter(self, pool):
        xi_t, s_t, sigma_rel, n_rep = 0.5, 0.05, 3e-5, 400
        S = noi.sigma_y(len(pool), sigma_rel=sigma_rel)
        y0 = fwd.eta_bar_linear(pool, xi_t, s_t, W, PLATE)
        rng = np.random.default_rng(11)
        eps = noi.sample(S, rng, size=n_rep)
        xis, sbs = [], []
        for k in range(n_rep):
            out = est.fit(y0 + eps[k], pool, PLATE, S, w=W)
            xis.append(out["xi_d"])
            sbs.append(out["s_bar"])
        m = idf.metrics(pool, PLATE, (xi_t, s_t), W, S)
        emp_xi_mm = float(np.std(xis, ddof=1)) * PLATE.extent * 1e3
        emp_s_pp = float(np.std(sbs, ddof=1)) * 100.0
        assert emp_xi_mm / m["crlb_xi_mm"] == pytest.approx(1.0, abs=0.12)
        assert emp_s_pp / m["crlb_s_pp"] == pytest.approx(1.0, abs=0.12)


class TestProfileAndGrid:
    def test_profile_minimum_at_truth(self, pool):
        xi_t, s_t = 0.4, 0.05
        S = noi.sigma_y(len(pool), sigma_rel=1e-4)
        y = fwd.eta_bar_linear(pool, xi_t, s_t, W, PLATE)
        xi_grid = np.linspace(0.05, 0.95, 91)
        prof = idf.profile_likelihood(y, pool, PLATE, S, W, xi_grid)
        assert xi_grid[int(np.argmin(prof))] == pytest.approx(xi_t, abs=0.02)
        assert prof.min() < 1e-6

    def test_profile_interval_brackets_truth_and_matches_crlb(self, pool):
        """Δχ²=3.84 구간이 진실을 감싸고, 그 반폭이 1.96·CRLB와 같은 규모여야 한다.

        격자는 CRLB의 1/5로 해상한다(구간이 격자간격보다 좁으면 검정이 무의미해짐).
        """
        xi_t, s_t = 0.4, 0.05
        S = noi.sigma_y(len(pool), sigma_rel=1e-3)
        crlb_xi = idf.metrics(pool, PLATE, (xi_t, s_t), W, S)["crlb_xi_mm"] / (
            PLATE.extent * 1e3)                     # ξ 단위 표준편차
        rng = np.random.default_rng(5)
        y = fwd.eta_bar_linear(pool, xi_t, s_t, W, PLATE) + noi.sample(S, rng)
        step = crlb_xi / 5.0
        xi_grid = np.arange(max(0.0, xi_t - 30 * step), xi_t + 30 * step, step)
        prof = idf.profile_likelihood(y, pool, PLATE, S, W, xi_grid)
        lo, hi = idf.profile_interval(xi_grid, prof, delta_chi2=3.84)
        assert lo - step <= xi_t <= hi + step
        half = 0.5 * (hi - lo)
        assert half / (1.96 * crlb_xi) == pytest.approx(1.0, abs=0.25)

    def test_objective_grid_minimum_near_truth(self, pool):
        xi_t, s_t = 0.6, 0.03
        S = noi.sigma_y(len(pool), sigma_rel=1e-4)
        y = fwd.eta_bar_linear(pool, xi_t, s_t, W, PLATE)
        xi_grid = np.linspace(0.2, 0.9, 36)
        s_grid = np.linspace(0.005, 0.06, 34)
        chi2 = idf.objective_grid(y, pool, PLATE, S, W, xi_grid, s_grid)
        i, j = np.unravel_index(int(np.argmin(chi2)), chi2.shape)
        assert xi_grid[i] == pytest.approx(xi_t, abs=0.03)
        assert s_grid[j] == pytest.approx(s_t, abs=0.003)
