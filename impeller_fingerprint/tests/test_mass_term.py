"""F12 — 질량항 포함 순방향·역식별 (설계서 §11.5 F12).

문제: 정본 §3.3의 순방향 η̄ = −∫γ^K d dr ≤ 0은 **양의 이동을 표현할 수 없다**. 그런데 림 포켓은
질량제거가 강성손실을 이겨 pair mean이 양수가 되고(3D·섭동이론 모두 확인), 추정기가 경계로 붙었다.
수정: η̄ = −∫γ^K d_K dr + ∫γ^M d_M dr, 포켓은 d_K = 1−(1−p)³ ≈ 3p, d_M = p → d_M = d_K/3.
"""
import numpy as np
import pytest

from impeller_fingerprint import degenerate as deg
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
    return [ker.mode_kernel(PLATE, m=m, n=n, n_grid=2001) for m, n in MODES]


class TestMassKernel:
    def test_mass_kernel_normalized(self, pool):
        for k in pool:
            assert np.trapezoid(k.gamma_mass, k.r) == pytest.approx(1.0, rel=1e-6)

    def test_mass_kernel_dominates_at_rim(self, pool):
        """림에서 γ^M ≫ γ^K — 질량이 강성을 이기는 구조적 이유."""
        k = pool[2]
        xi = (k.r - PLATE.a) / PLATE.extent
        outer = xi > 0.85
        assert np.trapezoid(k.gamma_mass[outer], k.r[outer]) > \
               np.trapezoid(k.gamma[outer], k.r[outer])

    def test_stiffness_kernel_dominates_mid_span(self, pool):
        k = pool[2]
        xi = (k.r - PLATE.a) / PLATE.extent
        mid = (xi > 0.3) & (xi < 0.6)
        assert np.trapezoid(k.gamma[mid], k.r[mid]) > \
               np.trapezoid(k.gamma_mass[mid], k.r[mid])


class TestF12SignRecovery:
    def test_rim_damage_gives_positive_eta(self, pool):
        """F12의 핵심: 질량항을 넣으면 림 손상에서 η̄ > 0이 나온다."""
        eta = fwd.eta_bar_linear_mass(pool, xi_d=0.9, s_bar=0.03, w=W, plate=PLATE)
        assert np.any(eta > 0), eta

    def test_stiffness_only_map_cannot_produce_positive(self, pool):
        """비교: 강성전용 맵은 어떤 (ξ, S̄)에서도 음수만 낸다."""
        for xi in (0.1, 0.5, 0.9, 0.99):
            assert np.all(fwd.eta_bar_linear(pool, xi, 0.03, W, PLATE) <= 0.0)

    def test_midspan_still_negative(self, pool):
        eta = fwd.eta_bar_linear_mass(pool, xi_d=0.4, s_bar=0.03, w=W, plate=PLATE)
        assert np.all(eta < 0), eta

    def test_coupling_zero_reduces_to_stiffness_only(self, pool):
        a = fwd.eta_bar_linear_mass(pool, 0.5, 0.03, W, PLATE, coupling=0.0)
        b = fwd.eta_bar_linear(pool, 0.5, 0.03, W, PLATE)
        assert np.allclose(a, b, rtol=1e-14)


class TestAgainstDegenerateTheory:
    def test_matches_axisymmetric_pocket_with_mass(self):
        """축대칭 포켓(Δθ=2π)에서 질량항 포함 섭동맵 = degenerate 모듈의 pair mean.

        degenerate는 δK−λδM을 직접 적분하므로 독립 경로다. 두 경로가 맞으면 결합비 1/3과
        질량커널 정의가 옳다는 뜻.
        """
        m, r1, r2, depth = 2, 0.030, 0.036, 0.15
        p = deg.Pocket(r1=r1, r2=r2, theta0=0.0, dtheta=2 * np.pi, depth_frac=depth)
        eta_ref = deg.observables(PLATE, m, p, mass_term=True)["eta_bar"]

        k = ker.mode_kernel(PLATE, m=m, n=0, n_grid=4001)
        beta_D = 1.0 - (1.0 - depth) ** 3
        beta_M = depth
        box = ((k.r >= r1) & (k.r <= r2)).astype(float)
        eta_mine = (-beta_D * float(np.trapezoid(k.gamma * box, k.r))
                    + beta_M * float(np.trapezoid(k.gamma_mass * box, k.r)))
        assert eta_mine == pytest.approx(eta_ref, rel=3e-3)

    def test_rim_pocket_positive_in_both_paths(self):
        p = deg.Pocket(r1=0.032, r2=0.036, theta0=0.4, dtheta=np.deg2rad(30),
                       depth_frac=0.5)
        assert deg.observables(PLATE, 2, p)["eta_bar"] > 0


class TestInversionWithMass:
    @pytest.mark.parametrize("xi_t", [0.3, 0.5, 0.85])
    def test_recovers_truth_including_rim(self, pool, xi_t):
        """질량항 맵으로 생성한 데이터를 같은 맵으로 역식별 — 림에서도 경계에 붙지 않는다."""
        s_t = 0.03
        y = fwd.eta_bar_linear_mass(pool, xi_t, s_t, W, PLATE)
        S = noi.sigma_y(len(pool), 1e-4)
        out = est.fit(y, pool, PLATE, S, w=W, n_starts=8, mass=True)
        assert out["mass_term"] is True
        assert out["xi_d"] == pytest.approx(xi_t, abs=0.02), out
        assert out["s_bar"] == pytest.approx(s_t, rel=0.05)
        assert not out["boundary_hit"]

    def test_stiffness_only_fit_fails_on_rim_data(self, pool):
        """대조: 같은 림 데이터를 강성전용 맵으로 적합하면 경계로 붙는다(F12가 보고한 실패)."""
        y = fwd.eta_bar_linear_mass(pool, 0.9, 0.03, W, PLATE)
        S = noi.sigma_y(len(pool), 1e-4)
        bad = est.fit(y, pool, PLATE, S, w=W, n_starts=8, mass=False)
        assert bad["boundary_hit"] or abs(bad["xi_d"] - 0.9) > 0.1

    def test_jacobian_mass_matches_finite_difference(self, pool):
        xi, s_bar = 0.8, 0.03
        J = fwd.jacobian_linear_mass(pool, xi, s_bar, W, PLATE)
        h = 1e-6
        fd = (fwd.eta_bar_linear_mass(pool, xi + h, s_bar, W, PLATE)
              - fwd.eta_bar_linear_mass(pool, xi - h, s_bar, W, PLATE)) / (2 * h)
        assert np.allclose(J[:, 0], fd, rtol=1e-5)
        assert np.allclose(J[:, 1],
                           fwd.eta_bar_linear_mass(pool, xi, s_bar, W, PLATE) / s_bar,
                           rtol=1e-12)


class TestIdentifiabilityWithMass:
    def test_mass_term_changes_crlb_at_rim(self, pool):
        """질량항은 림에서 민감도 구조를 바꾼다 — CRLB도 달라져야 한다."""
        S = noi.sigma_y(len(pool), 5e-4)
        a = idf.metrics(pool, PLATE, (0.9, 0.03), W, S, mass=False)["crlb_xi_mm"]
        b = idf.metrics(pool, PLATE, (0.9, 0.03), W, S, mass=True)["crlb_xi_mm"]
        assert not np.isclose(a, b, rtol=0.02)

    def test_null_locus_exists_where_stiffness_cancels_mass(self, pool):
        """γ^K = γ^M/3이 되는 반경 부근에서 η̄가 0을 지난다(감도 소실 지점).

        이 위치는 역식별이 구조적으로 불가능한 곳이므로 논문에 표시해야 한다.
        """
        xs = np.linspace(0.05, 0.99, 60)
        eta2 = np.array([fwd.eta_bar_linear_mass(pool, x, 0.03, W, PLATE)[2]
                         for x in xs])
        assert np.any(eta2 < 0) and np.any(eta2 > 0)
        cross = xs[np.argmin(np.abs(eta2))]
        assert 0.5 < cross < 1.0, cross


class TestA4MassWiring:
    """설계서 M7/A4 — 프로파일우도·목적함수면이 `mass`를 실제로 쓰는지 고정한다.

    사고 이력: `_chi2_at`가 `fwd.eta_bar_linear`를 하드코딩해 `cli a4 --mass` 산출물이
    강성전용 파일과 byte-identical했다. 아래 검정은 그 배선을 코드에서 고정한다.
    """

    def test_profile_likelihood_with_mass_recovers_rim_truth(self, pool):
        """정확결합에서 S̄는 해석해가 없다 — 수치 최소화가 진실을 정확히 맞춰야 한다."""
        xi_t, s_t = 0.9, 0.05
        S = noi.sigma_y(len(pool), 1e-3)
        y = fwd.eta_bar_linear_mass(pool, xi_t, s_t, W, PLATE, coupling="exact")
        xg = np.linspace(0.05, 0.98, 94)
        prof = idf.profile_likelihood(y, pool, PLATE, S, W, xg, mass="exact")
        assert prof.min() < 1e-6
        assert xg[int(np.argmin(prof))] == pytest.approx(xi_t, abs=0.02)

    def test_stiffness_only_profile_cannot_fit_mass_data(self, pool):
        """배선이 누락되면 이 χ²가 ~0이 된다(= 사고 재발 탐지기)."""
        xi_t, s_t = 0.9, 0.05
        S = noi.sigma_y(len(pool), 1e-3)
        y = fwd.eta_bar_linear_mass(pool, xi_t, s_t, W, PLATE, coupling="exact")
        xg = np.linspace(0.05, 0.98, 94)
        assert idf.profile_likelihood(y, pool, PLATE, S, W, xg).min() > 1.0

    def test_objective_grid_with_mass_locates_rim_truth(self, pool):
        xi_t, s_t = 0.9, 0.05
        S = noi.sigma_y(len(pool), 1e-4)
        y = fwd.eta_bar_linear_mass(pool, xi_t, s_t, W, PLATE, coupling="exact")
        xg = np.linspace(0.5, 0.98, 25)
        sg = np.linspace(0.02, 0.08, 25)
        g_m = idf.objective_grid(y, pool, PLATE, S, W, xg, sg, mass="exact")
        i, j = np.unravel_index(int(np.argmin(g_m)), g_m.shape)
        assert xg[i] == pytest.approx(xi_t, abs=0.03)
        assert sg[j] == pytest.approx(s_t, abs=0.006)
        g_k = idf.objective_grid(y, pool, PLATE, S, W, xg, sg)
        assert g_k.min() > g_m.min()

    def test_profile_likelihood_linear_path_unchanged(self, pool):
        """하위호환: mass=None 경로는 해석 최소화를 그대로 쓴다."""
        xi_t, s_t = 0.4, 0.03
        S = noi.sigma_y(len(pool), 1e-3)
        y = fwd.eta_bar_linear(pool, xi_t, s_t, W, PLATE)
        xg = np.linspace(0.1, 0.9, 41)
        prof = idf.profile_likelihood(y, pool, PLATE, S, W, xg)
        assert prof.min() < 1e-9
        assert xg[int(np.argmin(prof))] == pytest.approx(xi_t, abs=0.02)


class TestCouplingConvention:
    """M8 — `mass` 인자의 결합비 해석이 모듈마다 갈리지 않게 못박는다 (2026-08-05 감사).

    감사 시점의 실상: `mass=True`가 세 가지 다른 모델을 뜻했다.
      - identifiability·estimator : coupling = MASS_COUPLING = 1/3
      - modeselect                : coupling = float(True) = 1.0  ⇒ d_M = d_K (정확값의 3배)
      - validity·montecarlo       : 선형항 coupling = 1.0인데 정확재해는 정확결합 ⇒ 서로 다른 모델
    F20 ①과 동일한 사고 패턴(불리언이 연속 파라미터 자리로 새는 것)이므로
    `forward.resolve_coupling` 한 곳으로 정규화했고, 이 검정이 재발을 막는다.
    """

    def test_resolve_coupling_maps_true_to_first_order(self):
        assert fwd.resolve_coupling(True) == pytest.approx(fwd.MASS_COUPLING)
        assert fwd.resolve_coupling("exact") == "exact"
        assert fwd.resolve_coupling(0.25) == 0.25
        assert fwd.resolve_coupling(None) is None
        assert fwd.resolve_coupling(False) is False

    def test_true_is_first_order_not_unity_in_every_module(self, pool):
        """모든 소비 모듈에서 mass=True ≡ mass=1/3 이고, coupling=1.0과는 달라야 한다."""
        from impeller_fingerprint import modeselect as msel
        from impeller_fingerprint import validity as val
        theta, sigma = (0.8, 0.05), noi.sigma_y(len(pool), 1e-3)
        probes = {
            "identifiability": lambda c: idf.metrics(pool, PLATE, theta, W, sigma,
                                                     mass=c)["crlb_xi_mm"],
            "modeselect": lambda c: msel.subset_metrics(pool, PLATE, theta, W, sigma,
                                                        mass=c)["crlb_xi_mm"],
            "validity": lambda c: float(val.e_pert(PLATE, pool, MODES, theta[0],
                                                   theta[1], W, mass=c)[0]),
        }
        for name, fn in probes.items():
            assert fn(True) == pytest.approx(fn(fwd.MASS_COUPLING), rel=1e-12), name
            assert not np.isclose(fn(True), fn(1.0), rtol=1e-3), (
                f"{name}: mass=True가 coupling=1.0으로 새고 있다 (F20 ①)")

    def test_exact_coupling_differs_from_first_order(self, pool):
        """정확결합과 1/3 근사는 실제로 다른 값을 낸다 — 별칭이 무의미해지지 않았음을 확인."""
        eta_ex = fwd.eta_bar_linear_mass(pool, 0.8, 0.05, W, PLATE, coupling="exact")
        eta_13 = fwd.eta_bar_linear_mass(pool, 0.8, 0.05, W, PLATE, coupling=True)
        assert not np.allclose(eta_ex, eta_13, rtol=0.02)

    def test_production_cli_never_passes_boolean_mass(self):
        """생산 경로 검수: `--mass`는 nargs='?' const='exact'이므로 True가 들어갈 수 없다."""
        from impeller_fingerprint import cli
        parser = cli.build_parser() if hasattr(cli, "build_parser") else None
        if parser is None:                      # 파서 팩토리가 없으면 소스로 검사
            import inspect
            src = inspect.getsource(cli)
            assert 'action="store_true"' not in src.split("--mass")[0][-200:]
            assert src.count('"--mass", nargs="?", const="exact"') \
                   + src.count("'--mass', nargs='?', const='exact'") >= 1
            return
        for item in ("a2", "a3", "a4", "a5", "b1"):
            args = parser.parse_args([item])
            assert getattr(args, "mass", None) is None
            args = parser.parse_args([item, "--mass"])
            assert args.mass == "exact"


# ---------------------------------------------------------------------------
# 샌드위치 결합법칙 (설계서 §5.3 / 정본 §3.3 범위한정) — 리뷰 지적 5
# ---------------------------------------------------------------------------
def _I_two_flange(p: float, t_f: float, s: float) -> float:
    """면판 2장 단면의 관성모멘트(자체관성 포함). 손상면판을 외면에서 p·t_f 제거."""
    t1, c1 = t_f * (1.0 - p), s / 2 - p * t_f / 2
    t2, c2 = t_f, -s / 2
    zb = (t1 * c1 + t2 * c2) / (t1 + t2)
    return (t1 ** 3 / 12 + t1 * (c1 - zb) ** 2
            + t2 ** 3 / 12 + t2 * (c2 - zb) ** 2)


class TestSandwichLaw:
    """실제 슈라우드는 판재 2장 샌드위치다 — 판 법칙 d_K = 1−(1−p)³와 원리적으로 다르다."""

    @pytest.mark.parametrize("p", [0.02, 0.05, 0.1, 0.25, 0.5, 0.75])
    def test_thin_face_closed_form_is_the_section_limit(self, p):
        """d_K = p/(2−p)는 t_f/s → 0 극한의 정확한 값이고 **s에 무관**하다."""
        vals = []
        for ratio in (0.02, 0.01, 0.005):
            t_f = 1.0
            s = t_f / ratio
            vals.append(1.0 - _I_two_flange(p, t_f, s) / _I_two_flange(0.0, t_f, s))
        target = p / (2.0 - p)
        assert vals[-1] == pytest.approx(target, rel=0.02)
        # 수렴: 얇아질수록 닫힌형에 가까워진다
        assert abs(vals[-1] - target) < abs(vals[0] - target)

    @pytest.mark.parametrize("p,ratio", [(0.05, 0.05), (0.1, 0.1), (0.25, 0.05)])
    def test_first_order_thickness_correction(self, p, ratio):
        """d_K ≈ [p/(2−p)]·[1 + 2(1−p)·t_f/s] — 유한 면판두께의 1차 보정."""
        t_f = 1.0
        s = t_f / ratio
        exact = 1.0 - _I_two_flange(p, t_f, s) / _I_two_flange(0.0, t_f, s)
        approx = (p / (2.0 - p)) * (1.0 + 2.0 * (1.0 - p) * ratio)
        assert exact == pytest.approx(approx, rel=0.03)

    @pytest.mark.parametrize("p", [0.01, 0.05, 0.2, 0.5])
    def test_coupling_law_is_dK_over_one_plus_dK(self, p):
        """d_K = p/(2−p), d_M = p/2  ⇒  d_M = d_K/(1+d_K)."""
        d_K = p / (2.0 - p)
        assert fwd.mass_field_sandwich(d_K) == pytest.approx(p / 2.0, rel=1e-12)

    def test_sandwich_mass_is_about_three_times_monolithic(self):
        """같은 d_K에서 샌드위치의 질량손실이 판 법칙의 2.6–3.0배 — 세제곱↔선형 차이."""
        d_K = np.array([0.01, 0.05, 0.1, 0.2])
        ratio = fwd.mass_field_sandwich(d_K) / fwd.mass_field_exact(d_K)
        assert np.all(ratio > 2.3) and np.all(ratio < 3.0)
        assert ratio[0] == pytest.approx(3.0, abs=0.05)   # d_K→0 극한은 정확히 3

    def test_sandwich_moves_sign_reversal_inboard(self, pool):
        """ξ*는 γ^M/γ^K = 1/ζ를 만족하므로 ζ가 크면(샌드위치) 반전이 안쪽으로 온다."""
        from impeller_fingerprint import figures as figs
        _, _, loci_mono = figs.null_loci(pool, coupling="exact")
        _, _, loci_sw = figs.null_loci(pool, coupling="sandwich")
        assert np.all(np.isfinite(loci_sw))
        assert np.all(loci_sw < loci_mono - 0.05), (loci_mono, loci_sw)

    def test_sandwich_propagates_through_jacobian_and_metrics(self, pool):
        """배선 검정 — 샌드위치 법칙이 조용히 정확결합으로 폴백하면 실패한다."""
        J_sw = fwd.jacobian_linear_mass(pool, 0.6, 0.05, W, PLATE, coupling="sandwich")
        J_ex = fwd.jacobian_linear_mass(pool, 0.6, 0.05, W, PLATE, coupling="exact")
        assert not np.allclose(J_sw, J_ex, rtol=0.05)
        S = noi.sigma_y(len(pool), 1e-3)
        m_sw = idf.metrics(pool, PLATE, (0.6, 0.05), W, S, mass="sandwich")
        m_ex = idf.metrics(pool, PLATE, (0.6, 0.05), W, S, mass="exact")
        assert not np.isclose(m_sw["crlb_xi_mm"], m_ex["crlb_xi_mm"], rtol=0.05)

    def test_sandwich_reaches_exact_resolve(self, pool):
        """정확재해(비섭동)도 샌드위치 질량장을 쓴다 — `bool(mass)` 규약에 걸리지 않는지."""
        from impeller_fingerprint import validity as val
        e_sw = val.e_pert_abs(PLATE, pool, MODES, 0.6, 0.05, W, mass="sandwich")
        e_ex = val.e_pert_abs(PLATE, pool, MODES, 0.6, 0.05, W, mass="exact")
        assert not np.allclose(e_sw, e_ex, rtol=0.05)

    def test_exact_law_outputs_unchanged_by_refactor(self, pool):
        """회귀 고정 — 정확결합 경로의 값이 리팩터로 바뀌지 않았다(설계서 §11.8 교훈).

        `MASS_LAWS` 도입 전 `HEAD:forward.py`와 (ξ,S̄) 4셀 × 결합비 3종 × 정확재해에서
        **비트단위 동일**함을 확인하고 그 중 한 값을 고정한다. 이 값은 두께에 무관하므로
        (레일 t = 1.6 → 2.0 mm) 실측 2치수 반영 후에도 **불변**이다(F58).
        """
        eta = fwd.eta_bar_linear_mass(pool, 0.5, 0.05, W, PLATE, coupling="exact")
        assert float(eta[0]) == pytest.approx(-0.019943474503819385, rel=1e-12)


class TestAsBuiltSandwichLaw:
    """실측 t_f/s = 0.196은 얇은면판 극한이 아니다 — 정확 단면으로 법칙을 다시 낸다(F60)."""

    T_F, S = geo.T_SHEET, geo.FACE_SEPARATION

    def test_production_section_matches_test_helper(self):
        """`forward.sandwich_I`가 테스트 헬퍼(독립 구현)와 일치 — 단면식 교차검증."""
        for p in (0.0, 0.05, 0.25, 0.5):
            assert float(fwd.sandwich_I(p, self.T_F, self.S)) == pytest.approx(
                _I_two_flange(p, self.T_F, self.S), rel=1e-12)

    def test_thin_face_limit_recovers_published_law(self):
        """t_f/s → 0에서 as-built 법칙이 공표 법칙 d_M = d_K/(1+d_K)로 수렴한다."""
        for ratio in (1e-2, 1e-3, 1e-4):
            t_f, s = 1.0, 1.0 / ratio
            for p in (0.05, 0.25, 0.5):
                d_K = float(fwd.sandwich_dk_from_depth(p, t_f, s))
                d_M = 0.5 * float(fwd.sandwich_depth_from_dk(d_K, t_f, s))
                assert d_M == pytest.approx(d_K / (1 + d_K), rel=5 * ratio)

    def test_depth_inversion_round_trip(self):
        p = np.array([0.01, 0.05, 0.1, 0.25, 0.5, 0.8])
        d_K = fwd.sandwich_dk_from_depth(p)
        assert np.allclose(fwd.sandwich_depth_from_dk(d_K), p, atol=1e-10)

    @pytest.mark.parametrize("p,ratio_lo,ratio_hi", [(0.05, 1.38, 1.40),
                                                     (0.1, 1.36, 1.38),
                                                     (0.5, 1.19, 1.20)])
    def test_leading_term_underestimates_dK_at_measured_ratio(self, p, ratio_lo, ratio_hi):
        """선행차수 p/(2−p)는 실측 t_f/s에서 d_K를 19–39 % 과소평가한다."""
        d_K = float(fwd.sandwich_dk_from_depth(p))
        assert ratio_lo <= d_K / (p / (2 - p)) <= ratio_hi

    def test_published_law_is_an_upper_bound_on_mass(self):
        """공표 법칙은 질량효과의 **상계**(정본 §3.3) — as-built는 그보다 작다."""
        d_K = np.array([0.01, 0.05, 0.1, 0.2, 0.4])
        ab = fwd.mass_field_sandwich_asbuilt(d_K)
        thin = fwd.mass_field_sandwich(d_K)
        mono = fwd.mass_field_exact(d_K)
        assert np.all(ab < thin) and np.all(ab > mono)
        zeta = ab / d_K
        assert np.all(zeta > 0.60) and np.all(zeta < 0.72)

    def test_asbuilt_law_wired_into_maps(self):
        pool_ = [ker.mode_kernel(PLATE, m=m, n=n, n_grid=1001) for m, n in MODES]
        e_ab = fwd.eta_bar_linear_mass(pool_, 0.6, 0.05, W, PLATE,
                                       coupling="sandwich_asbuilt")
        e_thin = fwd.eta_bar_linear_mass(pool_, 0.6, 0.05, W, PLATE,
                                         coupling="sandwich")
        e_mono = fwd.eta_bar_linear_mass(pool_, 0.6, 0.05, W, PLATE, coupling="exact")
        assert not np.allclose(e_ab, e_thin, rtol=1e-3)
        assert not np.allclose(e_ab, e_mono, rtol=1e-3)
        from impeller_fingerprint import validity as val
        v_ab = val.e_pert_abs(PLATE, pool_, MODES, 0.6, 0.05, W,
                              mass="sandwich_asbuilt")
        v_thin = val.e_pert_abs(PLATE, pool_, MODES, 0.6, 0.05, W, mass="sandwich")
        assert not np.allclose(v_ab, v_thin, rtol=1e-3)


class TestThicknessInvarianceOfIdentifiability:
    """F58 — 실측 두께 반영(레일 t 1.6 → 2.0 mm)이 식별성 절대값까지 바꾸지 않는다.

    η̄는 무차원이고 σ_η = 2c도 상대량이므로 J·F·CRLB[mm]가 **두께에 무관**하다.
    바뀌는 것은 절대주파수(∝t)와 그것이 결정하는 **측정가능 대역**뿐이다(F59).
    """

    OLD = geo.Plate(a=0.0154, b=0.03656, t=0.0016, E=193e9, rho=7930.0, nu=0.29)

    def _pool(self, plate):
        return [ker.mode_kernel(plate, m=m, n=n, n_grid=2001) for m, n in MODES]

    def test_crlb_and_cond_are_bit_identical(self, pool):
        old = self._pool(self.OLD)
        S = noi.sigma_y(len(MODES), 1e-3)
        for xi in (0.2, 0.5, 0.8, 0.95):
            a = idf.metrics(old, self.OLD, (xi, 0.05), W, S, mass="exact")
            b = idf.metrics(pool, PLATE, (xi, 0.05), W, S, mass="exact")
            for key in ("crlb_xi_mm", "crlb_s_pp", "cond2", "det_F", "sigma_min"):
                assert a[key] == pytest.approx(b[key], rel=1e-14), (xi, key)

    def test_frequencies_scale_exactly_with_thickness(self, pool):
        old = self._pool(self.OLD)
        for ko, kn in zip(old, pool):
            assert kn.f / ko.f == pytest.approx(PLATE.t / self.OLD.t, rel=1e-12)
            assert kn.Lambda == pytest.approx(ko.Lambda, rel=1e-14)


class TestExactClosureIsPointwise:
    """생산 순방향 맵의 결합은 **손상장에 점별로 적용된 정확식**이다(정본 Eq. 7).

    2026-08-25 외부 검토: 원고가 정확식을 정의해 놓고 인쇄된 Eq. (7)은 dₘ ≈ dₖ/3
    저심각도 근사였다. 코드가 어느 쪽인지가 결론을 가르므로 검정으로 고정한다.
      (T-P1) `coupling="exact"`는 스칼라 비가 아니라 d_M(r) = 1−(1−d_K(r))^{1/3}을 적분한다.
      (T-P2) 1/3 근사와 정확식은 림 근처 고심각도에서 유의하게 다르다 — 근사로 대체 불가.
    """

    def test_exact_branch_integrates_the_pointwise_closure(self):
        import numpy as np
        from impeller_fingerprint import cli, forward as fwd
        pool, P, w = cli._pool(), cli.PLATE, cli.W_GAUSS
        xi, s = 0.8, 0.15
        got = fwd.eta_bar_linear_mass(pool, xi, s, w, P, coupling="exact")
        # 같은 식을 테스트 안에서 독립적으로 조립한다(스칼라 비가 아님을 확인)
        hand = []
        for k in pool:
            d = fwd._damage_on(k.r, xi, s, w, P)
            d_m = 1.0 - (1.0 - np.clip(d, 0, 1 - 1e-12)) ** (1.0 / 3.0)
            hand.append(-np.trapezoid(k.gamma * d, k.r)
                        + np.trapezoid(k.gamma_mass * d_m, k.r))
        assert got == pytest.approx(np.array(hand), rel=1e-12)

    def test_one_third_limit_is_not_a_substitute_near_the_rim(self):
        import numpy as np
        from impeller_fingerprint import cli, forward as fwd
        pool, P, w = cli._pool(), cli.PLATE, cli.W_GAUSS
        e = fwd.eta_bar_linear_mass(pool, 0.8, 0.15, w, P, coupling="exact")
        a = fwd.eta_bar_linear_mass(pool, 0.8, 0.15, w, P, coupling=1 / 3)
        rel = np.abs(a / e - 1)
        assert rel.max() > 0.5, rel          # m = 3에서 80 % 규모
        assert rel[0] > 0.2, rel             # m = 0에서도 20 % 이상
