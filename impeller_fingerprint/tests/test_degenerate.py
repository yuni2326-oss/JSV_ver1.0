"""degenerate 테스트 — 정본 §3.2 축퇴쌍 섭동이론(논문의 spine)을 코드로 고정.

T1: tr H^(m)의 θ₀ 회전 불변성 (pair mean이 방위에 무관 = 반경 역문제의 적격성 근거)
T2: 축대칭 손상 → 분리 0 / 국소 포켓 → 분리 > 0, Δθ = π/m에서 분리 null
추가: 2D 수치적분 경로와 해석 닫힌형 경로의 독립 일치, 배향 ψ_m = θ₀ (mod π/m),
      Δθ=2π·질량항0 극한에서 kernels의 반경 η̄와 일치(모듈 간 정합).
"""
import math
import numpy as np
import pytest

from impeller_fingerprint import degenerate as deg
from impeller_fingerprint import geometry as geo
from impeller_fingerprint import kernels as ker

PLATE = geo.DISK


@pytest.fixture(scope="module")
def pockets():
    return dict(
        mid=deg.Pocket(r1=0.021, r2=0.027, theta0=0.7, dtheta=np.deg2rad(30),
                       depth_frac=0.25),
        rim=deg.Pocket(r1=0.033, r2=0.036, theta0=1.9, dtheta=np.deg2rad(15),
                       depth_frac=0.4),
    )


class TestT1TraceInvariance:
    @pytest.mark.parametrize("m", [1, 2, 3])
    def test_trace_independent_of_azimuth(self, m, pockets):
        base = pockets["mid"]
        traces = []
        for th in np.linspace(0.0, 2 * np.pi, 13):
            p = deg.Pocket(base.r1, base.r2, float(th), base.dtheta, base.depth_frac)
            H = deg.pair_matrix(PLATE, m, p)
            traces.append(np.trace(H))
        traces = np.array(traces)
        assert np.ptp(traces) < 1e-10 * np.abs(traces).max()

    @pytest.mark.parametrize("m", [1, 2, 3])
    def test_pair_mean_independent_of_azimuth(self, m, pockets):
        base = pockets["rim"]
        obs = [deg.observables(PLATE, m, deg.Pocket(base.r1, base.r2, float(th),
                                                    base.dtheta, base.depth_frac))
               for th in (0.0, 0.3, 1.1, 2.7, 5.0)]
        means = np.array([o["eta_bar"] for o in obs])
        assert np.ptp(means) < 1e-10 * np.abs(means).max()


class TestT2Splitting:
    def test_axisymmetric_damage_gives_zero_splitting(self):
        p = deg.Pocket(r1=0.021, r2=0.027, theta0=0.0, dtheta=2 * np.pi,
                       depth_frac=0.25)
        for m in (1, 2, 3):
            o = deg.observables(PLATE, m, p)
            assert abs(o["delta_eta"]) < 1e-12

    def test_localized_pocket_splits(self, pockets):
        for m in (1, 2, 3):
            o = deg.observables(PLATE, m, pockets["mid"])
            assert o["delta_eta"] > 0.0

    @pytest.mark.parametrize("m", [1, 2, 3])
    def test_splitting_null_at_dtheta_pi_over_m(self, m):
        """|B| ∝ sin(mΔθ) → Δθ = π/m에서 분리가 사라진다(방위 2m차 푸리에 null)."""
        p = deg.Pocket(r1=0.022, r2=0.028, theta0=0.4, dtheta=np.pi / m,
                       depth_frac=0.3)
        o = deg.observables(PLATE, m, p)
        assert abs(o["delta_eta"]) < 1e-12
        # 그 근방에서는 0이 아니다
        p2 = deg.Pocket(p.r1, p.r2, p.theta0, np.pi / m * 0.6, p.depth_frac)
        assert deg.observables(PLATE, m, p2)["delta_eta"] > 1e-8

    def test_splitting_grows_with_localization(self):
        """각폭이 좁아질수록(국소화) 분리가 커진다 — 정본 §3.2의 진술."""
        m = 2
        widths = np.deg2rad([60.0, 40.0, 20.0, 10.0])
        vals = [deg.observables(PLATE, m,
                                deg.Pocket(0.022, 0.028, 0.3, float(dw), 0.3))["delta_eta"]
                / deg.observables(PLATE, m,
                                  deg.Pocket(0.022, 0.028, 0.3, float(dw), 0.3))["severity_s_bar"]
                for dw in widths]
        assert all(b > a for a, b in zip(vals, vals[1:])), vals


class TestOrientation:
    @pytest.mark.parametrize("m", [1, 2, 3])
    def test_model_recovers_azimuth_mod_pi_over_m(self, m):
        """모델의 sign(B̄)를 쓰면 배향에서 손상 방위가 복원된다(mod π/m)."""
        for th0 in (0.0, 0.35, 1.2, 2.9):
            p = deg.Pocket(0.022, 0.028, float(th0), np.deg2rad(25), 0.3)
            o = deg.observables(PLATE, m, p)
            period = np.pi / m
            diff = (o["theta0_hat"] - th0) % period
            diff = min(diff, period - diff)
            assert diff < 1e-8, (m, th0, o["theta0_hat"], o["B_signed"])

    def test_observable_orientation_is_half_period_off_when_B_positive(self):
        """관측 배향 psi_lower 자체는 B̄>0이면 손상 방위에서 반주기 어긋난다."""
        p = deg.Pocket(0.022, 0.028, 0.4, np.deg2rad(25), 0.3)
        o = deg.observables(PLATE, 3, p)
        period = np.pi / 3
        off = (o["psi_lower"] - 0.4) % period
        off = min(off, period - off)
        if o["B_signed"] > 0:
            assert off == pytest.approx(0.5 * period, abs=1e-8)
        else:
            assert off < 1e-8

    def test_B_sign_is_not_universal_across_m(self):
        """소견(F1): B̄의 부호가 m에 따라 바뀐다 — 비틀림·질량항이 굽힘항을 이길 수 있다.

        따라서 "손상 위 antinode를 갖는 짝이 더 떨어진다"는 보편적 진술이 아니다.
        """
        p = deg.Pocket(0.022, 0.028, 0.4, np.deg2rad(25), 0.3)
        signs = {m: np.sign(deg.observables(PLATE, m, p)["B_signed"])
                 for m in (1, 2, 3)}
        assert len(set(signs.values())) > 1, signs

    def test_pair_members_bracket_the_mean(self):
        p = deg.Pocket(0.022, 0.028, 0.5, np.deg2rad(25), 0.3)
        o = deg.observables(PLATE, 2, p)
        assert o["eta_minus"] < o["eta_plus"] <= 0.0
        assert o["eta_minus"] == pytest.approx(o["eta_bar"] - 0.5 * o["delta_eta"],
                                               rel=1e-12)
        assert o["eta_plus"] == pytest.approx(o["eta_bar"] + 0.5 * o["delta_eta"],
                                             rel=1e-12)


class TestQuadratureVsClosedForm:
    @pytest.mark.parametrize("m", [1, 2, 3])
    def test_two_independent_paths_agree(self, m, pockets):
        for key in ("mid", "rim"):
            Hc = deg.pair_matrix(PLATE, m, pockets[key])
            Hq = deg.pair_matrix_quadrature(PLATE, m, pockets[key],
                                            n_theta=4001, n_r=2001)
            scale = np.abs(Hc).max()
            assert np.max(np.abs(Hc - Hq)) < 2e-4 * scale, (m, key, Hc, Hq)

    def test_mass_term_included(self, pockets):
        """포켓은 질량도 제거한다 → δM 항이 pair mean을 강성단독보다 덜 떨어뜨린다."""
        p = pockets["mid"]
        with_mass = deg.observables(PLATE, 2, p)["eta_bar"]
        stiff_only = deg.observables(PLATE, 2, p, mass_term=False)["eta_bar"]
        assert with_mass > stiff_only          # 질량제거는 주파수를 올리는 방향


class TestCrossModuleConsistency:
    def test_axisymmetric_stiffness_only_matches_radial_kernel(self):
        """Δθ=2π·질량항 제외 극한에서 pair mean = −∫γ_m d dr (kernels 경로)."""
        m, r1, r2, depth = 2, 0.021, 0.027, 0.2
        p = deg.Pocket(r1=r1, r2=r2, theta0=0.0, dtheta=2 * np.pi, depth_frac=depth)
        eta_pair = deg.observables(PLATE, m, p, mass_term=False)["eta_bar"]

        k = ker.mode_kernel(PLATE, m=m, n=0, n_grid=4001)
        beta_D = 1.0 - (1.0 - depth) ** 3
        d = np.where((k.r >= r1) & (k.r <= r2), beta_D, 0.0)
        eta_kernel = -float(np.trapezoid(k.gamma * d, k.r))
        assert eta_pair == pytest.approx(eta_kernel, rel=2e-3)

    def test_severity_conversion_matches_severity_module(self):
        p = deg.Pocket(0.021, 0.027, 0.0, np.deg2rad(30), 0.25)
        o = deg.observables(PLATE, 2, p)
        from impeller_fingerprint import severity as sev
        expected = sev.pocket_depth_to_severity(0.25, p.r2 - p.r1, PLATE.extent)
        assert o["severity_s_bar_radial"] == pytest.approx(expected, rel=1e-12)


class TestDocumentedRadialFactors:
    """Eq. (5)의 R_U·R_V·R_M을 **부록 A가 적은 식 그대로** 재조립해 코드와 대조한다.

    2026-08-25 외부 검토: "R_U·R_V가 어떤 곡률 조합을 담는지 적지 않으면 Eq. (5)를
    재현할 수 없다 — B̄의 부호가 그 조합으로 정해지므로". 실제로 원고에는 세 인자의 정의가
    없었다. 부록 A에 적는 식을 여기서 독립 조립해 고정한다(문서가 코드와 갈라지면 실패).

        A = R″ + R′/r − m²R/r²,  B = R″,  C = R′/r − m²R/r²,  T = m(R′/r − R/r²)
        R_U = ∫[A² − 2(1−ν)BC] r dr,  R_V = ∫2(1−ν)T² r dr,  R_M = ∫R² r dr
    """

    @pytest.mark.parametrize("m", [1, 2, 3])
    def test_radial_factors_match_the_documented_integrals(self, m):
        plate = geo.DISK
        pocket = deg.Pocket(r1=0.024, r2=0.030, theta0=0.3, dtheta=math.radians(30),
                            depth_frac=0.2)
        co = deg.pair_coefficients(plate, m, pocket)

        r, R, dR, d2R, lam = deg._normalized_shape(plate, m)
        nu = plate.nu
        A = d2R + dR / r - m ** 2 * R / r ** 2
        B = d2R
        C = dR / r - m ** 2 * R / r ** 2
        T = m * (dR / r - R / r ** 2)
        rr = np.linspace(pocket.r1, pocket.r2, 4001)
        def I(f):
            return float(np.trapezoid(np.interp(rr, r, f) * rr, rr))
        assert co["R_U"] == pytest.approx(I(A ** 2 - 2 * (1 - nu) * B * C), rel=1e-12)
        assert co["R_V"] == pytest.approx(I(2 * (1 - nu) * T ** 2), rel=1e-12)
        assert co["R_M"] == pytest.approx(I(R ** 2), rel=1e-12)

    @pytest.mark.parametrize("m", [1, 2, 3])
    def test_mode_normalization_stated_in_the_appendix(self, m):
        """∫∫ ρh φ² r dr dθ = 1 — m > 0에서 ρh·π·∫R² r dr = 1."""
        plate = geo.DISK
        r, R, *_ = deg._normalized_shape(plate, m)
        assert plate.rhoh * math.pi * float(np.trapezoid(R ** 2 * r, r)) == \
            pytest.approx(1.0, rel=1e-10)

    @pytest.mark.parametrize("m", [1, 2, 3])
    def test_bar_coefficients_are_built_from_those_factors(self, m):
        """Ā·B̄가 (R_U+R_V)I₀·(R_U−R_V)I₁ 조합으로 조립된다 — 부호 논의의 근거."""
        plate = geo.DISK
        pocket = deg.Pocket(r1=0.024, r2=0.030, theta0=0.3, dtheta=math.radians(30),
                            depth_frac=0.2)
        co = deg.pair_coefficients(plate, m, pocket)
        kD = -pocket.beta_D * plate.D
        kM = co["lambda_m"] * pocket.beta_M * plate.rhoh
        assert co["A_bar"] == pytest.approx(
            kD * (co["R_U"] + co["R_V"]) * co["I0"] + kM * co["R_M"] * co["I0"], rel=1e-12)
        assert co["B_bar"] == pytest.approx(
            kD * (co["R_U"] - co["R_V"]) * co["I1"] + kM * co["R_M"] * co["I1"], rel=1e-12)
