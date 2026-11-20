"""geometry·severity 테스트 — 정의 동결(설계서 §4)이 코드에서 지켜지는지.

설계서 §4: S̄_D = S/(b−a), ξ_d = (r_d−a)/(b−a), d(r)=(S/(w√π))exp(−((r−r_d)/w)²), ∫d dr = S.
"""
import math

import numpy as np
import pytest

from impeller_fingerprint import geometry as geo
from impeller_fingerprint import severity as sev


class TestGeometry:
    def test_canon_g3_2026_10_30(self):
        """G3 정본(2026-10-30): 표준 두께 앞판 0.40·뒤판 0.60, 간격 3.00, 깃 1.0 mm(2번 시편), 41.32 g.

        08-13 기하(양면 1.0 mm·유로 4.1 mm)는 질량 68.7 g을 내 실측과 맞지 않아 폐기했고, G1(10-16, 1번 시편
        0.375/0.59/3.18/1.2 mm)은 `ASBUILT_G1`에 재현용으로 남겼다. 이 테스트가 그 회귀를 막는다.
        """
        assert geo.T_FRONT == pytest.approx(0.40e-3)
        assert geo.T_BACK == pytest.approx(0.60e-3)
        assert geo.GAP == pytest.approx(3.00e-3)
        assert geo.T_VANE == pytest.approx(1.0e-3)
        assert geo.VANE_WIDTH == pytest.approx(3.0e-3)
        g1 = geo.ASBUILT_G1
        assert (g1.t_front, g1.t_back, g1.gap, g1.t_vane) == pytest.approx((0.375e-3, 0.59e-3, 3.18e-3, 1.2e-3))
        assert (g1.a, g1.b, g1.r_bore, g1.collar_t, g1.mass_g) == pytest.approx(
            (0.014955, 0.036515, 0.006335, 0.000425, 40.2059))
        assert g1.collar_h is None and geo.VANE_WIDTH_G1 == pytest.approx(3.2e-3)
        assert geo.RIM_TOTAL == pytest.approx(geo.GAP + geo.T_FRONT + geo.T_BACK)
        assert geo.FACE_SEPARATION == pytest.approx(geo.GAP + 0.5 * (geo.T_FRONT + geo.T_BACK))
        assert (geo.SANDWICH.t_front, geo.SANDWICH.t_back, geo.SANDWICH.gap) == pytest.approx(
            (geo.T_FRONT, geo.T_BACK, geo.GAP))
        assert geo.VANE.h == pytest.approx(geo.T_VANE)
        ab = geo.ASBUILT
        assert (ab.a, ab.b, ab.r_bore, ab.collar_t, ab.collar_h, ab.mass_g) == pytest.approx(
            (0.01489, 0.036465, 0.00631, 0.00052, 0.0050, 41.3186))

    def test_geom0813_is_preserved_for_reproduction(self):
        g = geo.GEOM_0813
        assert (g.t_sheet, g.b2_channel, g.t_vane) == pytest.approx((0.0010, 0.0041, 0.0010))
        assert g.face_separation == pytest.approx(0.0051)
        assert g.rim_total == pytest.approx(0.0061)
        assert g.disk().t == pytest.approx(0.0020)
        assert g.vane().h == pytest.approx(0.0010)
        old = g.sandwich()
        assert old.D == pytest.approx(old.E * g.t_sheet * old.sep ** 2 / (2 * (1 - old.nu ** 2)),
                                      rel=1e-12)

    def test_vane_analytic_fundamental_matches_paper(self):
        """베인 EB 캔틸레버(G3 h = 1.0 mm) 해석 f₁ = 881.6 Hz(08-13과 같은 두께); G1 h = 1.2 mm는 1057.9 Hz (f₁ ∝ h)."""
        f = geo.VANE.eb_frequencies(n=3)
        assert f[0] == pytest.approx(881.6, abs=0.3)
        assert geo.GEOM_0813.vane().eb_frequencies(1)[0] == pytest.approx(881.6, abs=0.3)
        g1 = geo.Beam(L=geo.VANE.L, h=geo.ASBUILT_G1.t_vane, E=geo.VANE.E, rho=geo.VANE.rho, nu=geo.VANE.nu)
        assert g1.eb_frequencies(1)[0] == pytest.approx(1057.9, abs=0.3)
        assert g1.eb_frequencies(1)[0] == pytest.approx(f[0] * (1.2 / 1.0), rel=2e-4)
        # 고차 비 (βL)² 비율
        assert f[1] / f[0] == pytest.approx((4.6940911 / 1.8751041) ** 2, rel=1e-6)

    def test_disk_extent_matches_paper(self):
        """정본 §3.1: 반경 구간 b−a ≈ 21.2 mm (레일 반경은 유지, G2 결정 D1)."""
        assert geo.DISK.extent == pytest.approx(0.0212, abs=0.0002)
        assert (geo.DISK.a, geo.DISK.b) == pytest.approx((0.0154, 0.03656))

    def test_uniform_plate_rigidity(self):
        p = geo.DISK
        assert p.D == pytest.approx(p.E * p.t ** 3 / (12 * (1 - p.nu ** 2)), rel=1e-12)
        assert p.rhoh == pytest.approx(p.rho * p.t, rel=1e-12)

    def test_sandwich_props_formula(self):
        """샌드위치 유효물성(얇은면판): D_eff = E μ d²/(1−ν²), μ = t_f t_b/(t_f+t_b), ρh = ρ(t_f+t_b)."""
        s = geo.SANDWICH
        mu = s.t_front * s.t_back / (s.t_front + s.t_back)
        assert s.mu == pytest.approx(mu, rel=1e-12)
        assert s.D == pytest.approx(s.E * mu * s.sep ** 2 / (1 - s.nu ** 2), rel=1e-12)
        assert s.rhoh == pytest.approx(s.rho * (s.t_front + s.t_back), rel=1e-12)

    def test_rail_areal_mass_equals_sandwich(self):
        """설계서 §5.2: 레일 균일판의 면적질량 = 실물 두 면판의 면적질량(t = t_front + t_back = 1.00 mm, G3; G1 0.965)."""
        assert geo.DISK.t == pytest.approx(geo.T_FRONT + geo.T_BACK, rel=1e-12)
        assert geo.DISK.rhoh == pytest.approx(geo.SANDWICH.rhoh
                                              * (geo.DISK.rho / geo.SANDWICH.rho),
                                              rel=1e-12)

    def test_impeller_cad_spec_uses_asbuilt_canon(self):
        """CAD 스펙이 간격을 **직접** 받고 전체두께는 파생값 — 기본은 정본(G3)."""
        from impeller_fingerprint.impeller_cad import ImpellerSpec
        sp = ImpellerSpec()
        sp.check()
        ab = geo.ASBUILT
        assert sp.gap == pytest.approx(ab.gap)
        assert sp.total_thickness == pytest.approx(ab.gap + ab.t_front + ab.t_back)
        assert sp.face_separation == pytest.approx(ab.gap + 0.5 * (ab.t_front + ab.t_back))
        assert (sp.t_front, sp.t_back, sp.t_vane) == pytest.approx(
            (ab.t_front, ab.t_back, ab.t_vane))
        assert (sp.r_eye, sp.r_out, sp.wrap_deg, sp.r_vane_in) == pytest.approx(
            (ab.a, ab.b, ab.wrap_deg, ab.r_vane_in))


class TestSeverity:
    def test_s_bar_roundtrip(self):
        extent = geo.DISK.extent
        for s_bar in (0.001, 0.0567, 0.3):
            S = sev.S_from_s_bar(s_bar, extent)
            assert sev.s_bar_from_S(S, extent) == pytest.approx(s_bar, rel=1e-12)

    def test_pilot_severity_reparam(self):
        """파일럿 S = 1.2 mm → S̄_D ≈ 5.7 % (b−a = 21.16 mm)."""
        s_bar = sev.s_bar_from_S(0.0012, geo.DISK.extent)
        assert s_bar == pytest.approx(0.0567, abs=0.0005)

    def test_damage_field_integrates_to_S(self):
        a, b = geo.DISK.a, geo.DISK.b
        r = np.linspace(a, b, 20001)
        S, w = 0.0012, 0.003
        d = sev.damage_field(r, r_d=0.5 * (a + b), S=S, w=w)
        assert np.trapezoid(d, r) == pytest.approx(S, rel=2e-4)

    def test_damage_field_peak_and_shape(self):
        a, b = geo.DISK.a, geo.DISK.b
        r = np.linspace(a, b, 4001)
        S, w, r_d = 0.0012, 0.003, 0.5 * (a + b)
        d = sev.damage_field(r, r_d=r_d, S=S, w=w)
        assert d.max() == pytest.approx(S / (w * math.sqrt(math.pi)), rel=1e-6)
        assert r[int(np.argmax(d))] == pytest.approx(r_d, abs=(b - a) / 4000)

    def test_xi_r_roundtrip(self):
        a, b = geo.DISK.a, geo.DISK.b
        for xi in (0.0, 0.2, 0.5, 1.0):
            r = sev.xi_to_r(xi, a, b)
            assert sev.r_to_xi(r, a, b) == pytest.approx(xi, abs=1e-12)

    def test_damage_field_from_xi_matches_S_form(self):
        """(ξ_d, S̄_D) 파라미터화가 (r_d, S) 형태와 동일한 장을 준다."""
        a, b = geo.DISK.a, geo.DISK.b
        r = np.linspace(a, b, 4001)
        xi, s_bar, w = 0.35, 0.0567, 0.003
        d1 = sev.damage_field_xi(r, xi_d=xi, s_bar=s_bar, w=w, a=a, b=b)
        d2 = sev.damage_field(r, r_d=sev.xi_to_r(xi, a, b),
                              S=sev.S_from_s_bar(s_bar, b - a), w=w)
        assert np.allclose(d1, d2, rtol=1e-12, atol=0)
