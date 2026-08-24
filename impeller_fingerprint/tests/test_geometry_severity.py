"""geometry·severity 테스트 — 정의 동결(설계서 §4)이 코드에서 지켜지는지.

설계서 §4: S̄_D = S/(b−a), ξ_d = (r_d−a)/(b−a), d(r)=(S/(w√π))exp(−((r−r_d)/w)²), ∫d dr = S.
"""
import math

import numpy as np
import pytest

from impeller_fingerprint import geometry as geo
from impeller_fingerprint import severity as sev


class TestGeometry:
    def test_measured_two_dimensions(self):
        """2026-08-13 실측: 판두께 1.0 mm, 유로폭 4.1 mm ⇒ 전체 6.1, 중립면 간격 5.1 mm.

        이전 해석("4.1 = 전체두께" → t_f 0.8 / s 4.1)은 폐기됐다. 이 테스트가 그 회귀를 막는다.
        """
        assert geo.T_SHEET == pytest.approx(0.0010)
        assert geo.B2_CHANNEL == pytest.approx(0.0041)
        assert geo.RIM_TOTAL == pytest.approx(0.0061)
        assert geo.FACE_SEPARATION == pytest.approx(0.0051)
        assert geo.SANDWICH.t_face == pytest.approx(geo.T_SHEET)
        assert geo.SANDWICH.sep == pytest.approx(geo.FACE_SEPARATION)
        assert geo.VANE.h == pytest.approx(geo.T_SHEET)     # 베인도 같은 판재

    def test_vane_analytic_fundamental_matches_paper(self):
        """정본 §3.1: 베인 EB 캔틸레버(h = t_f = 1.0 mm) 해석 f₁ = 881.6 Hz.

        옛 값 1057.9 Hz는 h = 1.2 mm 가정의 값이고, f₁ ∝ h이므로 1057.9×(1.0/1.2) = 881.6이다.
        """
        f = geo.VANE.eb_frequencies(n=3)
        assert f[0] == pytest.approx(881.6, abs=0.3)
        assert f[0] == pytest.approx(1057.9 * (1.0 / 1.2), rel=2e-4)
        # 고차 비 (βL)² 비율
        assert f[1] / f[0] == pytest.approx((4.6940911 / 1.8751041) ** 2, rel=1e-6)

    def test_disk_extent_matches_paper(self):
        """정본 §3.1: 반경 구간 b−a ≈ 21.2 mm."""
        assert geo.DISK.extent == pytest.approx(0.0212, abs=0.0002)

    def test_uniform_plate_rigidity(self):
        p = geo.DISK
        assert p.D == pytest.approx(p.E * p.t ** 3 / (12 * (1 - p.nu ** 2)), rel=1e-12)
        assert p.rhoh == pytest.approx(p.rho * p.t, rel=1e-12)

    def test_sandwich_props_formula(self):
        """샌드위치 유효물성(정본 §3.1·논문1 §2): D_eff = E t_f s²/[2(1−ν²)], ρh = 2ρ t_f."""
        s = geo.SANDWICH
        assert s.D == pytest.approx(
            s.E * s.t_face * s.sep ** 2 / (2 * (1 - s.nu ** 2)), rel=1e-12)
        assert s.rhoh == pytest.approx(2 * s.rho * s.t_face, rel=1e-12)

    def test_rail_areal_mass_equals_sandwich(self):
        """설계서 §5.2: 레일 균일판의 면적질량 = 샌드위치 면적질량(2 t_face).

        실측 반영 후에도 t = 2 t_f = 2.0 mm로 유지된다 — §5.2의 정당화가 살아 있다.
        """
        assert geo.DISK.t == pytest.approx(2 * geo.SANDWICH.t_face, rel=1e-12)
        assert geo.DISK.rhoh == pytest.approx(geo.SANDWICH.rhoh
                                              * (geo.DISK.rho / geo.SANDWICH.rho),
                                              rel=1e-12)

    def test_impeller_cad_spec_uses_measured_channel_gap(self):
        """F62 — CAD 스펙이 유로폭을 **직접** 받고 전체두께는 파생값이다."""
        from impeller_fingerprint.impeller_cad import ImpellerSpec
        sp = ImpellerSpec()
        sp.check()
        assert sp.gap == pytest.approx(geo.B2_CHANNEL)
        assert sp.total_thickness == pytest.approx(geo.RIM_TOTAL)
        assert sp.face_separation == pytest.approx(geo.FACE_SEPARATION)
        assert (sp.t_front, sp.t_back, sp.t_vane) == pytest.approx(
            (geo.T_SHEET, geo.T_SHEET, geo.T_SHEET))


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
