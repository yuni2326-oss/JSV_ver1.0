"""E1-C(베인 쿠폰 노치 시험 사전설계) 회귀검정 — 시편 상수, 격자 형상, 모드 분류, 설계 산술."""
import math
import shutil

import numpy as np
import pytest

from impeller_fingerprint import geometry as geo
from impeller_fingerprint.rev1 import e1c_coupon as e
from impeller_fingerprint.rev1 import n3_crack as n3


def test_healthy_eb_f1_is_manuscript_value():
    """정본 깃 시편 30 × 3.0 × 1.0 mm(G3): EB f₁ = 881.6 Hz (G1 30 × 3.2 × 1.2 mm는 1057.9 Hz)."""
    assert (e.L_C, e.B_C, e.H_C) == pytest.approx((0.030, geo.VANE_WIDTH, geo.T_VANE))
    f1 = geo.Beam(L=e.L_C, h=e.H_C, E=e.E_C, rho=e.RHO_C, nu=e.NU_C).eb_frequencies(1)[0]
    assert f1 == pytest.approx(881.6, abs=0.05)


def test_coupon_width_cancels_in_frequency_ratios():
    # 폭 b는 EI·GA·ρA·ρI·균열유연도에서 같은 배율로 상쇄된다. 균열 연결 C⁻¹의 조건수 때문에
    # 고유해 반올림이 1e-7 규모로 남는다(N3 정본 대조와 같은 규모).
    mine = e.tm_ratio(0.2, 0.5, "tada", n_elem=300)
    n3_ratio = n3.crack_ratios([(0.2, n3.crack_compliance(0.5, "tada", True))], n_elem=300)
    assert np.allclose(mine, n3_ratio, rtol=1e-6)


def test_torsion_constant_known_limits():
    # 정사각 단면 β = 0.1406, 얇은 판 극한 J → b h³/3 (1 − 0.630 h/b)
    assert e.torsion_J(1.0, 1.0) == pytest.approx(0.1406, abs=2e-4)
    b, h = 50.0, 1.0
    assert e.torsion_J(b, h) == pytest.approx(b * h ** 3 / 3 * (1 - 0.6302 * h / b), rel=1e-3)


def test_slot_mesh_removes_exact_volume_and_keeps_nodes():
    kw = dict(xc_over_L=0.2, kerf=0.25e-3, depth_abar=0.3, n_left=6, n_kerf=2, n_right=12,
              n_below=3, n_above=3, ny=2)
    co0, cn0, _ = e.coupon_mesh3d(**kw, notch=False)
    co1, cn1, info = e.coupon_mesh3d(**kw, notch=True)
    removed = e.hex_volume(co0, cn0) - e.hex_volume(co1, cn1)
    depth = 0.3 * e.H_C                                  # ā = 0.3 of the vane thickness
    assert removed == pytest.approx(0.25e-3 * depth * e.B_C, rel=1e-10)
    assert info["kerf_m"] == pytest.approx(0.25e-3, rel=1e-12)
    assert info["depth_m"] == pytest.approx(depth, rel=1e-12)
    assert info["xc_m"] == pytest.approx(0.2 * e.L_C, rel=1e-12)
    keys0 = {tuple(np.round(p, 12)) for p in co0}
    assert all(tuple(np.round(p, 12)) in keys0 for p in co1)


@pytest.fixture(scope="module")
def coarse_modes():
    co, cn, _ = e.coupon_mesh3d(30, 4, 2)
    return e.solve3d(co, cn, n_modes=7)


def test_mode_classification_order(coarse_modes):
    cls = e.classify3d(coarse_modes)
    got = [(c["kind"], c["order"]) for c in cls[:6]]
    # G3 30 × 3.0 × 1.0 mm(폭/두께 3.0): 3차 면외 굽힘(≈15.3 kHz)이 2차 폭방향 굽힘(≈15.8 kHz)보다 낮다
    # (08-13 30 × 4.1 × 1.0 mm와 같은 순서). G1 30 × 3.2 × 1.2 mm(2.67)는 edge2가 flap3보다 낮았다.
    assert got == [("flap", 1), ("edge", 1), ("flap", 2), ("torsion", 1), ("flap", 3), ("edge", 2)]


def test_mode2_null_from_3d_near_beam_value(coarse_modes):
    i2 = e.find_mode(e.classify3d(coarse_modes), "flap", 2)
    assert e.null_from_3d(coarse_modes, i2) == pytest.approx(0.2166, abs=0.004)


def test_mass_limit_matches_first_order_formula(coarse_modes):
    i1 = e.find_mode(e.classify3d(coarse_modes), "flap", 1)
    ml = e.mass_limits(coarse_modes, i1)
    # 질량정규화 φ: m_limit = 2·budget/φ²_max, 1차 끝단 유효질량 ≈ 0.243 × 쿠폰질량(30 × 3.2 × 1.2 mm, 0.92 g)
    assert ml["m_limit_surface_mg_dff0.1pct"] == pytest.approx(
        2e-3 / ml["phi2_surface_max_per_kg"] * 1e6, rel=1e-12)
    m_coupon_g = e.RHO_C * e.L_C * e.B_C * e.H_C * 1e3
    assert ml["m_eff_surface_g"] == pytest.approx(0.2427 * m_coupon_g, rel=0.04)


@pytest.mark.skipif(shutil.which("gmsh") is None, reason="gmsh 없음")
def test_unotch_geometry_area():
    r = e.unotch_2d(0.2, 0.5, 0.25e-3)
    assert abs(r["removed_area_rel_err"]) < 2e-3
    assert 0.80 < r["ratio"][0] < 0.95 and abs(r["ratio"][1] - 1) < 3e-3


def test_required_repeats_arithmetic():
    assert e.required_repeats(0.021, 3e-3, 3.0, 2.0) == 1
    assert e.required_repeats(0.00141, 3e-3, 1.645, 2.0) == math.ceil(2 * (1.645 * 3e-3 / 0.00141) ** 2)
    assert e.required_repeats(-0.001, 3e-3, 3.0, 2.0) == -1


def test_crlb_scales_inversely_with_snr():
    a = e.crlb_damped_sine(5500.0, 2e-3, 204800.0, 0.3, 100.0)
    b = e.crlb_damped_sine(5500.0, 2e-3, 204800.0, 0.3, 200.0)
    assert np.isfinite(a) and a > 0
    assert b == pytest.approx(a / 2, rel=1e-9)


def test_contrast_equals_r21_difference():
    r = e.tm_ratio(0.2, 0.5, "tada")
    c = e.fingerprint_cols(r, 0.2)
    assert c["d2_pct"] < 0 and c["smooth_d2_pct"] < 0
    # δ₁ 정합(brentq xtol)의 잔차만큼 다르다
    assert c["contrast2_pct"] == pytest.approx(c["r21_pct"] - c["smooth_r21_pct"], abs=1e-6)


def test_damped_sine_fit_recovers_frequency_and_damping():
    fs, T = 204800.0, 0.5
    t = np.arange(int(fs * T)) / fs
    rng = np.random.default_rng(7)

    def mode(f, z, A, ph):
        wn = 2 * math.pi * f
        return A * np.exp(-z * wn * t) * np.cos(wn * math.sqrt(1 - z * z) * t + ph)
    x = mode(886.8, 3e-3, 2.0, 0.4) + mode(5528.2, 2e-3, 1.0, -1.1) + 0.003 * rng.standard_normal(t.size)
    r = e.fit_damped_sine(x, fs, 5500.0)
    # 재장착 산포 시나리오(3e-4–3e-3)보다 두 자릿수 이상 작아야 한다
    assert r["f_Hz"] == pytest.approx(5528.2, rel=1e-5)
    assert r["zeta"] == pytest.approx(2e-3, rel=0.02)
    r1 = e.fit_damped_sine(x, fs, 880.0)
    assert r1["f_Hz"] == pytest.approx(886.8, rel=1e-5)


def test_damped_sine_fit_noiseless_three_modes():
    fs, T = 204800.0, 0.5
    t = np.arange(int(fs * T)) / fs

    def mode(f, z, A, ph):
        wn = 2 * math.pi * f
        return A * np.exp(-z * wn * t) * np.cos(wn * math.sqrt(1 - z * z) * t + ph)
    x = mode(886.8, 3e-3, 2.0, 0.4) + mode(5528.2, 2e-3, 1.0, -1.1) + 0.6 * mode(15366.0, 2e-3, 1.0, 0.2)
    for fg, f in ((880.0, 886.8), (5500.0, 5528.2), (15300.0, 15366.0)):
        assert e.fit_damped_sine(x, fs, fg)["f_Hz"] == pytest.approx(f, rel=1e-5)
