"""N6 부착 센서 질량부하·근거리 마이크(E2-I, LDV 없이) — `rev1/n6_sensors.py`."""
import math

import numpy as np
import pytest

from impeller_fingerprint.rev1 import n1_mesh as nm
from impeller_fingerprint.rev1 import n6_freefree as n6
from impeller_fingerprint.rev1 import n6_sensors as ns


def _coarse_spec(**kw):
    base = nm.FitSpec(n_r=8, n_bay=5)
    return n6.FFSpec(base=base, n_hub=2, n_z_collar=2, **kw)


@pytest.fixture(scope="module")
def coarse():
    ff = _coarse_spec()
    fm = n6.build(ff)
    km = n6.assemble(fm)
    md, _ = n6.solve(km, 14)
    obs = n6.classify(fm, md)
    return ff, fm, km, md, obs


def test_with_mass_adds_rigid_translation_mass(coarse):
    _, fm, km, *_ = coarse
    nodes = ns.footprint(fm, "back", 0.010, 0.0, 2.5e-3)
    assert nodes.size >= 1
    km2 = ns.with_mass(km, nodes, 2e-3)
    for c in range(3):
        u = np.zeros(km.M.shape[0])
        u[c::3] = 1.0
        assert float(u @ (km2.M @ u)) - float(u @ (km.M @ u)) == pytest.approx(2e-3, rel=1e-9)


def test_added_mass_only_lowers_frequencies_and_splits_doublets(coarse):
    ff, fm, km, md, obs = coarse
    nodes = ns.footprint(fm, "back", ff.a + 0.6 * ff.extent, 0.0, 2.5e-3)
    md2, _ = n6.solve(ns.with_mass(km, nodes, 1e-3), 14)
    # 질량을 더하면 정렬된 고유진동수가 모두 내려간다(끼움 정리) — 이것이 엄밀한 검사다.
    assert np.all(md2.freqs[:12] <= md.freqs[:12] * (1 + 1e-12))
    eta, split, mac, idx = ns.match(md, md2, obs)
    ok = np.isfinite(eta)
    assert ok.sum() >= len(obs) - 1
    # MAC 짝짓기는 근접 무리에서 섞일 수 있다: G3 성긴 격자는 h1:m1n1과 h3:m3n1이 0.04 % 떨어져 있어
    # 비대칭 질량이 둘을 섞고, 짝지은 성분 평균이 +1.4e-5 남짓 올라간 것처럼 보인다.
    assert np.all(eta[ok] <= 1e-4)
    pairs = [j for j, o in enumerate(obs) if o.deg == 2 and np.isfinite(split[j])]
    assert pairs and np.nanmax(split[pairs]) > 1e-6
    assert all(len(idx[j]) == obs[j].deg for j in np.nonzero(ok)[0])


def test_small_mass_matches_first_order(coarse):
    """1 mg: 재해석 쌍평균 η = −Σ_j (m/N)|φ(x_j)|²의 성분 평균(1차)."""
    ff, fm, km, md, obs = coarse
    nodes = ns.footprint(fm, "front", ff.a + 0.9 * ff.extent, 0.3, 2.0e-3)
    m = 1e-6
    md2, _ = n6.solve(ns.with_mass(km, nodes, m), 14)
    eta, _, _, _ = ns.match(md, md2, obs)
    pred = ns.first_order_mass_eta(md, obs, nodes, m)
    big = np.abs(pred) > 1e-6
    assert big.any()
    np.testing.assert_allclose(eta[big], pred[big], rtol=0.02)


def test_rayleigh_point_patch():
    """한 점 넓이 A, 변위 w: |p| = ω²ρ₀ w A /(2πR)."""
    xs = np.array([[0.0, 0.0, 0.0]])
    f, R = 3000.0, 0.01
    p = ns.rayleigh_pressure(xs, np.array([1e-6]), np.array([1e-6]), f, np.array([0.0, 0.0, R]))
    want = (2 * math.pi * f) ** 2 * ns.RHO_AIR * 1e-6 * 1e-6 / (2 * math.pi * R)
    assert abs(p) == pytest.approx(want, rel=1e-12)


def test_face_areas_sum_to_annulus(coarse):
    ff, fm, *_ = coarse
    assert ns.node_areas(fm, "front").sum() == pytest.approx(math.pi * (ff.b ** 2 - (ff.a + ff.collar_t) ** 2),
                                                             rel=0.02)
    assert ns.node_areas(fm, "back").sum() == pytest.approx(math.pi * (ff.b ** 2 - ff.r_bore ** 2), rel=0.02)


def test_mic_on_axis_sees_only_axisymmetric_families(coarse):
    """축 위 마이크: m ≠ 0 계열은 둘레 적분이 0이라 거의 안 들린다(h0 singlet만 남는다)."""
    ff, fm, km, md, obs = coarse
    z_top = ff.z_front[1]
    p = ns.mic_pressure(fm, md, obs, np.array([0.0, 0.0, z_top + 0.02]), face="front")
    p_off = ns.mic_pressure(fm, md, obs, np.array([ff.a + 0.9 * ff.extent, 0.0, z_top + 0.005]), face="front")
    for j, o in enumerate(obs):
        if o.h in (1, 2) or (o.h == 3):
            assert p[j] < 1e-6 * max(p_off.max(), 1e-30) + 1e-9 * p_off[j]
