"""N1 조립체 모듈 회귀 — 거친 격자(6×36, 2차)로 대칭·감육·1차섭동 규약을 고정한다."""
import math

import numpy as np
import pytest

from impeller_fingerprint import forward as fwd
from impeller_fingerprint.rev1 import n1_assembly as n1


@pytest.fixture(scope="module")
def coarse():
    spec = n1.fit_spec(n_r=8, n_bay=5)
    am = n1.build(spec)
    km0 = n1.assemble(am)
    md = n1.solve(km0, 12)
    obs = n1.classify(am, md)
    return spec, am, km0, md, obs


def test_protected_doublets_exactly_degenerate(coarse):
    """정확 C6 격자: 보호 doublet(h=1,2)의 두 성분이 수치정밀도로 같다(사면체 CAD는 0.04–0.96 %)."""
    _, _, _, md, obs = coarse
    pairs = [o for o in obs if o.deg == 2]
    assert pairs, "doublet이 하나도 없다"
    for o in pairs:
        f = md.freqs[o.members]
        assert abs(f[1] - f[0]) / f[0] < 1e-9
        assert o.h in (1, 2)


def test_singlets_are_h0_or_h3(coarse):
    _, _, _, _, obs = coarse
    for o in obs:
        if o.deg == 1:
            assert o.h in (0, 3)


def test_depth_profile_round_trip(coarse):
    """p(r)를 정확 2면판식(실측 비대칭: 앞판 손상)에 다시 넣으면 Gaussian d_K(r)가 복원된다."""
    spec, am, *_ = coarse
    assert am.t_o is not None and am.t_f == spec.t_front and am.t_o == spec.t_back
    r = np.linspace(spec.a, spec.b, 301)
    p = n1.depth_profile_am(am, r, 0.5, 0.05, n1.W_GAUSS)
    dk = n1.dK_profile(r, 0.5, 0.05, n1.W_GAUSS, spec.a, spec.b)
    assert np.allclose(fwd.two_face_dk_from_depth(p, am.t_f, am.t_o, am.gap), dk, atol=1e-10)


def test_depth_profile_equal_faces_uses_0813_formula():
    """08-13 기하(같은 두께)에서는 기존 같은-두께 식이 그대로 쓰인다 — r4.0 산출물 재현 경로."""
    spec = n1.nm.FitSpec.geom0813(n_r=8, n_bay=5)
    am = n1.build(spec)
    assert am.t_o is None
    s_sep = spec.t_front + spec.channel
    r = np.linspace(spec.a, spec.b, 101)
    a = n1.depth_profile_am(am, r, 0.5, 0.05, n1.W_GAUSS)
    b = n1.depth_profile(r, 0.5, 0.05, n1.W_GAUSS, spec.a, spec.b, am.t_f, s_sep)
    assert np.array_equal(a, b)


def test_morph_removes_expected_volume(coarse):
    """절점 감육의 제거체적 = 2π∫p t_f r dr (조각선형 p의 이산오차 1 % 이내, 거친 격자)."""
    spec, am, km0, *_ = coarse
    xi, sb = 0.5, 0.05
    p = n1.depth_profile_am(am, am.r_v, xi, sb, n1.W_GAUSS)
    km1 = n1.assemble(am, n1.morph(am, p))
    pr = lambda r: n1.depth_profile_am(am, r, xi, sb, n1.W_GAUSS)
    ratio = (km0.volume - km1.volume) / n1.removed_volume_exact(am, pr)
    assert abs(ratio - 1.0) < 0.01
    # 안쪽면(베인 접합)은 움직이지 않는다
    c1 = n1.morph(am, p)
    inner = np.isclose(am.coors[:, 2], am.z_in)
    assert np.array_equal(c1[inner], am.coors[inner])


def test_axisymmetric_damage_keeps_degeneracy(coarse):
    """축대칭 감육은 C6을 보존 → 손상 후에도 doublet 축퇴 유지(분리 = 0)."""
    spec, am, km0, md, obs = coarse
    p = n1.depth_profile_am(am, am.r_v, 0.6, 0.05, n1.W_GAUSS)
    mdd = n1.solve(n1.assemble(am, n1.morph(am, p)), 12)
    rel = np.abs(np.diff(mdd.freqs)) / mdd.freqs[:-1]
    n_pairs = int(np.sum(rel < 1e-9))
    assert n_pairs == sum(1 for o in obs if o.deg == 2)


def test_first_order_matches_resolve_at_small_severity(coarse):
    """1차 커널(행렬 중심차분 + Rayleigh)과 재해석의 비 → 1 (S̄ = 0.2 %에서 2 % 이내)."""
    spec, am, km0, md, obs = coarse
    ks = n1.kernel_scan(am, md, obs, [0.5], s_probe=1e-3, km0=km0)
    eta, mac = n1.resolved_eta(am, md, obs, 0.5, 0.002)
    # 분해불가 군(베이 국부모드, 간격 < 0.2 %)은 근접 모드 상호작용으로 2차항이 커서 제외한다
    res = np.array([o.resolvable for o in obs])
    ok = np.isfinite(eta) & (np.abs(ks.k[0]) * 0.002 > 1e-6) & res
    assert ok.sum() >= 4
    assert np.all(np.abs(eta[ok] / (ks.k[0, ok] * 0.002) - 1.0) < 0.02)
    assert np.nanmin(mac) > 0.99


def test_zero_damage_zero_shift(coarse):
    spec, am, km0, md, obs = coarse
    eta, mac = n1.track(md, md, obs)
    assert np.allclose(eta[np.isfinite(eta)], 0.0, atol=1e-12)


def test_fitted_vane_is_continuous_with_exact_thickness():
    """캠버 정합 격자: 모든 반경행에 베인 셀이 있고(끊김 없음), 캠버에 수직인 두께가 t_v다."""
    from impeller_fingerprint.rev1 import n1_mesh as nm
    fs = n1.fit_spec(n_r=10, n_bay=5, wrap_deg=90.0)
    c, conn, info = nm.mesh(fs)
    vane0 = info["cell_vane"] == 0
    rows = np.unique(np.round(info["cell_r"][vane0], 9))
    r_nodes = np.linspace(fs.a, fs.b, fs.n_r + 1)
    rc = 0.5 * (r_nodes[1:] + r_nodes[:-1])
    assert set(np.round(rc[rc >= fs.r_vane_in], 9)) <= set(rows)
    beta = fs.blade_angle(0.03)
    gi = info["grid_idx"]
    th = np.arctan2(c[:, 1], c[:, 0])
    r = np.hypot(c[:, 0], c[:, 1])
    for ir in range(fs.n_r + 1):
        i0 = np.nonzero((gi[:, 0] == ir) & (gi[:, 1] == 0) & (gi[:, 2] == 0))[0][0]
        i1 = np.nonzero((gi[:, 0] == ir) & (gi[:, 1] == 1) & (gi[:, 2] == 0))[0][0]
        arc = r[i0] * abs(np.angle(np.exp(1j * (th[i1] - th[i0]))))
        if r[i0] >= fs.r_vane_in:
            assert arc * math.cos(beta) == pytest.approx(fs.t_vane, rel=1e-9)
