"""N6 자유-자유 실측 형상 조립체 — 거친 격자로 구조·대칭·질량·홈 규약을 고정한다."""
import math

import numpy as np
import pytest

from impeller_fingerprint import geometry as geo
from impeller_fingerprint.rev1 import n1_mesh as nm
from impeller_fingerprint.rev1 import n6_freefree as n6


def _coarse_spec(**kw):
    base = nm.FitSpec(n_r=8, n_bay=5)
    return n6.FFSpec(base=base, n_hub=2, n_z_collar=2, **kw)


@pytest.fixture(scope="module")
def coarse():
    ff = _coarse_spec()
    fm = n6.build(ff)
    km = n6.assemble(fm)
    md, rigid = n6.solve(km, 14)
    obs = n6.classify(fm, md)
    return ff, fm, km, md, rigid, obs


def test_radial_nodes_put_collar_on_first_ring():
    ff = _coarse_spec()
    r = ff.r_nodes
    assert r[0] == pytest.approx(ff.r_bore)
    assert r[ff.n_hub] == pytest.approx(ff.a)
    assert r[ff.n_hub + 1] - r[ff.n_hub] == pytest.approx(ff.collar_t)
    assert r[-1] == pytest.approx(ff.b)
    assert np.all(np.diff(r) > 0)


def test_six_rigid_modes_and_elastic_gap(coarse):
    *_, md, rigid, _ = coarse
    assert rigid.size == 6 and np.all(rigid < 1.0)
    assert md.freqs[0] > 300.0


def test_protected_doublets_exactly_degenerate(coarse):
    *_, md, _, obs = coarse
    pairs = [o for o in obs if o.deg == 2]
    assert pairs
    for o in pairs:
        f = md.freqs[o.members]
        assert abs(f[1] - f[0]) / f[0] < 1e-8
        assert o.h in (1, 2)
    assert all(o.h in (0, 3) for o in obs if o.deg == 1)


def test_model_mass_matches_part_sum(coarse):
    """모델 질량 = 뒤판(보어–외경) + 앞판(흡입구–외경) + 흡입구 테 + 베인 6매(캠버 길이 × 간격 × 두께)."""
    ff, fm, km, *_ = coarse
    fs = ff.base
    rho = fs.rho
    back = math.pi * (ff.b ** 2 - ff.r_bore ** 2) * fs.t_back
    front = math.pi * (ff.b ** 2 - ff.a ** 2) * fs.t_front
    collar = math.pi * ((ff.a + ff.collar_t) ** 2 - ff.a ** 2) * ff.collar_h
    beta = fs.blade_angle(0.03)
    vanes = fs.n_vane * (fs.b - fs.r_vane_in) / math.cos(beta) * fs.channel * fs.t_vane
    expect = rho * (back + front + collar + vanes)
    assert rho * km.volume == pytest.approx(expect, rel=0.02)


def test_groove_removes_expected_volume_and_keeps_inner_face(coarse):
    ff, fm, km, *_ = coarse
    xi, depth = 0.6, 0.1e-3
    p = n6.groove_depth_ratio(fm.r_v, ff, xi, depth)
    c1 = n6.morph_groove(fm, p)
    km1 = n6.assemble(fm, c1)
    removed = km.volume - km1.volume
    # 거친 격자(반경 8칸)에서 홈 가장자리가 절점 사이에 걸리므로 조각선형 근사 오차가 크다
    assert removed == pytest.approx(n6.groove_removed_volume(ff, xi, depth), rel=0.35)
    z_in, _ = ff.z_front
    inner = np.isclose(fm.coors[:, 2], z_in)
    assert np.array_equal(c1[inner], fm.coors[inner])
    collar = fm.coors[:, 2] > ff.z_front[1] + 1e-12
    assert np.array_equal(c1[collar], fm.coors[collar])


def test_axisymmetric_groove_keeps_c6_degeneracy(coarse):
    ff, fm, km, md, _, obs = coarse
    p = n6.groove_depth_ratio(fm.r_v, ff, 0.85, 0.06e-3)
    mdd, _ = n6.solve(n6.assemble(fm, n6.morph_groove(fm, p)), 14)
    rel = np.abs(np.diff(mdd.freqs)) / mdd.freqs[:-1]
    assert int(np.sum(rel < 1e-8)) == sum(1 for o in obs if o.deg == 2)


def test_first_order_matches_resolve_for_shallow_groove(coarse):
    ff, fm, km, md, _, obs = coarse
    eta1 = n6.groove_first_order(fm, md, obs, 0.6, 0.01e-3)
    eta, mac = n6.groove_resolved(fm, md, obs, 0.6, 0.01e-3)
    res = np.array([o.resolvable for o in obs])
    ok = np.isfinite(eta) & res & (np.abs(eta1) > 1e-7)
    assert ok.sum() >= 4
    assert np.all(np.abs(eta[ok] / eta1[ok] - 1.0) < 0.05)
    assert np.nanmin(mac[ok]) > 0.99


# ---------------------------------------------------------------- 가진·응답점(n6_excite)
from impeller_fingerprint.rev1 import n6_excite as nx


def test_rigid_hertz_impact_duration_matches_closed_form():
    """강체 반공간 Hertz 타격: t_c = 2.9432·δ_max/v0, δ_max = (5 m v0² / (4 k))^{2/5}, 접촉 1회."""
    E, nu, m, R, v0, dt = 193e9, 0.29, 2e-3, 1.27e-3, 0.5, 2e-8
    F = nx.impact(np.zeros(1), np.zeros(1), m, E=E, nu=nu, R=R, v0=v0, Z=1e15, dt=dt)
    k = nx.hertz_k(E, nu, R)
    d_max = (5 * m * v0 ** 2 / (4 * k)) ** 0.4
    band = nx.force_band(F, dt)
    assert band["n_contact"] == 1
    assert band["duration_us"] == pytest.approx(nx.HERTZ_DURATION * d_max / v0 * 1e6, rel=0.01)
    # 2 g 강철 팁의 −10 dB 대역은 PCB 086E80 매뉴얼 곡선(≈ 28 kHz) 근처
    assert 24e3 < band["f_m10dB_Hz"] < 34e3


def test_mass_props_match_assembled_volume_and_c6(coarse):
    ff, fm, km, *_ = coarse
    p = nx.mass_props(fm.coors, fm.conn, ff.base.rho)
    assert p["M"] == pytest.approx(ff.base.rho * km.volume, rel=1e-9)
    assert 0.0 < p["zc"] < ff.z_breaks[-1]
    assert p["Ixx"] == pytest.approx(p["Iyy"], rel=1e-6)


def test_visibility_is_one_at_front_reference_max(coarse):
    ff, fm, km, md, _, obs = coarse
    top = n6._front_top_nodes(fm)
    mask = np.zeros(fm.coors.shape[0], bool)
    mask[top] = True
    V = nx.visibility(md.vshape, [o.members for o in obs], mask)
    assert np.allclose(V[:, mask].max(axis=1), 1.0)


def test_h0_h3_singlets_split_by_adjacent_bay_sign(coarse):
    """h0 싱글릿은 여섯 베이 가운데가 같은 부호, h3은 교대 — 이웃 베이 두 점의 합·차로 가른다."""
    ff, fm, km, md, _, obs = coarse
    fs = ff.base
    per = fs.n_vc + fs.n_bay
    loc = fm.info["grid_idx"][:, 1] % per
    top = n6._front_top_nodes(fm)
    cen_all = top[loc[top] == fs.n_vc + fs.n_bay // 2]
    rr = fm.r_v[cen_all]
    checked = 0
    for o in obs:
        if o.deg != 1:
            continue
        uz = md.vshape[o.members[0]].reshape(-1, 3)[:, 2]
        ring = np.unique(np.round(rr, 9))
        r_best = max(ring, key=lambda r: np.abs(uz[cen_all[np.isclose(rr, r, atol=1e-9)]]).min())
        cen = cen_all[np.isclose(rr, r_best, atol=1e-9)]
        cen = cen[np.argsort(np.arctan2(fm.coors[cen, 1], fm.coors[cen, 0]))]
        s = np.sign(uz[cen])
        if np.abs(uz[cen]).min() < 1e-3 * np.abs(uz).max():
            continue
        if o.h == 0:
            assert np.all(s == s[0])
        else:
            assert np.all(s[1:] == -s[:-1])
        checked += 1
    assert checked >= 1


def test_small_ball_single_short_contact_on_thin_front_plate(coarse):
    ff, fm, km, md, _, obs = coarse
    fs = ff.base
    top = n6._front_top_nodes(fm)
    j = int(top[np.argmax(fm.r_v[top])])
    props = nx.mass_props(fm.coors, fm.conn, fs.rho)
    d = 1.5e-3
    mb = fs.rho * math.pi / 6 * d ** 3
    F = nx.impact(md.vshape[:, 3 * j + 2] ** 2, 2 * np.pi * md.freqs, mb, E=fs.E, nu=fs.nu, R=d / 2, v0=0.3,
                  rigid=nx.rigid_phi2(fm.coors[j], props), dt=2e-9, T=2e-4)
    band = nx.force_band(F, 2e-9)
    assert band["n_contact"] == 1
    assert band["duration_us"] < 15.0
    assert band["dB_25kHz"] > -1.0


# ---------------------------------------------------------------- 설치 상태(허브 고정)·가우시안 커널
@pytest.fixture(scope="module")
def coarse_hub():
    ff0 = _coarse_spec()
    ff = _coarse_spec(clamp_r=float(ff0.r_nodes[1]))
    fm = n6.build(ff)
    md, rigid = n6.solve(n6.assemble(fm), 14, n_rigid=ff.n_rigid)
    return ff, fm, md, rigid, n6.classify(fm, md)


def test_hub_clamp_removes_rigid_modes_and_interlaces(coarse, coarse_hub):
    """허브 고정: 강체 모드가 없고(n_rigid = 0), 구속 정리로 고정계 j번째 ≥ 자유계 (j − 6)번째 탄성."""
    *_, md_free, _, _ = coarse
    ff, fm, md, rigid, obs = coarse_hub
    assert ff.n_rigid == 0 and rigid.size == 0
    assert md.freqs[0] > 50.0
    n = min(md.freqs.size, md_free.freqs.size + 6)
    assert np.all(md.freqs[6:n] >= md_free.freqs[:n - 6] * (1 - 1e-9))
    with pytest.raises(RuntimeError):
        n6.solve(n6.assemble(n6.build(_coarse_spec())), 8, n_rigid=0)


def test_gaussian_depth_ratio_front_plate_only(coarse):
    ff, fm, *_ = coarse
    p = n6.gaussian_depth_ratio(fm, 0.5, 0.05)
    assert np.all(p[fm.r_v < ff.a - 1e-9] == 0.0)
    assert p.max() > 0.0
    r_peak = fm.r_v[np.argmax(p)]
    assert abs((r_peak - ff.a) / ff.extent - 0.5) < 0.1


def test_gaussian_kernel_matches_resolve_at_small_severity(coarse_hub):
    ff, fm, md, _, obs = coarse_hub
    ks = n6.kernel_scan(fm, md, obs, [0.6])
    eta, mac = n6.resolved_gaussian(fm, md, obs, 0.6, 0.01)
    res = np.array([o.resolvable for o in obs])
    ok = np.isfinite(eta) & res & (np.abs(ks.k[0]) > 1e-2)
    assert ok.sum() >= 4
    assert np.all(np.abs(eta[ok] / 0.01 / ks.k[0][ok] - 1.0) < 0.1)


# ---------------------------------------------------------------- 베인 뿌리 물리적 감육(b5 대체)
def test_vane_thinning_removes_window_volume_and_keeps_rest(coarse_hub):
    ff, fm, md, _, obs = coarse_hub
    fs = ff.base
    c0 = nm.vane_thin_coors(fm.coors, fm.info["grid_idx"], fs, vane=0, frac=0.0)
    assert np.array_equal(c0, fm.coors)
    c = nm.vane_thin_coors(fm.coors, fm.info["grid_idx"], fs, vane=0, span=(0.0, 0.25), frac=0.6)
    moved = np.any(np.abs(c - fm.coors) > 1e-15, axis=1)
    assert moved.any()
    # 반경·z는 그대로, 움직인 절점은 베인 0의 섹터 안
    assert np.allclose(np.hypot(c[:, 0], c[:, 1]), np.hypot(fm.coors[:, 0], fm.coors[:, 1]))
    assert np.array_equal(c[:, 2], fm.coors[:, 2])
    per = fs.n_vc + fs.n_bay
    assert np.all(fm.info["grid_idx"][moved, 1] // per == 0)
    # 제거 부피 ≈ 0.6 × (창 안 베인 셀 부피) — 창 경계 셀의 사다리꼴 때문에 거친 격자에서 넉넉히
    from impeller_fingerprint.rev1 import n6_excite as nx
    v_all = nx.mass_props(fm.coors, fm.conn, 1.0)["M"]
    v_thin = nx.mass_props(c, fm.conn, 1.0)["M"]
    ck, cv, cs = fm.info["cell_kind"], fm.info["cell_vane"], fm.info["cell_span"]
    win = (ck == 1) & (cv == 0) & (cs >= 0.0) & (cs <= 0.25)
    v_win = nx.mass_props(fm.coors, fm.conn[win], 1.0)["M"]
    assert v_all - v_thin == pytest.approx(0.6 * v_win, rel=0.35)


def test_vane_thinning_splits_doublets_and_lowers_pair_means(coarse_hub):
    from impeller_fingerprint.rev1 import n2_pocket as n2
    ff, fm, md, _, obs = coarse_hub
    c = nm.vane_thin_coors(fm.coors, fm.info["grid_idx"], ff.base, vane=0, frac=0.6)
    mdd, _ = n6.solve(n6.assemble(fm, c), md.phi.shape[1] + 6, n_rigid=ff.n_rigid)
    mean, split, mac = n2.track_split(md, mdd, obs)
    dbl = [j for j, o in enumerate(obs) if o.deg == 2 and np.isfinite(split[j])]
    assert dbl
    assert all(split[j] > 1e-6 for j in dbl)
    ok = np.isfinite(mean)
    assert ok.sum() >= 4 and np.nanmax(mean) < 1e-3     # 강성 손실이 지배 — 오르는 관측량 거의 없음
