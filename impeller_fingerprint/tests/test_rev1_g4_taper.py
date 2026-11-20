"""G4 깃 높이 테이퍼(출구 끝 3.00 → 흡입구 쪽 3.19 mm) — 정본 G3는 그대로이고, 테이퍼는 조립 좌표에만 들어간다."""
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from impeller_fingerprint import geometry as geo
from impeller_fingerprint.rev1 import g1_section as g1
from impeller_fingerprint.rev1 import n1_assembly as n1
from impeller_fingerprint.rev1 import n1_mesh as nm
from impeller_fingerprint.rev1 import n6_freefree as n6

C_IN = 3.19e-3
ROOT = Path(__file__).resolve().parents[2]


def _layers(fs, coors, grid_idx, r_target):
    """반지름 r_target 링에서 (뒤판 윗면, 앞판 안쪽면, 앞판 바깥면) z."""
    nzs, nzc = fs.n_z_shroud, fs.n_z_channel
    r = np.hypot(coors[:, 0], coors[:, 1])
    ring = np.isclose(r, r_target, atol=1e-9)
    iz = grid_idx[:, 2]

    def z(k):
        v = np.unique(np.round(coors[ring & (iz == k), 2], 12))
        assert v.size == 1
        return float(v[0])

    return z(nzs), z(nzs + nzc), z(2 * nzs + nzc)


def test_g4_constants_and_default_is_g3():
    assert geo.ASBUILT_G4.gap == pytest.approx(3.00e-3)
    assert geo.ASBUILT_G4.gap_in == pytest.approx(C_IN)
    assert geo.ASBUILT_G3.gap_in is None
    if os.environ.get("PAPER3_GEOM", "g3").lower() != "g4":
        assert geo.ASBUILT is geo.ASBUILT_G3
        assert nm.FitSpec().channel_in is None and not nm.FitSpec().tapered
    assert nm.FitSpec.geom0813().channel_in is None


def test_env_switch_selects_g4():
    code = ("from impeller_fingerprint import geometry as g; from impeller_fingerprint.rev1 import n1_mesh as nm; "
            "print(g.ASBUILT.gap_in, nm.FitSpec().channel_in, nm.FitSpec().tapered)")
    env = dict(os.environ, PAPER3_GEOM="g4", PYTHONPATH=str(ROOT))
    out = subprocess.run([sys.executable, "-c", code], env=env, cwd=ROOT, capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["0.00319", "0.00319", "True"]


def test_channel_at_linear_and_clipped():
    fs = nm.FitSpec(channel_in=C_IN)
    assert fs.channel_at(fs.a) == pytest.approx(C_IN)
    assert fs.channel_at(fs.b) == pytest.approx(fs.channel)
    assert fs.channel_at(0.5 * (fs.a + fs.b)) == pytest.approx(0.5 * (C_IN + fs.channel))
    assert fs.channel_at(0.5 * fs.a) == pytest.approx(C_IN)            # 허브 쪽은 자른다
    flat = nm.FitSpec()
    assert np.all(flat.channel_at([flat.a, flat.b]) == flat.channel)


def test_taper_coors_n1_gap_plates_and_identity():
    fs = nm.FitSpec(n_r=6, n_bay=4, channel_in=C_IN)
    coors, conn, info = nm.mesh(fs)
    gi = info["grid_idx"]
    X = fs.taper_coors(coors, gi)
    for r_t in (fs.a, 0.5 * (fs.a + fs.b), fs.b):
        zb, zi, zt = _layers(fs, X, gi, r_t)
        assert zb == pytest.approx(fs.t_back, abs=1e-15)                # 뒤판은 평판
        assert zi - zb == pytest.approx(float(fs.channel_at(r_t)), abs=1e-12)
        assert zt - zi == pytest.approx(fs.t_front, abs=1e-12)          # 앞판 두께 보존
    assert np.array_equal(X[:, :2], coors[:, :2])
    flat = replace(fs, channel_in=None)
    assert flat.taper_coors(coors, gi) is coors                          # G3면 그대로
    assert replace(fs, channel_in=fs.channel).taper_coors(coors, gi) is coors


def test_taper_coors_n6_hub_fixed_collar_lifted():
    ff = n6.FFSpec(base=nm.FitSpec(n_r=6, n_bay=4, channel_in=C_IN), n_hub=2, n_z_collar=2)
    fm = n6.build(ff)
    X = n1.physical_coors(fm, fm.coors)
    r = fm.r_v
    hub = r < ff.a - 1e-9
    assert np.array_equal(X[hub], fm.coors[hub])
    coll = fm.info["grid_idx"][:, 2] > 2 * ff.base.n_z_shroud + ff.base.n_z_channel
    dz = (X - fm.coors)[coll, 2]
    assert np.allclose(dz, ff.base.channel_at(r[coll]) - ff.base.channel, atol=1e-15)
    assert dz.max() == pytest.approx(C_IN - ff.base.channel)


def test_mass_increase_is_vane_height_only():
    fs = nm.FitSpec(n_r=6, n_bay=4, channel_in=C_IN)
    am_t, am_f = n1.build(fs), n1.build(replace(fs, channel_in=None))
    v_t, v_f = n1.assemble(am_t).volume, n1.assemble(am_f).volume
    vane = am_t.info["cell_kind"] == 1
    e0 = am_t.coors[am_t.conn[vane]]
    e1 = fs.taper_coors(am_t.coors, am_t.info["grid_idx"])[am_t.conn[vane]]

    def hexvol(P):
        d1 = (P[:, 1] + P[:, 2] + P[:, 5] + P[:, 6] - P[:, 0] - P[:, 3] - P[:, 4] - P[:, 7]) / 4
        d2 = (P[:, 3] + P[:, 2] + P[:, 7] + P[:, 6] - P[:, 0] - P[:, 1] - P[:, 4] - P[:, 5]) / 4
        d3 = (P[:, 4] + P[:, 5] + P[:, 6] + P[:, 7] - P[:, 0] - P[:, 1] - P[:, 2] - P[:, 3]) / 4
        return np.abs(np.einsum("ij,ij->i", np.cross(d1, d2), d3))

    assert v_t - v_f > 0
    assert v_t - v_f == pytest.approx(hexvol(e1).sum() - hexvol(e0).sum(), rel=2e-2)


def test_depth_law_uses_local_gap():
    fs = nm.FitSpec(n_r=6, n_bay=4, channel_in=C_IN)
    am = n1.build(fs)
    xi, s = 0.4, 0.05
    p = n1.depth_profile_am(am, am.r_v, xi, s, n1.W_GAUSS)
    dk = n1.dK_profile(am.r_v, xi, s, n1.W_GAUSS, fs.a, fs.b)
    for r_t in (fs.a, fs.b):
        k = np.argmin(np.abs(am.r_v - r_t))
        ref = g1.depth_from_dk(dk[k:k + 1], fs.t_front, fs.t_back, float(fs.channel_at(r_t)))[0]
        assert p[k] == pytest.approx(ref, rel=1e-6)
    flat = n1.build(replace(fs, channel_in=None))
    p0 = n1.depth_profile_am(flat, flat.r_v, xi, s, n1.W_GAUSS)
    k = np.argmin(np.abs(am.r_v - fs.a))
    assert p[k] != pytest.approx(p0[k], rel=1e-4)                       # 흡입구 쪽은 간격이 달라 깊이도 다르다
    kb = np.argmin(np.abs(am.r_v - fs.b))
    assert p[kb] == pytest.approx(p0[kb], rel=1e-9)                      # 출구 끝은 같다
