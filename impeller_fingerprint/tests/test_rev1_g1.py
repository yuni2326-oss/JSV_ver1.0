"""G1(2026-10-16 재측정, 비대칭 두 면판) — 단면식·N1 격자의 하위호환과 새 기하."""
import numpy as np
import pytest

from impeller_fingerprint import forward as fwd
from impeller_fingerprint.rev1 import g1_section as g1
from impeller_fingerprint.rev1 import n1_assembly as n1
from impeller_fingerprint.rev1 import n1_mesh as nm
from impeller_fingerprint.rev1 import n1_run as nr


class TestSection:
    def test_equal_faces_reduce_to_forward_sandwich(self):
        p = np.array([0.0, 0.05, 0.1, 0.25, 0.5, 0.9])
        a = fwd.sandwich_dk_from_depth(p, 1.0e-3, 5.1e-3)
        b = g1.dk_from_depth(p, 1.0e-3, 1.0e-3, 4.1e-3)
        assert np.max(np.abs(a - b)) < 1e-12

    def test_inverse_round_trip_asymmetric(self):
        p = np.linspace(0.0, 0.95, 20)
        dk = g1.dk_from_depth(p, g1.T_FRONT, g1.T_BACK, g1.GAP)
        assert np.all(np.diff(dk) > 0)
        back = g1.depth_from_dk(dk, g1.T_FRONT, g1.T_BACK, g1.GAP)
        assert np.max(np.abs(back - p)) < 1e-9

    def test_full_removal_leaves_other_face(self):
        t_d, t_o, gap = g1.T_FRONT, g1.T_BACK, g1.GAP
        I1 = float(g1.two_face_I(1.0, t_d, t_o, gap))
        assert I1 == pytest.approx(t_o ** 3 / 12.0, rel=1e-12)

    def test_zeta_depends_on_which_face_is_thinned(self):
        # 08-13 기하 0.708, 얇은 앞판 0.545, 두꺼운 판을 깎으면 1.09 (scratchpad impact.py와 같은 값)
        assert g1.zeta0(1.0e-3, 1.0e-3, 4.1e-3) == pytest.approx(0.708, abs=2e-3)
        assert g1.zeta0(g1.T_FRONT, g1.T_BACK, 3.2e-3) == pytest.approx(0.545, abs=2e-3)
        assert g1.zeta0(g1.T_BACK, g1.T_FRONT, 3.2e-3) == pytest.approx(1.089, abs=3e-3)


class TestFitSpec:
    def test_geom0813_z_breaks_unchanged(self):
        fs = nm.FitSpec.geom0813(n_z_shroud=2, n_z_channel=2)
        t, c = 0.0010, 0.0041
        old = np.concatenate([np.linspace(0.0, t, 3), np.linspace(t, t + c, 3)[1:],
                              np.linspace(t + c, 2 * t + c, 3)[1:]])
        assert np.array_equal(fs.z_breaks, old)
        assert not fs.asymmetric and fs.tf == fs.tb == t
        assert (fs.wrap_deg, fs.r_vane_in, fs.a, fs.b) == (90.0, 0.0170, 0.0154, 0.03656)

    def test_default_is_measured_canon(self):
        from impeller_fingerprint import geometry as geo
        fs = nm.FitSpec()
        ab = geo.ASBUILT
        assert (fs.t_front, fs.t_back, fs.channel, fs.t_vane) == (ab.t_front, ab.t_back, ab.gap, ab.t_vane)
        assert (fs.wrap_deg, fs.r_vane_in, fs.a, fs.b) == (ab.wrap_deg, ab.r_vane_in, ab.a, ab.b)
        assert fs.asymmetric

    def test_asymmetric_z_breaks(self):
        fs = nm.FitSpec(t_front=0.375e-3, t_back=0.59e-3, channel=3.02e-3)
        z = fs.z_breaks
        assert z[0] == 0.0
        assert z[fs.n_z_shroud] == pytest.approx(0.59e-3)
        assert z[fs.n_z_shroud + fs.n_z_channel] == pytest.approx(0.59e-3 + 3.02e-3)
        assert z[-1] == pytest.approx(0.59e-3 + 3.02e-3 + 0.375e-3)
        assert np.all(np.diff(z) > 0)


class TestBuild:
    def _small(self, **kw):
        return nm.FitSpec(n_r=8, n_bay=5, **kw)

    def test_symmetric_build_and_depth_unchanged(self):
        am = n1.build(nm.FitSpec.geom0813(n_r=8, n_bay=5))
        assert am.t_o is None and am.gap is None
        assert am.z_in == am.spec.t_sheet + am.spec.channel
        r = np.linspace(am.spec.a, am.spec.b, 50)
        a = n1.depth_profile_am(am, r, 0.5, 0.02, 3e-3)
        b = n1.depth_profile(r, 0.5, 0.02, 3e-3, am.spec.a, am.spec.b, am.t_f,
                             am.spec.t_sheet + am.spec.channel)
        assert np.array_equal(a, b)

    def test_asymmetric_build(self):
        fs = self._small(t_front=0.375e-3, t_back=0.59e-3, channel=3.02e-3, t_vane=1.2e-3)
        am = n1.build(fs)
        assert am.t_f == pytest.approx(0.375e-3)
        assert am.t_o == pytest.approx(0.59e-3)
        assert am.gap == pytest.approx(3.02e-3)
        assert am.z_in == pytest.approx(0.59e-3 + 3.02e-3)
        z = am.coors[:, 2]
        assert z.max() == pytest.approx(0.59e-3 + 3.02e-3 + 0.375e-3)
        assert np.all(z[am.top] >= am.z_in - 1e-12)
        # 깎는 깊이는 앞판(얇은 판) 기준: 같은 d_K에서 같은 두께 1.0 mm 판보다 깊이비가 크다
        r = np.linspace(fs.a, fs.b, 50)
        p = n1.depth_profile_am(am, r, 0.5, 0.02, 3e-3)
        dk = n1.dK_profile(r, 0.5, 0.02, 3e-3, fs.a, fs.b)
        assert np.max(np.abs(g1.dk_from_depth(p, am.t_f, am.t_o, am.gap) - dk)) < 1e-9

    def test_geom_tag(self):
        assert nr._geom_tag(nm.FitSpec()) == ""
        assert nr._geom_tag(nm.FitSpec.geom0813()) == "_gF1000B1000C4100V1000R1700A1540O3656"
        tag = nr._geom_tag(nm.FitSpec(channel=3.02e-3))
        assert tag.startswith("_gF400B600C3020V1000")
