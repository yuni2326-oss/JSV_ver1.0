"""crack2d 테스트 — A11: 2D 평면탄성 + 폭 0 균열(정본 §3.6-iv의 남은 arm).

핵심은 **격자가 진짜 폭 0 슬릿인가**다. 개발 중 `i >= i_c`로 이중절점을 걸어 오른쪽 열
전체가 찢어진 버그가 있었고(건전 1058 Hz → 550 Hz), 주파수만 보면 "깊은 균열"로 오인된다.
그래서 연결도 불변식을 직접 검사한다:

  T-A11-1 슬릿 아래: x=x_c 절점이 좌·우로 **분리**(각 절점은 한쪽 요소만)
  T-A11-2 선단·리거먼트: 절점이 좌·우 요소에 **공유**
  T-A11-3 이중절점은 균열선 한 열에만 — 오른쪽 두 번째 열 이후는 원래 절점
  T-A11-4 균열깊이가 격자에 스냅되지 않는다(선단이 절점에 정확히 놓인다)
"""
import numpy as np
import pytest

from impeller_fingerprint import crack2d as c2
from impeller_fingerprint import crack_shear as cs
from impeller_fingerprint import geometry as geo
from impeller_fingerprint import rail3d as r3

VANE = geo.VANE
BEAM = cs.TimoBeam(L=VANE.L, h=VANE.h, b=2 * VANE.h, E=VANE.E, rho=VANE.rho,
                   nu=VANE.nu)
GRID = dict(nx_left=8, nx_right=32, nz_below=4, nz_above=4, bias=12.0)


def _sides(coors, conn, node, xc):
    els = np.where((conn == node).any(axis=1))[0]
    return {("L" if coors[conn[e]][:, 0].mean() < xc else "R") for e in els}


class TestGradedNodes:
    def test_uniform_when_bias_one(self):
        g = c2.graded_nodes(0.0, 1.0, 5, bias=1.0)
        assert np.allclose(g, np.linspace(0, 1, 6))

    @pytest.mark.parametrize("toward", ["lo", "hi"])
    def test_endpoints_and_monotone(self, toward):
        g = c2.graded_nodes(2.0, 5.0, 7, toward=toward, bias=10.0)
        assert g[0] == pytest.approx(2.0) and g[-1] == pytest.approx(5.0)
        assert np.all(np.diff(g) > 0)

    def test_bias_clusters_toward_requested_end(self):
        d_hi = np.diff(c2.graded_nodes(0.0, 1.0, 8, "hi", 10.0))
        assert d_hi[-1] < d_hi[0]
        d_lo = np.diff(c2.graded_nodes(0.0, 1.0, 8, "lo", 10.0))
        assert d_lo[0] < d_lo[-1]
        assert d_hi[0] / d_hi[-1] == pytest.approx(10.0, rel=1e-9)


class TestSlitTopology:
    """폭 0 슬릿의 연결도 불변식 — 이게 이 arm의 정체성이다."""

    A_BAR = 0.5

    @pytest.fixture(scope="class")
    def mesh(self):
        return c2.slit_mesh(VANE.L, VANE.h, self.A_BAR, **GRID)

    def test_crack_line_nodes_are_separated_below_tip(self, mesh):
        coors, conn, info = mesh
        xc, ytip = info["tip_x"], info["tip_y"]
        on = np.where(np.abs(coors[:, 0] - xc) < 1e-12)[0]
        below = [n for n in on if coors[n, 1] < ytip - 1e-12]
        assert len(below) == 2 * GRID["nz_below"], len(below)
        for n in below:
            assert len(_sides(coors, conn, n, xc)) == 1, (n, coors[n, 1])

    def test_tip_and_ligament_nodes_are_shared(self, mesh):
        coors, conn, info = mesh
        xc, ytip = info["tip_x"], info["tip_y"]
        on = np.where(np.abs(coors[:, 0] - xc) < 1e-12)[0]
        atop = [n for n in on if coors[n, 1] >= ytip - 1e-12]
        assert len(atop) == GRID["nz_above"] + 1
        for n in atop:
            assert _sides(coors, conn, n, xc) == {"L", "R"}, (n, coors[n, 1])

    def test_duplication_is_local_to_the_crack_line(self, mesh):
        """오른쪽 두 번째 열 이후의 절점은 좌·우 요소를 정상적으로 공유한다(버그 회귀)."""
        coors, conn, info = mesh
        xc = info["tip_x"]
        xs = np.unique(coors[:, 0])
        x_next = xs[xs > xc + 1e-12][0]
        for n in np.where(np.abs(coors[:, 0] - x_next) < 1e-12)[0]:
            els = np.where((conn == n).any(axis=1))[0]
            assert els.size >= 2, n
        assert info["n_dup_nodes"] == GRID["nz_below"]

    def test_healthy_mesh_has_no_duplicates(self):
        _, _, info = c2.slit_mesh(VANE.L, VANE.h, self.A_BAR, crack=False, **GRID)
        assert info["n_dup_nodes"] == 0

    @pytest.mark.parametrize("ab", [0.1, 0.37, 0.5, 0.625, 0.8])
    def test_tip_sits_exactly_on_a_node_no_snapping(self, ab):
        """균열깊이가 격자에 스냅되지 않는다 — B2(3D)는 0.125 단위로 양자화됐다."""
        coors, _, info = c2.slit_mesh(VANE.L, VANE.h, ab, **GRID)
        assert info["tip_y"] == pytest.approx(-0.5 * VANE.h + ab * VANE.h)
        d = np.hypot(coors[:, 0] - info["tip_x"], coors[:, 1] - info["tip_y"])
        assert d.min() < 1e-15

    def test_finite_width_notch_removes_elements(self):
        _, cn0, i0 = c2.slit_mesh(VANE.L, VANE.h, 0.5, kerf_width=0.0, **GRID)
        _, cn1, i1 = c2.slit_mesh(VANE.L, VANE.h, 0.5, kerf_width=5e-4, **GRID)
        assert i1["n_dup_nodes"] == 0            # 노치는 이중화가 아니라 제거
        assert i1["kerf_width"] == pytest.approx(5e-4)


class TestElementDescriptorInference:
    """요소형 하드코딩 금지(과거 세그폴트 원인) — 차원·절점수에서 추론한다."""

    def test_infers_2d_quad_and_3d_hex(self):
        coors, conn, _ = c2.slit_mesh(VANE.L, VANE.h, 0.5, **GRID)
        assert r3.elem_desc(coors, conn) == "2_4"
        c3, cn3, _ = r3.vane_mesh(nx=4, ny=1, nz=2)
        assert r3.elem_desc(c3, cn3) == "3_8"

    def test_rejects_unknown(self):
        with pytest.raises(ValueError):
            r3.elem_desc(np.zeros((3, 2)), np.zeros((1, 5), dtype=np.int32))


class TestPlaneSolve:
    """sfepy 2D 해가 물리적으로 맞는가."""

    def test_healthy_matches_euler_bernoulli(self):
        coors, conn, _ = c2.slit_mesh(VANE.L, VANE.h, 0.5, crack=False, **GRID)
        f, kinds, _ = c2.flap_modes(coors, conn, VANE.E, VANE.nu, VANE.rho,
                                    plane="stress")
        eb = VANE.eb_frequencies(3)
        assert f[0] / eb[0] == pytest.approx(1.0, rel=0.01)
        assert f[1] / eb[1] == pytest.approx(1.0, rel=0.02)
        assert kinds[:3] == ["flap", "flap", "flap"]

    def test_plane_strain_is_stiffer_by_one_minus_nu_squared(self):
        coors, conn, _ = c2.slit_mesh(VANE.L, VANE.h, 0.5, crack=False, **GRID)
        fs, _, _ = c2.flap_modes(coors, conn, VANE.E, VANE.nu, VANE.rho,
                                 plane="stress")
        fe, _, _ = c2.flap_modes(coors, conn, VANE.E, VANE.nu, VANE.rho,
                                 plane="strain")
        assert fe[0] / fs[0] == pytest.approx(1.0 / np.sqrt(1 - VANE.nu ** 2),
                                             rel=0.02)

    def test_slit_softens_fundamental_but_not_mode2(self):
        """정본 §4.1의 곡률-null 실명이 폭 0 균열에서도 유지되는가(수치는 CSV가 정본)."""
        kw = dict(nx_left=8, nx_right=32, nz_below=4, nz_above=4, bias=12.0)
        c0, k0, _ = c2.slit_mesh(VANE.L, VANE.h, 0.5, crack=False, **kw)
        f0, _, _ = c2.flap_modes(c0, k0, VANE.E, VANE.nu, VANE.rho, plane="stress")
        c1, k1, _ = c2.slit_mesh(VANE.L, VANE.h, 0.5, **kw)
        f1, _, _ = c2.flap_modes(c1, k1, VANE.E, VANE.nu, VANE.rho, plane="stress")
        assert 0.80 < f1[0] / f0[0] < 0.92          # 기본모드는 크게 떨어진다
        assert abs(f1[1] / f0[1] - 1) < 0.01        # mode 2는 1 % 미만
        assert (1 - f1[0] / f0[0]) > 30 * abs(f1[1] / f0[1] - 1)


class TestEquivalentCompliance:
    def test_inversion_round_trips(self):
        target = 0.02513
        r = c2.beam_ratios_from_cmm(BEAM, target, n_elem=300, n_modes=1)[0]
        assert c2.invert_c_theta(BEAM, float(r)) == pytest.approx(target, rel=1e-3)

    def test_monotone_in_compliance(self):
        rs = [c2.beam_ratios_from_cmm(BEAM, c, n_elem=200, n_modes=1)[0]
              for c in (1e-3, 1e-2, 1e-1)]
        assert rs[0] > rs[1] > rs[2]

    def test_dimensionless_form_is_E_invariant(self):
        """무차원 c_θ·EI/L은 평면응력/평면변형의 E' 규약을 상쇄한다."""
        b2 = cs.TimoBeam(L=BEAM.L, h=BEAM.h, b=BEAM.b, E=BEAM.E / (1 - VANE.nu ** 2),
                         rho=BEAM.rho, nu=BEAM.nu)
        c_a = cs.compliance(0.5, BEAM.h, BEAM.b, BEAM.E, BEAM.nu,
                            convention="tada")["c_MM"]
        c_b = cs.compliance(0.5, BEAM.h, BEAM.b, BEAM.E, BEAM.nu,
                            plane_strain=True, convention="tada")["c_MM"]
        assert (c2.dimensionless_c_theta(c_a, BEAM)
                == pytest.approx(c2.dimensionless_c_theta(c_b, b2), rel=1e-12))

    def test_out_of_bracket_returns_nan(self):
        """브래킷 밖 요구는 NaN — 기하에 무관한 두 극단으로 검정한다.

        옛 판정값 0.999999999는 **기하 의존**이었다: 브래킷 하한 c_MM = 1e−6에서 이 헬퍼가
        내는 주파수비는 보 이산화(n_elem=300에서 균열 노드가 추가돼 격자가 미세히 달라진다)
        때문에 1 ± 1e−4에 있고 부호가 기하마다 다르다 — h = 1.2 mm에서 0.99980,
        h = 1.0 mm에서 1.00010이다. 그래서 그 1e−4 바닥보다 훨씬 멀리 있는 두 극단으로 바꿨다.
        """
        assert np.isnan(c2.invert_c_theta(BEAM, 1.05))       # 상승은 불가능
        assert np.isnan(c2.invert_c_theta(BEAM, 0.01))       # 상한 밖(강하 99 %)


class TestDeterminism:
    def test_flap_modes_are_bit_identical_across_calls(self):
        """같은 입력이면 같은 주파수 — `cli a11` 산출물이 실행마다 달라지던 원인 차단.

        `solve_modes`는 ARPACK 시작벡터를 전역 RNG에서 뽑으므로(F102), 시드를 고정하지
        않으면 커밋된 CSV를 재현할 수 없다(2026-08-24에 두 실행이 상대 1e-4까지 갈렸다).
        """
        coors, conn, _ = c2.slit_mesh(L=0.030, h=0.0010, a_bar=0.5,
                                      nx_left=4, nx_right=12, nz_below=3, nz_above=3)
        kw = dict(E=193e9, nu=0.29, rho=8000.0, n_modes=3)
        f1, _, _ = c2.flap_modes(coors, conn, **kw)
        f2, _, _ = c2.flap_modes(coors, conn, **kw)
        assert list(f1) == list(f2), (f1, f2)
