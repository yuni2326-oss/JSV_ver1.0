"""질량부하 한계 — 식 (ML)과 그 수치구현의 검정.

핵심은 **해석 앵커**다: EB 캔틸레버 1차의 팁 유효질량이 0.2427 m_total이므로 질량정규화
모드형은 |φ(L)|² = 1/(0.2427 m)을 만족해야 한다. 3D 레일이 이 값을 재현하면 (i) eigsh의
질량정규화 규약, (ii) 절점값 = 변위장 값, (iii) 식 (ML)이 한꺼번에 검증된다.
"""
import numpy as np
import pytest

from impeller_fingerprint import massload as ml
from impeller_fingerprint import rail3d as r3
from impeller_fingerprint.geometry import B2_CHANNEL, VANE


class TestFormula:
    def test_df_f_and_mass_limit_are_inverse(self):
        """m_a = mass_limit(φ², B) 를 (ML)에 넣으면 정확히 −B가 나온다."""
        for phi2 in (10.0, 137.0, 4076.0):
            for budget in (1.5e-4, 5e-4, 1e-3):
                m_a = ml.mass_limit(phi2, budget)
                assert float(ml.df_f_point_mass(m_a, phi2)) == pytest.approx(-budget,
                                                                             rel=1e-12)

    def test_df_f_is_linear_and_negative(self):
        """부착질량은 항상 주파수를 **내린다** — 부호가 뒤집히면 물리가 틀린 것이다."""
        assert float(ml.df_f_point_mass(1e-3, 100.0)) < 0
        assert float(ml.df_f_point_mass(2e-3, 100.0)) == pytest.approx(
            2 * float(ml.df_f_point_mass(1e-3, 100.0)), rel=1e-12)

    def test_modal_effective_mass_reading(self):
        """δf/f = −½ m_a/m_eff 로 읽어도 같은 값이어야 한다."""
        phi2, m_a = 58.0, 0.2e-3
        m_eff = ml.modal_effective_mass(phi2)
        assert float(ml.df_f_point_mass(m_a, phi2)) == pytest.approx(
            -0.5 * m_a / m_eff, rel=1e-12)

    def test_mass_limit_rejects_nonpositive_budget(self):
        with pytest.raises(ValueError):
            ml.mass_limit(100.0, 0.0)

    def test_nodal_phi2_sums_components(self):
        shape = np.array([1.0, 2.0, 2.0, 0.0, 3.0, 4.0])      # 2절점 × 3성분
        assert ml.nodal_phi2(shape) == pytest.approx([9.0, 25.0])


class TestMeshVolume:
    def test_hex_unit_box(self):
        """8절점 육면체 1개 = 정확한 부피(6-사면체 분할이 옳은지)."""
        coors = np.array([[0, 0, 0], [2, 0, 0], [2, 3, 0], [0, 3, 0],
                          [0, 0, 5], [2, 0, 5], [2, 3, 5], [0, 3, 5]], float)
        conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])
        assert ml.mesh_volume(coors, conn) == pytest.approx(30.0, rel=1e-12)

    def test_single_tet(self):
        coors = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
        assert ml.mesh_volume(coors, np.array([[0, 1, 2, 3]])) == pytest.approx(
            1 / 6, rel=1e-12)

    def test_rejects_unsupported_element(self):
        with pytest.raises(ValueError):
            ml.mesh_volume(np.zeros((3, 3)), np.array([[0, 1, 2]]))


class TestOuterSurfaceMask:
    def test_selects_top_face_and_outer_rim_only(self):
        coors = np.array([[0.030, 0.0, 0.0061],     # 상면·외주 둘 다
                          [0.020, 0.0, 0.0061],     # 상면
                          [0.03656, 0.0, 0.0030],   # 외주
                          [0.020, 0.0, 0.0030]],    # 내부(접근 불가)
                         float)
        mask = ml.outer_surface_mask(coors, r_out=0.03656, z_top=0.0061, tol=1e-4)
        assert mask.tolist() == [True, True, True, False]


class TestAnalyticAnchor:
    """느린 검정(3D 고유해 1회) — 이 앵커가 깨지면 논문의 mg 한계가 전부 무효다."""

    @pytest.fixture(scope="class")
    def vane(self):
        L, w, h = VANE.L, B2_CHANNEL, VANE.h
        c, cn, _ = r3.vane_mesh(L=L, w=w, h=h, nx=40, ny=3, nz=6)
        res = r3.solve_modes(c, cn, VANE.E, VANE.nu, VANE.rho, r3.clamp_root(),
                             n_modes=4, order=2, keep_shapes=True)
        return res, VANE.rho * L * w * h

    def test_tip_phi2_matches_eb_effective_mass(self, vane):
        """|φ(L)|²·m ≈ 1/0.24267 = 4.12 — 3D 전단·포아송 때문에 몇 % 낮게 나온다."""
        res, m_tot = vane
        d = ml.mode_phi2_max(res.full_shapes[0], res.field_coors)
        assert d["phi2_max"] * m_tot == pytest.approx(
            1.0 / ml.EB_TIP_EFFECTIVE_MASS_RATIO, rel=0.05)

    def test_antinode_is_at_the_tip(self, vane):
        """1차 굽힘의 |φ|² 최댓값은 자유단에 있어야 한다."""
        res, _ = vane
        d = ml.mode_phi2_max(res.full_shapes[0], res.field_coors)
        assert res.field_coors[d["node"], 0] == pytest.approx(VANE.L, abs=1e-9)

    def test_coupon_limit_is_sub_milligram(self, vane):
        """as-built 쿠폰(0.98 g)에서 floor 절반 예산의 한계는 **1 mg 미만**이다 —
        접촉센서가 원천 배제된다는 §5 E2의 판정 근거."""
        res, _ = vane
        d = ml.mode_phi2_max(res.full_shapes[0], res.field_coors)
        limit_mg = 1e6 * float(ml.mass_limit(d["phi2_max"], 5e-4))
        assert 0.1 < limit_mg < 1.0

    def test_normalization_holds_on_reduced_shapes(self, vane):
        """φᵀMφ = 1 — 대각질량 레일에서 직접 확인(육면체 조립체 레일과 같은 규약)."""
        from impeller_fingerprint import impeller_hex as ih
        spec = ih.HexImpellerSpec(n_r=6, n_theta=12, n_z_shroud=1, n_z_channel=1)
        spec.check()
        res, info = ih.solve_free_free(spec, n_modes=10)
        dev = max(abs(float((res.shapes[:, k] ** 2 * info["_M_diag"]).sum()) - 1.0)
                  for k in range(res.shapes.shape[1]))
        assert dev < 1e-9
