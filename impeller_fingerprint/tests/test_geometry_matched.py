"""a19 정합 **기하** 대조 — 고립 베인(as-built)의 메시·경계조건·구역 변형에너지 검정.

정본 §4.3-vii이 남긴 미결("a control that also matches the isolated-vane *geometry* — not
only the damage — is still outstanding")을 계산으로 닫기 위한 사다리의 하부 검정이다.
a18은 손상만 맞췄고 부품 쪽이 **직선 쿠폰·뿌리 캔틸레버**였으므로, 65–89배에는 기하와
경계조건 효과가 섞여 있다. 사다리는 한 단에 하나씩만 바꾼다:

    R0 직선 쿠폰(뿌리 클램프) → R1 as-built 베인(뿌리 클램프) → R2 as-built 베인
    (슈라우드면 클램프) → R3 조립체 pair mean

이 파일이 고정하는 주장
  (T-G1) 단일 베인 메시의 체적 = 압출 프로파일 면적 × 유로높이 — 메시가 요청한 기하를
         실현했는가의 자기검증(`impeller_hex.analytic_mass`와 같은 규약).
  (T-G2) 단일 베인 메시에는 슈라우드가 없다(z 범위가 유로층뿐).
  (T-G3) 손상은 조립체와 **같은 코드 경로**로 들어가고 뿌리쪽만 얇게 한다.
  (T-B1) 뿌리 클램프는 내경단 **한 면**만 고른다(면이지 선이 아니며, 외경단은 없다).
  (T-B2) 슈라우드면 클램프는 압출 **양면**만 고른다(대칭).
  (T-E1) 구역 변형에너지 분율은 **분할에 대해 1로 합쳐진다**(정의상 항등식의 기계적 검정).
  (T-E2) 캔틸레버 1차모드 에너지는 뿌리 1/4에 몰리고 팁 1/4에는 거의 없다 — 사다리가
         비율을 "왜"로 바꾸는 통화가 이것이고, 부호를 틀리면 결론이 뒤집힌다.
  (T-M1) 손상 마스크의 체적분율이 캠버 호길이로 독립 계산한 창 분율과 맞는다.

3D 고유해는 비싸므로 여기서는 메시·선택자 수준만 보고, 솔버를 부르는 검정은
`TestLadder`에서 최소 격자로 한 번씩만 돈다.
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from impeller_fingerprint import impeller_cad as icad
from impeller_fingerprint import rail3d as r3

COARSE = 0.0015          # [m] 검정용 성긴 격자 (생산은 1.2/1.0 mm)


def tet_volume(coors: np.ndarray, conn: np.ndarray) -> float:
    """사면체 메시 총 체적 — 메시가 실현한 기하의 독립 측정."""
    p = coors[conn]
    return float(np.abs(np.einsum("ij,ij->i",
                                  np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]),
                                  p[:, 3] - p[:, 0])).sum() / 6.0)


def polygon_area(px: np.ndarray, py: np.ndarray) -> float:
    """닫힌 다각형 면적(신발끈) — 프로파일의 독립 측정."""
    return float(abs(np.dot(px, np.roll(py, -1)) - np.dot(py, np.roll(px, -1))) / 2)


def profile_thickness(px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """프로파일 양쪽 오프셋 점을 짝지어 국소 두께를 잰다(0=뿌리, -1=팁)."""
    n = len(px) // 2
    j = np.arange(n)
    return np.hypot(px[j] - px[2 * n - 1 - j], py[j] - py[2 * n - 1 - j])


@pytest.fixture(scope="module")
def healthy(tmp_path_factory):
    spec = icad.ImpellerSpec()
    d = tmp_path_factory.mktemp("vane_h")
    coors, conn = icad.build_geometry(spec, mesh_size=COARSE, workdir=str(d),
                                     tag="vane_h", include_shrouds=False, vanes=(0,))
    return spec, coors, conn


@pytest.fixture(scope="module")
def damaged(tmp_path_factory):
    spec = replace(icad.ImpellerSpec(), damage_vane=0, damage_frac=0.6)
    d = tmp_path_factory.mktemp("vane_d")
    coors, conn = icad.build_geometry(spec, mesh_size=COARSE, workdir=str(d),
                                     tag="vane_d", include_shrouds=False, vanes=(0,))
    return spec, coors, conn


class TestSingleVaneMesh:
    def test_volume_matches_the_extruded_profile(self, healthy):
        """T-G1 — 사면체 체적이 다각형 면적 × 유로높이와 맞는다."""
        spec, coors, conn = healthy
        px, py = icad.profile_polygon(spec, 0)
        ref = polygon_area(px, py) * spec.gap
        assert tet_volume(coors, conn) == pytest.approx(ref, rel=0.03)

    def test_mesh_omits_the_shrouds(self, healthy):
        """T-G2 — z가 유로층 [t_back, t_back+b₂]뿐이다(슈라우드 없음)."""
        spec, coors, _ = healthy
        assert coors[:, 2].min() == pytest.approx(spec.t_back, abs=1e-9)
        assert coors[:, 2].max() == pytest.approx(spec.t_back + spec.gap, abs=1e-9)

    def test_damage_thins_only_the_root_quarter(self, healthy, damaged):
        """T-G3 — 창 안은 두께 40 %, 창 밖은 100 %이고 체적비가 프로파일과 일치한다."""
        spec_h, coors_h, conn_h = healthy
        spec_d, coors_d, conn_d = damaged
        th = profile_thickness(*icad.profile_polygon(spec_d, 0))
        assert th[0] == pytest.approx(0.4 * spec_d.t_vane, rel=1e-6)
        assert th[-1] == pytest.approx(spec_d.t_vane, rel=1e-6)
        ratio_poly = (polygon_area(*icad.profile_polygon(spec_d, 0))
                      / polygon_area(*icad.profile_polygon(spec_h, 0)))
        assert 0.85 < ratio_poly < 0.97          # 뿌리 1/4 × 60 % 제거 규모
        assert (tet_volume(coors_d, conn_d) / tet_volume(coors_h, conn_h)
                == pytest.approx(ratio_poly, rel=0.01))


class TestClamps:
    def test_root_clamp_selects_the_inner_end_face_only(self, healthy):
        """T-B1 — 내경단 한 면: z를 다 덮고(면), 반경은 뿌리 근처뿐이다."""
        spec, coors, _ = healthy
        idx = icad.clamp_vane_root(spec, 0)(coors)
        assert idx.size >= 4
        z = coors[idx, 2]
        assert z.max() - z.min() == pytest.approx(spec.gap, rel=1e-6)
        r = np.hypot(coors[idx, 0], coors[idx, 1])
        assert r.max() < spec.r_vane_in + spec.t_vane
        assert r.min() > spec.r_vane_in - spec.t_vane

    def test_shroud_clamp_selects_both_extruded_faces(self, healthy):
        """T-B2 — 압출 양면만, 대칭으로."""
        spec, coors, _ = healthy
        idx = icad.clamp_vane_shroud_faces(spec)(coors)
        z = coors[idx, 2]
        lo, hi = spec.t_back, spec.t_back + spec.gap
        assert np.all((np.abs(z - lo) < 1e-9) | (np.abs(z - hi) < 1e-9))
        assert (np.abs(z - lo) < 1e-9).sum() == (np.abs(z - hi) < 1e-9).sum()
        assert idx.size > 2 * icad.clamp_vane_root(spec, 0)(coors).size


class TestRegionEnergy:
    """T-E1·T-E2 — `solve_modes(region_mask=…)`의 구역 변형에너지 분율."""

    @staticmethod
    def _coupon():
        coors, conn, _ = r3.vane_mesh(L=0.030, w=0.0041, h=0.0010,
                                      nx=12, ny=2, nz=3)
        return coors, conn

    @staticmethod
    def _frac(coors, conn, mask):
        np.random.seed(20260824)
        res = r3.solve_modes(coors, conn, 193e9, 0.29, 8000.0, r3.clamp_root(),
                             n_modes=3, order=2, region_mask=mask)
        return res.region_energy_frac

    def test_whole_domain_fraction_is_exactly_one(self):
        """T-E1a — 구역 = 전 영역이면 분율은 기계정밀도로 1이다(한 해 안의 엄격 항등식)."""
        coors, conn = self._coupon()
        whole = self._frac(coors, conn, np.ones(len(conn), dtype=bool))
        assert whole == pytest.approx(np.ones(len(whole)), abs=1e-12)

    def test_partition_fractions_sum_to_one(self):
        """T-E1b — 상보 분할의 합도 1. 잔차 O(1e-9)는 **두 번의 독립 ARPACK 해**에서
        온 고유벡터 차이이지 항등식의 오차가 아니다(전 영역 검정이 그것을 분리한다)."""
        coors, conn = self._coupon()
        xc = coors[conn][:, :, 0].mean(axis=1)
        lo = self._frac(coors, conn, xc < 0.015)
        hi = self._frac(coors, conn, xc >= 0.015)
        assert lo + hi == pytest.approx(np.ones(len(lo)), abs=1e-7)

    def test_cantilever_energy_concentrates_at_the_root(self):
        coors, conn = self._coupon()
        xc = coors[conn][:, :, 0].mean(axis=1)
        root = self._frac(coors, conn, xc < 0.25 * 0.030)
        tip = self._frac(coors, conn, xc > 0.75 * 0.030)
        assert root[0] > 0.5
        assert tip[0] < 0.05

    @staticmethod
    def _solve(coors, conn, mask):
        np.random.seed(20260824)
        return r3.solve_modes(coors, conn, 193e9, 0.29, 8000.0, r3.clamp_root(),
                              n_modes=3, order=2, region_mask=mask)

    def test_whole_domain_kinetic_fraction_is_exactly_one(self):
        """T-E3a — 운동에너지 분율도 전 영역에서 1(질량 구역조립의 항등식)."""
        coors, conn = self._coupon()
        res = self._solve(coors, conn, np.ones(len(conn), dtype=bool))
        assert res.region_kinetic_frac == pytest.approx(
            np.ones(len(res.freqs)), abs=1e-12)

    def test_kinetic_and_strain_energy_localize_at_opposite_ends(self):
        """T-E3b — 캔틸레버 1차: 강성은 뿌리, 질량은 팁. 두 구역행렬이 뒤바뀌면 깨진다.

        감육은 강성(∝t³)과 질량(∝t)을 **함께** 줄이므로 1차 주파수변화는 두 분율의
        차로 결정된다 — 강성 분율만 쓰면 부호와 크기를 모두 틀린다(§3.2의 γ^K·γ^M).
        """
        coors, conn = self._coupon()
        xc = coors[conn][:, :, 0].mean(axis=1)
        root = self._solve(coors, conn, xc < 0.25 * 0.030)
        tip = self._solve(coors, conn, xc > 0.75 * 0.030)
        assert root.region_energy_frac[0] > root.region_kinetic_frac[0]
        assert tip.region_kinetic_frac[0] > tip.region_energy_frac[0]
        assert tip.region_kinetic_frac[0] > 0.5


class TestDamagedCellMask:
    def test_mask_volume_matches_the_camber_window(self, healthy):
        """T-M1 — 뿌리 창의 체적분율 = 캠버 호길이로 잰 창 분율."""
        spec, coors, conn = healthy
        mask = icad.damaged_cell_mask(spec, coors, conn, 0)
        assert mask.any() and not mask.all()
        p = coors[conn]
        vol = np.abs(np.einsum("ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]),
                               p[:, 3] - p[:, 0])) / 6.0
        got = vol[mask].sum() / vol.sum()
        xs, ys = icad.vane_camber(spec, 0, n_pts=4001)
        s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))])
        f_r = (np.hypot(xs, ys) - spec.r_vane_in) / (spec.r_out - spec.r_vane_in)
        ref = np.interp(spec.damage_span[1], f_r, s) / s[-1]
        assert got == pytest.approx(ref, rel=0.10)


class TestVaneModeKinds:
    """T-K1 — 곡면 베인 모드의 **형상 기반** 분류.

    설계서의 3D 함정 ③: "주파수 순서로 모드를 짝지으면 결론이 뒤집힌다". 직선 레일은
    `rail3d.beam_mode_kinds`가 데카르트 성분으로 골랐지만, 후곡 베인은 굽힘 방향이
    반경마다 회전하므로 **캠버 국소좌표**(접선·법선·z)로 투영해야 한다.
    """

    @staticmethod
    def _solve(spec, coors, conn, selector):
        np.random.seed(20260824)
        return r3.solve_modes(coors, conn, spec.E, spec.nu, spec.rho, selector,
                              n_modes=6, order=2, keep_shapes=True)

    def test_fundamental_is_flapwise_and_the_classifier_discriminates(self, healthy):
        spec, coors, conn = healthy
        res = self._solve(spec, coors, conn, icad.clamp_vane_root(spec, 0))
        kinds, part = icad.vane_mode_kinds(spec, res, 0)
        assert kinds[0] == "flap"
        assert part[0, 1] > 0.5                     # 법선(flap) 성분이 지배
        assert set(kinds) != {"flap"}               # 전부 flap으로 라벨하지 않는다
        assert part.sum(axis=1) == pytest.approx(np.ones(len(kinds)), abs=1e-12)


class TestSolveModalWiring:
    def test_solve_modal_forwards_the_region_mask(self, healthy):
        """T-W1 — `solve_modal`이 region_mask를 흘리지 않는다(조용한 단절 회귀).

        조립체 단(R3)의 에너지 분율은 이 경로로만 나온다. 인자가 조용히 버려지면
        `region_energy_frac`가 None이 되어 사다리의 마지막 칸이 빈다.
        """
        spec, coors, conn = healthy
        res = icad.solve_modal(spec, coors, conn, n_modes=3, order=2,
                              clamp_radius=spec.r_vane_in + spec.t_vane,
                              region_mask=np.ones(len(conn), dtype=bool))
        assert res.region_energy_frac == pytest.approx(np.ones(3), abs=1e-12)
        assert res.region_kinetic_frac == pytest.approx(np.ones(3), abs=1e-12)


class TestLadderFactors:
    def test_factors_multiply_to_the_total_dilution(self):
        """T-L1 — 단계별 인자의 곱 = R0→R3 총 희석비(회계 항등식).

        논문이 인용하는 것은 총비(65–89배)와 그 **분해**이므로, 분해가 총비를 재구성하지
        못하면 표가 자기모순이다.
        """
        from impeller_fingerprint import cli
        shifts = [-70.052, -22.5, -4.1, -0.787]
        fac = cli.ladder_factors(shifts)
        assert np.isnan(fac[0])
        assert np.prod(fac[1:]) == pytest.approx(shifts[0] / shifts[-1], rel=1e-12)
        assert fac[1] == pytest.approx(shifts[0] / shifts[1], rel=1e-12)

    def test_zero_shift_does_not_raise(self):
        """0 강하(수치해상도 이하)는 무한 인자가 아니라 nan으로 돌려준다."""
        from impeller_fingerprint import cli
        fac = cli.ladder_factors([-70.0, 0.0, -1.0])
        assert np.isnan(fac[1])


class TestCellVolumes:
    def test_cell_volumes_are_positive_and_sum_to_the_mesh_volume(self, healthy):
        """T-V1 — 셀 체적은 모두 양이고 합이 총 체적이다(부호·계수 오류 차단)."""
        _, coors, conn = healthy
        v = icad.cell_volumes(coors, conn)
        assert v.shape == (len(conn),)
        assert (v > 0).all()
        assert v.sum() == pytest.approx(tet_volume(coors, conn), rel=1e-12)


class TestA19Wiring:
    def test_a19_is_a_registered_subcommand(self):
        """T-W2 — 산출 명령이 파서에 붙어 있다(CSV만 있고 명령이 없던 a18의 재발 차단)."""
        from impeller_fingerprint import cli
        args = cli.build_parser().parse_args(["a19"])
        assert args.func is cli.cmd_a19
        assert args.mesh_size == [0.0012, 0.0010]
        assert args.damage_frac == 0.6 and args.damage_span == 0.25


class TestHexCellVolumes:
    def test_structured_box_hex_volumes_are_exact(self):
        """T-V2 — 육면체 메시에도 쓴다: 상자 메시의 셀 체적은 해석값과 **정확히** 같다.

        R0 단은 육면체(레일 쿠폰)라 사면체 공식을 그대로 쓰면 아래면 4절점만 읽어
        모든 셀이 0이 되고 창 체적분율이 0/0 = nan이 된다(스모크에서 실제로 났다).
        """
        L, w, h, nx, ny, nz = 0.030, 0.0041, 0.0010, 6, 2, 3
        coors, conn, _ = r3.vane_mesh(L=L, w=w, h=h, nx=nx, ny=ny, nz=nz)
        v = icad.cell_volumes(coors, conn)
        assert (v > 0).all()
        assert v.sum() == pytest.approx(L * w * h, rel=1e-12)
        assert v == pytest.approx(
            np.full(nx * ny * nz, (L / nx) * (w / ny) * (h / nz)), rel=1e-12)

    def test_unsupported_element_is_refused(self):
        """절점 수가 4·8이 아니면 조용히 틀린 값을 주지 않고 거절한다."""
        with pytest.raises(ValueError):
            icad.cell_volumes(np.zeros((6, 3)), np.arange(6).reshape(1, 6))
