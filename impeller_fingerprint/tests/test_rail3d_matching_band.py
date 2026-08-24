"""3D 레일의 **형상 기반 모드매칭**과 **밴드 손상 순방향** 검정 (설계서 F21·§3.6-iii).

이 파일이 고정하는 주장
  (T-M1) `disk_mesh`는 요청 포켓이 셀면에 정확히 놓일 때만 `shape_exact=True`를 준다
         — 형상보존 격자 규약(F11′)의 기계적 판정.
  (T-M2) `match_order`는 방위차수를 **차수→모드**로 고른다(주파수 순서 무관).
  (T-M3) `subspace_mac_match`는 건전 부분공간에 없는 국소모드를 **거부**한다
         (F21의 "억지 짝짓기 금지"를 코드로 강제).
  (T-B1) 밴드 1차 섭동식이 `degenerate`의 닫힌형 H^(m)과 Δθ=2π에서 일치한다(독립 2경로).
  (T-B2) 밴드 정확재해는 심각도→0에서 1차 섭동으로 수렴한다.
  (T-B3) 르장드르 기저·전처리는 **수학을 바꾸지 않는다**(단항 기저와 일치) — n_trial을
         올릴 수 있게 하는 순수 수치 개선.
  (T-S1) 대리모델은 κ=1에서 정확재해와 같고, κ≠1이면 회복오차에 모델형식 편향이 생긴다.

3D 고유해는 비싸므로 매칭 검정은 **합성 ModalResult**로 한다(솔버 호출 없음).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from impeller_fingerprint import degenerate as deg
from impeller_fingerprint import forward as fwd
from impeller_fingerprint import geometry as geo
from impeller_fingerprint import kernels as ker
from impeller_fingerprint import montecarlo as mc
from impeller_fingerprint import rail3d as r3

PLATE = geo.DISK
MODES = [(0, 0), (1, 0), (2, 0), (3, 0)]
W = 0.003


@pytest.fixture(scope="module")
def pool():
    return [ker.mode_kernel(PLATE, m=m, n=n, n_grid=2001) for m, n in MODES]


# --------------------------------------------------------------- 합성 모달해
def _cloud(nr=12, nt=64, nz=3):
    rs = np.linspace(PLATE.a, PLATE.b, nr + 1)
    ths = np.linspace(0.0, 2 * math.pi, nt, endpoint=False)
    zs = np.linspace(-0.5 * PLATE.t, 0.5 * PLATE.t, nz + 1)
    R, TH, Z = np.meshgrid(rs, ths, zs, indexing="ij")
    return np.stack([(R * np.cos(TH)).ravel(), (R * np.sin(TH)).ravel(),
                     Z.ravel()], axis=1)


def _synth(coors, specs):
    """specs = [(m, psi, radial_power, localize_theta or None), ...] → ModalResult."""
    rr = np.hypot(coors[:, 0], coors[:, 1])
    th = np.arctan2(coors[:, 1], coors[:, 0])
    x = (rr - PLATE.a) / PLATE.extent
    shapes = []
    for (m, psi, pw, loc) in specs:
        amp = x ** pw
        if loc is None:
            uz = amp * np.cos(m * (th - psi))
        else:                                   # 방위 국소 모드(가우시안 창)
            dd = np.angle(np.exp(1j * (th - loc)))
            uz = amp * np.exp(-(dd / 0.25) ** 2)
        u = np.zeros((coors.shape[0], 3))
        u[:, 2] = uz
        u /= np.linalg.norm(u)
        shapes.append(u.ravel())
    return r3.ModalResult(freqs=np.arange(1, len(specs) + 1) * 1000.0,
                          shapes=np.zeros((1, len(specs))), coors=coors,
                          ndof=coors.shape[0] * 3, field_coors=coors,
                          full_shapes=np.stack(shapes))


class TestShapePreservingGrid:
    """T-M1 — 형상보존 격자 규약(F11′)의 기계적 판정."""

    def _pk(self, xi1, xi2, dth_deg=30.0, depth=0.5):
        return {"r1": PLATE.a + xi1 * PLATE.extent,
                "r2": PLATE.a + xi2 * PLATE.extent, "theta0": math.pi / 8,
                "dtheta": math.radians(dth_deg), "depth_frac": depth}

    @pytest.mark.parametrize("nr", [10, 20, 30])
    def test_exact_on_conforming_grids(self, nr):
        _, _, info = r3.disk_mesh(nr=nr, ntheta=48, nz=4,
                                  pocket=self._pk(0.3, 0.6))
        assert info["shape_exact"]
        assert abs(info["snap_r1_mm"]) < 1e-9 and abs(info["snap_r2_mm"]) < 1e-9

    def test_nonconforming_nr_snaps_shape(self):
        """nr=12는 0.3·nr·0.6·nr가 정수가 아니라 반경경계가 다른 셀면으로 스냅된다 —
        F11이 '미수렴'으로 오진했던 원인."""
        _, _, info = r3.disk_mesh(nr=12, ntheta=48, nz=4, pocket=self._pk(0.3, 0.6))
        assert not info["shape_exact"]
        assert max(abs(info["snap_r1_mm"]), abs(info["snap_r2_mm"])) > 0.1

    def test_theta0_must_lie_on_a_cell_face(self):
        """옛 값 θ₀=0.4 rad(22.918°)은 nθ=48의 셀면(7.5° 배수)이 아니어서 형상이 어긋난다."""
        pk = self._pk(0.3, 0.6)
        pk["theta0"] = 0.4
        _, _, info = r3.disk_mesh(nr=20, ntheta=48, nz=4, pocket=pk)
        assert not info["shape_exact"]
        assert abs(info["snap_theta0_deg"]) > 0.1

    def test_depth_must_be_a_layer_multiple(self):
        pk = self._pk(0.3, 0.6, depth=0.3)
        _, _, info = r3.disk_mesh(nr=20, ntheta=48, nz=4, pocket=pk)
        assert not info["shape_exact"]
        assert info["depth_actual"] == pytest.approx(0.25)


class TestOrderMatching:
    """T-M2 — 차수→모드 선택(주파수 순서에 의존하지 않는다)."""

    def test_picks_requested_order_regardless_of_frequency_order(self):
        coors = _cloud()
        # 주파수 오름차순으로 m = 2, 0, 1, 1 (일부러 뒤섞음)
        res = _synth(coors, [(2, 0.0, 2, None), (0, 0.0, 2, None),
                             (1, 0.0, 2, None), (1, math.pi / 2, 2, None)])
        assert r3.match_order(res, 0, n_take=1)["idx"] == [1]
        assert r3.match_order(res, 2, n_take=1)["idx"] == [0]
        got = r3.match_order(res, 1, n_take=2)
        assert got["idx"] == [2, 3] and got["matched"]

    def test_purity_reported_and_gate_rejects_mixed_mode(self):
        coors = _cloud()
        res = _synth(coors, [(0, 0.0, 2, None), (0, 0.0, 2, 0.4)])
        pure = r3.match_order(res, 0, n_take=1, purity_min=0.9)
        assert pure["purity"][0] > 0.9
        # 국소모드만 있는 해에서는 m=0 순도가 낮아 거부돼야 한다
        res2 = _synth(coors, [(0, 0.0, 2, 0.4)])
        assert not r3.match_order(res2, 0, n_take=1, purity_min=0.9)["matched"]


class TestSubspaceMac:
    """T-M3 — MAC이 건전 부분공간 밖의 모드를 거부한다(F21 금지사항의 코드화)."""

    def test_identity_gives_unit_mac(self):
        coors = _cloud()
        res = _synth(coors, [(1, 0.0, 2, None), (1, math.pi / 2, 2, None)])
        out = r3.subspace_mac_match(res, [0, 1], res, n_take=2)
        assert out["matched"] and min(out["mac"]) > 0.999

    def test_rotated_pair_still_matches(self):
        """축퇴쌍은 배향이 회전해도 같은 2차원 부분공간에 있다."""
        coors = _cloud()
        h = _synth(coors, [(1, 0.0, 2, None), (1, math.pi / 2, 2, None)])
        d = _synth(coors, [(1, 0.3, 2, None), (1, 0.3 + math.pi / 2, 2, None)])
        out = r3.subspace_mac_match(h, [0, 1], d, n_take=2)
        assert out["matched"] and min(out["mac"]) > 0.99

    def test_localized_mode_is_rejected(self):
        coors = _cloud()
        h = _synth(coors, [(0, 0.0, 2, None)])
        d = _synth(coors, [(0, 0.0, 2, 0.4)])
        out = r3.subspace_mac_match(h, [0], d, n_take=1, mac_min=0.8)
        assert not out["matched"]

    def test_works_when_damaged_mesh_lost_nodes(self):
        """손상 메시는 절점이 줄어든다 — 공유절점만으로 MAC을 계산해야 한다."""
        coors = _cloud()
        h = _synth(coors, [(2, 0.0, 2, None), (2, math.pi / 4, 2, None)])
        keep = coors[:, 2] < 0.4 * PLATE.t
        d = _synth(coors[keep], [(2, 0.0, 2, None), (2, math.pi / 4, 2, None)])
        out = r3.subspace_mac_match(h, [0, 1], d, n_take=2)
        assert out["matched"] and min(out["mac"]) > 0.99


class TestBandForward:
    """T-B1·T-B2 — 밴드 손상 순방향의 독립 검증."""

    def test_s_bar_roundtrip(self):
        r1, r2 = PLATE.a + 0.3 * PLATE.extent, PLATE.a + 0.5 * PLATE.extent
        for s in (0.01, 0.05, 0.15):
            p = fwd.band_depth_for_s_bar(s, r1, r2, PLATE.extent)
            assert fwd.band_s_bar(r1, r2, p, PLATE.extent) == pytest.approx(s)

    def test_linear_band_matches_degenerate_closed_form(self, pool):
        """Δθ=2π(축대칭)에서 `eta_bar_linear_band`가 `degenerate`의 A̅/λ와 일치."""
        r1, r2 = PLATE.a + 0.3 * PLATE.extent, PLATE.a + 0.5 * PLATE.extent
        p = 0.2
        lin = fwd.eta_bar_linear_band(pool, r1, r2, p, PLATE, coupling="exact")
        for i, (m, n) in enumerate(MODES):
            if m == 0:
                continue
            o = deg.observables(PLATE, m,
                                deg.Pocket(r1, r2, 0.0, 2 * math.pi, p),
                                n_grid=2001, n_r=2001)
            assert lin[i] == pytest.approx(o["eta_bar"], rel=2e-3)

    def test_exact_band_converges_to_linear_at_small_severity(self, pool):
        r1, r2 = PLATE.a + 0.4 * PLATE.extent, PLATE.a + 0.6 * PLATE.extent
        prev = None
        for p in (0.02, 0.005, 0.001):
            lin = fwd.eta_bar_linear_band(pool, r1, r2, p, PLATE)
            ex = fwd.eta_bar_exact_band(PLATE, MODES, r1, r2, p)
            rel = float(np.max(np.abs(lin - ex) / np.abs(ex)))
            if prev is not None:
                assert rel < prev                    # 단조 감소
            prev = rel
        assert prev < 0.01

    def test_exact_band_sign_reversal_at_rim(self, pool):
        """림 밴드에서는 질량제거가 이겨 η̄ > 0 — 강성전용 맵이 표현 못하는 영역(F12)."""
        r1, r2 = PLATE.a + 0.8 * PLATE.extent, PLATE.b
        ex = fwd.eta_bar_exact_band(PLATE, MODES, r1, r2, 0.25)
        assert ex[0] > 0


class TestRitzNumerics:
    """T-B3 — 기저·전처리 변경은 수학을 바꾸지 않는다."""

    def test_precondition_does_not_change_eigenvalues(self):
        for m in (0, 2):
            a = ker.solve_eigenvalues_props(PLATE.a, PLATE.b, PLATE.nu, m,
                                            n_modes=3, n_trial=8, n_grid=2001)
            b = ker.solve_eigenvalues_props(PLATE.a, PLATE.b, PLATE.nu, m,
                                            n_modes=3, n_trial=8, n_grid=2001,
                                            precondition=True)
            assert np.allclose(a, b, rtol=1e-9)

    def test_legendre_basis_matches_monomial_where_both_solve(self):
        r1, r2 = PLATE.a + 0.4 * PLATE.extent, PLATE.a + 0.6 * PLATE.extent
        mono = fwd.eta_bar_exact_band(PLATE, MODES, r1, r2, 0.25, n_trial=8,
                                      basis="monomial")
        legd = fwd.eta_bar_exact_band(PLATE, MODES, r1, r2, 0.25, n_trial=8,
                                      basis="legendre")
        assert np.allclose(mono, legd, rtol=1e-5)

    def test_legendre_reaches_orders_the_monomial_basis_cannot(self):
        """단항 기저는 n_trial=12에서 Cholesky가 깨진다 — 르장드르는 20까지 푼다."""
        r1, r2 = PLATE.a + 0.4 * PLATE.extent, PLATE.a + 0.6 * PLATE.extent
        with pytest.raises(np.linalg.LinAlgError):
            fwd.eta_bar_exact_band(PLATE, MODES, r1, r2, 0.5, n_trial=14,
                                   basis="monomial")
        out = fwd.eta_bar_exact_band(PLATE, MODES, r1, r2, 0.5, n_trial=20,
                                     basis="legendre")
        assert np.all(np.isfinite(out))


class TestSurrogate:
    """T-S1 — 대리모델 진실과 그 모델형식 편향."""

    def test_kappa_one_reproduces_exact_band(self):
        kap = {"xi": [0.2, 0.8], "kappa": [[1.0, 1.0]] * 4, "half_xi": 0.1}
        y = mc.surrogate_truth(PLATE, MODES, 0.5, 0.05, kap)
        r1 = PLATE.a + 0.4 * PLATE.extent
        r2 = PLATE.a + 0.6 * PLATE.extent
        p = fwd.band_depth_for_s_bar(0.05, r1, r2, PLATE.extent)
        assert np.allclose(y, fwd.eta_bar_exact_band(PLATE, MODES, r1, r2, p))

    def test_kappa_interpolates_in_xi(self):
        kap = {"xi": [0.2, 0.8], "kappa": [[1.0, 2.0]] * 4, "half_xi": 0.1}
        y = mc.surrogate_truth(PLATE, MODES, 0.5, 0.05, kap)
        base = mc.surrogate_truth(PLATE, MODES, 0.5, 0.05,
                                  {**kap, "kappa": [[1.0, 1.0]] * 4})
        assert np.allclose(y / base, 1.5, rtol=1e-9)

    def test_surrogate_truth_biases_the_inverse(self):
        """κ≠1이면 자기일관 진실보다 회복오차가 커진다(=모델형식 penalty)."""
        cell = mc.Cell(0.5, 0.05, 1e-4, 12, 7)
        clean = mc.run_cell(cell, PLATE, MODES, W, n_grid=1001, mass="exact")
        kap = {"xi": [0.2, 0.8], "kappa": [[1.3, 1.3], [1.0, 1.0], [0.8, 0.8],
                                           [1.1, 1.1]], "half_xi": 0.1}
        biased = mc.run_cell(cell, PLATE, MODES, W, n_grid=1001, mass="exact",
                             surrogate=kap, n_starts=4)
        assert clean["truth_model"] == "self_consistent_linear"
        assert biased["truth_model"] == "3d_rail_surrogate"
        assert abs(biased["bias_xi_mm"]) > abs(clean["bias_xi_mm"])
        assert biased["abs_err_xi_mm_median"] > clean["abs_err_xi_mm_median"]
