"""modeselect 테스트 — A5: D-최적 모드부분집합과 모드셋 비교 (정본 §3.5·§4.3(iv)).

정본의 논지: γ₀≈γ₁ 커널 중복성에 대한 **구성적 답**이 D-최적 선택이다. 또한 모드셋
(i) 기본 m=0–3, (ii) D-최적 4모드, (iii) 측정가능 전체, (iv) 주파수만, (v) 주파수+분리
를 비교해 "대칭성 관측량이 무엇을 더 주는가"를 정량화한다.
"""
import numpy as np
import pytest

from impeller_fingerprint import geometry as geo
from impeller_fingerprint import kernels as ker
from impeller_fingerprint import modeselect as ms
from impeller_fingerprint import noise as noi

PLATE = geo.DISK
W = 0.003


@pytest.fixture(scope="module")
def pool():
    return ker.mode_pool(PLATE, ms=(0, 1, 2, 3, 4), ns=(0, 1), n_grid=1001)


class TestPoolAndMeasurability:
    def test_pool_sorted_and_filtered(self):
        full = ker.mode_pool(PLATE, ms=(0, 1, 2, 3, 4), ns=(0, 1), n_grid=801)
        assert len(full) == 10
        fs = [k.f for k in full]
        assert fs == sorted(fs)
        cut = ker.mode_pool(PLATE, ms=(0, 1, 2, 3, 4), ns=(0, 1), n_grid=801,
                            f_max=fs[3] + 1.0)
        assert len(cut) == 4

    def test_measurable_flags_use_daq_band(self, pool):
        rows = ms.pool_table(pool, f_max=2.0e4)
        assert all(r["measurable"] == (r["f_Hz"] <= 2.0e4) for r in rows)
        assert any(r["measurable"] for r in rows)

    def test_radial_order_one_family_is_out_of_band_at_measured_thickness(self, pool):
        """F59 — 실측 두께(레일 t = 2.0 mm)에서 n=1 족이 20 kHz **밖**으로 나간다.

        옛 두께 1.6 mm에서는 m0n1·m1n1·m2n1이 17.9/18.4/19.9 kHz로 대역 안이었고 A5의
        D-최적 집합이 그 모드에 의존했다. f ∝ t이므로 ×1.25 되어 22.3/23.0/24.9 kHz가 된다.
        """
        rows = ms.pool_table(pool, f_max=2.0e4)
        assert [r["label"] for r in rows if r["measurable"]] == [
            "m0n0", "m1n0", "m2n0", "m3n0", "m4n0"]
        assert all(not r["measurable"] for r in rows if r["n"] == 1)
        n1 = {r["label"]: r["f_Hz"] for r in rows if r["n"] == 1}
        assert n1["m0n1"] == pytest.approx(22326.0, rel=2e-3)
        assert n1["m1n1"] == pytest.approx(22965.0, rel=2e-3)
        assert n1["m2n1"] == pytest.approx(24902.0, rel=2e-3)


class TestDOptimal:
    def test_d_optimal_beats_arbitrary_subset(self, pool):
        S4 = noi.sigma_y(4, sigma_rel=5e-4)
        theta = (0.5, 0.05)
        best = ms.d_optimal_subset(pool, PLATE, theta, W, S4, k=4)
        arbitrary = [pool[i] for i in range(4)]
        det_arb = ms.subset_metrics(arbitrary, PLATE, theta, W, S4)["det_F"]
        assert best["det_F"] >= det_arb

    def test_d_optimal_is_exhaustive_maximum(self, pool):
        """조합 수가 작으므로 전수탐색과 일치해야 한다(구현 정확성)."""
        from itertools import combinations
        S3 = noi.sigma_y(3, sigma_rel=1e-3)
        theta = (0.35, 0.03)
        best = ms.d_optimal_subset(pool, PLATE, theta, W, S3, k=3)
        dets = [ms.subset_metrics(list(c), PLATE, theta, W, S3)["det_F"]
                for c in combinations(pool, 3)]
        assert best["det_F"] == pytest.approx(max(dets), rel=1e-12)

    def test_more_modes_never_hurt_information(self, pool):
        """정보는 모드를 추가할 때 감소하지 않는다(같은 σ 가정)."""
        theta = (0.6, 0.04)
        d4 = ms.subset_metrics(pool[:4], PLATE, theta, W,
                               noi.sigma_y(4, sigma_rel=5e-4))["det_F"]
        d6 = ms.subset_metrics(pool[:6], PLATE, theta, W,
                               noi.sigma_y(6, sigma_rel=5e-4))["det_F"]
        assert d6 >= d4


class TestModeSetComparison:
    def test_comparison_table_covers_five_sets(self, pool):
        table = ms.compare_mode_sets(PLATE, pool, theta=(0.5, 0.05), w=W,
                                     sigma_rel=5e-4, f_max=2.0e4)
        names = {row["set"] for row in table}
        assert names == {"default_m0_3", "d_optimal_4", "all_measurable",
                         "freq_only", "freq_plus_splitting"}
        for row in table:
            assert row["crlb_xi_mm"] > 0 and row["crlb_s_pp"] > 0

    def test_splitting_observables_improve_crlb(self, pool):
        """(v) 주파수+분리가 (iv) 주파수만보다 위치 CRLB를 개선한다 (F2의 귀결)."""
        table = {r["set"]: r for r in
                 ms.compare_mode_sets(PLATE, pool, theta=(0.5, 0.05), w=W,
                                      sigma_rel=5e-4, f_max=2.0e4)}
        assert (table["freq_plus_splitting"]["crlb_xi_mm"]
                <= table["freq_only"]["crlb_xi_mm"])

    def test_kernel_collinearity_is_quantified(self, pool):
        """γ₀≈γ₁ 중복성이 부분집합 선택으로 완화되는지 수치로 남긴다."""
        rep = ms.collinearity_report(pool)
        pairs = {(r["a"], r["b"]): r["cos"] for r in rep}
        assert pairs[("m0n0", "m1n0")] > 0.99
        assert min(pairs.values()) < 0.99      # 모든 쌍이 공선은 아니다


class TestMassPropagation:
    """설계서 M7/A5 — `mass`가 내부 호출(subset_metrics·d_optimal_subset)까지 전달되는지.

    사고 이력: `compare_mode_sets`가 `mass`를 받지 않아 `cli a5 --mass` 산출물이
    강성전용 파일과 byte-identical했다.
    """

    def test_d_optimal_subset_uses_mass_forward(self, pool):
        theta = (0.95, 0.05)
        S4 = noi.sigma_y(4, sigma_rel=1e-3)
        k_only = ms.d_optimal_subset(pool, PLATE, theta, W, S4, k=4)
        with_mass = ms.d_optimal_subset(pool, PLATE, theta, W, S4, k=4, mass="exact")
        assert with_mass["det_F"] != pytest.approx(k_only["det_F"], rel=1e-6)

    def test_radial_rows_gain_location_information_at_rim(self, pool):
        """림에서 γ^M이 위치정보를 더한다 → 세 가우시안 행의 CRLB_ξ가 작아진다."""
        theta = (0.95, 0.05)
        a = {r["set"]: r for r in
             ms.compare_mode_sets(PLATE, pool, theta, W, 1e-3)}
        b = {r["set"]: r for r in
             ms.compare_mode_sets(PLATE, pool, theta, W, 1e-3, mass="exact")}
        for name in ("default_m0_3", "d_optimal_4", "all_measurable"):
            assert b[name]["crlb_xi_mm"] < a[name]["crlb_xi_mm"], name

    def test_pocket_rows_are_mass_inclusive_by_construction(self, pool):
        """(iv)/(v)는 `degenerate`가 δK−λδM을 직접 다루므로 `mass` 인자와 무관하다."""
        theta = (0.5, 0.05)
        a = {r["set"]: r for r in
             ms.compare_mode_sets(PLATE, pool, theta, W, 1e-3)}
        b = {r["set"]: r for r in
             ms.compare_mode_sets(PLATE, pool, theta, W, 1e-3, mass="exact")}
        for name in ("freq_only", "freq_plus_splitting"):
            assert b[name]["crlb_xi_mm"] == pytest.approx(a[name]["crlb_xi_mm"],
                                                          rel=1e-12), name
