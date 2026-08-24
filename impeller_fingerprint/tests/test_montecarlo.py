"""montecarlo 테스트 — B1 생산 MC의 계약(소규모로 검정, 생산은 cli에서)."""
import numpy as np
import pytest

from impeller_fingerprint import geometry as geo
from impeller_fingerprint import montecarlo as mc

PLATE = geo.DISK
MODES = [(0, 0), (1, 0), (2, 0), (3, 0)]
W = 0.003


def test_cell_reports_all_required_fields():
    row = mc.run_cell(mc.Cell(0.5, 0.05, 3e-4, 60, 1), PLATE, MODES, W, n_grid=601)
    for key in ("abs_err_xi_mm_median", "abs_err_xi_mm_p95", "abs_err_s_pp_median",
                "boundary_hit_prob", "coverage95_xi", "coverage95_s",
                "ratio_std_over_crlb_xi", "crlb_xi_mm", "ellipse_major",
                "ellipse_angle_deg", "corr_xi_s"):
        assert key in row, key
    assert 0.0 <= row["boundary_hit_prob"] <= 1.0
    assert 0.0 <= row["coverage95_xi"] <= 1.0


def test_coverage_near_nominal_in_asymptotic_regime():
    """소노이즈·중간span에서 CRLB 기반 95 % 구간의 커버리지가 공칭에 가깝다."""
    row = mc.run_cell(mc.Cell(0.5, 0.05, 3e-5, 400, 7), PLATE, MODES, W, n_grid=601)
    assert row["coverage95_xi"] == pytest.approx(0.95, abs=0.05)
    assert row["ratio_std_over_crlb_xi"] == pytest.approx(1.0, abs=0.15)


def test_rim_cell_degrades_and_hits_boundary_more():
    """정본 §4.2의 관측 재현: rim 쪽에서 오차가 커지고 경계접촉이 늘어난다."""
    mid = mc.run_cell(mc.Cell(0.5, 0.02, 1e-3, 200, 3), PLATE, MODES, W, n_grid=601)
    rim = mc.run_cell(mc.Cell(0.95, 0.02, 1e-3, 200, 4), PLATE, MODES, W, n_grid=601)
    assert rim["abs_err_xi_mm_median"] > mid["abs_err_xi_mm_median"]
    assert rim["boundary_hit_prob"] >= mid["boundary_hit_prob"]


def test_production_grid_shape_and_reduction_flag():
    rows = mc.run_production(PLATE, MODES, [0.3, 0.7], [0.05], [1e-3],
                             w=W, n_real=40, n_workers=2, n_grid=601,
                             n_real_requested=5000)
    assert len(rows) == 2
    assert all(r["reduced"] and r["n_real_requested"] == 5000 for r in rows)


def test_deterministic_given_seed():
    a = mc.run_cell(mc.Cell(0.4, 0.03, 1e-3, 50, 42), PLATE, MODES, W, n_grid=601)
    b = mc.run_cell(mc.Cell(0.4, 0.03, 1e-3, 50, 42), PLATE, MODES, W, n_grid=601)
    assert a["abs_err_xi_mm_median"] == b["abs_err_xi_mm_median"]
