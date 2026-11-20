"""N5·b7 FEM 재실행(`rev1/n5_fem.py`)의 회귀 테스트 — 진실·δ·표의 정의 일치."""
import numpy as np
import pytest

from impeller_fingerprint import montecarlo as mc
from impeller_fingerprint.rev1 import n4_width as n4
from impeller_fingerprint.rev1 import n5_fem as nf
from impeller_fingerprint.rev1 import n5_heldout as n5


def test_fem_surrogate_truth_is_fem_band_plus_interpolated_delta():
    spec = {"xi": [0.2, 0.9], "s_bar": [0.1156, 0.175],
            "delta": (np.ones((4, 2, 2)) * 1e-3).tolist(), "half_xi": 0.10}
    xi, s = 0.55, 0.13
    r1, r2 = n5.band_edges(xi)
    want = n4.band_fem_eta(r1, r2, s * n5.EXT / (r2 - r1)) + 1e-3
    got = nf.fem_surrogate_truth(n5.PLATE, n5.MODES, xi, s, spec)
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-14)


def test_anchor_delta_is_3d_minus_fem_kirchhoff():
    a = nf.anchors_fem()
    r = a.iloc[3]
    kf = nf._k_fem(r.r1_mm, r.r2_mm, r.depth_frac)
    for i, m in enumerate(n5.MS):
        assert r[f"delta_m{m}"] == pytest.approx(r[f"eta3d_m{m}"] - kf[i], abs=1e-14)
        assert r[f"etak_m{m}"] == pytest.approx(kf[i], abs=1e-14)


def test_fem_nodes_never_create_degenerate_elements():
    """경계가 기준 절점과 반올림으로 갈라져도(ξ = 0.73) 요소 길이 ≥ 0.05h."""
    h = n4.EXT / n4.FEM_N
    for xi in (0.73, 0.6, 0.1 + 1e-12, 0.35):
        r1, r2 = n5.band_edges(xi)
        nd = n4._fem_nodes(n4.FEM_N, (r1, r2))
        assert np.all(np.diff(nd) > 0.049 * h)
        eta = n4.band_fem_eta(r1, r2, 0.3)
        assert np.all(np.abs(eta) < 1.0)
