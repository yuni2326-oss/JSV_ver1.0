"""N5(held-out 불일치 보정·b7 편향 분해) 회귀 테스트.

무거운 표(ξ 0.01 × p 0.02)는 쓰지 않고, 같은 코드 경로를 거친 **거친 표**로 검정한다.
데이터 의존 테스트(b6·b7 CSV)는 산출물이 없는 트리에서 건너뛴다(JSV_ver1.0 배포 규약).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from impeller_fingerprint import estimator as est
from impeller_fingerprint import forward as fwd
from impeller_fingerprint import montecarlo as mc
from impeller_fingerprint.rev1 import n5_heldout as n5

HAVE_B6 = (n5.DATA / "b6_epert_fem.csv").exists()
HAVE_B7 = (n5.DATA / "b7_mc_surrogate.csv").exists()


@pytest.fixture(scope="module")
def coarse_band():
    tab = n5.build_band_table(dxi=0.05, dp=0.1, p_max=0.8, workers=4, save=False)
    return n5.BandMap(tab)


@pytest.fixture(scope="module")
def anchors():
    if not HAVE_B6:
        pytest.skip("b6_epert_fem.csv 없음")
    return n5.load_anchors(n5.HALF_REF)


# ------------------------------------------------------------------ 표 보간
def test_piecewise_table_respects_kink():
    """꺾임점을 노드로 두고 조각을 나누면, 꺾인 함수도 노드 사이에서 정확하다."""
    xs = np.linspace(0.0, 1.0, 21)
    ys = np.linspace(0.0, 1.0, 11)
    f = lambda x, y: np.abs(x - 0.5) + np.sin(3 * y)               # noqa: E731
    vals = np.array([[[f(x, y)] for y in ys] for x in xs])
    split = n5.PiecewiseTable(xs, ys, vals, breaks=(0.5,))
    whole = n5.PiecewiseTable(xs, ys, vals)
    q = [(0.47, 0.33), (0.53, 0.71), (0.52, 0.05)]
    err_split = max(abs(split(x, y)[0] - f(x, y)) for x, y in q)
    err_whole = max(abs(whole(x, y)[0] - f(x, y)) for x, y in q)
    assert err_split < 1e-3
    assert err_whole > 3 * err_split


def test_band_table_requires_kink_nodes():
    with pytest.raises(ValueError):
        n5.build_band_table(dxi=0.03, dp=0.4, workers=1, save=False)


def test_band_map_matches_direct(coarse_band):
    """표 보간 ↔ 정본 `eta_bar_exact_band` 직접 계산(거친 표라도 1e-3 안)."""
    for xi, s in [(0.5, 0.135), (0.33, 0.12), (0.88, 0.15)]:
        p = float(n5.depth_from_dk(n5.band_dk(xi, s)))
        assert np.max(np.abs(coarse_band(xi, s) - n5.eta_band_direct(xi, p))) < 1e-3


@pytest.mark.skipif(not (n5.DATA / "rev1_n5_band_table_h0.10.npz").exists(),
                    reason="생산 표 없음")
def test_production_band_table_accuracy():
    """생산 표(ξ 0.01 × p 0.02)는 격자 사이 임의 점에서 직접 계산과 4e-4 안에서 맞는다.

    기준은 표가 아니라 **솔버**가 정한다: 같은 점에서 정본 Ritz(36개)와 44개의 차가 최대
    4.8e-4이므로(2026-10-06 실측), 표 보간오차(중앙 4.8e-5, 최대 3.0e-4)는 그 아래다."""
    bm = n5.BandMap(n5.load_table("band"))
    rng = np.random.default_rng(7)
    for xi, s in zip(rng.uniform(0.12, 0.98, 6), rng.uniform(0.05, 0.18, 6)):
        if bm.penalty(xi, s) > 0:
            continue
        p = float(n5.depth_from_dk(n5.band_dk(xi, s)))
        assert np.max(np.abs(bm(xi, s) - n5.eta_band_direct(xi, p))) < 4e-4


@pytest.mark.skipif(not (n5.DATA / "rev1_n5_gauss_table.npz").exists(), reason="생산 표 없음")
def test_production_gauss_table_accuracy():
    gm = n5.GaussExactMap(n5.load_table("gauss"))
    rng = np.random.default_rng(11)
    for xi, s in zip(rng.uniform(0.02, 0.98, 4), rng.uniform(0.01, 0.23, 4)):
        assert np.max(np.abs(gm(xi, s) - n5.eta_gauss_direct(xi, s))) < 1e-5


def test_band_penalty_only_outside_admissible(coarse_band):
    assert coarse_band.penalty(0.5, 0.15) == 0.0                   # d_K = 0.75
    assert coarse_band.penalty(0.5, 0.199) > 0.0                   # d_K = 0.995 > D_MAX
    assert coarse_band.penalty(0.95, 0.15) > 0.0                   # 경계에서 폭이 줄어 d_K > 1


# ------------------------------------------------------------------ 적합
def test_wls_self_consistent_recovery(coarse_band):
    """같은 맵이 만든 자료는 같은 맵으로 수치정밀도까지 회복된다(역범죄 대조)."""
    sig = n5.sigma_y(1e-3)
    for xi, s in [(0.45, 0.13), (0.72, 0.16)]:
        f = n5.wls_fit(coarse_band(xi, s), coarse_band, sig)
        assert abs(f["xi"] - xi) < 1e-6 and abs(f["s"] - s) < 1e-6
        assert f["chi2"] < 1e-12 and not f["boundary_xi"]


def test_noiseless_optimum_independent_of_noise_scale(coarse_band):
    """Σ_y ∝ c²Σ₀이면 잡음 없는 WLS 최적점은 c와 무관하다(답변서 Eq. M.2의 근거)."""
    y = coarse_band(0.55, 0.14) + np.array([0.004, -0.006, 0.003, -0.002])
    f1 = n5.wls_fit(y, coarse_band, n5.sigma_y(1e-3))
    f2 = n5.wls_fit(y, coarse_band, n5.sigma_y(3e-5))
    assert abs(f1["xi"] - f2["xi"]) < 1e-7 and abs(f1["s"] - f2["s"]) < 1e-7


def test_crlb_scales_linearly_with_c(coarse_band):
    J = n5.jac_fd(coarse_band, 0.5, 0.135)
    a = n5.crlb(J, n5.sigma_y(1e-3))
    b = n5.crlb(J, n5.sigma_y(1e-4))
    assert a[0] / b[0] == pytest.approx(10.0, rel=1e-9)


# ------------------------------------------------------------------ Gaussian 정확맵 수렴
def test_gauss_exact_production_ritz_unconverged_at_high_severity():
    """정본 `eta_bar_exact`(단항 8개)는 S̄ = 15 %의 국소 Gaussian에서 미수렴 — 여기서 Legendre로
    다시 푸는 이유. Legendre 16 ↔ 24는 1e-4 안에서 일치한다."""
    e16 = n5.eta_gauss_direct(0.5, 0.15, n_trial=16)
    e24 = n5.eta_gauss_direct(0.5, 0.15, n_trial=24)
    assert np.max(np.abs(e16 - e24)) < 2e-4
    mono8 = fwd.eta_bar_exact(n5.PLATE, n5.MODES, 0.5, 0.15, n5.W_GAUSS, mass="exact")
    assert np.max(np.abs(mono8 - e24)) > 2e-3


# ------------------------------------------------------------------ δ 격자·fold
def test_delta_spec_nodes_reproduce_anchor_delta(anchors):
    spec = n5.delta_spec(anchors)
    for _, r in anchors.iterrows():
        dh = mc.kappa_eval(spec, r.xi_c, r.s_key / 100.0, key="delta")
        assert np.allclose(dh, [r[f"delta_m{m}"] for m in n5.MS], atol=1e-12)


def test_leave_one_anchor_fills_by_xi_interpolation(anchors):
    """빠진 노드는 같은 S̄ 행의 ξ 이웃 선형보간으로 메운다(`cmd_b7.fill` 규칙)."""
    idx = anchors.index[np.isclose(anchors.xi_c, 0.5) & np.isclose(anchors.s_key, 11.56)][0]
    tr = anchors.drop(index=idx)
    spec = n5.delta_spec(tr)
    lo = anchors[np.isclose(anchors.xi_c, 0.4) & np.isclose(anchors.s_key, 11.56)].iloc[0]
    hi = anchors[np.isclose(anchors.xi_c, 0.6) & np.isclose(anchors.s_key, 11.56)].iloc[0]
    want = [0.5 * (lo[f"delta_m{m}"] + hi[f"delta_m{m}"]) for m in n5.MS]
    assert np.allclose(mc.kappa_eval(spec, 0.5, 0.1156, key="delta"), want, atol=1e-12)


def test_folds_have_no_leakage(anchors):
    for fo in n5.make_folds(anchors):
        tr = fo["train"]
        for idx in fo["held"]:
            r = anchors.loc[idx]
            clash = tr[np.isclose(tr.xi_c, r.xi_c) & np.isclose(tr.s_key, r.s_key)]
            assert clash.empty, fo["desc"]


def test_inside_hull_flags(anchors):
    def tr_without(xi, s=None):
        sel = np.isclose(anchors.xi_c, xi)
        if s is not None:
            sel &= np.isclose(anchors.s_key, s)
        return anchors[~sel]
    assert n5.inside_hull(tr_without(0.5, 11.56), 0.5, 11.56)
    assert not n5.inside_hull(tr_without(0.2, 11.56), 0.2, 11.56)       # 모서리
    assert not n5.inside_hull(tr_without(0.9), 0.9, 17.5)                # ξ 끝 통째
    assert n5.inside_hull(tr_without(0.6), 0.6, 17.5)


def test_nested_loo_cov_is_psd(anchors, coarse_band):
    tr = anchors[~np.isclose(anchors.xi_c, 0.5)]
    C = n5.nested_loo_cov(tr, coarse_band)
    assert C.shape == (4, 4)
    assert np.all(np.linalg.eigvalsh(C) > -1e-15)


def test_corrected_map_self_consistent(anchors, coarse_band):
    corr = n5.CorrectedMap(coarse_band, n5.delta_spec(anchors))
    y = corr(0.47, 0.14)
    f = n5.wls_fit(y, corr, n5.sigma_y(1e-3))
    assert abs(f["xi"] - 0.47) < 1e-5 and abs(f["s"] - 0.14) < 1e-5


# ------------------------------------------------------------------ b7 재현
@pytest.mark.skipif(not (HAVE_B6 and HAVE_B7), reason="b6·b7 산출물 없음")
def test_b7_production_bias_matches_its_covariance(anchors):
    """잡음 없는 생산맵 적합이 b7 MC(c = 0.01 %, 1,000 실현 평균) 편향을 재현한다 — 파일을 만든 공분산 규약으로만.

    08-14 파일(r3.9 §4.3 인용값)은 **옛 균일 2c 가중**으로 계산됐다(F175: 원고 §3.5의 관측량별 바닥이 아니었다).
    현행 `montecarlo.run_cell`(관측량별 바닥, m = 0에 2√2c)로 다시 만든 파일(G1, 2026-10-16)은 현행 규약과 맞는다.
    (ξ, S̄) = (0.5, 13.5 %)에서 두 규약은 0.02 mm 넘게 갈리므로 어느 쪽인지 판별된다."""
    spec = n5.delta_spec(anchors)
    b7 = pd.read_csv(n5.DATA / "b7_mc_surrogate.csv")
    b7 = b7[(b7.truth_model == "3d_rail_surrogate") & np.isclose(b7.sigma_rel, 1e-4)]
    sigmas = {"legacy": n5.sigma_y_legacy(1e-3), "current": n5.sigma_y(1e-3)}

    def noiseless_bias(xi, s, sig):
        y = mc.surrogate_truth(n5.PLATE, n5.MODES, xi, s, spec)
        f = est.fit(y, n5.pool_kernels(), n5.PLATE, sig, w=n5.W_GAUSS, mass="exact", n_starts=4)
        return (f["xi_d"] - xi) * n5.EXT * 1e3

    row = b7[np.isclose(b7.xi_d, 0.5) & np.isclose(b7.s_bar, 0.135)].iloc[0]
    bias = {k: noiseless_bias(0.5, 0.135, sig) for k, sig in sigmas.items()}
    assert abs(bias["legacy"] - bias["current"]) > 0.02
    match = [k for k, v in bias.items() if abs(v - row.bias_xi_mm) < 0.005]
    assert len(match) == 1, (bias, row.bias_xi_mm)
    row = b7[np.isclose(b7.xi_d, 0.75) & np.isclose(b7.s_bar, 0.12)].iloc[0]
    assert noiseless_bias(0.75, 0.12, sigmas[match[0]]) == pytest.approx(row.bias_xi_mm, abs=0.005)
