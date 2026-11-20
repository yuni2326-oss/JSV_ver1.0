"""N4(폭·발자국 불일치) 계산의 회귀 테스트 — 앵커 재현, 솔버 동치, 탐색점 보호."""
import math

import numpy as np
import pytest

from impeller_fingerprint import forward as fwd
from impeller_fingerprint import kernels as ker
from impeller_fingerprint import severity as sev
from impeller_fingerprint.rev1 import n4_width as n4


def test_fast_ritz_matches_canonical_solver():
    """미리계산 적분행렬 솔버 = `kernels.solve_eigenvalues_props`(같은 기저·전처리)."""
    P = n4.PLATE
    xi, s, w = 0.4, 0.05, 0.002
    dmg = lambda r: sev.damage_field_xi(r, xi, s, w, P.a, P.b)
    dmg_m = lambda r: fwd.mass_field_exact(dmg(r))
    R = n4.RitzRail(n4.MODES_DEF, n_trial=20, n_grid=2001)
    got = R.eta(n4.Profile("gauss", w), xi, s)
    ref = []
    for m, n in n4.MODES_DEF:
        h = ker.solve_eigenvalues_props(P.a, P.b, P.nu, m, n_modes=n + 1, n_trial=20,
                                        n_grid=2001, precondition=True, basis="legendre")
        d = ker.solve_eigenvalues_props(P.a, P.b, P.nu, m, n_modes=n + 1, damage=dmg,
                                        damage_mass=dmg_m, n_trial=20, n_grid=2001,
                                        precondition=True, basis="legendre")
        ref.append(d[n] / h[n] - 1)
    # 같은 수학 — 차이는 르장드르 Gram 행렬(cond ~1e8)의 반올림뿐(η 단위 ~1e-7, floor의 5e-5)
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-6)


def test_table3a_default_anchor():
    """r3.9 Table 3a 기본집합: CRLB_ξ 0.279 mm, CRLB_S̄ 0.326 %p (ξ=0.5, S̄=5 %, c=0.1 %)."""
    c = n4.crlb_subset(n4.MODES_DEF, 0.5, 0.05, n4.W_REF)
    assert c["crlb_xi_fixed_mm"] == pytest.approx(0.279286, abs=2e-5)
    assert c["crlb_s_fixed_pp"] == pytest.approx(0.326019, abs=2e-5)


def test_sign_reversal_anchor():
    """r3.9 §4.2: ξ* = 0.629, 0.633, 0.706, 0.966 (S̄=3 %, w=3 mm, 생산 격자정의)."""
    xs = n4.sign_reversal_linear(n4.W_REF, 0.03)
    np.testing.assert_allclose(xs, [0.6293125, 0.6332341, 0.7057829, 0.9660765], atol=1e-6)


def test_ceilings_match_plan_table():
    """받은 계획서 §2.16: w=1.5 mm의 물리 상한 12.565 %, 3 mm 25.129 %."""
    assert n4.Profile("gauss", 0.0015).ceiling() == pytest.approx(0.12565, abs=5e-5)
    assert n4.Profile("gauss", 0.003).ceiling() == pytest.approx(0.25129, abs=5e-5)
    assert n4.Profile("band", 0.006).ceiling() == pytest.approx(0.006 / n4.EXT)


def test_in_domain_matches_q_factor():
    p = n4.Profile("gauss", 0.003)
    assert p.in_domain(0.05, 0.05) == pytest.approx(0.05 * 0.691, abs=5e-4)   # 정본 A.3
    assert n4.q_factor(0.2, 0.003) == pytest.approx(0.977, abs=5e-4)


def test_same_profile_control_recovers_exactly():
    """진실 폭 = 3 mm면 정확 역맵(같은 솔버)이 진실을 복원해야 한다."""
    y = n4.truth_eta(n4.GAUSS_REF, 0.37, 0.05)
    f = n4.fit_fixed(y, n4.sigma_obs(n4.MODES_DEF), n4.W_REF, model="exact",
                     starts=(0.2, 0.5, 0.8))
    assert f["xi"] == pytest.approx(0.37, abs=1e-5)
    assert f["s_bar"] == pytest.approx(0.05, abs=1e-6)
    assert f["chi2"] < 1e-6


def test_free_width_never_evaluates_inadmissible_field(monkeypatch):
    """폭 자유 역산의 (ξ, peak, w) 매개화는 어떤 탐색점에서도 peak d_K < 1을 지킨다."""
    seen = []
    orig = n4.FemRail.eta_fields

    def spy(self, d_K):
        seen.append(float(np.max(d_K)))
        return orig(self, d_K)

    y = n4.truth_eta(n4.Profile("band", 0.006), 0.6, 0.15)
    monkeypatch.setattr(n4.FemRail, "eta_fields", spy)
    n4.fit_free(y, n4.sigma_obs(n4.MODES_DEF), starts=((0.6, 3e-3),), a0=0.9)
    assert seen and max(seen) < 1.0


def test_fixed_width_bound_respects_ceiling():
    """폭 고정 역산의 S̄ 상한은 그 폭의 물리 상한(탐색점 보호)."""
    y = n4.truth_eta(n4.Profile("gauss", 0.009), 0.5, 0.15)
    f = n4.fit_fixed(y, n4.sigma_obs(n4.MODES_DEF), 0.0015, model="exact",
                     starts=(0.5,))
    assert f["s_bar"] < n4.Profile("gauss", 0.0015).ceiling()


def test_schur_complement_never_tightens():
    J3 = n4.jac_linear(n4.MODES_DEF, 0.5, 0.05, n4.W_REF)
    c = n4.crlb_fixed_free(J3, n4.sigma_obs(n4.MODES_DEF))
    assert c["crlb_xi_free_mm"] >= c["crlb_xi_fixed_mm"] * (1 - 1e-12)
    assert c["crlb_s_free_pp"] >= c["crlb_s_fixed_pp"] * (1 - 1e-12)


def test_rail_rejects_inadmissible_severity():
    with pytest.raises(ValueError):
        n4.rail().eta(n4.Profile("gauss", 0.0015), 0.5, 0.15)


def test_fem_converged_against_refinement():
    """생산 FEM(N=160) 대 N=320 — 좁고 깊은 손상에서도 0.05 floor 이내."""
    p = n4.Profile("gauss", 0.0015)
    fl = np.sqrt(np.diag(n4.sigma_obs(n4.MODES_EXT)))
    a = n4.FemRail(n4.MODES_EXT, n_elem=160).eta(p, 0.5, 0.10)
    b = n4.FemRail(n4.MODES_EXT, n_elem=320).eta(p, 0.5, 0.10)
    assert np.max(np.abs(a - b) / fl) < 0.05


def test_fem_matches_production_kernels_in_small_damage_limit():
    """S̄ → 0에서 FEM 재해 = 생산 1차 맵(커널) — 모드 기저가 같은 판을 푼다는 확인."""
    for xi in (0.3, 0.7):
        lin = n4.eta_linear(n4.MODES_EXT, xi, 1e-4, 0.003)
        fem = n4.rail(n4.MODES_EXT).eta(n4.GAUSS_REF, xi, 1e-4)
        np.testing.assert_allclose(fem, lin, rtol=2e-3, atol=1e-8)


def test_band_truth_puts_nodes_on_band_edges():
    p = n4.Profile("band", 0.006)
    r1, r2 = p.bounds_r(0.5)
    nodes = n4._fem_nodes(n4.FEM_N, (r1, r2))
    assert np.min(np.abs(nodes - r1)) < 1e-12 and np.min(np.abs(nodes - r2)) < 1e-12


# ---------------------------------------------------------------- N5 대조(리드 후속)
def test_band_fem_converged_deep_band():
    """b6 최심 앵커(d_K = 0.875) — 경계 절점 FEM은 N = 100에서 이미 수렴(≤ 0.01 floor)."""
    r1, r2 = 17.516e-3, 21.748e-3
    a = n4.band_fem_eta(r1, r2, 0.875)
    R = n4.FemRail(n4.MODES_DEF, n_elem=400, cuts=(r1, r2))
    b = R.eta_fields(np.where((R.r >= r1) & (R.r <= r2), 0.875, 0.0))
    fl = np.sqrt(np.diag(n4.sigma_obs(n4.MODES_DEF)))
    assert np.max(np.abs(a - b) / fl) < 0.01


def test_b7_delta_grid_reproduces_b6_anchor_column():
    """정본 δ 열로 조립한 격자는 앵커 노드에서 b6 `delta_modelform`을 그대로 돌려준다."""
    import pandas as pd
    from impeller_fingerprint import cli
    from impeller_fingerprint import montecarlo as mc
    d = pd.read_csv(cli.DATA / "b6_epert_fem.csv")
    spec = n4.b7_delta_spec(d.assign(**{"delta_modelform": d.delta_modelform}),
                            "delta_modelform")
    fam = d[np.isclose(d.half_xi, 0.10) & d.matched.astype(bool)]
    # b7 규약: S̄ 노드는 소수 둘째 자리로 반올림된 값(11.5625 → 11.56 %)이다
    for (x, sp), g in fam.groupby(["xi_c", fam.s_bar_pct.round(2)]):
        got = mc.kappa_eval(spec, float(x), float(sp) / 100, key="delta")
        for _, row in g.iterrows():
            assert got[int(row.m)] == pytest.approx(float(row.delta_modelform), abs=1e-12)


def test_exact_band_map_recovers_its_own_truth():
    fn, hi2 = n4._map_fn("ex_B")
    r1, r2 = n4._b7_band(0.55)
    y = n4.band_fem_eta(r1, r2, 0.135 * n4.EXT / (r2 - r1))
    f = n4._grid_then_lsq(y, n4.sigma_obs(n4.MODES_DEF), fn, hi2, 0.55, n_xi=21, n_2=7)
    assert abs(f["xi"] - 0.55) * n4.EXT * 1e3 < 0.01


def test_two_factor_decomposition_closes():
    import pandas as pd
    rows = []
    for mp, v in zip(n4.MAPS, (-2.0, -1.2, -2.5, 0.3)):
        rows.append({"truth": "t", "xi_d": 0.5, "s_bar": 0.1, "map": mp, "bias_xi_mm": v})
    dec = n4._decompose(pd.DataFrame(rows)).iloc[0]
    assert dec.closure_resid == pytest.approx(0.0, abs=1e-12)
    assert dec.interaction == pytest.approx(-2.0 + 1.2 + 2.5 + 0.3)
    assert dec.path1_lin - dec.path2_lin == pytest.approx(dec.interaction)


def test_decomp_summary_uses_fem_nonsaturated_cells():
    """원고 분해 문단의 부분집합 = FEM b7에서 어느 c에서도 경계에 닿지 않은 셀(N4 §10.2)."""
    import pandas as pd
    b7 = pd.DataFrame([
        {"truth_model": "3d_rail_surrogate", "xi_d": 0.5, "s_bar": 0.12, "sigma_rel": c, "boundary_hit_prob": 0.0}
        for c in (1e-4, 1e-3)] + [
        {"truth_model": "3d_rail_surrogate", "xi_d": 0.2, "s_bar": 0.12, "sigma_rel": 1e-4, "boundary_hit_prob": 0.0},
        {"truth_model": "3d_rail_surrogate", "xi_d": 0.2, "s_bar": 0.12, "sigma_rel": 1e-3, "boundary_hit_prob": 0.2},
        {"truth_model": "kirchhoff_control", "xi_d": 0.2, "s_bar": 0.12, "sigma_rel": 1e-4, "boundary_hit_prob": 0.0}])
    nosat = n4.nosat_fem_cells(b7)
    assert bool(nosat.loc[(0.5, 0.12)]) and not bool(nosat.loc[(0.2, 0.12)])
    rows = []
    for (x, vals) in ((0.5, (-2.0, -1.2, -2.5, 0.3)), (0.2, (-9.0, -9.0, -9.0, -9.0))):
        for tr in ("y3D_fem", "K_band_fem"):
            for mp, v in zip(n4.MAPS, vals):
                rows.append({"truth": tr, "xi_d": x, "s_bar": 0.12, "map": mp,
                             "bias_xi_mm": v if tr == "y3D_fem" else 0.5 * v})
    dec = n4._decompose(pd.DataFrame(rows))
    mf = pd.DataFrame({"xi_d": [0.5, 0.2], "s_bar": [0.12, 0.12],
                       **{f"ratio_c{c:g}": [2.0, 0.5] for c in (1e-4, 3e-4, 1e-3, 3e-3)}})
    out = n4.decomp_summary(dec, mf, nosat)
    by = dict(zip(n4.MAPS, (-2.0, -1.2, -2.5, 0.3)))
    assert out["n_cells"] == 1 and out["n_cells_crlb"] == 1
    assert out["abs_bias_median_mm"]["lin_G"] == pytest.approx(abs(by["lin_G"]))
    assert out["abs_bias_median_mm"]["ex_B"] == pytest.approx(abs(by["ex_B"]))
    sh = out["shapley_share_median"]
    assert sh["lin"] + sh["foot"] + sh["modelform"] == pytest.approx(1.0)
    assert out["n_modelform_below_production"] == 1
    assert out["modelform_gt_crlb"]["0.001"] == 1
    assert out["kirchhoff_only_production_abs_median_mm"] == pytest.approx(0.5 * abs(by["lin_G"]))
