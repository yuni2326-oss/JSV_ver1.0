"""정본 형상(G3, 2026-10-30 표준 두께; 이전 G1) 수정본 수치 — 원고 생성기(`paper3_rev1_redline.py`)의 글자 그대로 쓴
값이 산출 CSV·정본 모듈과 같은지 대조한다. 생성기는 수치를 문자열로 박아 두므로(출처는 주석), 재계산 뒤 어긋남을 여기서
잡는다. 깃 1.0 mm(G3)에서 제출본 r3.9 문장이 그대로 맞는 수치는 생성기의 `G3_UNCHANGED`에 있다.

정본 데이터가 아니면(3D 쿠폰 arm이 정본 폭 `geometry.VANE_WIDTH`가 아니면) 건너뛴다 — 옛 산출물은 그 판의 기록이다.
"""
import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from impeller_fingerprint import geometry as geo
from impeller_fingerprint import kernels as ker

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("PAPER3_OUT", ROOT / "docs" / "_generated")) / "data" / "paper3"
GEN = ROOT / "docs" / "_generated" / "paper3_rev1_redline.py"
ARM_3D = f"c_3d_notch_w{geo.VANE_WIDTH * 1e3:.1f}"
#: N6 키의 흡입구 테 높이 꼬리표(정본 G3 5.0 mm → c50)
_C = f"c{geo.ASBUILT.collar_h * 1e4:.0f}"


def _csv(name: str) -> pd.DataFrame:
    p = DATA / name
    if not p.exists():
        pytest.skip(f"산출물 없음: {p}")
    return pd.read_csv(p)


@pytest.fixture(scope="module")
def gen() -> str:
    if not GEN.exists():
        pytest.skip(f"생성기 없음: {GEN}")
    arms = DATA / "a11_arm_comparison.csv"
    if not arms.exists() or ARM_3D not in set(pd.read_csv(arms).arm):
        pytest.skip("G1 데이터가 아니다 — 수정본 수치 대조는 해당 없음")
    # 생성기는 긴 문장을 암묵 연결 문자열로 나눠 쓴다 — 파서가 이어 붙인 문자열 상수만 모은다
    import ast
    tree = ast.parse(GEN.read_text(encoding="utf-8"))
    return "\n".join(n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str))


def test_table1_cells_equal_csv(gen):
    import importlib.util
    spec = importlib.util.spec_from_file_location("g", GEN)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    t1 = _csv("a11_table1_conventions.csv")
    cols = ["ratio_f1_spring_dim", "ratio_f1_spring_tada", "ratio_f1_2d_stress", "ratio_f2_spring_dim",
            "ratio_f2_2d_stress", "ratio_f3_spring_dim", "ratio_f3_2d_stress"]
    for row, a_bar in zip((1, 2, 3, 4), (0.1, 0.3, 0.5, 0.6)):
        r = t1[np.isclose(t1.a_bar, a_bar)].iloc[0]
        assert m.TABLE1_G1[row] == tuple(f"{r[c]:.4f}" for c in cols), a_bar


def test_rail_and_sandwich_frequencies(gen):
    f = {(mm, n): ker.mode_kernel(geo.DISK, m=mm, n=n).f for mm, n in
         [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (0, 1), (1, 1), (2, 1)]}
    n0 = ", ".join(f"{f[(mm, 0)] / 1e3:.2f}" for mm in range(4)) + f", and {f[(4, 0)] / 1e3:.2f} kHz"
    n1 = ", ".join(f"{f[(mm, 1)] / 1e3:.1f}" for mm in range(2)) + f", and {f[(2, 1)] / 1e3:.1f} kHz"
    assert f"occur at {n0}" in gen
    assert f"occur at {n1}" in gen
    sw = geo.SANDWICH
    d_r = geo.DISK.E * geo.DISK.t ** 3 / (12 * (1 - geo.DISK.nu ** 2))
    scale = math.sqrt((sw.D / sw.rhoh) / (d_r / (geo.DISK.rho * geo.DISK.t)))
    assert f"= {scale:.2f}, which maps" in gen
    sc = ", ".join(f"{f[(mm, 0)] * scale / 1e3:.1f}" for mm in range(4)) + f", and {f[(4, 0)] * scale / 1e3:.1f} kHz"
    assert f"rail frequencies to {sc}" in gen
    assert f"{sw.D:.1f} N·m" in gen and f"{d_r:.2f} N·m" in gen


def test_vane_f1(gen):
    b = geo.VANE
    f1 = (1.8751040687 ** 2 / (2 * math.pi * b.L ** 2)) * math.sqrt(b.E * b.h ** 2 / (12 * b.rho))
    assert f"f₁ = {f1:.1f} Hz" in gen


def test_measured_section_transfer(gen):
    a10 = _csv("a10_sandwich_law.csv")
    sr = a10[(a10.block == "sign_reversal") & (a10.law == "sandwich_asbuilt")].sort_values("m")
    v = [f"{x:.3f}" for x in sr.xi_star]
    assert f"ξ* = {v[0]}, {v[1]}, {v[2]}, and {v[3]} for m = 0 to 3" in gen
    ir = a10[a10.block == "identifiability_ratio"]
    r5 = ir[np.isclose(ir.xi_d, 0.5)].crlb_xi_ratio_asbuilt_over_mono
    r8 = ir[np.isclose(ir.xi_d, 0.8) & np.isclose(ir.s_bar, 0.05)].crlb_xi_ratio_asbuilt_over_mono.iloc[0]
    r95 = ir[np.isclose(ir.xi_d, 0.95) & np.isclose(ir.s_bar, 0.05)].crlb_xi_ratio_asbuilt_over_mono.iloc[0]
    assert f"factors of {r5.min():.2f}–{r5.max():.2f}, {r8:.2f}, and {r95:.2f}" in gen
    lin = a10[(a10.block == "linearization") & np.isclose(a10.xi_d, 0.8) & np.isclose(a10.s_bar, 0.05)]
    e = lin.set_index("law").abs_err_over_floor_max
    assert f"from {e['exact']:.2f} to {e['sandwich_asbuilt']:.2f} times the measurement floor" in gen
    sec = a10[(a10.block == "asbuilt_section") & (a10.p_depth_frac >= 0.05 - 1e-9)]
    z_ab, z_th = sec.zeta_asbuilt, sec.zeta_thin_face
    assert f"= {z_ab.min():.2f}–{z_ab.max():.2f} over *p* = 0.05–0.5" in gen
    assert f"the {z_th.min():.2f}–{z_th.max():.2f} range obtained for the thin-face limit" in gen


def test_crack_numbers(gen):
    t1 = _csv("a11_table1_conventions.csv").set_index("a_bar")
    f1 = [t1.loc[a, "shift_f1_pct_2d_stress"] for a in (0.3, 0.5, 0.6)]
    f2 = [t1.loc[a, "shift_f2_pct_2d_stress"] for a in (0.3, 0.5, 0.6)]
    assert f"decreases by {f1[0]:.1f}%, {f1[1]:.1f}%, and {f1[2]:.1f}%" in gen
    assert f"confined to {f2[0]:.3f}%, {f2[1]:.3f}%, and {f2[2]:.2f}%" in gen
    assert f"approximately {t1.loc[0.5, 'crack2d_ratio_m2_over_m1_pct']:.2f}% for the zero-width crack" in gen
    b2 = _csv("b2_vane3d.csv")
    s = b2[np.isclose(b2.vane_width_mm, geo.VANE_WIDTH * 1e3) & np.isclose(b2.kerf_width_actual_mm, 0.25)
           & np.isclose(b2.depth_frac_actual, 0.5)].iloc[0]
    assert f"gives {s.abs_shift_f2_pct:.3f}% against a {100 * (1 - s.ratio_f1):.1f}% fundamental drop" in gen
    a7 = _csv("a7_crack_shear.csv")
    c9 = a7[(a7.arm == "modeI+II_c0.9") & np.isclose(a7.a_bar, 0.5) & (a7.convention == "tada")
            & np.isclose(a7.h_over_L, geo.VANE.h / geo.VANE.L)].iloc[0]
    assert f"increases only to {100 * (1 - c9.ratio_f2):.3f}%" in gen
    cv = _csv("a11_convergence.csv")
    fin = cv[cv.ref == cv.ref.max()][["d_shift_f1_pct_rel", "d_shift_f2_pct_rel", "d_shift_f3_pct_rel"]]
    assert f"frequency shift by {100 * fin.abs().max().max():.2f}% at the finest step" in gen


def test_n3_robustness_numbers(gen):
    sw = _csv("rev1_n3_location_sweep.csv")
    r35 = sw[np.isclose(sw.xc_over_L, 0.35) & np.isclose(sw.a_bar, 0.5)].iloc[0]
    assert f"coincided ({r35.D21_crack_pct:.1f}%)" in gen
    mc = _csv("rev1_n3_multicrack.csv")
    away = mc[mc.case.str.startswith("x2_")].D21_crack_pct
    assert f"ratios of {away.min():.0f}%–{away.max():.0f}%" in gen
    bc = _csv("rev1_n3_breathing_contrast.csv")
    o = bc[bc.model == "open"].set_index(["xc_over_L", "a_bar"])
    fac = []
    for mdl in ("harmonic_mean", "td_unilateral"):
        t = bc[bc.model == mdl].set_index(["xc_over_L", "a_bar"])
        fac += list(t.d_f1_pct / o.d_f1_pct) + list(t.d_f2_pct / o.d_f2_pct)
    assert f"same factor of {min(fac):.2f}–{max(fac):.2f}" in gen
    fl = _csv("rev1_n3_root_fillet_2d.csv").drop_duplicates("case")
    nulls = fl[fl.case.str.startswith("base")].mode2_null_xL_2d
    assert f"moved the null to {nulls.min():.3f}–{nulls.max():.3f}" in gen
    di = _csv("rev1_n3_depth_inversion.csv")
    deep = di[(di.generator == "2d_stress_slit") & (di.a_bar_true >= 0.6)]["resid_noise_units_c0.001"].abs()
    shallow = di[(di.generator == "2d_stress_slit") & (di.a_bar_true <= 0.5)]["resid_noise_units_c0.001"].abs()
    assert f"reached {deep.min():.1f}–{deep.max():.1f} standard deviations" in gen
    assert shallow.max() < 0.9 and "below 0.9 standard deviations" in gen


def test_disk3d_reference_and_refinement(gen):
    """§4.3 [369]: 건전 3D 원판 대 Kirchhoff–Ritz 레일, 단방향 격자 세분(작업 격자 20×48×4 기준)."""
    b4 = _csv("b4_convergence.csv")
    w = b4[(b4.nr == 20) & (b4.ntheta == 48) & (b4.nz == 4)].iloc[0]
    ref = b4[b4.ndof > w.ndof]
    de = (100 * (ref.eta_bar_m2 / w.eta_bar_m2 - 1)).abs().max()
    dd = (100 * (ref.delta_eta_m2 / w.delta_eta_m2 - 1)).abs().max()
    assert f"pair-mean observable by at most {de:.2f}% and the splitting by at most {dd:.2f}%" in gen
    # m = 0이 양의 끝, m = 2는 범위 안(m = 1·3·4는 b3 로그에만 있다)
    k0 = 100 * (w.f_m0_healthy / ker.mode_kernel(geo.DISK, m=0, n=0).f - 1)
    k2 = 100 * (w.f_m2_healthy / ker.mode_kernel(geo.DISK, m=2, n=0).f - 1)
    # 아래 끝 −0.33 %는 m = 3(b3 로그 건전 줄 2754.4 Hz; G1은 m = 3의 −0.29 %)
    assert f"Kirchhoff–Ritz rail by −0.33% to +{k0:.2f}%" in gen
    assert -0.33 < k2 < k0


def test_installed_reversal_numbers(gen):
    """§4.2 조립체: 설치 상태(32×16) 부호반전과 24×12 감도(좁은 허브 고정·wrap 55°/70°)."""
    def lowest3(key):
        d = _csv(f"rev1_n6_signrev_{key}.csv").set_index("label")
        return ", ".join(f"{float(d.loc[k, 'xi_star']):.3f}" for k in ("h1:m1n0", "h0:m0n0")) + \
            f", and {float(d.loc['h2:m2n0', 'xi_star']):.3f}"
    base = f"ff32x16h6w62{_C}k92"
    sr = _csv(f"rev1_n6_signrev_{base}.csv").set_index("label")
    rg = _csv(f"rev1_n6_registry_{base}.csv").set_index("label")
    m3 = sr.loc["h3:m3n1"]
    assert (f"= {lowest3(base)}, as does the m3n1 family at {m3.f_Hz / 1e3:.2f} kHz, with "
            f"{100 * rg.loc['h3:m3n1', 'Efrac_front']:.0f}% in the front shroud, at {float(m3.xi_star):.3f}") \
        in gen.replace("*ξ*_{d} ", "")
    s = (f"the three lowest families reverse at {lowest3(f'ff24x12h6w62{_C}k92')}; at "
         f"{lowest3(f'ff24x12h6w62{_C}k78')} with the hub clamped only to *r* = 7.78 mm; and at "
         f"{lowest3(f'ff24x12h6w55{_C}k92')} and at {lowest3(f'ff24x12h6w70{_C}k92')} with wraps of 55° and 70°.")
    assert s in gen


def test_rail_support_wrap_numbers(gen):
    """§4.2 조립체: 레일 지지 24×12의 m0n0(과 wrap 70°의 m1n0) 부호반전 — wrap 55/62/70°."""
    def row(w, lab):
        key = f"fit24x12v1z2c2w{w}"
        s = _csv(f"rev1_n1_signrev_{key}.csv").set_index("label")
        r = _csv(f"rev1_n1_registry_{key}.csv").set_index("label")
        return s.loc[lab, "xi_star"], 100 * r.loc[lab, "Efrac_front_shroud"]
    x55, e55 = row(55, "h0:m0n0")
    x62, e62 = row(62, "h0:m0n0")
    x70, e70 = row(70, "h0:m0n0")
    x70b, _ = row(70, "h1:m1n0")
    assert not np.isfinite(x55)
    s = (f"keeps one sign with a 55° wrap, with {e55:.0f}% of its strain energy in the front shroud, "
         f"and reverses at {x62:.3f} with the 62° wrap ({e62:.0f}%) and at {x70:.3f} with a 70° wrap ({e70:.0f}%), "
         f"where the m1n0 family also reverses, at {x70b:.3f}.")
    assert s in gen


def test_common_mode_offset_numbers(gen):
    """§5 운전 범위: 균일 오프셋(공통모드 nuisance)을 둔 중간 경간 CRLB — 20 kHz 아래 전부(레일 지지·설치 상태)."""
    vals = []
    for pre, key in (("n1", "fit32x16v1z2c2w62"), ("n6", f"ff32x16h6w62{_C}k92")):
        s = _csv(f"rev1_{pre}_selection_{key}.csv")
        q = s[np.isclose(s.xi, 0.5) & np.isclose(s.s_bar, 0.05) & (s.pool == "f<=20.0kHz")
              & (s.set == "all_resolvable") & s.model.str.startswith("assembly")].iloc[0]
        vals.append((q.crlb_xi_mm, q.crlb_xi_mm_cm))
    pct = [100 * (b / a - 1) for a, b in vals]
    s = (f"but by only {pct[0]:.0f}% and {pct[1]:.0f}% (to {vals[0][1]:.3f} and {vals[1][1]:.3f} mm with the "
         "rail support")
    assert s in gen


def test_pocket_assembly_numbers(gen):
    """§4.3 국소 포켓(레일 지지 24×12): 날개 상대 방위 의존 비와 Table S13 팔 비교 — 생성기 N2 사전."""
    N2 = _gen_module().N2
    key = "fit24x12v1z2c2w62"
    sw = _csv(f"rev1_n2_sweep_{key}.csv")
    s = sw[np.isclose(sw.xi, 0.2) & np.isclose(sw.dtheta_deg, 30.0)]

    def big(x):
        return f"{x:.0f}" if x >= 10 else f"{x:.1f}"

    for k, lab in (("r_h1", "h1:m1n0"), ("r_h2", "h2:m2n0"), ("r_h0", "h0:m0n0"), ("r_h3", "h3:m3n0")):
        m = s[s.label == lab]["mean"].abs()
        assert N2[k] == big(m.max() / m.min()), (k, m.max() / m.min())

    def arms(fn):
        a = _csv(fn)
        m = a.groupby(["xi", "dtheta_deg", "arm"]).crlb_xi_mm.median().unstack("arm")
        A, Ap = m["A_mean_theta_known"], m["A'_mean_theta_nuisance"]
        return (Ap / A, 100 * (1 - m["B_mean+split_theta_known"] / A),
                100 * (1 - m["C_mean+split_theta_nuisance"] / Ap))

    sup = (ROOT / "docs" / "_generated" / "paper3_rev1_supp_redline.py").read_text(encoding="utf-8")
    cut = re.search(r"^S13_FCUT_K = ([0-9.]+)", sup, re.M).group(1)
    pen, b, c = arms(f"rev1_n2_arms_{key}_f{float(cut):g}k.csv")
    assert N2["low_pen"] == f"{pen.min():.1f}–{big(pen.max())}"
    assert N2["low_B"] == f"{b.min():.0f}%–{b.max():.0f}%" and N2["low_C"] == f"{c.min():.0f}%–{c.max():.0f}%"
    pen, b, c = arms(f"rev1_n2_arms_{key}.csv")
    assert N2["all_pen"] == f"{pen.min():.2f}–{pen.max():.2f}"
    assert N2["all_B"] == f"{b.min():.0f}%–{b.max():.0f}%" and N2["all_C"] == f"{c.min():.0f}%–{c.max():.0f}%"
    # P['n2_nl']: 낮은 네 계열의 재해석/1차 쌍평균 비(짝지은 것 = 부분공간 MAC ≥ 0.8)와 짝짓기 실패 수
    nl = _csv(f"rev1_n2_nonlin_{key}.csv")
    s = nl[nl.label.isin(["h0:m0n0", "h1:m1n0", "h2:m2n0", "h3:m3n0"])]
    ok = s.mac.notna() & (s.mac >= 0.8)
    r = (s.mean_res / s.mean_lin)[ok]
    P = _gen_module().P
    assert P["n2_nl"].startswith(f"{r.min():.2f}–{r.max():.1f} (median {r.median():.1f}), and "
                                 f"{_num(int((~ok).sum()))} of the {len(s)} lowest-family responses")


def test_pocket3d_numbers(gen):
    """§4.3 [370]–[373]: 3D 포켓 원판 9개 형상의 MAC 유지·쌍평균 비·60° 잔여 분리·생산 맵 역식별."""
    d = _csv("b3_disk3d.csv")
    if len(d) != 9:
        pytest.skip("b3가 원고의 9개 형상(Δθ 15/30/60°)이 아니다")
    words = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
    n0, n1 = int(d.m0_matched.sum()), int(d.m1_matched.sum())
    assert d.m2_matched.all() and d.m3_matched.all()
    if n0 == n1:
        assert f"are each retained in only {words[n0]} cases." in gen
    else:                                   # G3: m0 4·m1 3 = r3.9 문장 그대로
        assert f"are retained in only {words[n0]} and {words[n1]} cases, respectively." in gen
    rat, same = [], 0
    for m in (1, 2, 3):
        ok = d[d[f"m{m}_matched"]]
        rat += list(ok[f"eta_bar_3d_m{m}"] / ok[f"eta_bar_theory_m{m}"])
        same += int((np.sign(ok[f"eta_bar_3d_m{m}"]) == np.sign(ok[f"eta_bar_theory_m{m}"])).sum())
    assert same == len(rat) and f"All {len(rat)} matched m > 0" in gen
    assert f"ranges from {min(rat):.2f} to {max(rat):.2f}, with a median of {np.median(rat):.2f}." in gen
    s60 = d[np.isclose(d.dtheta_deg, 60)].delta_eta_3d_m3 * 1e3
    assert np.allclose(d[np.isclose(d.dtheta_deg, 60)].delta_eta_theory_m3, 0, atol=1e-12)
    assert f"residual splitting of {s60.min():.1f} to {s60.max():.1f} × 10⁻³" in gen
    x = _csv("b3_modelform_mass.csv")
    sat = np.isclose(x.xi_hat_3d_mgt0, 1.0)
    ns = x[~sat].err_xi_mm_3d_mgt0.abs()
    assert f"ranges from {ns.min():.2f} to {ns.max():.2f} mm for the nonsaturated cases." in gen
    assert int(sat.sum()) == 3 and np.allclose(x[sat].err_xi_mm_3d_mgt0.abs(), 4.232)
    assert (x[sat].xi_d_true == 0.8).all() and "with an absolute location error of 4.23 mm" in gen
    assert int((x.penalty_over_crlb_mgt0 > 1).sum()) == 8          # r3.9 그대로: "eight of the nine configurations"
    assert f"median error-to-CRLB ratio of {x.penalty_over_crlb_mgt0.median():.1f}." in gen
    assert x.err_xi_mm_samemodel.abs().max() < 1e-9


def _gen_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("g", GEN)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_WORDS = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
          8: "eight", 9: "nine"}


def _num(n: int) -> str:
    return _WORDS.get(int(n), str(int(n)))


def test_band_study_numbers(gen):
    """§4.3 띠 연구(b6 재확인 FEM·모드별 floor)·F36·LOO — P 사전 대 산출 CSV."""
    P = _gen_module().P
    r = _csv("rev1_n4_reconcile_b6_recheck.csv")
    r = r[(r.floor == "mode_specific") & (r.kirchhoff == "fem")].iloc[0]
    assert (P["b6_lin"], P["b6_mf"], P["b6_total"], P["b6_factor"]) == (
        f"{r.median_e_lin_floors:.1f}", f"{r.median_e_modelform_floors:.1f}",
        f"{r.median_e_total_floors:.0f}", f"{r.ratio_lin_over_mf:.1f}")
    n_opp = int(r.n_opposite_sign_K_vs_3D)
    assert P["b6_opp"] == str(n_opp) and P["b6_opp_word"] == _num(n_opp)
    f = _csv("rev1_n4_f36_ratio_recheck.csv")
    assert P["b6_gauss_vs_band"] == f"{f.ratio_fem.median():.1f} (maximum {f.ratio_fem.max():.0f})"
    lo = _csv("rev1_n4_b7_loo_recheck.csv")
    lo = lo[lo.solver == "fem"].resid_rel
    assert (P["loo_median"], P["loo_p95"]) == (f"{100 * lo.median():.1f}%", f"{100 * lo.quantile(0.95):.0f}%")


def test_b7_surrogate_numbers(gen):
    """§4.3 대리 MC(FEM 진실) — `n5_heldout.b7_stats`와 같은 규칙, 포화 셀은 ξ < 0.5 = 안쪽 경계."""
    from impeller_fingerprint.rev1 import n5_heldout as n5
    P = _gen_module().P
    d = _csv("rev1_n4_b7_fem.csv")
    st = n5.b7_stats(d)
    lo, hi = st["nonsat_abs_bias_range_mm"]
    assert P["b7_range"] == (f"{lo:.2f}" if lo >= 0.01 else f"{lo:.1g}") + f" to {hi:.2f}"
    assert P["b7_median"] == f"{st['nonsat_abs_bias_median_mm']:.2f}"
    assert P["b7_std_crlb"] == f"{st['std_over_crlb_median']:.2f}"
    assert P["b7_n_gt"] == str(st["cells_bias_gt_crlb"]) and st["n_cells"] == 180
    assert P["b7_ratio"] == f"{st['bias_over_crlb_median']:.0f}"
    assert P["b7_change"] == f"{100 * st['bias_change_30x_median']:.1f}%"
    s = d[(d.truth_model == "3d_rail_surrogate") & (d.boundary_hit_prob > 0)]
    n_in, n_out = int((s.xi_d < 0.5).sum()), int((s.xi_d >= 0.5).sum())
    assert n_in + n_out == st["n_saturated"]
    want = (f"{st['n_saturated']} cells in which the estimate reached a parameter boundary "
            f"({_num(n_in)} at the inner and {_num(n_out)} at the outer limit)" if n_out
            else f"{st['n_saturated']} cells in which the estimate reached the inner parameter boundary")
    assert P["b7_sat"] == want


def test_decomposition_numbers(gen):
    """§4.3 분해 문단 — `n4_width decompsum` 요약(FEM b7 비포화 부분집합)."""
    import json
    p = DATA / "rev1_n4_decomposition_summary.json"
    if not p.exists():
        pytest.skip("decompsum 요약 없음")
    js = json.loads(p.read_text())
    D = _gen_module().D
    ab, sh = js["abs_bias_median_mm"], js["shapley_share_median"]
    sgn = lambda v: f"{v:+.2f}".replace("-", "−")
    gt = js["modelform_gt_crlb"]
    assert D == {"n": str(js["n_cells"]), "a": f"{ab['lin_G']:.2f}", "b": f"{ab['ex_G']:.2f}",
                 "ap": f"{ab['lin_B']:.2f}", "c": f"{ab['ex_B']:.2f}",
                 "n_c_lt_a": str(js["n_modelform_below_production"]),
                 "inter": f"{js['interaction_median_mm']:.2f}",
                 "foot1": sgn(js["path1_median_mm"]["foot"]), "foot2": sgn(js["path2_median_mm"]["foot"]),
                 "sh_lin": f"{100 * sh['lin']:.0f}%", "sh_foot": f"{100 * sh['foot']:.0f}%",
                 "sh_mf": f"{100 * sh['modelform']:.0f}%",
                 "gt": f"{gt['0.0003']}, {gt['0.001']}, and {gt['0.003']}"}
    assert js["n_cells_crlb"] == js["n_cells"]


def test_heldout_numbers(gen):
    """§4.3 held-out 보정 문단·Table 6 — `n5_fem summary` JSON."""
    import json
    p = DATA / "rev1_n5_heldout_fem_summary.json"
    if not p.exists():
        pytest.skip("N5 FEM 요약 없음")
    h = json.loads(p.read_text())
    H = _gen_module().H
    k = "ho_leave_one_anchor_"
    f2, f3 = (lambda v: f"{v:.2f}"), (lambda v: f"{v:.3f}")
    want = {"n_in": str(h[k + "interp_n"]), "n_out": str(h[k + "extrap_n"]),
            "in_prod": f2(h[k + "interp_prod_lin_gauss_abs_err_xi_median_mm"]),
            "in_unc": f2(h[k + "interp_unc_band_abs_err_xi_median_mm"]),
            "in_cor": f3(h[k + "interp_cor_band_abs_err_xi_median_mm"]),
            "in_s_unc": f2(h[k + "interp_unc_band_abs_err_s_median_pp"]),
            "in_s_cor": f3(h[k + "interp_cor_band_abs_err_s_median_pp"]),
            "out_prod": f2(h[k + "extrap_prod_lin_gauss_abs_err_xi_median_mm"]),
            "out_unc": f2(h[k + "extrap_unc_band_abs_err_xi_median_mm"]),
            "out_cor": f2(h[k + "extrap_cor_band_abs_err_xi_median_mm"]),
            "out_s_cor": f2(h[k + "extrap_cor_band_abs_err_s_median_pp"]),
            "ext_prod": f2(h["ext_prod_lin_gauss_abs_err_xi_median_mm"]),
            "ext_unc": f2(h["ext_unc_band_abs_err_xi_median_mm"]),
            "ext_cor": f3(h["ext_cor_band_abs_err_xi_median_mm"]),
            "ext_s_cor": f3(h["ext_cor_band_abs_err_s_median_pp"]),
            "cov_in": f2(h["mc_leave_one_anchor_c0.001_cor_band_coverage95_xi_median"]),
            "cov_ext": f2(h["extmc_c0.001_cor_band_coverage95_xi_median"]),
            "cov_in_unc": f2(h["mc_leave_one_anchor_c0.001_unc_band_coverage95_xi_median"]),
            "cov_ext_unc": f2(h["extmc_c0.001_unc_band_coverage95_xi_median"])}
    assert {kk: H[kk] for kk in want} == want
    n_gt, n_fl = h["corrected_gt_0p1mm_n"], h["corrected_gt_0p1mm_flagged"]
    if n_fl == 0:
        assert f"none of the {_num(n_gt)} corrected cases with location errors above 0.1 mm exceeded" in H["gof"]
    else:
        assert (f"of the {_num(n_gt)} corrected cases with location errors above 0.1 mm, only {_num(n_fl)} "
                "exceeded") in H["gof"]
    cs = h["corrected_gt_0p1mm_cases"]
    e, x = [abs(c["e_xi_mm"]) for c in cs], [c["xi"] for c in cs]
    assert f"(*ξ*_{{d}} ≈ {min(x):g}–{max(x):g}), where location errors of {min(e):.2f}–{max(e):.2f} mm" in H["resid"]
    assert f"where they reached {h[k + 'extrap_cor_band_abs_err_xi_max_mm']:.2f} mm" in H["resid"]
    ext = _csv("rev1_n5_heldout_fem_external_test.csv")
    n_ext = len(ext.drop_duplicates(["xi_true", "s_true"]))
    assert H["n_ext"] == str(n_ext) and H["n_ext_word"] == _num(n_ext)
