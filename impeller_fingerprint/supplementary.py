"""보충자료 문서 생성 — 정본이 약속한 표를 **CSV에서 직접** 조판한다.

정본은 세 곳에서 보충자료를 참조한다(§Data availability, Appendix A.6, Appendix B.4). 그
약속을 문서로 만들어 두지 않으면 제출 패키지가 불완전하다. 이 모듈은 산출 CSV를 읽어 표를
찍으므로 **손으로 옮겨 적는 단계가 없다** — 재실행하면 문서도 따라 바뀐다.

수록 원칙
  * 본문이 인용한 **바로 그 파일**에서 읽는다(`docs/_generated/data/paper3/`).
  * 열은 부록 B.2가 약속한 것을 우선하되, 조판 가능한 폭으로 **명시적으로 고른다** — 고르고
    남은 열이 CSV에 있다는 사실을 표 아래에 적어 독자가 원본을 찾을 수 있게 한다.
  * 표마다 **생성 명령**을 적는다. 재현 경로가 문서 안에 있어야 보충자료 구실을 한다.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

#: (제목, 파일, 표시할 열, 생성 명령, 설명)
SUPP_TABLES = [
    ("S1", "Production Monte-Carlo — summary by severity and noise level",
     "b1_mc_summary_mass.csv",
     ["xi_d", "s_bar", "sigma_rel", "abs_err_xi_mm_median", "abs_err_xi_mm_p90",
      "abs_err_xi_mm_p95", "bias_xi_mm", "std_xi_mm", "crlb_xi_mm",
      "ratio_std_over_crlb_xi", "coverage95_xi", "boundary_hit_prob"],
     "cli b1 --mass",
     "20 radial locations × 3 severities × 4 noise levels, 5 000 realizations per cell "
     "(1.2 × 10⁶ fits). Appendix B.2 lists the full column set; the CSV carries all 30 "
     "columns including the severity-error quantiles, the parameter correlation and the "
     "joint 95 % ellipse axes."),
    ("S2", "Contact-sensor mass-loading limits, per rail and mode",
     "a14_massload.csv",
     ["rail", "mode", "f_Hz", "mass_total_g", "phi2_max_all_per_kg", "m_eff_all_g",
      "m_limit_all_half_floor_0p05pct_mg", "m_limit_all_floor_0p1pct_mg",
      "dff_pct_all_0p2g", "antinode_r_mm", "antinode_z_mm"],
     "cli a14",
     "δf/f = −½ m_a |φ(x)|² with mass-normalized modes. `m_limit_*` are the admissible "
     "attached masses at the stated budget; `dff_pct_all_0p2g` is the first-order shift a "
     "0.2 g sensor would cause at the antinode."),
    ("S3", "Crack-model arms — frequency ratios at matched depth and width",
     "a11_arm_comparison.csv",
     ["arm", "convention", "a_bar", "width_mm", "ratio_f1", "ratio_f2", "ratio_f3",
      "shift_f1_pct", "shift_f2_pct", "shift_f3_pct"],
     "cli a11",
     "The four independent arms of §3.6-iv at matched crack depth and kerf width. "
     "Table S5's pre-specified centres and bands are read from these rows."),
    ("S4", "Assembly mesh refinement at 1.0 mm",
     "b5_mesh_refine_1p0mm.csv",
     ["config", "damage_frac", "mode", "f_Hz", "ndof", "pair1_mean_Hz",
      "pair1_split_Hz", "pair1_split_pct_of_f"],
     "cli b5 --configs asbuilt --mesh-size 0.0010 --damage 0.0 0.6",
     "Refinement of the nominal assembly configuration used to bracket the splitting SNR "
     "of Table S5 (1.27–1.66 % of f across the 1.2 and 1.0 mm meshes)."),
    ("S6", "Boundary degradation — truncation versus kernel geometry",
     "a15_truncation_vs_kernel.csv",
     ["sigma_rel", "xi_d", "in_domain_fraction", "crlb_xi_mm_at_nominal",
      "crlb_xi_mm_at_effective", "truncation_factor", "kernel_factor_vs_midspan"],
     "in-paper computation, §4.2",
     "The Gaussian is not renormalized on the finite annulus (Appendix A.3), so a boundary "
     "cell carries less integrated damage than its nominal label. Holding the *effective* "
     "severity fixed instead separates the two contributions: `truncation_factor` is the part "
     "the labelling causes, `kernel_factor_vs_midspan` the part the kernel geometry causes."),
    ("S7", "Raw-frequency correlation sweep",
     "a16_rho_sweep.csv",
     ["rho", "crlb_xi_mm_default", "crlb_s_pp_default", "crlb_xi_mm_freq_only",
      "crlb_xi_mm_freq_plus_split", "split_gain_pct"],
     "in-paper computation, §3.5",
     "Correlation is introduced in Σ_f, on the raw frequencies, and propagated through "
     "y = A f. Every individual variance grows with ρ, yet the location bound improves, "
     "because a session-common drift is nearly rank one in observable space; the independent "
     "case reported in §4 is conservative **for the location bound in this configuration** — "
     "the severity bound moves the other way, 0.326 → 0.347 %p from ρ = 0 to 0.6, so the "
     "statement is not a blanket upper bound. No D-optimal subset is listed: the subset search "
     "was not re-run through the production pipeline under correlation, and an ad-hoc search "
     "does not reproduce the pipeline's selection (design record F116)."),
    ("S8", "Assembly mode families — harmonic index and radial order",
     "a17_assembly_mode_families.csv",
     ["mode", "f_Hz", "h_index", "m_dominant", "radial_order_n", "family",
      "m_second", "m_second_share", "in_band_20kHz"],
     "in-paper computation, §3.2",
     "Every elastic mode of the six-vane assembly on the production grid, labelled by cyclic "
     "harmonic index *and* by structural family. The h = 2 representation contains four "
     "distinct (m, n) families inside 20 kHz, including an m4n0 doublet at 12.26 kHz — "
     "sharing an index does not merge two modes into one observable."),
    ("S9", "Matched-damage component/assembly control",
     "a18_matched_damage_control.csv",
     ["case", "f1_flap_Hz", "f2_flap_Hz", "f3_flap_Hz", "ndof", "damage", "note"],
     "in-paper computation, §4.3",
     "The same 60 % root thinning applied to an isolated coupon and to the assembly, so that "
     "the contrast measures assembly coupling rather than a change of damage idealization."),
    ("S10", "Geometry-matched control — component/assembly ladder",
     "a19_geometry_matched_control.csv",
     ["rung", "boundary", "mesh_mm", "f_healthy_Hz", "f_damaged_Hz", "shift_pct",
      "UK_window", "UM_window", "factor_vs_prev", "dilution_matched_geometry"],
     "cli a19",
     "One rung changes one thing: R0 the straight coupon of Table S9, R1 the assembly's own "
     "vane (as-built camber and span) isolated with the same root clamp and the same damage "
     "window, R2 the same vane with both shroud faces clamped, R3 the assembly pair mean read "
     "from Table S4 and the b5 sweep. `UK_window`/`UM_window` are the damaged window's share "
     "of the healthy mode's strain and kinetic energy, which is the first-order sensitivity to "
     "the local rigidity and mass loss and therefore the currency that makes rungs with "
     "different observables comparable. R2 carries no shift: the rigid-shroud limit produces a "
     "dense mode cluster whose damaged counterpart localizes into the thinned quarter, so no "
     "single matched mode exists and frequency-order matching is not permitted; its "
     "`f_healthy_Hz` cell is the lower edge of that cluster, and the CSV carries the upper "
     "edge, the spacing and the count of localized modes below it. R3's `f_damaged_Hz` is its "
     "healthy pair mean carried by the b5 shift rather than a second solve. The CSV also "
     "carries the window volumes, which agree to 1.3 % between the isolated and assembled "
     "meshes and so confirm that both rungs weight the same physical region, and the "
     "first-order predicted shift."),
]


#: r3.6 제출 계열(2026-08-17 개정판 — §5 프로토콜 삭제) 전용 표 목록.
#: 본문 포인터가 이미 박혀 있으므로 **번호가 본문을 따른다**: §3.5·A.5가 "Table S5"로
#: ρ 스윕을 가리키기 때문에 ρ 스윕이 S5가 되도록 배열했다(기본 계열과 번호가 다른 이유).
#: 캠페인 표(기본 계열의 수기 S5)와 Note S3은 본문에서 삭제됐으므로 없다.
SUPP_TABLES_R36 = [
    SUPP_TABLES[0],                                  # S1  생산 MC (b1)
    SUPP_TABLES[1],                                  # S2  질량부하 한계 (a14)
    ("S3", "Crack-model arms — frequency ratios at matched depth and width",
     "a11_arm_comparison.csv",
     ["arm", "convention", "a_bar", "width_mm", "ratio_f1", "ratio_f2", "ratio_f3",
      "shift_f1_pct", "shift_f2_pct", "shift_f3_pct"],
     "cli a11",
     "The four independent arms of Section 4.1 at matched crack depth and kerf width."),
    ("S4", "Assembly mesh refinement at 1.0 mm",
     "b5_mesh_refine_1p0mm.csv",
     ["config", "damage_frac", "mode", "f_Hz", "ndof", "pair1_mean_Hz",
      "pair1_split_Hz", "pair1_split_pct_of_f"],
     "cli b5 --configs asbuilt --mesh-size 0.0010 --damage 0.0 0.6",
     "Refinement of the nominal assembly configuration used to bracket the splitting "
     "response (1.27–1.66 % of f across the 1.2 and 1.0 mm meshes)."),
    ("S5", "Raw-frequency correlation sweep",
     "a16_rho_sweep.csv",
     ["rho", "crlb_xi_mm_default", "crlb_s_pp_default", "crlb_xi_mm_freq_only",
      "crlb_xi_mm_freq_plus_split", "split_gain_pct"],
     "in-paper computation, Section 3.5",
     "Correlation is introduced in Σ_f, on the raw frequencies, and propagated through "
     "y = A f. Every individual variance grows with ρ, yet the location bound improves, "
     "because a session-common drift is nearly rank one in observable space; the independent "
     "case reported in Section 4 is conservative **for the location bound in this "
     "configuration** — the severity bound moves the other way, 0.326 → 0.347 %p from "
     "ρ = 0 to 0.6, so the statement is not a blanket upper bound. No D-optimal subset is "
     "listed, because the subset search was not re-run through the production pipeline "
     "under correlation."),
    ("S6", "Boundary degradation — truncation versus kernel geometry",
     "a15_truncation_vs_kernel.csv",
     ["sigma_rel", "xi_d", "in_domain_fraction", "crlb_xi_mm_at_nominal",
      "crlb_xi_mm_at_effective", "truncation_factor", "kernel_factor_vs_midspan"],
     "in-paper computation, Section 4.2",
     "The Gaussian is not renormalized on the finite annulus (Appendix A.3), so a boundary "
     "cell carries less integrated damage than its nominal label. Holding the *effective* "
     "severity fixed instead separates the two contributions: `truncation_factor` is the part "
     "the labelling causes, `kernel_factor_vs_midspan` the part the kernel geometry causes."),
    ("S7", "Assembly mode families — harmonic index and radial order",
     "a17_assembly_mode_families.csv",
     ["mode", "f_Hz", "h_index", "m_dominant", "radial_order_n", "family",
      "m_second", "m_second_share", "in_band_20kHz"],
     "in-paper computation, Section 3.2",
     "Every elastic mode of the six-vane assembly on the production grid, labelled by cyclic "
     "harmonic index *and* by structural family. The h = 2 representation contains four "
     "distinct (m, n) families inside 20 kHz, including an m4n0 doublet at 12.26 kHz — "
     "sharing an index does not merge two modes into one observable."),
    ("S8", "Matched-damage component/assembly control",
     "a18_matched_damage_control.csv",
     ["case", "f1_flap_Hz", "f2_flap_Hz", "f3_flap_Hz", "ndof", "damage", "note"],
     "in-paper computation, Section 4.3",
     "The same 60 % root thinning applied to an isolated coupon and to the assembly, so that "
     "the contrast measures assembly coupling rather than a change of damage idealization."),
    ("S9", "Geometry-matched control — component/assembly ladder",
     "a19_geometry_matched_control.csv",
     ["rung", "boundary", "mesh_mm", "f_healthy_Hz", "f_damaged_Hz", "shift_pct",
      "UK_window", "UM_window", "factor_vs_prev", "dilution_matched_geometry"],
     "cli a19",
     "One rung changes one thing: R0 the straight coupon of Table S8, R1 the assembly's own "
     "vane (as-built camber and span) isolated with the same root clamp and the same damage "
     "window, R2 the same vane with both shroud faces clamped, R3 the assembly pair mean read "
     "from Table S4 and the b5 sweep. `UK_window`/`UM_window` are the damaged window's share "
     "of the healthy mode's strain and kinetic energy, which is the first-order sensitivity to "
     "the local rigidity and mass loss and therefore the currency that makes rungs with "
     "different observables comparable. R2 carries no shift: the rigid-shroud limit produces a "
     "dense mode cluster whose damaged counterpart localizes into the thinned quarter, so no "
     "single matched mode exists and frequency-order matching is not permitted; its "
     "`f_healthy_Hz` cell is the lower edge of that cluster, and the CSV carries the upper "
     "edge, the spacing and the count of localized modes below it. R3's `f_damaged_Hz` is its "
     "healthy pair mean carried by the shift of Table S4 rather than a second solve. The CSV "
     "also carries the window volumes, which agree to 1.3 % between the isolated and "
     "assembled meshes and so confirm that both rungs weight the same physical region, and "
     "the first-order predicted shift."),
]


def _fmt(v) -> str:
    """셀 값 → 마크다운 안전 문자열.

    **파이프를 반드시 이스케이프한다**: `d_optimal_set` 같은 값이 `m1n0|m2n0|…` 형태였을 때
    그 파이프가 열 구분자로 읽혀 한 셀이 네 칸으로 흘러가고 그 뒤 열 전체가 제목과 어긋났다
    (외부 검토 6차, Table S7). 값이 표를 깨뜨릴 수 없게 만드는 것이 옳은 자리다.
    """
    if isinstance(v, str) and "|" in v:
        return v.replace("|", "\\|")
    if isinstance(v, float):
        if v == 0 or 1e-3 <= abs(v) < 1e5:
            return f"{v:.4g}"
        return f"{v:.3e}"
    return str(v)


def table_markdown(df: pd.DataFrame, cols: list[str]) -> str:
    use = [c for c in cols if c in df.columns]
    head = "| " + " | ".join(use) + " |"
    sep = "|" + "---|" * len(use)
    rows = ["| " + " | ".join(_fmt(r[c]) for c in use) + " |"
            for _, r in df.iterrows()]
    return "\n".join([head, sep] + rows)


#: S1은 240행 전부를 싣지 않는다 — 워드/PDF에서 12쪽을 잡아먹고 열 이름이 글자 단위로 줄바꿈돼
#: 사람이 읽을 수 없다(외부 검토 4차 #8). 전체 표는 데이터 묶음의 CSV가 정본이고, 문서에는
#: **심각도 × 노이즈 12셀 집계**를 싣는다. 집계는 20개 반경에 대한 중앙값이다.
def _summarize_mc(df):
    g = (df.groupby(["s_bar", "sigma_rel"])
           .agg(n_cells=("xi_d", "size"),
                med_abs_err_xi_mm=("abs_err_xi_mm_median", "median"),
                p95_abs_err_xi_mm=("abs_err_xi_mm_p95", "median"),
                med_crlb_xi_mm=("crlb_xi_mm", "median"),
                med_std_over_crlb=("ratio_std_over_crlb_xi", "median"),
                med_coverage95=("coverage95_xi", "median"),
                max_boundary_hit=("boundary_hit_prob", "max"))
           .reset_index())
    return g


def build(data_dir, title: str = "Supplementary material",
          tables=None) -> dict:
    """보충자료 md 문자열과 통계를 만든다. 없는 파일은 **건너뛰지 않고 보고**한다.

    `tables`로 표 목록을 갈아끼울 수 있다(기본 = `SUPP_TABLES`; 제출 계열은
    `SUPP_TABLES_R36`). 목록만 다르고 조판 규약은 같다.
    """
    data_dir = Path(data_dir)
    out = [f"# {title}",
           "",
           "*Symmetry-derived frequency observables and the model-form limit of damage "
           "identification in a pump impeller*",
           "",
           "[Author placeholders]",
           "",
           "Every table below is written directly from the artifact named in its heading; "
           "no value is transcribed by hand. The command that regenerates each artifact is "
           "given with it. The artifacts themselves travel with this document as a separate data "
           "package, listed with byte counts and SHA-256 digests in its manifest; no repository "
           "identifier is quoted, because no permanent data record has been minted yet.",
           ""]
    made, missing = [], []
    for tag, title_, fname, cols, cmd, note in (SUPP_TABLES if tables is None
                                                else tables):
        f = data_dir / fname
        if not f.exists():
            missing.append(fname)
            continue
        df = pd.read_csv(f)
        n_raw = len(df)
        if tag == "S1":
            df = _summarize_mc(df)
            cols = list(df.columns)
        out += [f"## Table {tag}. {title_}",
                "",
                note,
                "",
                (f"*Source:* `{fname}` ({n_raw} rows"
                 + (f", summarized here to {len(df)} rows — the full table is in the "
                    "supplementary data package" if tag == "S1" else "")
                 + f"). *Regenerate with:* `{cmd}`."),
                "",
                table_markdown(df, cols),
                ""]
        made.append((tag, fname, n_raw))
    return {"text": "\n".join(out), "tables": made, "missing": missing}


def build_file(data_dir, out_path) -> dict:
    res = build(data_dir)
    Path(out_path).write_text(res["text"], encoding="utf-8")
    res["out"] = str(out_path)
    return res
