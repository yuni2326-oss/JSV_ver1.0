"""Fig 2 정합성 회귀검정 (설계서 F77) — 참고문헌(A12)·그림번호(F73) 검정과 같은 방식.

Fig 2는 2026-08-14까지 **리포에 생성기가 없는 유일한 제출 그림**이었다. 생성기를 신설한
지금 위험은 하나로 옮겨간다: *그림이 그리는 값*과 *정본 캡션·본문이 인용하는 값*이 조용히
갈라지는 것. 그래서 여기서 고정하는 것은 그림의 모양이 아니라 **수치의 단일 출처**다.

  T-F2-1  네 arm이 `a11_arm_comparison.csv`에 실제로 있고 그림이 그 행을 쓴다
  T-F2-2  판별비 D = |Δf₂/f₂|/|Δf₁/f₁|를 테스트가 CSV에서 **독립 재계산**해 일치
  T-F2-3  경쟁 가우시안이 정말 Δf₁으로 정합돼 있다(정합 오차 ≪ 인용 자릿수)
  T-F2-4  곡률-null 실명이 전 arm·전 깊이에서 유지된다(D < 1.1 %)
  T-F2-5  **정본 Fig 2 캡션**의 인용값이 CSV 값을 그 자릿수로 적은 문자열과 일치
  T-F2-6  **정본 §4.1 본문**의 인용값 4계열(2D f₁·2D mode 2·스프링 f₁·가우시안)도 일치
  T-F2-7  `CANON_FIGURES[2]`·`make_all`·CLI 배선이 실제 생성기를 가리킨다
  T-F2-8  생성기가 저장된 CSV만으로 돌아 png를 낸다(맵·FEM 재계산 없음)

하드코딩 금지 규약: 이 파일에 등장하는 균열 수치는 **전부 CSV에서 읽은 값**이고, 리터럴은
정본이 인용하는 *자릿수*(`.2f` 등)와 문장 골격뿐이다.
"""
from __future__ import annotations

import inspect
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from impeller_fingerprint import figures as F

#: 기본값은 **이 체크아웃의** 산출 디렉터리다 — 절대경로를 박으면 클론에서 동작하지
#: 않고 다른 워킹트리의 데이터를 검정한다(설계서 F153). `PAPER3_OUT`으로 덮어쓴다.
DATA = Path(os.environ.get(
    "PAPER3_OUT",
    Path(__file__).resolve().parents[2] / "docs" / "_generated")) / "data" / "paper3"
CANON = (Path(__file__).resolve().parents[2] / "docs" / "paper3-jsv"
         / "2026-07-31-paperB-jsv-v2.1.md")
MINUS = "−"                                   # 정본이 쓰는 유니코드 마이너스(U+2212)
DASH = "–"                                    # 구간 표기용 en dash(U+2013) — 부호와 구별
ARMS_CSV = DATA / "a11_arm_comparison.csv"
T1_CSV = DATA / "a11_table1_conventions.csv"


@pytest.fixture(scope="module")
def series():
    for p in (ARMS_CSV, T1_CSV):
        if not p.exists():
            pytest.skip(f"산출물 없음: {p}")
    return F.fig2_series(DATA)


@pytest.fixture(scope="module")
def arms_csv():
    if not ARMS_CSV.exists():
        pytest.skip(f"산출물 없음: {ARMS_CSV}")
    return pd.read_csv(ARMS_CSV)


@pytest.fixture(scope="module")
def t1_csv():
    if not T1_CSV.exists():
        pytest.skip(f"산출물 없음: {T1_CSV}")
    return pd.read_csv(T1_CSV)



def _canon_number() -> int:
    """`fig2_crack_signature.png`가 지금 **몇 번 그림인지**를 레지스트리에서 찾는다.

    이 검정은 원래 "Figure 2"를 하드코딩했는데, 2026-08-15 재번호로 균열 지문이 Figure 3이
    되면서 전부 깨졌다. 번호는 편집 판단이라 또 바뀔 수 있으므로 **파일명으로 역인용**한다 —
    파일명은 안정 식별자이고 `CANON_FIGURES`가 번호의 단일 출처다.
    """
    for n, e in F.CANON_FIGURES.items():
        if "fig2_crack_signature.png" in e["files"]:
            return n
    raise AssertionError("CANON_FIGURES에 fig2_crack_signature.png이 없다")


@pytest.fixture(scope="module")
def caption():
    if not CANON.exists():
        pytest.skip(f"정본 md 없음: {CANON}")
    n = _canon_number()
    for ln in CANON.read_text(encoding="utf-8").split("\n"):
        if ln.startswith(f"**Figure {n}.**"):
            return ln
    pytest.fail(f"정본에 Figure {n} 캡션이 없다")


@pytest.fixture(scope="module")
def body():
    """§4.1의 균열 지문 단락(STATUS NOTE는 옛 수치를 기록하므로 제외)."""
    if not CANON.exists():
        pytest.skip(f"정본 md 없음: {CANON}")
    return "\n".join(ln for ln in CANON.read_text(encoding="utf-8").split("\n")
                     if not ln.startswith("*[STATUS NOTE"))


def _shift(d: pd.DataFrame, arm: str, conv: str, w: float, a_bar: float, mode: int):
    """CSV에서 (arm, 규약, 커프폭, 깊이, 모드)의 이동량 [%] — 테스트의 독립 경로."""
    g = d[(d.arm == arm) & (d.convention == conv)
          & np.isclose(d.width_mm.to_numpy(float), w)
          & np.isclose(d.a_bar.to_numpy(float), a_bar)]
    assert len(g) == 1, (arm, conv, w, a_bar, len(g))
    return float(g[f"shift_f{mode}_pct"].iloc[0])


def _t1(t: pd.DataFrame, a_bar: float, col: str):
    g = t[np.isclose(t.a_bar.to_numpy(float), a_bar)]
    assert len(g) == 1, (a_bar, col, len(g))
    return float(g[col].iloc[0])


class TestSeriesComesFromCsv:
    def test_four_arms_are_the_canonical_ones(self, series):
        assert [a["key"] for a in series["arms"]] == [
            ("a_exact_spring_EB", "dimarogonas", 0.0),
            ("a_exact_spring_EB", "tada", 0.0),
            ("d_2d_stress_slit", "fem", 0.0),
            ("c_3d_notch_w4.1", "fem", 0.25)]

    def test_arm_rows_exist_in_csv(self, arms_csv):
        for arm, conv, w, *_ in F.FIG2_ARMS + F.FIG2_PATTERN:
            g = arms_csv[(arms_csv.arm == arm) & (arms_csv.convention == conv)
                         & np.isclose(arms_csv.width_mm.to_numpy(float), w)]
            assert not g.empty, f"{arm}/{conv}/w={w} 행이 CSV에 없다"

    def test_missing_arm_raises_instead_of_drawing_empty_line(self, arms_csv):
        with pytest.raises(KeyError):
            F._fig2_arm_rows(arms_csv, "no_such_arm", "fem", 0.0)

    def test_discriminant_recomputed_independently(self, series, arms_csv):
        """T-F2-2 — 그림의 D를 테스트가 CSV에서 다시 계산해 대조한다."""
        for a in series["arms"]:
            arm, conv, w = a["key"]
            for i, ab in enumerate(a["a_bar"]):
                d1 = _shift(arms_csv, arm, conv, w, float(ab), 1)
                d2 = _shift(arms_csv, arm, conv, w, float(ab), 2)
                assert a["shift"][0][i] == pytest.approx(d1, rel=1e-12)
                assert a["shift"][1][i] == pytest.approx(d2, rel=1e-12)
                assert a["discrim_pct"][i] == pytest.approx(100 * d2 / d1, rel=1e-12)

    def test_pattern_is_at_the_anchor_depth(self, series, arms_csv):
        assert series["anchor"] == F.FIG2_ANCHOR
        for p in series["pattern"]:
            arm, conv, w = p["key"]
            for m in (1, 2, 3):
                assert p["shift"][m - 1] == pytest.approx(
                    _shift(arms_csv, arm, conv, w, F.FIG2_ANCHOR, m), rel=1e-12)

    def test_gaussian_is_matched_on_delta_f1(self, series, t1_csv):
        """T-F2-3 — 정합이 성립하지 않으면 (b)의 진술 자체가 무의미하다."""
        assert series["gauss"]["shift"][0] == pytest.approx(
            _t1(t1_csv, F.FIG2_ANCHOR, "shift_f1_pct_2d_stress"), rel=1e-5)
        # 인용 자릿수(0.01 %p)보다 세 자리 이상 좋은 정합이어야 한다.
        assert series["match_rel_err"] < 1e-4

    def test_blindness_holds_in_every_arm_and_depth(self, series):
        """T-F2-4 — mode 2는 어디서도 f₁ 이동의 1.1 %를 넘지 않는다."""
        allD = np.concatenate([a["discrim_pct"] for a in series["arms"]])
        assert allD.max() < 1.1, allD
        assert allD.min() > 0.2, allD
        # 경쟁 매끄러운 장은 같은 지표에서 한 자리 이상 위여야 판별이 성립한다.
        assert series["gauss"]["discrim_pct"] > 10 * allD.max()


class TestCaptionAgreesWithCsv:
    """T-F2-5 — 캡션이 인용한 값 = CSV 값을 그 자릿수로 적은 문자열."""

    def test_discriminants(self, caption, series, t1_csv):
        d2d = _t1(t1_csv, F.FIG2_ANCHOR, "crack2d_ratio_m2_over_m1_pct")
        slide = next(p for p in series["pattern"]
                     if p["key"][0] == "b_timoshenko_modeI+II")["discrim_pct"]
        d3d = next(p for p in series["pattern"]
                   if p["key"][0].startswith("c_3d_notch"))["discrim_pct"]
        gauss = _t1(t1_csv, F.FIG2_ANCHOR, "gauss_ratio_m2_over_m1_pct")
        assert f"{d2d:.2f} % for the zero-width crack" in caption
        assert f"{slide:.2f} % for the spring with Mode-II sliding" in caption
        assert f"{d3d:.2f} % in 3D" in caption
        assert f"against {gauss:.0f} % for the smooth field" in caption

    def test_fundamental_range_and_discriminant_range(self, caption, series):
        f1 = np.concatenate([a["shift"][0] for a in series["arms"]])
        allD = np.concatenate([a["discrim_pct"] for a in series["arms"]])
        assert f"falls by {f1.min():.1f}{DASH}{f1.max():.0f} %" in caption
        assert f"(D = {allD.min():.2f}{DASH}{allD.max():.2f} %)" in caption

    def test_matching_quality_and_gaussian_amplitude(self, caption, series):
        assert (f"matched to {series['match_rel_err'] * 1e6:.1f} × 10⁻⁶"
                in caption)
        assert f"d_max = {series['gauss']['d_max']:.3f}" in caption

    def test_mode2_separation_factor(self, caption, series):
        assert f"factor of {series['gauss_over_crack_f2']:.0f}" in caption

    def test_caption_cites_its_data_inputs(self, caption):
        """캡션은 **데이터 출처**를 인용한다. 생성기 이름·CLI 명령은 2026-08-15 편집에서
        캡션에서 뺐다(외부 검토 4차 #8: 최종 원고에 개발기록성 문구를 두지 않는다) — 그
        배선은 `CANON_FIGURES[n]["maker"]`가 보유하고 `test_figure_numbering::
        TestRegistryMatchesCanonical::test_makers_exist`가 강제하므로 재현성은 잃지 않는다."""
        for tok in ("`fig2_crack_signature.png`", "`a11_arm_comparison.csv`",
                    "`a11_table1_conventions.csv`"):
            assert tok in caption, tok
        assert "cli fig2" not in caption, "생성 명령은 캡션이 아니라 레지스트리가 갖는다"


class TestBodyAgreesWithCsv:
    """T-F2-6 — §4.1 본문의 네 계열도 같은 CSV에서 나온 값이어야 한다."""

    def test_2d_fundamental_drops(self, body, arms_csv):
        v = [_shift(arms_csv, "d_2d_stress_slit", "fem", 0.0, ab, 1)
             for ab in (0.6, 0.5, 0.3)]
        assert (f"{MINUS}{v[0]:.1f} % at ā = 0.6**, {MINUS}{v[1]:.1f} % at "
                f"ā = 0.5, {MINUS}{v[2]:.1f} % at ā = 0.3") in body

    def test_2d_mode2_shifts(self, body, arms_csv):
        v = [_shift(arms_csv, "d_2d_stress_slit", "fem", 0.0, ab, 2)
             for ab in (0.6, 0.5, 0.3)]
        assert f"mode 2 moves {v[0]:.2f} %, {v[1]:.3f} % and {v[2]:.3f} %" in body

    def test_spring_dimarogonas_fundamental_drops(self, body, arms_csv):
        v = [_shift(arms_csv, "a_exact_spring_EB", "dimarogonas", 0.0, ab, 1)
             for ab in (0.6, 0.5, 0.3)]
        assert (f"gives {MINUS}{v[0]:.1f} %, {MINUS}{v[1]:.1f} % and "
                f"{MINUS}{v[2]:.1f} %") in body

    def test_matched_gaussian_mode2_shift(self, body, t1_csv):
        g = _t1(t1_csv, F.FIG2_ANCHOR, "gauss_shift_f2_pct")
        assert f"shifts mode 2 by {g:.1f} %" in body

    def test_discriminant_sentence(self, body, series, t1_csv):
        d2d = _t1(t1_csv, F.FIG2_ANCHOR, "crack2d_ratio_m2_over_m1_pct")
        slide = next(p for p in series["pattern"]
                     if p["key"][0] == "b_timoshenko_modeI+II")["discrim_pct"]
        d3d = next(p for p in series["pattern"]
                   if p["key"][0].startswith("c_3d_notch"))["discrim_pct"]
        gauss = _t1(t1_csv, F.FIG2_ANCHOR, "gauss_ratio_m2_over_m1_pct")
        assert (f"is {d2d:.2f} % for the zero-width elastic crack, {slide:.2f} % for "
                f"the spring model with sliding flexibility and {d3d:.2f} % in 3D, "
                f"against {gauss:.0f} % for the smooth-field competitor") in body


class TestWiring:
    """T-F2-7 — 표·make_all·CLI가 실제 생성기를 가리키는가(빈 생성기 회귀 방지)."""

    def test_registry_entry(self):
        e = F.CANON_FIGURES[_canon_number()]
        assert e["maker"] == "fig2_crack_signature"
        assert e["files"] == ("fig2_crack_signature.png",)
        assert callable(getattr(F, e["maker"]))

    def test_make_all_calls_it(self):
        assert "fig2_crack_signature(" in inspect.getsource(F.make_all)

    def test_cli_has_a_subcommand(self):
        from impeller_fingerprint import cli
        args = cli.build_parser().parse_args(["fig2"])
        assert args.func is cli.cmd_fig2
        assert "fig2_crack_signature" in inspect.getsource(cli.cmd_fig2)


class TestGeneratorRuns:
    """T-F2-8 — 저장된 CSV만으로 돌아야 한다(FEM·Ritz 재계산 금지, F54와 같은 규약)."""

    def test_writes_png_from_csv_only(self, tmp_path):
        for p in (ARMS_CSV, T1_CSV):
            if not p.exists():
                pytest.skip(f"산출물 없음: {p}")
        out = F.fig2_crack_signature(DATA, tmp_path)
        assert out.name == "fig2_crack_signature.png"
        assert out.exists() and out.stat().st_size > 20_000

    def test_missing_input_is_reported_not_guessed(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            F.fig2_series(tmp_path)

    def test_generator_reads_no_other_csv(self):
        """소스가 인용하는 입력 파일이 정확히 두 CSV인지 — 조용한 입력 추가 방지."""
        src = inspect.getsource(F.fig2_series) + inspect.getsource(
            F.fig2_crack_signature)
        assert set(re.findall(r'"([a-z0-9_]+\.(?:csv|npz))"', src)) == {
            "a11_arm_comparison.csv", "a11_table1_conventions.csv"}
