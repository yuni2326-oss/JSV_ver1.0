"""정본이 인용하는 질량부하 수치가 `a14_massload.csv`와 일치하는가 (F57).

설계서 §11.6의 규약: **정본 md 안에만 존재하는 사실은 검정할 수 없다.** §5 E1의 mg 한계와
|φ|² 범위는 프로토콜을 바꾸는 수치이므로(접촉센서 배제) 산출 CSV에 대해 회귀검정한다.
격자·레일을 바꿔 값이 달라지면 이 검정이 먼저 깨진다.
"""
import os
import re
from pathlib import Path

import pandas as pd
import pytest

#: 기본값은 **이 체크아웃의** 산출 디렉터리다 — 절대경로를 박으면 클론에서 동작하지
#: 않고 다른 워킹트리의 데이터를 검정한다(설계서 F153). `PAPER3_OUT`으로 덮어쓴다.
DATA = Path(os.environ.get(
    "PAPER3_OUT",
    Path(__file__).resolve().parents[2] / "docs" / "_generated")) / "data" / "paper3"
A14 = DATA / "a14_massload.csv"
CANON = (Path(__file__).resolve().parents[2] / "docs" / "paper3-jsv"
         / "2026-07-31-paperB-jsv-v2.1.md")

#: 이 파일은 전부 **원고 ↔ CSV 대조**다. 코드·데이터만 배포한 트리에는 원고가 없으므로
#: 건너뛴다(원고가 있으면 전부 실행된다).
pytestmark = pytest.mark.skipif(not CANON.exists(),
                                reason=f"원고 없음: {CANON}")


@pytest.fixture(scope="module")
def a14():
    if not A14.exists():
        pytest.skip(f"{A14} 없음 — `cli a14`를 먼저 실행한다")
    return pd.read_csv(A14)


@pytest.fixture(scope="module")
def canon():
    txt = CANON.read_text()
    # STATUS NOTE는 이력이라 옛 수치를 담고 있다 — 본문만 본다
    return "\n".join(l for l in txt.split("\n") if "STATUS NOTE" not in l)


def _ref_num(doi: str) -> int:
    """DOI → 현재 참고문헌 번호. 번호는 첫 인용 순서라 편집 때마다 바뀌므로 하드코딩하지 않는다."""
    from impeller_fingerprint import references as R
    for r in R.REFERENCES:
        if r.doi.lower() == doi.lower():
            return r.num
    raise AssertionError(f"참고문헌 표에 {doi}가 없다")


def _rng(df, rail, col):
    v = df.loc[df["rail"] == rail, col]
    return float(v.min()), float(v.max())


class TestArtifactShape:
    def test_three_rails_present(self, a14):
        assert set(a14["rail"]) == {"canonical_b5_clamped", "free_free_hex",
                                    "vane_coupon"}

    def test_canonical_rail_is_the_b5_rail(self, a14):
        """정본 §3.6-ii가 인용하는 레일과 **같은 격자**여야 한다(mesh 1.2 mm·ndof 278520)."""
        r = a14[a14["rail"] == "canonical_b5_clamped"].iloc[0]
        assert r["mesh_size_mm"] == pytest.approx(1.2)
        assert int(r["ndof"]) == 278520
        assert r["config"] == "asbuilt"

    def test_limits_are_consistent_with_the_exact_relation(self, a14):
        """m_limit = 2·budget/|φ|² — CSV 내부 정합(어느 열이 밀려도 잡힌다)."""
        for _, r in a14.iterrows():
            assert r["m_limit_all_half_floor_0p05pct_mg"] == pytest.approx(
                1e6 * 2 * 5e-4 / r["phi2_max_all_per_kg"], rel=1e-9)
            assert r["dff_pct_all_0p2g"] == pytest.approx(
                100 * 0.5 * 0.2e-3 * r["phi2_max_all_per_kg"], rel=1e-9)

    def test_surface_limit_is_never_tighter_than_the_worst_case(self, a14):
        """접근 가능한 외부면의 |φ|²는 전 절점 최댓값을 넘을 수 없다."""
        assert (a14["phi2_max_surface_per_kg"]
                <= a14["phi2_max_all_per_kg"] * (1 + 1e-12)).all()


class TestCanonQuotesMatchArtifact:
    def test_phi2_ranges(self, a14, canon):
        lo, hi = _rng(a14, "canonical_b5_clamped", "phi2_max_all_per_kg")
        assert f"{lo:.1f}–{hi:.0f} kg⁻¹" == "22.5–114 kg⁻¹"
        assert "22.5–114 kg⁻¹" in canon
        lo, hi = _rng(a14, "free_free_hex", "phi2_max_all_per_kg")
        assert f"{lo:.1f}–{hi:.0f} kg⁻¹" == "53.9–243 kg⁻¹"
        assert "53.9–243 kg⁻¹" in canon

    def test_assembly_mass_limits(self, a14, canon):
        lo, hi = _rng(a14, "canonical_b5_clamped",
                      "m_limit_all_half_floor_0p05pct_mg")
        assert (round(lo, 1), round(hi, 1)) == (8.8, 44.5)
        lo2, hi2 = _rng(a14, "free_free_hex", "m_limit_all_half_floor_0p05pct_mg")
        assert (round(lo2, 1), round(hi2, 1)) == (4.1, 18.6)
        assert "8.8–44.5 mg clamped and 4.1–18.6 mg free–free" in canon

    def test_percent_of_part_mass(self, a14, canon):
        col = "m_limit_all_half_floor_0p05pct_pct_of_part"
        lo = min(_rng(a14, r, col)[0] for r in ("canonical_b5_clamped",
                                                "free_free_hex"))
        hi = max(_rng(a14, r, col)[1] for r in ("canonical_b5_clamped",
                                                "free_free_hex"))
        assert (round(lo, 3), round(hi, 3)) == (0.007, 0.065)
        assert "0.007–0.065 % of the part mass" in canon

    def test_coupon_limit_and_first_order_breakdown(self, a14, canon):
        lo, hi = _rng(a14, "vane_coupon", "m_limit_all_half_floor_0p05pct_mg")
        assert (round(lo, 2), round(hi, 2)) == (0.15, 0.49)
        assert "0.15–0.49 mg" in canon
        dlo, dhi = _rng(a14, "vane_coupon", "dff_pct_all_0p2g")
        assert (round(dlo), round(dhi)) == (20, 66)
        assert "first-order shift of 20–66 %" in canon

    def test_0p2g_sensor_excess_factor_and_shift(self, a14, canon):
        """0.2 g가 자유-자유 한계를 넘는 배수와, 그때의 이동량 범위."""
        lo, hi = _rng(a14, "free_free_hex", "m_limit_all_half_floor_0p05pct_mg")
        assert (round(0.2e3 / hi), round(0.2e3 / lo)) == (11, 49)
        assert "factor of **11–49**" in canon
        shifts = [_rng(a14, r, "dff_pct_all_0p2g")
                  for r in ("canonical_b5_clamped", "free_free_hex")]
        assert (round(min(s[0] for s in shifts), 2),
                round(max(s[1] for s in shifts), 1)) == (0.22, 2.4)
        assert "**0.22–2.4 %**" in canon

    def test_exact_relation_is_stated_not_paraphrased(self, canon):
        """식 자체가 본문에 있어야 한다 — 독자가 다른 센서로 재계산할 수 있어야 하므로."""
        assert "δf/f = −½ m_a |φ(x)|²" in canon

    def test_analytic_anchor_is_quoted(self, canon):
        """검증 근거(EB 팁 유효질량 0.2427 m, 2.7 % 일치)가 본문에 남아 있는가."""
        assert "0.2427 m" in canon and "2.7 %" in canon

    def test_no_fill_left_anywhere_in_the_body(self, canon):
        """본문 `[FILL]`은 0건이다.

        E1의 두 건은 계산·규칙으로 대체됐고(v2.14), Appendix A 자리표시자 2건은 부록을
        실제로 집필하면서 사라졌다(v2.16). STATUS NOTE는 이력이라 제외하고 본문만 본다 —
        `canon` 픽스처가 이미 걸러 준다. E1 문단은 따로 한 번 더 확인한다.
        """
        e1 = next(l for l in canon.split("\n") if l.startswith("**E1 —"))
        block = canon.split("**E1 —")[1].split("\n**E2 —")[0]
        assert "[FILL" not in e1 and "[FILL" not in block, "E1에 [FILL]이 남아 있다"
        assert re.findall(r"\[FILL[^\]]*\]", canon) == []

    def test_appendices_are_written(self, canon):
        """부록 제목만 있고 본문이 없던 상태가 되돌아오지 않게 한다."""
        assert "**A.1 Section and rails.**" in canon
        assert "**B.1 Design.**" in canon


class TestWallClearanceGate:
    """[FILL] 1은 값이 아니라 **규칙**으로 대체됐다 — 그 규칙이 본문에 있는지."""

    def test_provisional_gate_is_one_diameter(self, canon):
        assert "≥ one impeller diameter, 73.1 mm" in canon

    def test_diameter_matches_geometry(self):
        from impeller_fingerprint.geometry import DISK
        assert 2e3 * DISK.b == pytest.approx(73.1, abs=0.05)

    def test_reason_no_computed_dstar(self, canon):
        """계산된 d*를 사전등록하지 못한 정량적 이유가 본문에 있어야 한다."""
        assert "δβ/β < 0.26 %" in canon
        assert "0.43–1.12" in canon and "19–126 %" in canon

    def test_empirical_fallback_is_falsifiable(self, canon):
        assert "at twice it" in canon and "½σ_f" in canon


class TestSigmaYAnchor:
    """§3.5 Σ_y 앵커 — [34] 전문에서 확인한 수치가 본문과 일치하는가 (F93).

    이 값들은 CSV가 아니라 **문헌 전문**에서 온 것이므로 출처를 표에 못박고 정본과 대조한다.
    문헌 표에서 재계산한 산포(f_n = 71.556 Hz 기준)와 정본 인용이 어긋나면 실패한다.
    """

    FN = 71.556
    #: Mituletu et al. MSSP 116 (2019) — 세트별 주파수 [Hz]
    SETS = {
        "swept_sine_avg": [71.555, 71.560, 71.561, 71.559, 71.557],       # Table 8
        "short_sine": [71.56967, 71.54202, 71.55807, 71.57075,            # Table 9 F
                       71.56334, 71.54006, 71.54365, 71.56933,            #         G
                       71.56846, 71.55114, 71.54313, 71.56427],           #         H
        "mechanical": [71.53496, 71.57562, 71.5771, 71.53226,             # Table 10 I
                       71.54971, 71.50389, 71.53109, 71.56917],           #          J
    }

    def _spread(self, key):
        v = self.SETS[key]
        return (max(v) - min(v)) / self.FN

    def test_three_levels_reproduce_the_quoted_relative_spreads(self):
        """정본이 **인용한 자릿수로** 반올림했을 때 일치해야 한다.

        허용오차를 임의로 잡으면 인용이 반올림 규칙과 어긋나도 통과한다 — 실제 값
        8.39e−5 / 4.29e−4 / 1.023e−3이 각각 8.4e−5 / 4.3e−4 / 1.0e−3으로 반올림되는지를 본다.
        """
        assert round(self._spread("swept_sine_avg") * 1e5, 1) == 8.4
        assert round(self._spread("short_sine") * 1e4, 1) == 4.3
        assert round(self._spread("mechanical") * 1e3, 1) == 1.0

    def test_levels_are_ordered_by_excitation_quality(self):
        """스윕사인 < short-sine < 해머 — 순서가 뒤집히면 §3.5의 논거가 무너진다."""
        assert (self._spread("swept_sine_avg") < self._spread("short_sine")
                < self._spread("mechanical"))

    def test_absolute_ranges_in_hz(self):
        for key, hz in (("swept_sine_avg", 0.006), ("short_sine", 0.031),
                        ("mechanical", 0.073)):
            v = self.SETS[key]
            assert max(v) - min(v) == pytest.approx(hz, abs=0.0006), key

    def test_canon_quotes_all_three(self, canon):
        assert "0.006 Hz, **8.4 × 10⁻⁵**" in canon
        assert "0.031 Hz with short-time sine, **4.3 × 10⁻⁴**" in canon
        assert "0.073 Hz with hammer impact, **1.0 × 10⁻³**" in canon

    def test_canon_states_the_millihertz_resolution_and_its_relative_size(self, canon):
        assert "millihertz (1.4 × 10⁻⁵ of f)" in canon

    def test_millihertz_relative_size_is_right(self):
        assert 1e-3 / self.FN == pytest.approx(1.4e-5, abs=0.05e-5)

    def test_canon_flags_the_papers_looser_envelope(self, canon):
        """문헌 자체의 상계(0.5 %/1 %)가 자기 표보다 느슨하다는 사실을 숨기지 않았는가."""
        assert "conservative envelope of 0.5 % for short sine and 1 % for impact" in canon
        assert "order of magnitude looser than the dispersions in its own tables" in canon

    def test_canon_credits_the_decade_to_excitation_not_only_remounting(self, canon):
        """정정된 귀속 — 10⁻³은 재장착 전에 가진·창 선택만으로 도달된다."""
        assert "before any re-mounting is involved" in canon

    def test_protocol_consequence_is_pre_registered(self, canon):
        assert "excitation method and analysed signal window are fixed in advance" in canon


class TestClassicalAttribution:
    """§2 — 보 극한의 분해와 변곡점 실명을 [22]에 귀속했는가 (F94)."""

    def test_severity_times_squared_curvature_is_attributed(self, canon):
        assert "depth-only severity times the normalized squared modal curvature" in canon
        assert "independent of crack position, boundary condition and mode number" in canon

    def test_inflection_point_blindness_is_attributed(self, canon):
        n = _ref_num("10.3390/s22031118")          # Gillich et al., Sensors 2022
        assert (f"a crack at an inflection point of a mode produces no shift in that mode [{n}]"
                in canon)

    def test_novelty_is_restated_as_where_it_is_pointed(self, canon):
        assert "What is new here is not that factorization but where it is pointed" in canon

    def test_weak_clamping_mechanism_is_cited_in_e1(self, canon):
        n = _ref_num("10.3390/s22031118")
        block = canon.split("**E1 —")[1].split("\n**E2 —")[0]
        assert "equivalent to a crack at the root" in block and f"[{n}]" in block
