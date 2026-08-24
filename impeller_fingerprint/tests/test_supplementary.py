"""보충자료 — 정본이 약속한 표가 실제로 존재하고 CSV에서 조판되는가.

정본은 §Data availability·Appendix A.6·B.4에서 보충자료를 참조한다. 그 약속과 실제 문서가
어긋나면 제출 패키지가 불완전해진다 — 이 검정이 둘을 묶는다.
"""
import os
from pathlib import Path

import pytest

from impeller_fingerprint import supplementary as supp

#: 기본값은 **이 체크아웃의** 산출 디렉터리다. 절대경로(다른 워킹트리)를 기본으로 두면
#: 브랜치에 커밋된 데이터가 문서와 어긋나도 검정이 통과한다 — 2026-08-24에 실제로 9개가
#: 어긋나 있었다(설계서 F153). 다른 스냅샷을 보려면 `PAPER3_OUT`으로 지정한다.
DATA = Path(os.environ.get(
    "PAPER3_OUT",
    Path(__file__).resolve().parents[2] / "docs" / "_generated")) / "data" / "paper3"
CANON = (Path(__file__).resolve().parents[2] / "docs" / "paper3-jsv"
         / "2026-07-31-paperB-jsv-v2.1.md")

@pytest.fixture(scope="module")
def built():
    if not DATA.exists():
        pytest.skip(f"{DATA} 없음")
    return supp.build(DATA)


@pytest.fixture(scope="module")
def canon():
    #: 원고는 코드·데이터만 배포한 트리에 없을 수 있다 — 그 검정만 건너뛴다.
    if not CANON.exists():
        pytest.skip(f"원고 없음: {CANON}")
    return "\n".join(l for l in CANON.read_text().split("\n")
                     if "STATUS NOTE" not in l)


class TestPromisesAreKept:
    def test_every_file_the_canon_calls_supplementary_is_a_table(self, canon, built):
        """정본이 '보충자료'라고 부른 파일은 보충자료 문서에 표로 들어 있어야 한다."""
        names = {f for _, f, _ in built["tables"]}
        for promised in ("b1_mc_summary_mass.csv", "a14_massload.csv"):
            assert promised in names, f"정본이 약속한 {promised}가 보충자료에 없다"

    def test_no_missing_artifact(self, built):
        assert built["missing"] == []

    def test_row_counts_are_the_full_tables(self, built):
        n = dict((f, k) for _, f, k in built["tables"])
        assert n["b1_mc_summary_mass.csv"] == 240, "240셀 전부가 들어가야 한다"
        assert n["a14_massload.csv"] == 30


class TestDocumentShape:
    def test_table_tags_are_unique_and_ordered(self, built):
        """생성기가 만드는 표는 번호가 **오름차순·중복 없음**이어야 한다.

        연속일 필요는 없다: S5는 본문 Table 4에서 보충으로 옮겨 온 **손으로 조판한** 표라
        생성기가 만들지 않는다(수치가 여러 CSV를 조합한 것이라 자동 조판 대상이 아니다).
        빈 번호를 허용하되 순서와 유일성은 강제한다 — 최종 문서에서 S1…S9가 빠짐없이
        나타나는지는 `test_final_document_has_every_tag`가 본다.
        """
        tags = [t for t, _, _ in built["tables"]]
        nums = [int(t[1:]) for t in tags]
        assert nums == sorted(nums) and len(set(nums)) == len(nums), tags

    def test_final_document_has_every_tag(self):
        """조립된 최종 보충문서에 생성기의 모든 표와 Note S1–S3가 빠짐없이 있는가.

        기대 목록은 **생성기에서 유도한다**(+ 손으로 조판한 S5). 번호를 하드코딩하면
        표를 하나 늘릴 때마다 이 검정이 낡아 실패한다 — S10을 추가하며 실제로 그랬다.
        """
        import re
        f = (Path(__file__).resolve().parents[2] / "docs" / "paper3-jsv"
             / "2026-08-16-paperB-jsv-supplementary-final.md")
        if not f.exists():
            pytest.skip("최종 보충문서 미생성")
        expect = sorted({t for t, *_ in supp.SUPP_TABLES} | {"S5"},
                        key=lambda s: int(s[1:]))
        heads = re.findall(r"^## (Table S\d+|Supplementary Note S\d+)", f.read_text(), re.M)
        assert [h for h in heads if h.startswith("Table")] == \
            [f"Table {t}" for t in expect], heads
        assert sum(h.startswith("Supplementary Note") for h in heads) == 3

    def test_each_table_states_its_source_and_command(self, built):
        t = built["text"]
        for tag, fname, _ in built["tables"]:
            blk = t.split(f"## Table {tag}.")[1].split("\n## ")[0]
            assert f"`{fname}`" in blk, tag
            assert "Regenerate with:" in blk, tag

    def test_author_placeholder_not_invented(self, built):
        assert "[Author placeholders]" in built["text"]

    def test_no_internal_markers(self, built):
        for tag in ("[SW", "STATUS NOTE", "[FILL", "[TO RUN"):
            assert tag not in built["text"], tag


class TestR36Layout:
    """r3.6 제출 계열(프로토콜 §5 삭제판) 보충자료 — 본문 포인터와 번호가 맞아야 한다.

    r3.5 본문이 가리키는 것: Table S1(생산 MC), **Table S5(ρ 스윕)**, Note S1·S2.
    캠페인 표(옛 S5)와 Note S3은 본문에서 사라졌으므로 표 목록에서 빠지고,
    정합기하 대조(a19)가 들어간다.
    """

    def test_r36_numbering_matches_the_body_pointers(self):
        tags = {t: f for t, _, f, *_ in supp.SUPP_TABLES_R36}
        assert tags["S1"] == "b1_mc_summary_mass.csv"
        assert tags["S5"] == "a16_rho_sweep.csv"          # §3.5·A.5의 "Table S5"
        assert tags["S9"] == "a19_geometry_matched_control.csv"
        assert "a18_matched_damage_control.csv" in tags.values()
        assert not any("seeded" in t.lower() or "campaign" in t.lower()
                       for _, t, *_ in supp.SUPP_TABLES_R36)
        nums = [int(t[1:]) for t, *_ in supp.SUPP_TABLES_R36]
        assert nums == list(range(1, 10)), "r36은 S1–S9 연속"

    def test_r36_descriptions_carry_no_internal_or_stale_refs(self):
        """설계서 F-번호·캠페인 표 참조·옛 번호(S10)가 제출 계열에 남으면 안 된다."""
        for tag, _, _, _, _, note in supp.SUPP_TABLES_R36:
            assert "design record" not in note, tag
            assert "S10" not in note, tag
            if tag in ("S3", "S4"):
                assert "S5" not in note, f"{tag}가 캠페인 표를 참조"
        # 정합기하(S9)가 직선 쿠폰 표를 새 번호(S8)로 가리킨다
        s9 = [n for t, _, _, _, _, n in supp.SUPP_TABLES_R36 if t == "S9"][0]
        assert "Table S8" in s9 and "Table S9" not in s9

    def test_r36_build_renders_all_nine(self):
        if not DATA.exists():
            pytest.skip(f"{DATA} 없음")
        r = supp.build(DATA, tables=supp.SUPP_TABLES_R36)
        assert r["missing"] == []
        assert [t for t, _, _ in r["tables"]] == [f"S{i}" for i in range(1, 10)]
