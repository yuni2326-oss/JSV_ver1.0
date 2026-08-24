"""A12 — 참고문헌 표와 정본 md의 **번호 정합성** 회귀검정 (설계서 M4).

M3에서 참고문헌 16번이 두 문헌을 담고 있었고(중복), v2.7까지 §5 E1에 `[16-wet]`이라는
목록에 없는 인용이 남아 있었다. 이 검정은 그 계열의 실패가 다시 들어오면 깨진다:

* 번호가 1..N 연속인가 / DOI가 유일한가
* 정본 md 본문의 모든 인용번호가 표에 있는가(dangling citation 금지)
* 표의 모든 항목이 본문에서 최소 1회 인용되는가(미인용 문헌 금지)
* 번호가 **첫 인용 순서**인가(Elsevier/JSV 규정)
* `[verify]`가 남은 항목은 `status`에 이유가 적혀 있는가
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from impeller_fingerprint import references as R

CANON = (Path(__file__).resolve().parents[2] / "docs" / "paper3-jsv"
         / "2026-07-31-paperB-jsv-v2.1.md")

# 인용 토큰: [12] / [1–4] / [11,12,17–19]. ξ_d ∈ [0,1] 같은 구간 리터럴은 0 때문에 걸러진다.
_TOKEN = re.compile(r"\[\s*\d[\d\s,–—-]*\]")


def _numbers(token: str) -> list[int]:
    out: list[int] = []
    for part in re.split(r"[,\s]+", token.strip("[] ")):
        if not part:
            continue
        m = re.fullmatch(r"(\d+)[–—-](\d+)", part)
        if m:
            out.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            out.append(int(part))
    return out


def citation_order(text: str) -> list[int]:
    """본문 등장 순서대로 인용번호를 나열한다(STATUS NOTE는 내부 메모라 제외)."""
    seq: list[int] = []
    for line in text.split("\n"):
        if line.startswith("*[STATUS NOTE"):
            continue
        for m in _TOKEN.finditer(line):
            nums = _numbers(m.group(0))
            if nums and min(nums) >= 1:          # 구간 리터럴 배제
                seq.extend(nums)
    return seq


@pytest.fixture(scope="module")
def canon_text():
    if not CANON.exists():                        # 정본이 없는 체크아웃에서는 건너뛴다
        pytest.skip(f"정본 md 없음: {CANON}")
    return CANON.read_text(encoding="utf-8")


class TestTable:
    def test_numbers_contiguous_from_one(self):
        assert [r.num for r in R.REFERENCES] == list(range(1, len(R.REFERENCES) + 1))

    def test_dois_unique_and_present(self):
        dois = [r.doi for r in R.REFERENCES]
        assert all(dois), "DOI 없는 항목이 있으면 서지 확정이라 부를 수 없다"
        assert len(set(dois)) == len(dois)

    def test_required_fields_nonempty(self):
        for r in R.REFERENCES:
            assert r.authors and r.title and r.journal and r.year, r.num
            assert r.source_url.startswith("http"), r.num

    def test_unverified_entries_state_the_reason(self):
        """추측 금지 규약: 확정 못 한 항목은 이유를 적고 [verify]를 유지한다."""
        for r in R.REFERENCES:
            if not r.verified:
                assert r.status.startswith("unverified:"), r.num
                assert len(r.status) > len("unverified:") + 5, r.num
                assert "[verify" in R.format_ref(r), r.num

    def test_verified_entries_carry_no_verify_tag(self):
        for r in R.REFERENCES:
            if r.verified:
                assert "[verify" not in R.format_ref(r), r.num

    def test_csv_roundtrip(self, tmp_path):
        import csv
        p = R.write_csv(tmp_path / "a12_references.csv")
        with p.open(encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
        assert tuple(rows[0]) == R.CSV_HEADER
        assert len(rows) == len(R.REFERENCES) + 1
        assert rows[1][0] == "1"


class TestCanonicalConsistency:
    def test_no_dangling_citation(self, canon_text):
        known = {r.num for r in R.REFERENCES}
        cited = set(citation_order(canon_text))
        assert not (cited - known), f"목록에 없는 인용번호: {sorted(cited - known)}"

    def test_no_uncited_reference(self, canon_text):
        cited = set(citation_order(canon_text))
        known = {r.num for r in R.REFERENCES}
        assert not (known - cited), f"본문에서 인용되지 않은 문헌: {sorted(known - cited)}"

    def test_numbering_follows_first_citation_order(self, canon_text):
        first: list[int] = []
        for n in citation_order(canon_text):
            if n not in first:
                first.append(n)
        assert first == sorted(first), (
            "번호가 첫 인용 순서가 아니다(JSV 규정). 첫 인용 순서: " + str(first))

    def test_no_leftover_author_name_or_verify_tags_in_references(self, canon_text):
        """번호 항목 줄에 미확정 표시나 대괄호 자리표시자가 남아 있으면 실패.

        절 머리의 설명 문단은 규약 자체를 서술하므로(`[verify]`라는 낱말이 등장한다) 제외하고,
        `N. ` 로 시작하는 **항목 줄만** 본다.
        """
        body = canon_text.split("\n## References")[-1].split("\n## Appendix")[0]
        entries = [ln for ln in body.split("\n") if re.match(r"^\d+\. ", ln)]
        assert len(entries) == len(R.REFERENCES)
        for ln in entries:
            n = ln.split(".", 1)[0]
            assert "[verify]" not in ln, n
            # 저자·제목을 비워 둔 대괄호 자리표시자(예: "[Gao et al.]", "[exact title]")
            assert not re.search(r"\[[A-Za-z][^\]\n]{3,}\]", ln), \
                f"{n}번 줄에 대괄호 자리표시자가 남아 있다"

    def test_canonical_reference_lines_match_table(self, canon_text):
        """정본 md의 References 줄이 `references.py`에서 생성한 것과 문자열까지 같은가."""
        body = canon_text.split("\n## References")[-1].split("\n## Appendix")[0]
        for r in R.REFERENCES:
            assert R.format_ref(r) in body, f"{r.num}번 줄이 정본과 다르다"
