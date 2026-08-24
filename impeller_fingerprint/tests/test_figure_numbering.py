"""그림 번호 정합성 회귀검정 (설계서 F73) — 참고문헌 검정(`test_references.py`)과 같은 방식.

2026-08-14에 §3.1의 임펠러 기하·모드형이 **Figure 1로 신설**되어 이후 번호가 하나씩 밀렸다
(옛 1–5 → 새 2–6). 이 계열의 실패가 다시 들어오면 여기서 깨진다:

* 캡션 번호가 1..N 연속인가 / 중복이 없는가 / 문서 등장순서가 오름차순인가
* 본문의 모든 `Fig. N` 참조가 존재하는 캡션을 가리키는가(dangling 금지)
* 모든 캡션이 본문에서 최소 1회 참조되는가(참조 없는 그림 금지 — JSV 조판 결함)
* `figures.CANON_FIGURES` 표의 번호가 정본 캡션 번호와 정확히 일치하는가
* 파일명이 `figN…`으로 번호를 안고 있으면 그 숫자가 **정본 번호와 같은가**
  (`fig3_kernels_and_sign.png`가 정본 Figure 4에 붙는 종류의 어긋남을 잡는다)
* 코드가 쓰는 모든 `figN….png` 리터럴이 표에 그 번호로 등재돼 있는가
* 캡션이 백틱으로 인용한 `*.png`가 그 번호의 표 항목에 있는가
* 표의 생성함수가 실제로 존재하는가 / 생성기 없는 항목은 이유를 적었는가
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from impeller_fingerprint import figures as F

CANON = (Path(__file__).resolve().parents[2] / "docs" / "paper3-jsv"
         / "2026-07-31-paperB-jsv-v2.1.md")
SRC_DIR = Path(F.__file__).resolve().parent

_CAPTION = re.compile(r"\*\*Figure\s+(\d+)\.\*\*")
_REF = re.compile(r"\bFig(?:ure|\.)\s*(\d+)")
_PNG_IN_CAPTION = re.compile(r"`([A-Za-z0-9_.\-]+\.png)`")
#: 소스에 박힌 번호 있는 그림 파일명 리터럴
_NUMBERED_PNG = re.compile(r"\"(fig(\d+)[A-Za-z0-9_.\-]*\.png)\"")


def _body_lines(text: str) -> list[tuple[int, str]]:
    """STATUS NOTE(내부 메모)는 옛 번호를 그대로 기록하므로 제외한다."""
    return [(i, ln) for i, ln in enumerate(text.split("\n"), 1)
            if not ln.startswith("*[STATUS NOTE")]


@pytest.fixture(scope="module")
def body():
    if not CANON.exists():
        pytest.skip(f"정본 md 없음: {CANON}")
    return _body_lines(CANON.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def captions(body):
    """번호 → 캡션 줄 전체. 등장순서를 유지한 리스트도 함께."""
    seq, out = [], {}
    for _, ln in body:
        for m in _CAPTION.finditer(ln):
            n = int(m.group(1))
            seq.append(n)
            out[n] = ln
    return seq, out


@pytest.fixture(scope="module")
def refs(body):
    """캡션이 아닌 위치의 그림 참조 번호."""
    got = []
    for _, ln in body:
        cap_spans = [m.span() for m in _CAPTION.finditer(ln)]
        for m in _REF.finditer(ln):
            if any(s <= m.start() < e for s, e in cap_spans):
                continue
            got.append(int(m.group(1)))
    return got


class TestCanonicalNumbering:
    def test_captions_contiguous_and_unique(self, captions):
        seq, _ = captions
        assert len(seq) == len(set(seq)), f"중복 캡션 번호: {seq}"
        assert sorted(seq) == list(range(1, len(seq) + 1)), seq

    def test_captions_in_ascending_document_order(self, captions):
        seq, _ = captions
        assert seq == sorted(seq), f"캡션이 문서 순서대로 번호되지 않았다: {seq}"

    def test_no_dangling_figure_reference(self, captions, refs):
        _, known = captions
        bad = sorted(set(refs) - set(known))
        assert not bad, f"존재하지 않는 그림을 참조한다: {bad}"

    def test_every_figure_is_referenced(self, captions, refs):
        _, known = captions
        bad = sorted(set(known) - set(refs))
        assert not bad, f"본문에서 참조되지 않은 그림: {bad}"


class TestRegistryMatchesCanonical:
    def test_registry_numbers_equal_caption_numbers(self, captions):
        _, known = captions
        assert set(F.CANON_FIGURES) == set(known), (
            f"표 {sorted(F.CANON_FIGURES)} vs 정본 {sorted(known)}")

    def test_filename_number_mismatch_must_be_documented(self):
        """파일명의 숫자 ≠ 그림 번호일 수 있다 — 단 **표에 이유가 적혀 있어야** 한다.

        2026-08-15 편집에서 등장 순서가 바뀌었지만(구 2→3, 3→4, 4→5, 5→2) 파일명은 그대로
        뒀다: 이름이 산출 명령·설계서·CSV 주석 여러 곳에 박혀 있어 한꺼번에 바꾸면 추적이
        끊기고, 그림 **안에는 번호가 없으므로** 재번호는 배치 문제일 뿐이다. 그래서 옛
        규칙(파일명 == 번호)을 **문서화 의무**로 바꾼다 — 어긋난 항목은 `note`에 "구 Fig"로
        내력을 남겨야 하고, 그러지 않으면 우연한 드리프트와 구별할 수 없다.
        """
        for n, e in F.CANON_FIGURES.items():
            for fn in e["files"]:
                m = re.match(r"fig(\d+)", fn)
                if m and int(m.group(1)) != n:
                    assert "구 Fig" in (e.get("note") or ""), (
                        f"Figure {n}의 파일명이 fig{m.group(1)}…인데(={fn}) 표의 note에 "
                        f"내력이 없다 — 재번호였다면 '구 Fig N'을 적어라")

    def test_caption_cited_png_is_in_registry(self, captions):
        _, known = captions
        for n, ln in known.items():
            for fn in _PNG_IN_CAPTION.findall(ln):
                assert fn in F.CANON_FIGURES[n]["files"], (
                    f"Figure {n} 캡션이 인용한 {fn}이 표에 없다")

    def test_makers_exist(self):
        for n, e in F.CANON_FIGURES.items():
            if e["maker"] is None:
                assert e.get("note"), f"Figure {n}: 생성기가 없으면 이유를 적어야 한다"
            else:
                assert callable(getattr(F, e["maker"], None)), (n, e["maker"])

    def test_source_numbered_png_literals_match_registry(self):
        """코드가 쓰는 `figN….png`가 표의 같은 번호에 등재돼 있는가."""
        known = {fn: n for n, e in F.CANON_FIGURES.items() for fn in e["files"]}
        for src in ("figures.py", "cli.py"):
            text = (SRC_DIR / src).read_text(encoding="utf-8")
            for fn, num in _NUMBERED_PNG.findall(text):
                if fn.startswith("fig1a") or fn.startswith("fig1b") \
                        or fn.startswith("fig1c"):
                    continue                       # 패널별 파일(조판 대안)
                assert fn in known, f"{src}가 쓰는 {fn}이 CANON_FIGURES에 없다"
                # 번호 일치는 요구하지 않는다(위 검정 참조) — 등재 여부만 본다. 코드가 쓰는
                # 파일이 표에 없으면 그 그림은 어느 번호에도 배치되지 않은 채 생성된다.


class TestCitationOrder:
    """Elsevier 규정 — 그림은 **본문에서 처음 인용되는 순서**로 번호가 붙어야 한다.

    캡션이 문서 순서대로 놓여 있어도 본문 인용이 뒤바뀌면 조판에서 지적된다. 2026-08-15에
    실제로 그랬다: §4.2의 항목 (iii)이 Figure 6을 인용하는데 커널 그림(Figure 5) 인용은 그
    문단 끝에 있어서 6이 5보다 먼저 나왔다. 항목 (i)의 "map panels"에 Figure 5를 달아 고쳤다.
    """

    def test_first_citation_order_is_ascending(self, refs):
        first: list[int] = []
        for n in refs:
            if n not in first:
                first.append(n)
        assert first == sorted(first), (
            "본문 첫 인용 순서가 오름차순이 아니다 — 번호를 바꾸거나 인용을 옮겨야 한다: "
            f"{first}")
