"""정본 md → **JSV 제출본** md (그림 삽입·내부표시 제거·절 제목 정규화).

**왜 변환기인가.** 정본 md는 *작업본*이다 — 버전별 STATUS NOTE가 이력으로 쌓여 있고,
`[SW]`(그 진술이 계산으로만 라이선스된다는 내부 표시)가 42곳에 박혀 있고, 절 제목 일부가
검토용 문구("status per item", "pilot status labelled")를 달고 있다. 제출본은 그것들이 전부
없어야 하지만, 정본에서 **지우면 이력이 사라진다**. 그래서 정본은 그대로 두고 제출본을
파생시킨다. 변환은 전부 기계적이며 산문을 다시 쓰지 않는다 — 문장 내용이 바뀌는 편집은
정본에서 해야 하고, 이 모듈은 그것을 강제하기 위해 **치환 규칙만** 갖는다.

**하는 일**
  1. STATUS NOTE 문단 제거(`*[STATUS NOTE`로 시작하는 줄).
  2. 내부 표시 제거 — `**[SW]**` / `[SW]` / `**[SW — …]**`, 잔존 `[TO RUN]`.
  3. 절 제목 정규화(아래 `HEADING_MAP`) — 검토용 부제를 떼고 JSV 관례에 맞춘다.
  4. **그림 삽입** — `**Figure N.**` 캡션 **앞**에 `![](경로)`를 넣는다. 경로는 제출본 md
     위치 기준 상대경로이고 파일은 `figures.CANON_FIGURES`가 정한다(그림 번호의 단일 출처).
     외부 변환기가 `![alt](path)`를 그림으로 렌더하므로 docx에 실제로 박힌다 — 지금까지
     생성된 docx에 그림이 없던 이유는 정본에 이미지 참조가 **0건**이었기 때문이다.
  5. 그림 참조 표기를 Elsevier 관례로 — 캡션은 `**Fig. N.**`, 본문은 `Fig. N`. 다만
     **문장 시작**(마침표·줄머리 직후)에서는 `Figure N`을 유지한다.

**하지 않는 일**: 문장 재작성, 수치 변경, 절 순서 변경, 저자·감사문 채우기. 저자는 정본의
`[Author placeholders]`를 그대로 통과시킨다 — 이름을 만들어 넣지 않는다.
"""
from __future__ import annotations

import re
from pathlib import Path

from .figures import CANON_FIGURES

#: 검토용 부제를 뗀 절 제목. 좌변은 정본의 제목 줄 **전체**(앞의 `#` 포함).
HEADING_MAP: dict[str, str] = {
    "### 3.2 Unified degenerate-perturbation framework (the paper’s spine)":
        "### 3.2 Unified degenerate-perturbation framework",
    "### 3.6 Independent-model FEM study (inverse-crime rail) — status per item":
        "### 3.6 Independent-model FEM study on an independent rail",
    "## 4. Software results (pilot status labelled)":
        "## 4. Results",
}

#: 내부 표시 — 문장 안에서 지운다(앞뒤 공백 정리는 `_squeeze`가 한다).
INTERNAL_TAGS = (
    re.compile(r"\s*\*\*\[SW[^\]]*\]\*\*"),
    re.compile(r"\s*\[SW[^\]]*\]"),
    re.compile(r"\s*\*\*\[TO RUN[^\]]*\]\*\*"),
    re.compile(r"\s*\[TO RUN[^\]]*\]"),
)

STATUS_NOTE = re.compile(r"^\*\[STATUS NOTE")
CAPTION = re.compile(r"^\*\*Figure (\d+)\.\*\*")


def _squeeze(s: str) -> str:
    """표시를 뗀 자리에 생긴 이중공백·구두점 앞 공백을 정리한다."""
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r"\s+([.,;:)])", r"\1", s)
    s = re.sub(r"—\s*$", "", s.rstrip()) if s.rstrip().endswith("—") else s
    return s.rstrip()


def strip_status_notes(lines: list[str]) -> tuple[list[str], int]:
    """STATUS NOTE 줄과 그 뒤에 붙은 빈 줄 하나를 제거한다."""
    out, removed, i = [], 0, 0
    while i < len(lines):
        if STATUS_NOTE.match(lines[i].strip()):
            removed += 1
            i += 1
            if i < len(lines) and not lines[i].strip():
                i += 1
            continue
        out.append(lines[i])
        i += 1
    return out, removed


def strip_internal_tags(line: str) -> str:
    for pat in INTERNAL_TAGS:
        line = pat.sub("", line)
    return _squeeze(line)


def normalize_heading(line: str) -> str:
    return HEADING_MAP.get(line.rstrip(), line)


def figure_reference_style(text: str) -> str:
    """`Figure N` → `Fig. N`. 문장 시작(줄머리 또는 `. `/`? `/`! ` 직후)은 유지한다."""
    def repl(m: re.Match) -> str:
        start = m.start()
        prefix = text[max(0, start - 2):start]
        at_sentence_start = start == 0 or prefix.endswith((". ", "? ", "! ")) \
            or text[start - 1] == "\n"
        return m.group(0) if at_sentence_start else "Fig. " + m.group(1)
    return re.sub(r"Figure (\d+)", repl, text)


def figure_path(num: int, fig_dir: str) -> list[str]:
    """그림 번호 → 삽입할 상대경로 목록(다중 패널 그림은 2개 이상)."""
    if num not in CANON_FIGURES:
        raise KeyError(f"CANON_FIGURES에 그림 {num}번이 없다 — 번호의 단일 출처를 어겼다")
    return [f"{fig_dir}/{f}" for f in CANON_FIGURES[num]["files"]]


def build(canon: str, fig_dir: str = "../_generated/figures/paper3") -> dict:
    """정본 md 문자열 → (제출본 md 문자열, 통계) 를 담은 dict."""
    lines, n_notes = strip_status_notes(canon.split("\n"))
    out: list[str] = []
    figs_inserted: dict[int, list[str]] = {}
    for line in lines:
        m = CAPTION.match(line.strip())
        if m:
            num = int(m.group(1))
            paths = figure_path(num, fig_dir)
            for p in paths:
                out.append(f"![Figure {num}]({p})")
            out.append("")
            figs_inserted[num] = paths
        out.append(strip_internal_tags(normalize_heading(line)))
    text = "\n".join(out)
    text = figure_reference_style(text)
    text = text.replace("**Figure ", "**Fig. ")          # 캡션 표기
    return {"text": text, "status_notes_removed": n_notes,
            "figures_inserted": figs_inserted,
            "n_figure_files": sum(len(v) for v in figs_inserted.values())}


def build_file(canon_path: str | Path, out_path: str | Path,
               fig_dir: str | None = None) -> dict:
    canon_path, out_path = Path(canon_path), Path(out_path)
    if fig_dir is None:
        fig_dir = "../_generated/figures/paper3"
    res = build(canon_path.read_text(encoding="utf-8"), fig_dir=fig_dir)
    out_path.write_text(res["text"], encoding="utf-8")
    res["out"] = str(out_path)
    return res


#: 본문이 인용하는 산출물 파일명을 찾는 패턴 — 백틱 안의 `*.csv` / `*.npz` / `*.png`.
ARTIFACT = re.compile(r"`([A-Za-z0-9_]+\.(?:csv|npz|png))`")


def cited_artifacts(canon: str) -> list[str]:
    """정본 본문이 **이름으로 인용한** 산출물 목록(중복 제거, 등장 순서).

    기탁 목록을 손으로 적으면 빠진다 — 본문이 인용한 것과 기탁한 것이 어긋나면 독자가
    확인할 수 없는 수치가 생긴다. 그래서 목록을 본문에서 **뽑는다**. STATUS NOTE는 이력이므로
    제외한다(옛 파일명을 담고 있다).
    """
    body = "\n".join(l for l in canon.split("\n") if "STATUS NOTE" not in l)
    out: list[str] = []
    for m in ARTIFACT.finditer(body):
        if m.group(1) not in out:
            out.append(m.group(1))
    return out


def deposition_manifest(canon: str, data_dir, figs_dir) -> dict:
    """기탁용 목록 + 존재 확인. 업로드는 하지 않는다(권한·기탁자 정보가 필요하다).

    반환: present / missing / extra_on_disk. `missing`이 비어 있지 않으면 본문이 인용하는
    산출물이 디스크에 없다는 뜻이므로 기탁 전에 반드시 해소해야 한다.
    """
    from pathlib import Path as _P
    data_dir, figs_dir = _P(data_dir), _P(figs_dir)
    cited = cited_artifacts(canon)
    present, missing = [], []
    for name in cited:
        for d in (data_dir, figs_dir):
            if (d / name).exists():
                present.append(str(d / name))
                break
        else:
            missing.append(name)
    on_disk = {p.name for p in list(data_dir.glob("*.csv")) + list(data_dir.glob("*.npz"))}
    return {"cited": cited, "present": present, "missing": missing,
            "extra_on_disk": sorted(on_disk - set(cited))}


def build_data_package(canon: str, data_dir, figs_dir, out_zip) -> dict:
    """정본이 인용한 산출물 + 그림 벡터본을 zip 하나로 묶고 `MANIFEST.md`를 함께 넣는다.

    **왜 명령으로 두는가**: 손으로 묶은 패키지는 산출을 다시 돌릴 때마다 조용히 낡는다 —
    2026-08-16에 묶인 패키지는 그날 오후 재생성된 a16·a17을 이미 담지 못했고, 새로 인용된
    산출물(a19)도 없었다(a18이 CSV만 있고 명령이 없던 것과 같은 부류의 결함). 목록을
    **본문 인용에서 뽑으므로** 문서와 패키지가 갈라지지 않는다.

    반환: files(보관된 이름), missing(인용됐으나 디스크에 없는 것), manifest, out.
    `missing`이 비어 있지 않으면 문서가 확인 불가능한 수치를 담는다는 뜻이다.
    """
    import hashlib
    import zipfile
    from pathlib import Path as _P

    from . import supplementary as _supp

    data_dir, figs_dir, out_zip = _P(data_dir), _P(figs_dir), _P(out_zip)
    cited = cited_artifacts(canon)
    picked: list[tuple[str, _P]] = []
    missing = []
    for name in cited:
        for d in (data_dir, figs_dir):
            if (d / name).exists():
                picked.append((name, d / name))
                break
        else:
            missing.append(name)
    for pdf in sorted(figs_dir.glob("*.pdf")):          # 그림 벡터본은 전부 넣는다
        picked.append((f"figures/{pdf.name}", pdf))
    picked.sort(key=lambda t: t[0])

    last = max(int(t[1:]) for t, *_ in _supp.SUPP_TABLES)
    rows = [f"| {name} | {p.stat().st_size:,} | "
            f"{hashlib.sha256(p.read_bytes()).hexdigest()[:16]} |"
            for name, p in picked]
    manifest = "\n".join([
        "# Supplementary data — file manifest",
        "",
        f"Artifacts behind Supplementary Tables S1–S{last} and the figures. Values in the "
        "paper and in the supplementary tables are read from these files, not transcribed. "
        "Table S1's full 240-cell table is `b1_mc_summary_mass.csv`; the document carries "
        "its 12-cell summary.",
        "",
        "| file | bytes | sha256 (first 16) |",
        "|---|---|---|",
        *rows,
        ""])
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("MANIFEST.md", manifest)
        for name, p in picked:
            z.write(p, name)
    return {"files": [n for n, _ in picked], "missing": missing,
            "manifest": manifest, "out": str(out_zip)}
