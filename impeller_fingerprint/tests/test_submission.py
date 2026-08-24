"""제출본 빌더 — 정본은 손대지 않고, 제출본에서만 내부표시가 사라지고 그림이 박히는가.

이 검정이 지키는 계약 두 개.
  (1) **정본 불변**: STATUS NOTE와 `[SW]`는 이력이므로 정본에는 남아 있어야 한다. 누가
      "제출본이 깨끗하니 정본도 지우자"고 하면 이 검정이 막는다.
  (2) **그림 번호의 단일 출처**: 삽입되는 파일은 `figures.CANON_FIGURES`가 정하고, 정본
      캡션 번호와 1:1이어야 한다. 캡션을 하나 늘리고 매핑을 안 늘리면 실패한다.
"""
import os
import re
from pathlib import Path

import pytest

from impeller_fingerprint import submission as sm
from impeller_fingerprint.figures import CANON_FIGURES

CANON = (Path(__file__).resolve().parents[2] / "docs" / "paper3-jsv"
         / "2026-07-31-paperB-jsv-v2.1.md")

#: 원고는 이 배포본에 포함되지 않을 수 있다(코드·데이터만 배포하는 경우) — 그때는
#: 원고 정합 검정을 **건너뛴다**. 원고가 있으면 전부 실행된다.
pytestmark = pytest.mark.skipif(not CANON.exists(),
                                reason=f"원고 없음: {CANON}")
#: 기본값은 **이 체크아웃의** 산출 디렉터리다 — 절대경로를 박으면 클론에서 동작하지
#: 않고 다른 워킹트리의 데이터를 검정한다(설계서 F153). `PAPER3_OUT`으로 덮어쓴다.
FIGS = Path(os.environ.get(
    "PAPER3_OUT",
    Path(__file__).resolve().parents[2] / "docs" / "_generated")) / "figures" / "paper3"


@pytest.fixture(scope="module")
def canon_text():
    return CANON.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def built(canon_text):
    return sm.build(canon_text)


class TestCanonicalIsUntouched:
    def test_canon_still_carries_its_history(self, canon_text):
        assert canon_text.count("[STATUS NOTE") >= 14
        assert "[SW]" in canon_text

    def test_builder_does_not_write_to_canon(self, canon_text, built):
        assert CANON.read_text(encoding="utf-8") == canon_text
        assert built["text"] != canon_text


class TestInternalMarkersRemoved:
    def test_no_status_notes(self, built):
        assert "STATUS NOTE" not in built["text"]
        assert built["status_notes_removed"] >= 14

    def test_no_internal_tags(self, built):
        for tag in ("[SW", "[TO RUN", "[FILL", "[EXP]"):
            assert tag not in built["text"], tag

    def test_headings_normalized(self, built):
        t = built["text"]
        assert "## 4. Results" in t
        assert "Software results" not in t
        assert "the paper’s spine" not in t
        assert "status per item" not in t

    def test_author_placeholder_is_passed_through_not_invented(self, built):
        """저자 이름을 만들어 넣지 않는다 — 자리표시자가 그대로 남아야 한다."""
        assert "[Author placeholders]" in built["text"]


class TestFigureInsertion:
    def test_all_six_figures_inserted(self, built):
        assert sorted(built["figures_inserted"]) == [1, 2, 3, 4, 5, 6]

    def test_paths_come_from_canon_figures(self, built):
        for num, paths in built["figures_inserted"].items():
            names = [Path(p).name for p in paths]
            assert tuple(names) == CANON_FIGURES[num]["files"]

    def test_image_line_precedes_its_caption(self, built):
        lines = built["text"].split("\n")
        for i, l in enumerate(lines):
            m = re.match(r"^\*\*Fig\. (\d+)\.\*\*", l.strip())
            if not m:
                continue
            num = int(m.group(1))
            n_files = len(CANON_FIGURES[num]["files"])
            imgs = [x for x in lines[max(0, i - n_files - 2):i] if x.startswith("![")]
            assert len(imgs) == n_files, f"Fig. {num} 앞의 이미지 줄 {len(imgs)}개"

    def test_caption_count_matches_the_mapping(self, canon_text):
        """정본 캡션 번호 집합 == CANON_FIGURES 키 집합."""
        nums = {int(m.group(1)) for m in
                re.finditer(r"^\*\*Figure (\d+)\.\*\*", canon_text, re.M)}
        assert nums == set(CANON_FIGURES)

    def test_unknown_figure_number_is_an_error(self):
        with pytest.raises(KeyError):
            sm.figure_path(99, "figs")

    def test_figure_files_exist_on_disk(self):
        missing = [f for spec in CANON_FIGURES.values() for f in spec["files"]
                   if not (FIGS / f).exists()]
        assert not missing, f"`cli figs` 필요: {missing}"


class TestFigureReferenceStyle:
    def test_captions_use_elsevier_abbreviation(self, built):
        assert "**Fig. 1.**" in built["text"]
        assert "**Figure 1.**" not in built["text"]

    def test_mid_sentence_reference_is_abbreviated(self):
        out = sm.figure_reference_style("as shown in Figure 3 the error grows")
        assert out == "as shown in Fig. 3 the error grows"

    def test_sentence_initial_reference_is_kept_long(self):
        out = sm.figure_reference_style("The bound holds. Figure 4 shows the kernels.")
        assert "Figure 4 shows" in out

    def test_line_initial_reference_is_kept_long(self):
        assert sm.figure_reference_style("Figure 2 is the fingerprint.").startswith(
            "Figure 2")


class TestJsvBackMatter:
    """Elsevier 관례 — 선언문·데이터·감사는 결론 뒤, References 앞."""

    ORDER = ("## 7. Conclusion", "## Declaration of competing interest",
             "## Data availability", "## Acknowledgements", "## References",
             "## Appendix A", "## Appendix B")

    def test_sections_present_and_in_order(self, built):
        pos = [built["text"].find(h) for h in self.ORDER]
        assert all(p >= 0 for p in pos), dict(zip(self.ORDER, pos))
        assert pos == sorted(pos), dict(zip(self.ORDER, pos))

    def test_nomenclature_follows_keywords(self, built):
        t = built["text"]
        assert t.find("**Keywords:**") < t.find("## Nomenclature") < t.find(
            "## 1. Introduction")

    def test_appendices_have_bodies_not_just_headings(self, built):
        t = built["text"]
        for head, marker in (("## Appendix A", "**A.1 Section and rails.**"),
                             ("## Appendix B", "**B.1 Design.**")):
            assert marker in t, head


class TestDataPackage:
    """제출 데이터 패키지 — 인용 목록에서 뽑아 zip으로 묶는다.

    이 검정이 고정하는 주장
      (T-D1) 정본이 이름으로 인용한 산출물과 그림 벡터본이 들어가고, 인용되지 않은 파일은
             들어가지 않는다.
      (T-D2) MANIFEST의 크기·해시가 **보관된 바이트**와 맞는다(압축 전 값을 적어두고
             다른 파일을 넣는 사고를 막는다).
      (T-D3) 인용됐지만 디스크에 없는 산출물은 조용히 빠지지 않고 보고된다.
    """

    @staticmethod
    def _dirs(tmp_path):
        data = tmp_path / "data"; figs = tmp_path / "figs"
        data.mkdir(); figs.mkdir()
        (data / "x1_cited.csv").write_text("a,b\n1,2\n")
        (data / "x2_uncited.csv").write_text("c\n3\n")
        (figs / "fig_x.png").write_bytes(b"\x89PNG fake")
        (figs / "fig_x.pdf").write_bytes(b"%PDF fake")
        return data, figs

    def test_package_holds_cited_artifacts_and_figure_vectors(self, tmp_path):
        import zipfile
        data, figs = self._dirs(tmp_path)
        canon = "text citing `x1_cited.csv` and `fig_x.png` only.\n"
        out = tmp_path / "pkg.zip"
        res = sm.build_data_package(canon, data, figs, out)
        with zipfile.ZipFile(out) as z:
            names = set(z.namelist())
        assert names == {"MANIFEST.md", "x1_cited.csv", "fig_x.png", "figures/fig_x.pdf"}
        assert res["missing"] == []

    def test_manifest_matches_the_archived_bytes(self, tmp_path):
        import hashlib
        import zipfile
        data, figs = self._dirs(tmp_path)
        out = tmp_path / "pkg.zip"
        sm.build_data_package("cites `x1_cited.csv`", data, figs, out)
        with zipfile.ZipFile(out) as z:
            man = z.read("MANIFEST.md").decode()
            for name in z.namelist():
                if name == "MANIFEST.md":
                    continue
                raw = z.read(name)
                row = [l for l in man.split("\n") if l.startswith(f"| {name} ")]
                assert len(row) == 1, f"{name}의 매니페스트 행이 1개가 아니다"
                assert f"{len(raw):,}" in row[0]
                assert hashlib.sha256(raw).hexdigest()[:16] in row[0]

    def test_missing_cited_artifact_is_reported(self, tmp_path):
        data, figs = self._dirs(tmp_path)
        res = sm.build_data_package("cites `nowhere.csv`", data, figs,
                                     tmp_path / "pkg.zip")
        assert res["missing"] == ["nowhere.csv"]

    def test_datapackage_is_a_registered_subcommand(self):
        """T-D4 — 패키지도 명령으로 만든다(손으로 묶으면 조용히 낡는다)."""
        from impeller_fingerprint import cli
        args = cli.build_parser().parse_args(["datapackage"])
        assert args.func is cli.cmd_datapackage
        assert args.out.endswith("supplementary-data.zip")


class TestDocxConverterBoundary:
    """`--docx`가 부르는 **외부 변환기**의 경계 — 배포본에는 그 파일이 없다.

    이 저장소는 markdown을 내보내고 docx 변환은 외부 스크립트에 맡긴다. 경로를 코드에
    박아 두면 배포본에서 `--docx`가 `FileNotFoundError` 같은 불친절한 예외로 죽는다.
      (T-C1) 변환기가 없으면 **경로를 말하는 SystemExit**으로 실패한다.
      (T-C2) `MD2DOCX` 환경변수로 다른 변환기를 지정할 수 있다.
      (T-C3) 변환기가 있으면 `<python> <converter> <md> <docx>`로 호출한다.
    """

    def test_missing_converter_fails_with_the_path(self, tmp_path, monkeypatch):
        from impeller_fingerprint import cli
        monkeypatch.setenv("MD2DOCX", str(tmp_path / "nope.py"))
        with pytest.raises(SystemExit) as e:
            cli.render_docx(tmp_path / "in.md", str(tmp_path / "out.docx"))
        assert "nope.py" in str(e.value)

    def test_env_override_selects_the_converter(self, tmp_path, monkeypatch):
        from impeller_fingerprint import cli
        conv = tmp_path / "conv.py"
        conv.write_text("import sys\nopen(sys.argv[2], 'w').write(sys.argv[1])\n")
        monkeypatch.setenv("MD2DOCX", str(conv))
        md = tmp_path / "in.md"
        md.write_text("# x\n")
        out = tmp_path / "out.docx"
        cli.render_docx(md, str(out))
        assert out.read_text() == str(md)

    def test_default_is_repo_relative(self, monkeypatch):
        from impeller_fingerprint import cli
        monkeypatch.delenv("MD2DOCX", raising=False)
        assert cli.docx_converter().name == "md2docx.py"
