"""A12 — 정본 참고문헌 표(확정본). 설계서 M4.

**왜 코드에 두는가.** v2.7까지 참고문헌은 정본 md 안의 자유서식 문단이었고 24건이
`[verify]`(저자·제목·권호·페이지 공란)였다. 서지사항을 코드의 표로 옮기면 세 가지가 생긴다 —
(1) `cli a12`로 `a12_references.csv`를 재생성할 수 있고, (2) 정본 md의 본문 인용번호가 이 표와
1:1로 맞는지 **회귀검정**할 수 있고(`tests/test_references.py`), (3) 각 항목이 어떤 URL로
검증됐는지 `source_url`에 남는다.

**검증 방법.** 전부 Crossref REST API(`api.crossref.org/works/<DOI>`)로 저자 전체·제목·저널·권·호·
페이지·연도·DOI를 대조했다. 내용(주제)이 본문 서술과 맞는지 의심스러운 2건은 초록까지 확인했다
(36 습식 감쇠의 벽 근접 의존, 40 균열면 접촉의 비선형 부가강성).

**2026-08-14 추가 4건 / 2026-08-15 재배치.** Σ_y 스윕 앵커(34 Mituletu·Gillich·Maia),
수조 벽 이격(37 Østby, 38 Khalfaoui), 접촉센서 질량부하(39 Çakar & Sanlıtürk)가 §3.5와
§5 E1의 `[FILL]`을 계산으로 대체하면서 필요해졌다. 번호는 두 번 밀렸다 — ① 34가 §3.5에서
처음 인용되어 그 뒤가 밀렸고, ② 2026-08-15 §3.7 재작성으로 **Stenius·Moraga의 첫 인용이
§3.7로 앞당겨져** 다시 4-사이클로 돌았다(38→35, 35→36, 36→37, 37→38). 최종:
35 Stenius, 36 Moraga, 37 Østby, 38 Khalfaoui, 39 Çakar, 40 Zhang(옛 36).
**교훈**: 문단을 옮기면 번호가 바뀐다. 이 표를 손으로 세지 말고 `test_references.py::
TestCanonicalConsistency::test_numbering_follows_first_citation_order`를 돌려서 확인한다.

**절대 추측하지 않는다.** 확인하지 못한 항목은 `status`를 `verified`가 아닌 값으로 두고 그 이유를
`note`에 적는다. 정본 md 생성 시 `verified`가 아닌 항목에는 `[verify]` 태그가 다시 붙는다
(`markdown_block()`).

**번호는 정본 md의 첫 인용 순서**다(Elsevier/JSV 규정). 번호를 바꾸면 정본 본문의 인용도 함께
바꿔야 하고, `test_references.py::TestCanonicalConsistency`가 그것을 강제한다.
"""
from __future__ import annotations

import csv
import re
import urllib.parse
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Sequence

CROSSREF = "https://api.crossref.org/works/"


@dataclass(frozen=True)
class Ref:
    num: int
    authors: str          # "A.B. Kim; C.D. Lee" — 세미콜론 구분
    title: str
    journal: str
    volume: str
    issue: str
    pages: str            # 페이지 범위 또는 article number
    year: str
    doi: str
    status: str           # "verified" | "unverified: <이유>"
    source_url: str
    note: str

    @property
    def verified(self) -> bool:
        return self.status == "verified"


def _r(num, authors, title, journal, volume, issue, pages, year, doi, note,
       status="verified", source_url=None):
    return Ref(num, authors, title, journal, volume, issue, pages, year, doi,
               status, source_url if source_url is not None else CROSSREF + doi, note)


# ---------------------------------------------------------------------------
# 번호 = 정본 md 첫 인용 순서.
# §1 [1–7] → §2 [8–28] → §2 고전 [29–33] → §3.5 [34] → §5 E1 [35–39] → §6 [40]
# 2026-08-15 §3.7 재작성으로 Stenius·Moraga의 첫 인용이 §3.7(§5보다 앞)로 옮겨졌다:
# §3.7 순서 = 35 Stenius(연성 음향) → 36 Moraga(벽 근접 감쇠), 이어서 §5 E1의
# 37 Østby → 38 Khalfaoui → 39 Çakar. 문장 순서가 번호를 정한다 — 문단을 옮기면
# 번호를 다시 잡아야 하고 test_references가 그것을 강제한다.
# ---------------------------------------------------------------------------
REFERENCES: tuple[Ref, ...] = (
    _r(1, "Z.A. Jassim; N.N. Ali; F. Mustapha; N.A. Abdul Jalil",
       "A review on the vibration analysis for a damage occurrence of a cantilever beam",
       "Engineering Failure Analysis", "31", "", "442-461", "2013",
       "10.1016/j.engfailanal.2013.02.016",
       "저자 4명·제목·페이지 신규 확정(초안은 'Z.A. Jassim et al.'과 권·연도만)."),
    _r(2, "A.D. Dimarogonas",
       "Vibration of cracked structures: a state of the art review",
       "Engineering Fracture Mechanics", "55", "5", "831-857", "1996",
       "10.1016/0013-7944(94)00175-8",
       "초안의 권·페이지가 맞았다. 제목·호·DOI 신규."),
    _r(3, "W.M. Ostachowicz; M. Krawczuk",
       "Analysis of the effect of cracks on the natural frequencies of a cantilever beam",
       "Journal of Sound and Vibration", "150", "2", "191-201", "1991",
       "10.1016/0022-460X(91)90615-Q",
       "초안에 DOI만 있었다(유일하게 [verify]가 없던 항목). 제목·호 신규. "
       "정본 §3.1의 c_θ = 5.346(h/EI)·J(ā) 다항식 출처."),
    _r(4, "G.Y. Xu; W.D. Zhu; B.H. Emory",
       "Experimental and numerical investigation of structural damage detection using changes "
       "in natural frequencies",
       "ASME Journal of Vibration and Acoustics", "129", "6", "686-700", "2007",
       "10.1115/1.2731409", "제목·호·페이지 신규."),
    _r(5, "J.P. Kaipio; E. Somersalo",
       "Statistical and Computational Inverse Problems (Applied Mathematical Sciences, vol. 160)",
       "Springer, New York", "", "", "", "2005",
       "10.1007/b138659",
       "ISBN 978-0-387-22073-4 / 978-0-387-27132-3. 초안이 한 항목에 묶어 두었던 "
       "Kaipio–Somersalo + Colton–Kress를 5/6/7로 분리했다(서지 확정 가능 단위)."),
    _r(6, "J. Kaipio; E. Somersalo",
       "Statistical inverse problems: discretization, model reduction and inverse crimes",
       "Journal of Computational and Applied Mathematics", "198", "2", "493-504", "2007",
       "10.1016/j.cam.2005.09.027",
       "신규 추가. 'inverse crime'을 제목에 담은 정본 인용으로, 초안이 이름만 적었던 "
       "용어 출처 문제를 실제 서지로 해소한다."),
    _r(7, "D. Colton; R. Kress",
       "Inverse Acoustic and Electromagnetic Scattering Theory, 3rd ed. "
       "(Applied Mathematical Sciences, vol. 93)",
       "Springer, New York", "", "", "", "2013",
       "10.1007/978-1-4614-4942-3",
       "초안은 '(inverse-crime terminology)'만 적고 서지사항이 전혀 없었다. "
       "용어의 원출처로 남기되 확인된 판(3rd ed., 2013)으로 고정. "
       "ISBN 978-1-4614-4941-6 / 978-1-4614-4942-3."),
    _r(8, "M.N. Cerri; F. Vestroni",
       "Detection of damage in beams subjected to diffused cracking",
       "Journal of Sound and Vibration", "234", "2", "259-276", "2000",
       "10.1006/jsvi.1999.2887", "초안 서지가 맞았다. 호·DOI 신규."),
    _r(9, "A. Morassi",
       "Identification of a crack in a rod based on changes in a pair of natural frequencies",
       "Journal of Sound and Vibration", "242", "4", "577-596", "2001",
       "10.1006/jsvi.2000.3380", "초안 서지가 맞았다. 호·DOI 신규."),
    _r(10, "C.-P. Fritzen; D. Jennewein; T. Kiefer",
       "Damage detection based on model updating methods",
       "Mechanical Systems and Signal Processing", "12", "1", "163-186", "1998",
       "10.1006/mssp.1997.0139", "초안 서지가 맞았다. 호·DOI 신규."),
    _r(11, "A. Kumar; C.M. Krousgrill",
       "Mode-splitting and quasi-degeneracies in circular plate vibration problems: the example "
       "of free vibrations of the stator of a traveling wave ultrasonic motor",
       "Journal of Sound and Vibration", "331", "26", "5788-5802", "2012",
       "10.1016/j.jsv.2012.07.032",
       "**내용 불일치 정정**: 초안 설명 '[split frequencies / cracked rotationally periodic "
       "structures]'는 틀렸다 — 원판(초음파모터 스테이터)의 mode-splitting·준축퇴 논문이며 "
       "bladed disk도 crack도 아니다. §2·§3.2의 서술을 이 내용에 맞게 고쳤다."),
    _r(12, "J.H. Kuang; B.W. Huang",
       "The effect of blade crack on mode localization in rotating bladed disks",
       "Journal of Sound and Vibration", "227", "1", "85-103", "1999",
       "10.1006/jsvi.1999.2329", "초안 권·페이지가 맞았다. 제목·호·DOI 신규."),
    _r(13, "S. Wang; Y. Zi; B. Li; C. Zhang; Z. He",
       "Reduced-order modeling for mistuned centrifugal impellers with crack damages",
       "Journal of Sound and Vibration", "333", "25", "6979-6995", "2014",
       "10.1016/j.jsv.2014.07.009",
       "§2 — **원심임펠러 + 균열 + 미스튜닝**을 함께 다루는 유일한 직접 비교대상. [12] bladed disk와 [13] 산업용 임펠러 사이에 놓인다."),
    _r(14, "M.P. Castanier; C. Pierre",
       "Modeling and analysis of mistuned bladed disk vibration: current status and emerging directions",
       "Journal of Propulsion and Power", "22", "2", "384-396", "2006",
       "10.2514/1.16345",
       "§2 — mode families·localization·ROM의 고전 리뷰. 조립체에서 단일베인 응답이 그대로 살아남지 않는 이유를 넓은 미스튜닝 문헌과 잇는다."),
    _r(15, "X. Zhao; H. Li; S. Yang; Z. Fan; J. Dong; H. Cao",
       "Blade vibration measurement and numerical analysis of a mistuned industrial impeller in "
       "a single-stage centrifugal compressor",
       "Journal of Sound and Vibration", "501", "", "116068", "2021",
       "10.1016/j.jsv.2021.116068",
       "저자 6명·제목 확정. **초안에서 목록에만 있고 본문 인용이 없었다** — §2의 mistuning "
       "문헌 문장에 배선했다(초안은 그 자리에 [9,16]을 썼는데 11은 원판, 12는 bladed disk라 "
       "'mistuning 문헌'의 근거가 되지 못했다)."),
    _r(16, "N. Gao; Z. Wei; C. Wang; D. Zhu; S. Wang",
       "Experimental and analytical studies on eliminating natural frequency splitting of an "
       "axisymmetric structure with cyclically symmetric feature groups",
       "Journal of Sound and Vibration", "562", "", "117829", "2023",
       "10.1016/j.jsv.2023.117829",
       "**내용 불일치 정정**: 초안 설명 'split-frequency-based disk crack detection'은 틀렸다 "
       "— 순환대칭 특징군이 만드는 분리를 *제거·보정*하는 논문이다(30 Fox의 correction 문제에 "
       "대응). §2 서술을 교체했다."),
    _r(17, "L. Zhang; W. Liao; J. Fan",
       "A novel surrogate-based crack identification method for cantilever beam based on the "
       "change of natural frequencies",
       "Computers & Structures", "292", "", "107243", "2024",
       "10.1016/j.compstruc.2023.107243",
       "저자(Long Zhang, Wenlin Liao, Juntao Fan)·제목 확정. 초안 설명과 내용 일치."),
    _r(18, "N. Dwek; V. Dimopoulos; D. Janssens; M. Kirchner; E. Deckers; F. Naets",
       "Damage identification in plate-like structures using frequency-coupled "
       "ℓ₁-based sparse estimation",
       "Mechanical Systems and Signal Processing", "224", "", "112084", "2025",
       "10.1016/j.ymssp.2024.112084",
       "저자 6명·제목 확정. **중복 표기 제거**: 초안은 이 항목 끝에 '17b. [Dessi et al. — "
       "see 26.]'를 붙여 한 번호에 두 문헌을 담고 있었다(M3 중복 이력의 잔재) — Dessi는 23번 "
       "단일 항목으로 정리했다. 초안이 '실험 포함'이라 적은 부분은 판형 구조 실험이다."),
    _r(19, "M. Rahai; R. Sarfaraz; M. Nourafkan; M. Hamdaoui",
       "Damage localization and quantification in viscoelastic sandwich beams via "
       "uncertainty-aware FE model updating with FRF-derived complex modes",
       "Journal of Sound and Vibration", "639", "", "119846", "2026",
       "10.1016/j.jsv.2026.119846",
       "권 639 신규 확정(초안은 권 공란). 저자 4명·제목 신규."),
    _r(20, "D. Jana; S. Mukhopadhyay; S. Ray-Chaudhuri",
       "Fisher information-based optimal input locations for modal identification",
       "Journal of Sound and Vibration", "459", "", "114833", "2019",
       "10.1016/j.jsv.2019.06.040", "저자 3명·제목 확정."),
    _r(21, "K. Zhang; X. Yan",
       "Multi-cracks identification method for cantilever beam structure with variable "
       "cross-sections based on measured natural frequency changes",
       "Journal of Sound and Vibration", "387", "", "53-65", "2017",
       "10.1016/j.jsv.2016.09.028",
       "제목·페이지 신규. **초안에서 미인용**이었고 §2의 주파수기반 식별 문장에 배선."),
    _r(22, "A.C. Altunışık; F.Y. Okur; V. Kahya",
       "Modal parameter identification and vibration based damage detection of a multiple "
       "cracked cantilever beam",
       "Engineering Failure Analysis", "79", "", "154-170", "2017",
       "10.1016/j.engfailanal.2017.04.026",
       "제목·페이지 신규. **초안에서 미인용**이었고 §2에 배선."),
    _r(23, "S.S.B. Chinka; S.R. Putti; B.K. Adavi",
       "Modal testing and evaluation of cracks on cantilever beam using mode shape curvatures "
       "and natural frequencies",
       "Structures", "32", "", "1386-1397", "2021",
       "10.1016/j.istruc.2021.03.049",
       "공저자 2명·제목·페이지 신규(초안은 'et al.'). **초안에서 미인용**이었고 §2에 배선."),
    _r(24, "N. Gillich; C. Tufisi; C. Sacarea; C.V. Rusu; G.-R. Gillich; Z.-I. Praisach; "
           "M. Ardeljan",
       "Beam damage assessment using natural frequency shift and machine learning",
       "Sensors", "22", "3", "1118", "2022",
       "10.3390/s22031118",
       "**저자 정정**: 제1저자는 N.(Nicoleta) Gillich이고 초안이 제1저자로 적은 "
       "G.-R. Gillich는 제5저자다. 논문번호 1118. **초안에서 미인용**이었고 §2에 배선. "
       "**2026-08-14 전문 확인 — 인용을 [15–23] 블록에서 떼어 특정 인용으로 승격했다.** 이 문헌의 "
       "식 (1)·(2)가 정본 순방향 사상의 **보 극한과 같은 분해**다: RFS Δf̄ = γ(a)·[φ̄''ᵢ(x)]², 곧 "
       "깊이만의 심각도 × 정규화 곡률²이고 γ(a)는 **균열위치·경계조건·모드번호에 무관**하다(p.4). "
       "모드간 패턴을 곡률 인자가 전부 나르므로 정본 §3.3의 심각도×커널 분리와 같은 구조다. "
       "둘째, **변곡점의 균열은 그 모드 주파수를 낮추지 않는다**가 p.4에 명시돼 있어(그들의 ref [31]) "
       "정본의 '곡률-null 실명'은 **고전 결과로 귀속**해야 하고 신규성은 그것이 Mode-II 슬라이딩 "
       "유연도·유한 커프·2D 탄성에서 **살아남는다**는 정량화에 있다. 셋째, **약한 클램프를 뿌리의 "
       "균열로 모형화**해 RFS를 중첩한다(식 (6)) — §5 E1이 free-free를 일차 지지로 택한 이유가 막연한 "
       "우려가 아니라 식별된 기제임을 뒷받침한다. 균열 위치 오차 < 0.6 %(초록). 대상은 보(캔틸레버·"
       "다경간)이고 축퇴쌍·2×2 관측량은 다루지 않으므로 정본의 2×2 골격과 충돌하지 않는다. "
       "**심각도 규약 주의**: 이 문헌의 γ(a)는 에너지법 기반(식 (3))이고 정본 Table 1은 컴플라이언스 "
       "규약(Tada/Dimarogonas)을 쓴다 — 제3의 규약이므로 Table 1 정합 과제에 함께 올려야 한다."),
    _r(25, "D. Dessi; F. Passacantilli; A. Venturi",
       "Analysis and mitigation of uncertainties in damage identification by modal-curvature "
       "based methods",
       "Journal of Sound and Vibration", "596", "", "118769", "2025",
       "10.1016/j.jsv.2024.118769",
       "제목 확정. 초안 설명 '노이즈/이산화/편향 전파, Monte-Carlo thresholds'는 "
       "모달곡률법의 불확실성 분석이라는 실제 주제와 부합한다. **초안에서 미인용**이었고 "
       "§2에 배선(16번의 '17b' 중복 표기도 함께 제거)."),
    _r(26, "T. Lim; H.W. Park",
       "Investigating the modal behaviors of a beam with a transverse crack on a high-frequency "
       "bending node",
       "International Journal of Mechanical Sciences", "221", "", "107217", "2022",
       "10.1016/j.ijmecsci.2022.107217",
       "초안이 요청한 '저자 이니셜' 확정: Taejeong Lim, Hyun Woo Park. 제목 신규."),
    _r(27, "T. Lim; H.W. Park",
       "Investigating the modal behaviors of a deep beam with a transverse open crack",
       "Journal of Sound and Vibration", "590", "", "118613", "2024",
       "10.1016/j.jsv.2024.118613", "저자·제목 확정."),
    _r(28, "H. Jalali; F. Noohi",
       "A modal-energy based equivalent lumped model for open cracks",
       "Mechanical Systems and Signal Processing", "98", "", "50-62", "2018",
       "10.1016/j.ymssp.2017.04.038",
       "제목 확정. 초안이 'notch-crack equivalence'라 적었으나 정확히는 '열린 균열의 "
       "모달에너지 기반 등가 집중(스프링) 모델'이며, §5 E2의 등가 유연도 보정 절차가 "
       "인용하는 대상으로는 맞다 — §2 서술만 그 표현으로 고쳤다."),
    _r(29, "C. Papadimitriou",
       "Optimal sensor placement methodology for parametric identification of structural systems",
       "Journal of Sound and Vibration", "278", "4-5", "923-947", "2004",
       "10.1016/j.jsv.2003.10.063",
       "초안의 '[정본 참조문헌을 선정하라 — 예: Kammer; Papadimitriou]' 지시를 해소. "
       "Fisher 정보/정보엔트로피 기반 최적배치의 정본으로 이 논문을 선택했다(28과 함께 인용)."),
    _r(30, "D.C. Kammer",
       "Sensor placement for on-orbit modal identification and correlation of large space "
       "structures",
       "Journal of Guidance, Control, and Dynamics", "14", "2", "251-259", "1991",
       "10.2514/3.20635",
       "초안의 'e.g. Kammer' 후보를 확정 — effective independence의 원전. 27과 함께 인용."),
    _r(31, "S.A. Tobias; R.N. Arnold",
       "The influence of dynamical imperfection on the vibration of rotating disks",
       "Proceedings of the Institution of Mechanical Engineers", "171", "1", "669-690", "1957",
       "10.1243/PIME_PROC_1957_171_056_02",
       "권 171(1)·페이지 669-690 확정(초안은 '권/페이지 미확인'). 제목도 초안 추정과 일치."),
    _r(32, "C.H.J. Fox",
       "A simple theory for the analysis and correction of frequency splitting in slightly "
       "imperfect rings",
       "Journal of Sound and Vibration", "142", "2", "227-243", "1990",
       "10.1016/0022-460X(90)90554-D",
       "권 142(2)·페이지 227-243 확정(초안은 '권/페이지 미확인'). 제목도 초안 추정과 일치."),
    _r(33, "R. Perrin; T. Charnley; J. dePont",
       "Normal modes of the modern English church bell",
       "Journal of Sound and Vibration", "90", "1", "29-49", "1983",
       "10.1016/0022-460X(83)90401-7",
       "초안의 '시리즈에서 정본 1편을 고르라'를 해소 — 1983년 시리즈 최초편을 선택했다. "
       "제3저자 J. dePont 신규(초안은 'Perrin, Charnley et al.')."),
    _r(34, "J. Lin; R.G. Parker",
       "Structured vibration characteristics of planetary gears with unequally spaced planets",
       "Journal of Sound and Vibration", "233", "5", "921-928", "2000",
       "10.1006/jsvi.1999.2581",
       "초안의 '정확한 제목 미확인(2000)'을 해소. 대칭성 파괴(불균등 배치)에서의 구조화 "
       "모달 특성 — §2가 요구한 내용의 절반."),
    _r(35, "J. Lin; R.G. Parker",
       "Sensitivity of planetary gear natural frequencies and vibration modes to model parameters",
       "Journal of Sound and Vibration", "228", "1", "109-128", "1999",
       "10.1006/jsvi.1999.2398",
       "초안이 한 항목(2000)에 담았던 '구조화 모달 특성 + 1차 감도' 중 **감도** 쪽 원전은 "
       "1999년 논문이다. 32와 함께 인용해 §2 서술을 서지로 뒷받침한다."),
    _r(36, "R.N. Wake; J.S. Burgess; J.T. Evans",
       "Changes in the natural frequencies of repeated mode pairs induced by cracks in a vibrating ring",
       "Journal of Sound and Vibration", "214", "4", "761-770", "1998",
       "10.1006/jsvi.1998.1606",
       "§2 신규성 문단의 직접 선행연구 — 균열이 축대칭 링의 repeated pair를 split시키고 성장에 따라 orientation까지 바꾼다. **Crossref 제목이 전부 대문자**(옛 JSV 레코드)라 title case로 옮겼다."),
    _r(37, "T.J. Royston; T. Spohnholtz; W.A. Ellingson",
       "Use of non-degeneracy in nominally axisymmetric structures for fault detection with application to cylindrical geometries",
       "Journal of Sound and Vibration", "230", "4", "791-808", "2000",
       "10.1006/jsvi.1999.2653",
       "§2 — 국소결함이 degeneracy를 깨 만드는 splitting·모드형 변화를 결함검출에 쓰고 수치·실험 검증까지 한다. **Crossref 제목 대문자**를 title case로 옮겼다."),
    _r(38, "O.E. Esu; Y. Wang; M.K. Chryssanthopoulos",
       "Local vibration mode pairs for damage identification in axisymmetric tubular structures",
       "Journal of Sound and Vibration", "494", "", "115845", "2021",
       "10.1016/j.jsv.2020.115845",
       "§2 — 축대칭 관형구조의 국소부식을 repeated mode pair로 식별한다. 정본의 pair 관측량 사용과 가장 가까운 최근 JSV 선행연구."),
    _r(39, "O.E. Esu; Y. Wang; M.K. Chryssanthopoulos",
       "A baseline-free method for damage identification in pipes from local vibration mode pair frequencies",
       "Structural Health Monitoring", "21", "5", "2152-2189", "2022",
       "10.1177/14759217211052335",
       "§2 — 위 JSV 연구의 후속(검출·국재화·정량화). **연도 주의**: Crossref date-parts는 2021(online first)이고 발행호는 21(5) 2022다 — 호 연도를 적었다(35 Moraga와 같은 처리)."),
    _r(40, "H. Tada; P.C. Paris; G.R. Irwin",
       "The Stress Analysis of Cracks Handbook, 3rd ed.",
       "ASME Press, New York", "", "", "", "2000",
       "10.1115/1.801535",
       "§3.1·§4.1이 핵심 기준해로 쓰는 **Tada 적분형의 원전**인데 목록에 없었다(검토 지적). 단행본이라 Crossref에 권·호·페이지가 없다 — DOI·저자·제목만 대조했다."),
    _r(41, "I.C. Mituletu; G.-R. Gillich; N.M.M. Maia",
       "A method for an accurate estimation of natural frequencies using swept-sine acoustic "
       "excitation",
       "Mechanical Systems and Signal Processing", "116", "", "693-709", "2019",
       "10.1016/j.ymssp.2018.07.018",
       "§3.5 Σ_y 스윕 **네 레벨 전부**의 앵커. **2026-08-14 전문 확인**(초기 등재 시에는 Crossref에 "
       "초록이 없어 '달성 정밀도 미확인'으로 두었고 정본도 10⁻⁴를 우리 선택으로만 썼다 — 전문에서 "
       "수치가 확인되어 인용을 승격했다). 시편은 캔틸레버 3차 굽힘 f_n = 71.556 Hz, 비접촉 음향 "
       "스윕사인. 확인된 값: 알고리즘 분해능 **밀리헤르츠**(1.4e−5 of f, p.694·결론), 크롭 후 빈 간격 "
       "0.04 Hz(5.6e−4), 두 독립 절차(RFA 교점 / FECI 평균) 일치 **0.006 Hz = 8.4e−5**(Table 8), "
       "고정구를 그대로 두고 **가진법·분석창만** 바꿀 때 short-time sine **0.031 Hz = 4.3e−4**(Table 9), "
       "해머 충격 **0.073 Hz = 1.0e−3**(Table 10). **불일치 1건**: p.708의 서술적 상계(short-sine 0.5 %, "
       "mechanical 1 %)는 자기 표의 산포보다 한 자릿수 느슨하다 — 정본은 **표 값을 인용**하고 상계는 "
       "'우리 최상단 3e−3도 그 안에 든다'는 확인으로만 쓴다. 부수 확인: 이 문헌이 음향 비접촉 가진을 "
       "택한 이유가 '접촉 없이 힘을 전달해 구조의 동특성을 바꾸지 않기 위해'(p.694)이며, 이는 F90의 "
       "접촉센서 배제와 독립적으로 같은 논거다. 제3저자 Maia는 [13,14]의 Lim & Park 계열과 무관하다."),
    _r(42, "I. Stenius; L. Fagerberg; A. Säther",
       "Predicting the natural frequency of submerged structures using coupled solid-acoustic "
       "finite element simulations",
       "Ocean Engineering", "159", "", "37-46", "2018",
       "10.1016/j.oceaneng.2018.03.069",
       "제목 확정. **초안에서 미인용**이었고 §5 E1 습식 baseline(습식 주파수 예측)에 배선."),
    _r(43, "G. Moraga; X. Xia; S. Roig; C. Valero; D. Valentín; M. Egusquiza; L. Zhou; "
           "E. Egusquiza; A. Presas",
       "Experimental study on the influence of vibration amplitude on the fluid damping of a "
       "submerged disk",
       "Journal of Sound and Vibration", "569", "", "118099", "2024",
       "10.1016/j.jsv.2023.118099",
       "**번호 정합성 정정**: 초안 §5 E1의 인용 '[16-wet]'은 목록의 어느 번호도 아니었다 "
       "(M3 중복 이력의 잔재). 그것이 가리키던 문헌이 이것이며 34로 정규화했다. "
       "제1저자 성은 Moraga(전체 Greco Alonso Moraga González)이므로 초안의 "
       "'[Greco / Moraga et al.]' 양자택일도 해소된다. 초록 확인: 진동 진폭과 강체벽 "
       "근접거리 **둘 다** 유체감쇠를 좌우한다 — 정본 E1의 '진폭·벽 민감' 서술과 일치. "
       "온라인 2023-10, 권 569 인쇄 2024-01(DOI 연도는 2023)."),
    _r(44, "T. Krizak; K. D'Souza",
       "A generalized model of mistuning for bladed disks",
       "Journal of Sound and Vibration", "606", "", "119003", "2025",
       "10.1016/j.jsv.2025.119003",
       "§4.2 — 같은 nodal-diameter/harmonic index에 여러 radial·structural family가 존재한다는 현대적 근거. m=4를 'm=2의 alias'라는 이유만으로 관측량에서 지울 수 없다는 논거(F107)."),
    _r(45, "S. Fortunati; F. Gini; M.S. Greco; C.D. Richmond",
       "Performance bounds for parameter estimation under misspecified models: fundamental findings and applications",
       "IEEE Signal Processing Magazine", "34", "6", "142-157", "2017",
       "10.1109/MSP.2017.2738017",
       "§4.3 — misspecified 모델에서는 classical CRB를 그대로 쓸 수 없다는 이론적 배경. 정본이 self-consistent 데이터에서는 CRLB가 맞고 독립 3D 데이터에서 coverage가 무너지는 것을 핵심 결과로 삼으므로 그 통계적 해석을 받쳐 준다."),
    _r(46, "P.T.K. Østby; K. Sivertsen; J.T. Billdal; B. Haugen",
       "Experimental investigation on the effect off near walls on the eigen frequency of a low "
       "specific speed francis runner",
       "Mechanical Systems and Signal Processing", "118", "", "757-766", "2019",
       "10.1016/j.ymssp.2018.08.060",
       "§5 E1 [FILL] 1(수조 벽 이격)의 정성 근거 — 근접벽이 러너 고유진동수를 낮춘다는 "
       "실험. 제목의 'effect off'는 **원 논문의 오타를 그대로 옮긴 것**이며 Crossref 정본 "
       "표기와 일치한다(임의 교정은 서지 불일치가 된다). 임계거리 자체는 기하·모드 의존이라 "
       "이 문헌에서 우리 기하로 옮길 수 있는 수치는 없다."),
    _r(47, "K. Khalfaoui; G. Moraga; J. Bareis; M. Zorn; A. Presas; D. Valentín; "
           "S. Riedelbauch",
       "On the numerical prediction of added damping and added mass of vibrating disc-like "
       "structures in heavy fluids",
       "Journal of Sound and Vibration", "618", "", "119305", "2025",
       "10.1016/j.jsv.2025.119305",
       "§5 E1 [FILL] 1 — 벽과의 거리를 변수로 한 added mass·감쇠의 **수치예측** 선례. "
       "우리가 d* 스윕을 사전등록만 하고 값을 내지 못한 이유(설계서 F86: ka = 0.43–1.12로 "
       "비압축 근사가 성립하지 않고, 예산이 요구하는 δβ/β < 0.26 %는 NAVMI 스트립이론의 "
       "정확도 밖)를 밝히는 대조 문헌이기도 하다."),
    _r(48, "Ö. Çakar; K.Y. Sanlıtürk",
       "Elimination of transducer mass loading effects from frequency response functions",
       "Mechanical Systems and Signal Processing", "19", "1", "87-104", "2005",
       "10.1016/S0888-3270(03)00086-4",
       "§5 E1 [FILL] 2(접촉센서 질량부하)의 표준 문헌. Crossref는 저자를 ASCII로 "
       "'O Cakar; K.Y Sanliturk'로 주지만 원 논문 표기는 Ö. Çakar·K.Y. Sanlıtürk이므로 "
       "발음기호를 복원했다(제목·권1·페이지 87-104·2005는 Crossref 그대로). **주의**: 이 "
       "문헌은 FRF에서 부하효과를 **사후 제거**하는 방법이고, 정본이 인용하는 맥락은 "
       "'제거가 필요 없는 수준으로 애초에 부하를 없애라'다 — A14의 mg 한계는 우리 계산이며 "
       "이 문헌으로부터 나온 값이 아니다."),
    _r(49, "M. Zhang; D. Valentín; C. Valero; A. Presas; M. Egusquiza; E. Egusquiza",
       "Experimental and numerical investigation on the influence of a large crack on the modal "
       "behaviour of a Kaplan turbine blade",
       "Engineering Failure Analysis", "109", "", "104389", "2020",
       "10.1016/j.engfailanal.2020.104389",
       "저자 6명·제목 확정. 초록 확인: 균열이 커서 자중으로 균열면이 접촉하고 그 접촉이 "
       "부가강성을 주므로 수치모델에 비선형으로 넣어야 한다 — 초안이 적은 'crack-face "
       "contact'과 일치하며 정본 §6의 'linear open crack(breathing 제외)' 한계 서술의 "
       "근거가 된다. **초안에서 미인용**이었고 §6에 배선."),    _r(50, "P. Gardner; C. Lord; R.J. Barthorpe",
       "Bayesian history matching for structural dynamics applications",
       "Mechanical Systems and Signal Processing", "143", "", "106828", "2020",
       "10.1016/j.ymssp.2020.106828",
       "§4.3 — model discrepancy를 additive term으로 명시해 다루는 구조동역학 보정 문헌. 정본의 '3D–Kirchhoff 불일치를 additive surrogate로 넣는다'는 접근의 배경."),

)

CSV_HEADER = tuple(f.name for f in fields(Ref))


#: 2026-08-15 외부 검토 4차로 추가된 10건. 번호는 `renumber_from_canon`이 첫 인용 순서로
#: 정하므로 여기서는 **토큰**으로만 식별한다(정본에 `{{TOKEN}}`으로 심고 스크립트가 치환).
PENDING: dict[str, tuple] = {
 "WAKE": ("R.N. Wake; J.S. Burgess; J.T. Evans",
   "Changes in the natural frequencies of repeated mode pairs induced by cracks in a vibrating ring",
   "Journal of Sound and Vibration", "214", "4", "761-770", "1998", "10.1006/jsvi.1998.1606",
   "§2 신규성 문단의 직접 선행연구 — 균열이 축대칭 링의 repeated pair를 split시키고 성장에 따라 "
   "orientation까지 바꾼다. **Crossref 제목이 전부 대문자**(옛 JSV 레코드)라 title case로 옮겼다."),
 "ROYSTON": ("T.J. Royston; T. Spohnholtz; W.A. Ellingson",
   "Use of non-degeneracy in nominally axisymmetric structures for fault detection with application "
   "to cylindrical geometries",
   "Journal of Sound and Vibration", "230", "4", "791-808", "2000", "10.1006/jsvi.1999.2653",
   "§2 — 국소결함이 degeneracy를 깨 만드는 splitting·모드형 변화를 결함검출에 쓰고 수치·실험 "
   "검증까지 한다. **Crossref 제목 대문자**를 title case로 옮겼다."),
 "ESUJSV": ("O.E. Esu; Y. Wang; M.K. Chryssanthopoulos",
   "Local vibration mode pairs for damage identification in axisymmetric tubular structures",
   "Journal of Sound and Vibration", "494", "", "115845", "2021", "10.1016/j.jsv.2020.115845",
   "§2 — 축대칭 관형구조의 국소부식을 repeated mode pair로 식별한다. 정본의 pair 관측량 사용과 "
   "가장 가까운 최근 JSV 선행연구."),
 "ESUSHM": ("O.E. Esu; Y. Wang; M.K. Chryssanthopoulos",
   "A baseline-free method for damage identification in pipes from local vibration mode pair "
   "frequencies",
   "Structural Health Monitoring", "21", "5", "2152-2189", "2022", "10.1177/14759217211052335",
   "§2 — 위 JSV 연구의 후속(검출·국재화·정량화). **연도 주의**: Crossref date-parts는 2021"
   "(online first)이고 발행호는 21(5) 2022다 — 호 연도를 적었다(35 Moraga와 같은 처리)."),
 "WANG": ("S. Wang; Y. Zi; B. Li; C. Zhang; Z. He",
   "Reduced-order modeling for mistuned centrifugal impellers with crack damages",
   "Journal of Sound and Vibration", "333", "25", "6979-6995", "2014", "10.1016/j.jsv.2014.07.009",
   "§2 — **원심임펠러 + 균열 + 미스튜닝**을 함께 다루는 유일한 직접 비교대상. [12] bladed disk와 "
   "[13] 산업용 임펠러 사이에 놓인다."),
 "CASTANIER": ("M.P. Castanier; C. Pierre",
   "Modeling and analysis of mistuned bladed disk vibration: current status and emerging directions",
   "Journal of Propulsion and Power", "22", "2", "384-396", "2006", "10.2514/1.16345",
   "§2 — mode families·localization·ROM의 고전 리뷰. 조립체에서 단일베인 응답이 그대로 살아남지 "
   "않는 이유를 넓은 미스튜닝 문헌과 잇는다."),
 "TADA": ("H. Tada; P.C. Paris; G.R. Irwin",
   "The Stress Analysis of Cracks Handbook, 3rd ed.",
   "ASME Press, New York", "", "", "", "2000", "10.1115/1.801535",
   "§3.1·§4.1이 핵심 기준해로 쓰는 **Tada 적분형의 원전**인데 목록에 없었다(검토 지적). "
   "단행본이라 Crossref에 권·호·페이지가 없다 — DOI·저자·제목만 대조했다."),
 "KRIZAK": ("T. Krizak; K. D'Souza",
   "A generalized model of mistuning for bladed disks",
   "Journal of Sound and Vibration", "606", "", "119003", "2025", "10.1016/j.jsv.2025.119003",
   "§4.2 — 같은 nodal-diameter/harmonic index에 여러 radial·structural family가 존재한다는 "
   "현대적 근거. m=4를 'm=2의 alias'라는 이유만으로 관측량에서 지울 수 없다는 논거(F107)."),
 "GARDNER": ("P. Gardner; C. Lord; R.J. Barthorpe",
   "Bayesian history matching for structural dynamics applications",
   "Mechanical Systems and Signal Processing", "143", "", "106828", "2020",
   "10.1016/j.ymssp.2020.106828",
   "§4.3 — model discrepancy를 additive term으로 명시해 다루는 구조동역학 보정 문헌. 정본의 "
   "'3D–Kirchhoff 불일치를 additive surrogate로 넣는다'는 접근의 배경."),
 "FORTUNATI": ("S. Fortunati; F. Gini; M.S. Greco; C.D. Richmond",
   "Performance bounds for parameter estimation under misspecified models: fundamental findings "
   "and applications",
   "IEEE Signal Processing Magazine", "34", "6", "142-157", "2017", "10.1109/MSP.2017.2738017",
   "§4.3 — misspecified 모델에서는 classical CRB를 그대로 쓸 수 없다는 이론적 배경. 정본이 "
   "self-consistent 데이터에서는 CRLB가 맞고 독립 3D 데이터에서 coverage가 무너지는 것을 "
   "핵심 결과로 삼으므로 그 통계적 해석을 받쳐 준다."),
}


def _norm(s: str) -> str:
    """비교용 정규화 — 대소문자·구두점·공백·발음기호 차이를 지운다.

    Crossref는 저자를 ASCII로 주기도 하고(예: 'O Cakar'), 제목의 하이픈·콜론 표기가 다르므로
    문자열을 그대로 비교하면 전부 불일치로 나온다. 서지의 **실체**만 비교한다.
    """
    import html
    import unicodedata
    s = html.unescape(s or "")                     # '&amp;' → '&'
    s = re.sub(r"<[^>]+>", "", s)                  # MathML/HTML 마크업 제거(ℓ₁ 등)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def verify_against_crossref(refs: Sequence[Ref] | None = None,
                            delay: float = 1.0, timeout: float = 30.0,
                            mailto: str | None = None) -> list[dict]:
    """각 항목을 Crossref로 **다시** 조회해 저장값과 대조한다.

    수동 검증은 한 번 맞았다는 사실만 남기고 그 뒤의 표 수정을 잡지 못한다(번호 재배치처럼
    표를 손대는 작업 뒤에는 특히 위험하다). 이 함수가 그 구멍을 메운다 — 필드별 불일치를
    **보고**하고 예외를 던지지 않으므로, 호출자가 무엇을 고칠지 판단한다.

    책(권·호·페이지가 없는 항목)은 해당 필드를 비교에서 제외한다.
    """
    import json
    import os
    import time
    import urllib.request

    # Crossref polite pool은 연락처를 요구하지 않지만 있으면 우대한다. 개인 주소를
    # 코드에 박으면 공개 저장소에 그대로 실리므로 환경변수로 받는다(없으면 생략).
    if mailto is None:
        mailto = os.environ.get("CROSSREF_MAILTO", "")

    out: list[dict] = []
    for r in (refs if refs is not None else REFERENCES):
        row: dict = {"num": r.num, "doi": r.doi, "status": "ok", "mismatch": {}}
        try:
            req = urllib.request.Request(
                CROSSREF + urllib.parse.quote(r.doi, safe="/:"),
                headers={"User-Agent": "paper3-refs/1.0"
                         + (f" (mailto:{mailto})" if mailto else "")})
            msg = json.load(urllib.request.urlopen(req, timeout=timeout))["message"]
        except Exception as exc:                       # 네트워크·404 모두 보고 대상
            row["status"] = f"fetch failed: {type(exc).__name__}"
            out.append(row)
            continue
        got = {
            "title": (msg.get("title") or [""])[0],
            "journal": (msg.get("container-title") or [""])[0],
            "volume": msg.get("volume", "") or "",
            "issue": msg.get("issue", "") or "",
            "pages": msg.get("page") or msg.get("article-number") or "",
            "year": str(msg.get("issued", {}).get("date-parts", [[""]])[0][0] or ""),
            "first_author": (msg.get("author") or [{}])[0].get("family", ""),
            "n_authors": len(msg.get("author") or []),
        }
        ours = {"title": r.title, "journal": r.journal, "volume": r.volume,
                "issue": r.issue, "pages": r.pages, "year": r.year,
                "first_author": r.authors.split(";")[0].split()[-1],
                "n_authors": len(r.authors.split(";"))}
        is_book = not r.volume and not r.pages
        for k in ("title", "journal", "volume", "issue", "pages", "year",
                  "first_author", "n_authors"):
            if is_book and k in ("journal", "volume", "issue", "pages"):
                continue
            if k == "n_authors":
                if got[k] and ours[k] != got[k]:
                    row["mismatch"][k] = (ours[k], got[k])
                continue
            if k == "title" and got[k] and _norm(ours[k]) != _norm(got[k]):
                # 책은 표 쪽이 총서·판 표기를 덧붙이므로 Crossref 제목이 접두이면 통과
                if not _norm(ours[k]).startswith(_norm(got[k])):
                    row["mismatch"][k] = (ours[k], got[k])
                continue
            if k == "journal" and got[k] and _norm(ours[k]) != _norm(got[k]):
                # 저널명은 축약 표기 차이가 흔하므로 포함관계면 통과시킨다
                if _norm(got[k]) not in _norm(ours[k]) and _norm(ours[k]) not in _norm(got[k]):
                    row["mismatch"][k] = (ours[k], got[k])
                continue
            if _norm(ours[k]) != _norm(got[k]):
                row["mismatch"][k] = (ours[k], got[k])
        if row["mismatch"]:
            row["status"] = "mismatch"
        out.append(row)
        time.sleep(delay)                              # Crossref 예절
    return out


def by_num(num: int) -> Ref:
    for r in REFERENCES:
        if r.num == num:
            return r
    raise KeyError(num)


def rows() -> list[tuple]:
    return [tuple(getattr(r, f) for f in CSV_HEADER) for r in REFERENCES]


def write_csv(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(CSV_HEADER)
        w.writerows(rows())
    return path


def _authors_md(authors: str) -> str:
    return ", ".join(a.strip() for a in authors.split(";") if a.strip())


def format_ref(r: Ref) -> str:
    """Elsevier 번호식 한 줄. 확정되지 않은 항목에는 `[verify]`가 다시 붙는다."""
    loc = r.volume + (f" ({r.issue})" if r.issue else "") if r.volume else ""
    if r.year:
        loc = (loc + " " if loc else "") + f"({r.year})"
    if r.pages:
        loc = (loc + " " if loc else "") + r.pages
    s = f"{r.num}. {_authors_md(r.authors)}, {r.title}, {r.journal}"
    s += f" {loc}." if loc else "."
    if r.doi:
        s += f" doi:{r.doi}."
    if not r.verified:
        s += f" **[verify — {r.status}]**"
    return s


def markdown_block(refs: Sequence[Ref] | None = None) -> str:
    return "\n\n".join(format_ref(r) for r in (refs or REFERENCES))
