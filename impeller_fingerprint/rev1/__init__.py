"""JSVD2602583 1차 수정(R1) 계산 — 심사 대응 N1–N6.

정본 모듈(`forward`·`crack_shear`·`rail3d` …)은 **읽기만** 한다. 수정 계산이 정본 동작을 바꿔야
하면 여기서 고치지 말고 해당 모듈에 테스트와 함께 별도 커밋으로 반영한다(제출본 재현성 보존).

산출 경로는 `cli.py`와 같은 규약을 쓴다 — `$PAPER3_OUT/data/paper3/rev1_*.csv`,
`$PAPER3_OUT/figures/paper3/rev1_*.png`. 워크트리에서 실행할 때는 `PAPER3_OUT`이 본 리포의
`docs/_generated`를 가리켜야 한다(설계서 F153: 산출 후 워크트리로 동기화).

작업 기록: `docs/paper3-jsv/revision-r1/`.
"""
from __future__ import annotations

import os
from pathlib import Path

OUT_ROOT = Path(os.environ.get(
    "PAPER3_OUT", Path(__file__).resolve().parents[2] / "docs" / "_generated"))
DATA = OUT_ROOT / "data" / "paper3"
FIGS = OUT_ROOT / "figures" / "paper3"
REV_DOCS = Path(__file__).resolve().parents[2] / "docs" / "paper3-jsv" / "revision-r1"


def out_csv(name: str) -> Path:
    """`rev1_<name>.csv` 경로(디렉터리 생성 포함)."""
    DATA.mkdir(parents=True, exist_ok=True)
    stem = name if name.startswith("rev1_") else f"rev1_{name}"
    return DATA / (stem if stem.endswith(".csv") else f"{stem}.csv")


def out_fig(name: str) -> Path:
    """`rev1_<name>.png` 경로(디렉터리 생성 포함)."""
    FIGS.mkdir(parents=True, exist_ok=True)
    stem = name if name.startswith("rev1_") else f"rev1_{name}"
    return FIGS / (stem if stem.endswith(".png") else f"{stem}.png")
