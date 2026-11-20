"""상태별 예측을 측정 극점마다 한 표로(42번 채택안, 40번 §5) — 분석 도구(groove_state·groove_compare)의 기본 예측 표.

출처: 고리 B1–B3·C1·C2 = s3_pred_states.csv(재해석), 창 L1a·L1·L6과 테이프 T-a·T-b·T-c·M75·M150(점 조각 계획)·
M120·M240(11-12부터 실제로 붙인 띠 20 mg) = s3_pred_windows.csv (h2 짝은 측정 성분별), R = B3(깃 사이 여섯 + 깃 위 여섯 = B 고리 0.06 mm).
열: state, mode(측정 극점 이름), fem, kind, judge_mode(극점 수준 판정 여부), judge_state(이 상태에서 판정 여부),
    pred_pct(재해석이 있으면 재해석, 없으면 1차), lin_pct, res_pct, half_split_pct, member_known, removed_mg(FEM, 붙인 질량은 음수), chain
- B 상태(B1–B3·L1a·L1·L6·R)의 3748은 judge_state = 거짓(41·42번: 빠진 질량 위치에 따라 부호가 바뀜).
- 국부 상태(L1a·L1·T-a·T-b·T-c)에서 h2 밖의 짝은 성분 방향을 모르므로 member_known = 거짓, 쌍 평균과 반 갈라짐만 준다.
  재해석이 짝을 못 따라간 경우(L1의 h2:m4n0 — 창 하나가 패널 짝을 섞는다)는 1차 값과 42번 s3_local의 1차 반 갈라짐을 쓴다.
사용: python s3_pred_poles.py → fe-nominal/s3_pred_poles.csv와 화면 표(마크다운)"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import groove_lib as gl  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FE = os.path.join(HERE, "..", "fe-nominal")
RING = pd.read_csv(os.path.join(FE, "s3_pred_states.csv"))
WIN = pd.read_csv(os.path.join(FE, "s3_pred_windows.csv"))
# 창 하나의 1차 반 갈라짐(42번 s3_local: B 30° 창, 깃 기준 30°, 0.02 mm). C6 대칭이라 섹터 3의 창과 갈라짐 크기가 같다.
# 재해석에서 짝을 못 따라간 경우(패널 짝이 섞임) 이 값을 쓴다. Δf/f 반 갈라짐[%] = 25·(η 고윳값 차) × 깊이/0.02 mm.
_LB = pd.read_csv(os.path.join(FE, "s3_local", "local_B.csv"))
_LB = _LB[(_LB.width_deg == 30.0) & (_LB.rel_deg == 30.0) & (_LB.deg == 2)].set_index("label")
HALF_LIN = {st: {lab: 25.0 * float(_LB.loc[lab, "split"]) * k for lab in _LB.index} for st, k in (("L1a", 1.5), ("L1", 3.0))}
CHAIN = {"B1": "ring", "B2": "ring", "B3": "ring", "L1a": "window", "L1": "window", "L6": "window", "R": "window",
         "C1": "c", "C2": "c", "T-a": "tape", "T-b": "tape", "T-c": "tape", "M75": "tape", "M150": "tape",
         "M120": "tape", "M240": "tape"}
B_STATES = {"B1", "B2", "B3", "L1a", "L1", "L6", "R"}
LOCAL = {"L1a", "L1", "T-a", "T-b", "T-c"}
best = lambda res, lin: float(res) if np.isfinite(res) else float(lin)

rows = []
for st in CHAIN:
    src_state = "B3" if st == "R" else st
    for m in gl.MODES:
        r = dict(state=st, mode=m.name, fem=m.fem or "", kind=m.kind, judge_mode=m.judge,
                 judge_state=not (st in B_STATES and m.name == "3748"), chain=CHAIN[st])
        if not m.fem:
            rows.append(dict(r, pred_pct=np.nan, lin_pct=np.nan, res_pct=np.nan, half_split_pct=0.0, member_known=True,
                             removed_mg=np.nan))
            continue
        if src_state in set(RING.state):
            q = RING[(RING.state == src_state) & (RING.label == m.fem)]
            if q.empty:
                continue
            q = q.iloc[0]
            rows.append(dict(r, pred_pct=best(q.df_pct_res, q.df_pct_lin), lin_pct=float(q.df_pct_lin),
                             res_pct=float(q.df_pct_res), half_split_pct=0.0, member_known=True, removed_mg=float(q.removed_mg)))
            continue
        g = WIN[(WIN.state == src_state) & (WIN.label == m.fem)]
        if g.empty:
            continue
        q = g[g.member == m.name].iloc[0] if m.fem == "h2:m2n0" else g.iloc[0]
        hs = float(q.half_split_pct) if np.isfinite(q.half_split_pct) else HALF_LIN.get(st, {}).get(m.fem, 0.0)
        known = bool(q.deg == 1 or m.fem == "h2:m2n0" or st not in LOCAL or hs < 0.01)
        rows.append(dict(r, pred_pct=best(q.df_pct_res, q.df_pct_lin), lin_pct=float(q.df_pct_lin),
                         res_pct=float(q.df_pct_res), half_split_pct=hs, member_known=known, removed_mg=float(q.removed_mg)))
T = pd.DataFrame(rows)
T.to_csv(os.path.join(FE, "s3_pred_poles.csv"), index=False)

KO = {"disk": "원판형", "panel": "패널", "mixed": "섞임", "back": "뒤판", "hub": "허브", "cand": "후보", "none": "—"}


def table(states, rel_to=None, title=""):
    print(f"\n{title}")
    print("| 측정 극점 [Hz] | FEM 짝 | 종류 | " + " | ".join(states) + " |")
    print("|---|---|---|" + "---|" * len(states))
    for m in gl.MODES:
        if not m.fem:
            continue
        cells = []
        for st in states:
            q = T[(T.state == st) & (T["mode"] == m.name)]
            if q.empty or not np.isfinite(q.pred_pct.iloc[0]):
                cells.append("—")
                continue
            v = float(q.pred_pct.iloc[0])
            if rel_to:
                b = float(T[(T.state == rel_to) & (T["mode"] == m.name)].pred_pct.iloc[0])
                v = 100 * ((1 + v / 100) / (1 + b / 100) - 1)
            s = f"{v:+.2f}"
            if not q.member_known.iloc[0]:
                s += f" (±{q.half_split_pct.iloc[0]:.2f})"
            if not q.judge_state.iloc[0]:
                s = f"[{s}]"
            cells.append(s)
        lab = m.name if m.judge else f"({m.name})"
        print(f"| {lab} | {m.fem} | {KO[m.kind]} | " + " | ".join(cells) + " |")


mass = T.groupby("state").removed_mg.first()
print("FEM 질량[mg] (붙인 질량은 음수): " + ", ".join(f"{s} {mass[s]:.1f}" for s in CHAIN))
table(["T-a", "T-b", "T-c", "M75", "M150", "M120", "M240"], title="A 단계 테이프(건전 대비, 1차 질량점; M120·M240 = 띠 20 mg)")
table(["L1a", "L1", "L6", "R"], title="B 창(건전 대비 누적; L1a·L1·L6 재해석, R = B 고리 0.06 mm 재해석)")
table(["C1", "C2"], rel_to="R", title="C 고리(R 대비 증분)")
print("\n표기: (±x) = 짝 성분 방향을 모름(쌍 평균 ± 반 갈라짐), [ ] = 이 상태에서 판정하지 않음, ( ) 극점 = 허브·후보(판정 밖).")
d = T[np.isfinite(T.res_pct) & np.isfinite(T.lin_pct)]
d = d.assign(diff=(d.res_pct - d.lin_pct).abs())
print("\n재해석 − 1차 최대 [%p]:", ", ".join(f"{s} {g['diff'].max():.3f} ({g.loc[g['diff'].idxmax(), 'mode']})"
                                         for s, g in d.groupby("state", sort=False)))
print(f"→ {os.path.join(FE, 's3_pred_poles.csv')}")
