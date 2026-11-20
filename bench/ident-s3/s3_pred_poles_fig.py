"""그림(40번 §6, 42번 채택안): 상태별 예측(극점별) — (a) 테이프 T·M, (b) B 창 L1a·L1·L6·R(누적), (c) 단계 증분(L6 − L1, R − L6, C1·C2 − R).
판정 극점만. 회색 띠 = 문턱(테이프 한 점 ±0.06 %, 그 밖 ±0.15 %). 빗금 = 그 상태에서 판정하지 않음(성분 방향 모름·B의 3748).
사용: python s3_pred_poles_fig.py → groove-test/s3-pred-poles.png"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import groove_lib as gl  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
T = pd.read_csv(os.path.join(HERE, "..", "fe-nominal", "s3_pred_poles.csv"))
T["mode"] = T["mode"].astype(str)
MODES = [m for m in gl.MODES if m.judge and m.fem]
KIND = {"disk": "disk", "panel": "panel", "mixed": "mixed", "back": "back"}


def val(st, n, rel=None):
    q = T[(T.state == st) & (T["mode"] == n)].iloc[0]
    v = float(q.pred_pct)
    if rel:
        b = float(T[(T.state == rel) & (T["mode"] == n)].pred_pct.iloc[0])
        v = 100 * ((1 + v / 100) / (1 + b / 100) - 1)
    return v, bool(q.member_known) and bool(q.judge_state)


def panel(ax, groups, thr, title, ylabel, loc="lower left"):
    x = np.arange(len(MODES))
    w = 0.8 / len(groups)
    ax.axhspan(-thr, thr, color="0.9", zorder=0)
    for i, (lab, st, rel, col) in enumerate(groups):
        for j, m in enumerate(MODES):
            v, ok = val(st, m.name, rel)
            ax.bar(x[j] + (i - (len(groups) - 1) / 2) * w, v, width=w, color=col if ok else "white", edgecolor=col,
                   hatch=None if ok else "////", lw=0.8, label=lab if j == 0 else None)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m.name}\n{m.fem}\n{KIND.get(m.kind, m.kind)}" for m in MODES], fontsize=6.8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9, loc="left")
    ax.legend(fontsize=7.5, ncol=len(groups), loc=loc)


fig, ax = plt.subplots(3, 1, figsize=(13, 11.5))
panel(ax[0], [("T-a 30 mg (vane 2)", "T-a", None, "#8c510a"), ("T-b 30 mg (vane 3 - 14°)", "T-b", None, "#d8b365"),
              ("T-c 10 mg (mid 3-4)", "T-c", None, "#5ab4ac"), ("M75", "M75", None, "#80cdc1"), ("M150", "M150", None, "#01665e")],
      0.06, "(a) tape before cutting (point-mass first order): one-point T moves one h2 member only; grey = +/-0.06 % (same hanging)",
      "change vs A1 [%]", loc="lower right")
panel(ax[1], [("L1a 0.03 mm (window 3-4)", "L1a", None, "#fdae6b"), ("L1 0.06 mm", "L1", None, "#e6550d"),
              ("L6 (+5 mid-bay windows)", "L6", None, "#a63603"), ("R (+6 over-vane = B ring)", "R", None, "#3182bd")],
      0.15, "(b) B windows, cumulative from healthy (re-solved); hatched = not judged in that state (member unknown / 3748 at B)",
      "change vs healthy [%]")
panel(ax[2], [("L6 - L1", "L6", "L1", "#a63603"), ("R - L6", "R", "L6", "#3182bd"), ("C1 - R", "C1", "R", "#92c5de"),
              ("C2 - R", "C2", "R", "#2166ac")],
      0.15, "(c) step increments: mid-bay windows raise, over-vane windows lower, inner ring lowers (disc-type most)",
      "increment [%]", loc="upper right")
fig.suptitle("G3 FEM 32x16 predictions per measured pole (specimen 2, 10-30 frequencies) — 40/42", fontsize=10)
fig.tight_layout()
out = os.path.join(HERE, "..", "groove-test", "s3-pred-poles.png")
fig.savefig(out, dpi=125)
print(out)
