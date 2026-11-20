"""그림(40번): 상태별 예측 주파수 변화(G3 FEM 32×16 재해석) — 판정 모드만, 회색 띠 = 문턱 ±0.15 %.
사용: python s3_pred_fig.py → groove-test/s3-pred.png"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import groove_lib as gl  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
T = pd.read_csv(os.path.join(HERE, "..", "fe-nominal", "s3_pred_table.csv"))
T = T[T.judge.astype(bool)].reset_index(drop=True)
STATES = ["B1", "B2", "B3", "C1", "C2"]
COL = {"B1": "#f4a582", "B2": "#d6604d", "B3": "#b2182b", "C1": "#92c5de", "C2": "#2166ac"}
KIND = {"disk": "disk-type", "panel": "front panel", "mixed": "mixed", "back": "back shroud"}
fig, (ax, bx) = plt.subplots(2, 1, figsize=(12, 8.2), gridspec_kw={"height_ratios": [1.25, 1]})
x = np.arange(len(T)); w = 0.16
ax.axhspan(-0.15, 0.15, color="0.9", zorder=0)
for i, st in enumerate(STATES):
    ax.bar(x + (i - 2) * w, T[f"{st}_pct"], width=w, color=COL[st], label=st)
ax.axhline(0, color="k", lw=0.6)
ax.set_xticks(x)
ax.set_xticklabels([f"{m}\n{f}\n{KIND.get(k, k)}" for m, f, k in zip(T["mode"], T.fem, T.kind)], fontsize=7)
ax.set_ylabel("change vs healthy [%]")
ax.legend(ncol=5, fontsize=8, loc="lower left",
          title="B = outer groove 0.02/0.04/0.06 mm; C = B3 + inner groove 0.02/0.04 mm", title_fontsize=7.5)
ax.set_title("(a) G3 FEM (32x16, re-solved), cumulative from the healthy state; grey = +/-0.15 % threshold",
             fontsize=9, loc="left")
bx.axhspan(-0.15, 0.15, color="0.9", zorder=0)
for i, st in enumerate(["C1", "C2"]):
    inc = 100 * ((1 + T[f"{st}_pct"] / 100) / (1 + T["B3_pct"] / 100) - 1)
    bx.bar(x + (i - 0.5) * 0.3, inc, width=0.3, color=COL[st], label=f"{st} vs B3")
bx.axhline(0, color="k", lw=0.6)
bx.set_xticks(x)
bx.set_xticklabels([f"{m}\n{f}" for m, f in zip(T["mode"], T.fem)], fontsize=7)
bx.set_ylabel("change vs B3 [%]")
bx.legend(fontsize=8, loc="lower left")
bx.set_title("(b) inner groove alone: C1, C2 relative to B3 (every judged pole falls)", fontsize=9, loc="left")
fig.tight_layout()
out = os.path.join(HERE, "..", "groove-test", "s3-pred.png")
fig.savefig(out, dpi=130)
print(out)
