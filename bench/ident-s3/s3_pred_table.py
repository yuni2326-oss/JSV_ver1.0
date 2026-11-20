"""상태별 예측표(40번 §5) — s3_pred_states.csv(재해석)를 측정 극점(10-30 기준 주파수)에 붙이고,
상태마다 예측 주파수가 0.3 % 안으로 붙는 극점 쌍(교차 위험)과 35번 표(증분 합)와의 차를 낸다.
사용: python s3_pred_table.py → 화면(마크다운)과 fe-nominal/s3_pred_table.csv"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import groove_lib as gl  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FE = os.path.join(HERE, "..", "fe-nominal")
P = pd.read_csv(os.path.join(FE, "s3_pred_states.csv"))
OLD = pd.read_csv(os.path.join(FE, "s2l_plan_pred.csv")).set_index("FEM")
STATES = ["B1", "B2", "B3", "C1", "C2"]
OLD_COL = {"B1": "x0.9_d0.02", "B2": "x0.9_d0.04", "B3": "x0.9_d0.06"}
KO = {"disk": "원판형", "panel": "패널", "mixed": "섞임", "back": "뒤판", "hub": "허브", "cand": "후보", "none": "—"}

rows = []
for m in gl.MODES:
    r = {"mode": m.name, "fem": m.fem or "", "kind": m.kind, "judge": m.judge, "f_base": m.f}
    for st in STATES:
        s = P[(P.state == st) & (P.label == m.fem)] if m.fem else P.iloc[0:0]
        if len(s):
            d = float(s.df_pct_res.iloc[0]) if np.isfinite(s.df_pct_res.iloc[0]) else float(s.df_pct_lin.iloc[0])
            r[f"{st}_pct"] = d; r[f"{st}_lin"] = float(s.df_pct_lin.iloc[0]); r[f"{st}_mac"] = float(s.mac.iloc[0])
        else:
            r[f"{st}_pct"] = np.nan
        r[f"{st}_Hz"] = m.f * (1 + (r[f"{st}_pct"] if np.isfinite(r[f"{st}_pct"]) else 0.0) / 100)
    if m.fem in OLD.index:
        for st, c in OLD_COL.items():
            r[f"{st}_35"] = float(OLD.loc[m.fem, c])
        for st, d in (("C1", "x0.3_d0.02"), ("C2", "x0.3_d0.04")):
            r[f"{st}_35"] = float(OLD.loc[m.fem, "x0.9_d0.06"]) + float(OLD.loc[m.fem, d])
    rows.append(r)
T = pd.DataFrame(rows)
T.to_csv(os.path.join(FE, "s3_pred_table.csv"), index=False)

mass = P.groupby("state").removed_mg.first()
print("FEM이 깎은 질량[mg]: " + ", ".join(f"{s} {mass[s]:.1f}" for s in STATES))
print()
print("| 극점 [Hz] | FEM | 종류 | " + " | ".join(STATES) + " |")
print("|---|---|---|" + "---|" * len(STATES))
for r in rows:
    cells = []
    for st in STATES:
        d = r[f"{st}_pct"]
        cells.append("—" if not np.isfinite(d) else f"{d:+.2f} % ({r[f'{st}_Hz']:.0f})")
    lab = r["mode"] if r["judge"] else f"({r['mode']})"
    print(f"| {lab} | {r['fem']} | {KO[r['kind']]} | " + " | ".join(cells) + " |")
print()
print("35번(증분 합) 대비 차 [%p], 판정 모드 최대:")
for st in STATES:
    d = [(abs(r[f"{st}_pct"] - r[f"{st}_35"]), r["mode"]) for r in rows if r["judge"] and np.isfinite(r.get(f"{st}_35", np.nan))
         and np.isfinite(r[f"{st}_pct"])]
    if d:
        mx = max(d)
        print(f"  {st}: 최대 {mx[0]:.3f} %p ({mx[1]})")
print()
print("교차 위험(예측 주파수 0.3 % 안, h2 짝 제외):")
for st in STATES:
    fs = sorted((r[f"{st}_Hz"], r["mode"]) for r in rows)
    w = [f"{a[1]}·{b[1]} ({a[0]:.0f}/{b[0]:.0f} Hz)" for a, b in zip(fs, fs[1:])
         if b[0] / a[0] - 1 < 0.003 and {a[1], b[1]} != set(gl.PAIR)]
    print(f"  {st}: " + ("; ".join(w) if w else "없음"))
