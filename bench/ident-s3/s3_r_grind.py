"""3번 시편 R(11-19 S18, 메모 s3_R) — 깃 4 위를 17.9 mg(39941.4 → 39923.5 mg) 깎은 경우의 G3 FEM(54번).

깎은 모양은 아직 모른다(사진 대기). s3_edge_grind의 모양 함수를 깃 4 선 가운데로 옮겨 몇 가지로 푼다.
  RB20  끝 띠 ξ ≥ 0.9(테두리에서 2.2 mm), 깃 4 선 ±10°,   RB10  같은 띠 ±5°(더 깊게)
  RB20p5·RB20m5  RB20의 가운데를 깃 3 쪽(+)·깃 5 쪽(−)으로 5° 옮김
  RV25  테두리 경사면(테두리에서 가장 깊고 ξ0.8에서 0), ±12.5°
  RW20  창 ξ ≥ 0.8(4.3 mm), ±10°,   RW30  창 ±15°(51번 R30과 같은 모양)
  (55번) 깃 사이로 번진 끝 띠: RB50 ±25°, RB60 ±30°(양쪽 칸 가운데까지), RB30p15·RB30m15 깃 선에서 한쪽 칸 가운데까지
  (55번 사진) 깃 4 선 끝에서 깃 5 쪽(−)으로만: RS25b 경사면(ξ0.8에서 0, 테두리 최대) 0 ~ −25°, RS25f 같은 자리 고르게(ξ ≥ 0.8),
       RS25w 경사면이 더 안쪽까지(ξ0.6에서 0), RS20b 경사면 0 ~ −20°
출력: fe-nominal/s3_r_grind/<경우>.csv, summary.csv, key.csv
사용(PYTHONPATH=저장소, env -u PAPER3_GEOM, OMP_NUM_THREADS=2):
      python s3_r_grind.py --cases RB20,RB10 --cache <pkl>    경우마다 약 4분
      python s3_r_grind.py --summary"""
import argparse
from pathlib import Path

import s3_edge_grind as eg

MG = 17.9
eg.SPEC = {"RB20": ("face", 0, "trap", 20, 0.9, "flat", MG), "RB10": ("face", 0, "trap", 10, 0.9, "flat", MG),
           "RB20p5": ("face", 5, "trap", 20, 0.9, "flat", MG), "RB20m5": ("face", -5, "trap", 20, 0.9, "flat", MG),
           "RV25": ("face", 0, "trap", 25, 0.8, "ramp", MG), "RW20": ("face", 0, "trap", 20, 0.8, "flat", MG),
           "RW30": ("face", 0, "trap", 30, 0.8, "flat", MG),
           # 55번: 깃 사이로 넓게 번진 경우(12자리에서 오른 모드가 있어서)
           "RB50": ("face", 0, "trap", 50, 0.9, "flat", MG), "RB60": ("face", 0, "trap", 60, 0.9, "flat", MG),
           "RB30p15": ("face", 15, "trap", 30, 0.9, "flat", MG), "RB30m15": ("face", -15, "trap", 30, 0.9, "flat", MG),
           # 55번 사진: 깃 4 선 끝에서 깃 5 쪽으로만 테두리를 따라 약 15 mm(25°), 테두리 모서리가 가장 깊음
           "RS25b": ("face", -12.5, "trap", 25, 0.8, "ramp", MG), "RS25f": ("face", -12.5, "trap", 25, 0.8, "flat", MG),
           "RS25w": ("face", -12.5, "trap", 25, 0.6, "ramp", MG), "RS20b": ("face", -10, "trap", 20, 0.8, "ramp", MG)}
eg.CASES = list(eg.SPEC)
eg.OUTD = Path(__file__).resolve().parents[1] / "fe-nominal" / "s3_r_grind"
eg.GROUPS = [("(a) on the vane-4 line, 17.9 mg: shape varied", ["RB20", "RB10", "RV25", "RW20", "RW30"]),
             ("(b) centre moved 5° off the vane line", ["RB20", "RB20p5", "RB20m5"]),
             ("(c) -", [])]

def cli():
    """--cases·--summary 공통 진입점(eg.SPEC·eg.OUTD를 바꿔 다른 시편 단계에도 쓴다 — s3_l_grind)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--cache", required=False, default="")
    a = ap.parse_args()
    if a.cases:
        eg.run_cases([c for c in a.cases.split(",") if c], a.cache)
    if a.summary:
        S = eg.load_summary()
        S.to_csv(eg.OUTD / "summary.csv", index=False, float_format="%.5g")
        KEY = ["h2:m2n0", "h3:m3n0", "h0:m0n1", "h1:m1n1", "h2:m4n0", "h3:m3n1", "h0:m6n1", "h1:m1n1#2", "h2:m4n0#2",
               "h0:m6n0", "h1:m5n0", "h2:m2n1", "h1:m7n0", "h3:m3n2", "h3:m3n1#2"]
        print("[FEM %, 짝은 '작은/큰' 성분과 합] " + " ".join(f"{eg.LABS[l]:>14s}" for l in KEY if l in eg.LABS))
        rows = []
        for name in eg.CASES:
            x = S[S.case == name].set_index("label")
            if x.empty:
                continue
            cells, row = [], dict(case=name, size_mm=x.iloc[0].size_mm)
            for l in KEY:
                if l not in x.index:
                    cells.append("--")
                    continue
                r = x.loc[l]
                star = "*" if r.s1 < 0.6 else ""
                if r.deg == 2:
                    a_, b_ = sorted([r.df1, r.df2], key=abs)
                    cells.append(f"{a_:+.2f}/{b_:+.2f}={a_ + b_:+.2f}{star}")
                    row[eg.LABS[l]] = a_ + b_
                else:
                    cells.append(f"{r.df1:+.2f}{star}")
                    row[eg.LABS[l]] = r.df1
            rows.append(row)
            print(f"  {name:7s} {x.iloc[0].size_mm:.3f} " + " ".join(f"{c:>14s}" for c in cells))
        import pandas as pd
        pd.DataFrame(rows).to_csv(eg.OUTD / "key.csv", index=False, float_format="%.4g")


if __name__ == "__main__":
    cli()
