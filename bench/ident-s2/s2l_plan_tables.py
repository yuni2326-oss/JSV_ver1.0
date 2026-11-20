"""다음 실험 계획(35번) 표 — G3 자유-자유 FEM(32×16) 홈 예측에서 측정된 모드별 예측 변화를 뽑는다.
측정 극점 ↔ FEM 짝은 32번·34번(간격 3.00 mm) 그대로."""
import numpy as np
import pandas as pd

D = "/home/super/project/docs/_generated_g3/data/paper3/"
OUT = "/tmp/claude-1000/-home-super-project/efb54ce3-bcf9-473d-9dc8-eb835c653517/scratchpad/impact1022/out/s2/"
g = pd.read_csv(D + "rev1_n6_grooves_ff32x16h6w62c50.csv")
# 측정 극점(10-30) → FEM 이름(32×16 표준 두께 FEM), 원판형/패널형(FEM 앞판 몫)
PAIRS = [(2138.0, "h2:m2n0", "h2 짝 2134/2142"), (3163.7, "h3:m3n0", "3164"), (3249.9, "h0:m0n1", "3250"),
         (3458.9, "h1:m1n1", "3459"), (3639.0, "h2:m4n0", "3639"), (3747.9, "h3:m3n1", "3748"),
         (3804.1, "h0:m6n1", "3804"), (4225.5, "h1:m1n1#2", "4225"), (4967.0, "h2:m4n0#2", "4967"),
         (5617.2, "h0:m6n0", "5617"), (5744.7, "h1:m5n0", "5745"), (6438.9, "h2:m2n1", "6439"),
         (8321.2, "h3:m3n1#2", "8321")]
HUB = {"h1:m1n1#2", "h0:m6n1"}          # 매달기(보어 클립)에 민감했던 모드 — 판정에서 뺀다


def val(xi, lab, d):
    s = g[np.isclose(g.xi, xi) & (g.label == lab) & np.isclose(g.depth_mm, d)]
    if not len(s):
        return np.nan
    r = s.iloc[0]
    return r.df_pct_res if pd.notna(r.df_pct_res) else r.df_pct_lin


rows = []
for f, lab, name in PAIRS:
    row = {"measured": name, "FEM": lab}
    for xi in (0.3, 0.85, 0.9, 0.95):
        for d in (0.02, 0.04, 0.06, 0.09):
            row[f"x{xi}_d{d}"] = val(xi, lab, d)
    rows.append(row)
T = pd.DataFrame(rows)
T.to_csv(OUT + "s2l_plan_pred.csv", index=False)
pd.set_option("display.width", 250)
for xi in (0.3, 0.85, 0.9, 0.95):
    cols = [f"x{xi}_d{d}" for d in (0.02, 0.04, 0.06, 0.09)]
    print(f"== ξ {xi}")
    print(T[["measured", "FEM"] + cols].round(3).to_string(index=False))
# 질량: 홈 넓이 × 깊이 × ρ
a, b, rho = 14.89, 36.465, 7.93e-3
w = 0.2 * (b - a)
for xi in (0.3, 0.85, 0.9, 0.95):
    rc = a + xi * (b - a)
    lo, hi = rc - w / 2, min(rc + w / 2, b)
    area = np.pi * (hi ** 2 - lo ** 2)
    print(f"ξ {xi}: r {lo:.2f}–{hi:.2f} mm, area {area:.0f} mm², 0.01 mm = {area * 0.01 * rho * 1e3:.1f} mg")
