"""그림: 측정 대 FE — 옛 FE(1번 시편 치수), 2번 시편 치수 FE, 그리고 빠진 질량을 FE 분포대로 더한 경우."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/tmp/claude-1000/-home-super-project/efb54ce3-bcf9-473d-9dc8-eb835c653517/scratchpad/impact1022/out/s2/"
P = pd.read_csv(OUT + "s2f_mass2_pairs.csv")
P = P[P.pair != "h2:m4n0"].sort_values("T_back_main").reset_index(drop=True)
uni = 1 / np.sqrt(1 + 4.787 / 36.531) - 1
x = np.arange(len(P))
fig, (ax, bx) = plt.subplots(2, 1, figsize=(9, 6.6), sharex=True, gridspec_kw={"height_ratios": [1.6, 1]})
ax.plot(x, P.err_old, "s", mfc="none", color="0.5", label="FE with specimen-1 dims (front 0.375, vane 1.2 mm; 36.78 g)")
ax.plot(x, P.err_new, "o", color="C3", label="FE with specimen-2 dims (front 0.40, back 0.58, vane 1.0 mm; 36.53 g)")
res = 100 * ((1 + P.err_new / 100) / (1 + uni) - 1)
ax.plot(x, res, "^", color="C0", label="same + missing 4.79 g spread like the FE mass (-5.97 % on every mode)")
ax.axhline(0, color="k", lw=0.6)
ax.set_ylabel("measured / FE - 1  [%]")
ax.legend(fontsize=6.8, loc="upper left")
ax.set_ylim(-8, 11.5)
ax.set_title("specimen 2 (41.32 g): 10-30 poles vs free-free FE (continuous vane joints)", fontsize=9, loc="left")
bot = np.zeros(len(P))
for k, c, n in [("front", "C0", "front plate"), ("back_main", "C3", "back plate"), ("hub", "C4", "hub"), ("vanes", "C1", "vanes"), ("collar", "C2", "inlet collar")]:
    bx.bar(x, P[f"T_{k}"], bottom=bot, color=c, width=0.6, label=n); bot += P[f"T_{k}"]
bx.set_ylabel("FE kinetic share\n(specimen-2 FE)")
bx.set_xticks(x)
bx.set_xticklabels([f"{p.replace('h3:m3n0+h0:m0n1', 'm3n0+m0n1')}\n{f:.0f} Hz" for p, f in zip(P.pair, P.f_meas)], fontsize=6.8, rotation=35, ha="right")
bx.legend(fontsize=6.8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.55))
fig.tight_layout()
fig.savefig(OUT + "s2f_fe.png", dpi=170)
print(P[["pair", "err_old", "err_new", "T_back_main"]].round(3).to_string())
print("residual after uniform:", np.round(res.to_numpy(), 2))
