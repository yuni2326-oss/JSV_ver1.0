"""그림: FE 대비 측정 차와 '빠진 질량 4.54 g' 배치 가설별 예측, 모드별 운동에너지 분율."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/tmp/claude-1000/-home-super-project/efb54ce3-bcf9-473d-9dc8-eb835c653517/scratchpad/impact1022/out/s2/"
ft = pd.read_csv(OUT + "s2c_mass_fe_fractions.csv").set_index("label")
REG = ["front", "back_main", "hub", "vanes", "collar"]
MASS = {"front": 10.361, "back_main": 16.301, "hub": 2.696, "vanes": 6.139, "collar": 1.281}
DM = 41.3186 - 36.777
meas = [("h2:m2n0", 2138.04), (("h3:m3n0", "h0:m0n1"), (3163.65 + 3249.89) / 2), ("h1:m1n1", 3458.89),
        ("h3:m3n1", 3747.87), ("h0:m6n1", 3804.07), ("h1:m1n1#2", 4225.49), ("h2:m4n1", 4966.99),
        ("h0:m6n0", 5617.22), ("h1:m5n1", 5744.69), ("h2:m2n1", 6438.86), ("h3:m3n1#2", 8321.24)]
rows = []
for lab, fm in meas:
    labs = lab if isinstance(lab, tuple) else (lab,)
    T = ft.loc[list(labs), [f"T_{k}" for k in REG]].mean().to_numpy()
    f0 = ft.loc[list(labs), "f_FE"].mean()
    name = "h3:m3n0+h0:m0n1\n(mixed pair, mean)" if len(labs) == 2 else lab
    rows.append({"name": name, "f_meas": fm, "f_FE": f0, "err": (fm / f0 - 1) * 100, **dict(zip(REG, T))})
d = pd.DataFrame(rows).sort_values("front", ascending=False).reset_index(drop=True)

def pred(dist):
    """dist: {구역: 추가 g} → 모드별 Δf/f [%], 모양 고정 레일리 몫 f/f0 = 1/sqrt(1 + Σ ε T)."""
    s = sum((g / MASS[k]) * d[k] for k, g in dist.items())
    return (1 / np.sqrt(1 + s) - 1) * 100

hyp = {"4.54 g spread like the FE mass": {k: DM * MASS[k] / sum(MASS.values()) for k in REG},
       "4.54 g in back plate + hub": {"back_main": DM * MASS["back_main"] / 18.997, "hub": DM * MASS["hub"] / 18.997},
       "4.54 g in vanes": {"vanes": DM},
       "4.54 g in hub + inlet collar": {"hub": DM * MASS["hub"] / 3.977, "collar": DM * MASS["collar"] / 3.977}}
fig, (ax, bx) = plt.subplots(2, 1, figsize=(7.6, 7.0), sharex=True, gridspec_kw={"height_ratios": [1.5, 1]})
x = np.arange(len(d))
ax.bar(x, d.err, color="0.55", width=0.6, label="measured (10-30 poles) vs FE")
mk = ["_", "o", "s", "^"]
for (name, dist), m, c in zip(hyp.items(), mk, ["k", "C0", "C1", "C2"]):
    p = pred(dist)
    if m == "_":
        ax.plot(x, p, ls="--", color=c, lw=1, label=name)
    else:
        ax.plot(x, p, m, color=c, ms=5, mfc="none", mew=1.3, label=name)
ax.axhline(0, color="k", lw=0.6)
ax.set_ylabel("f / f_FE - 1  [%]")
ax.set_ylim(-16, 2)
ax.legend(fontsize=7, loc="lower left", ncol=2)
ax.set_title("FE 36.78 g vs specimen 2 41.32 g (+4.54 g).  Largest extra mass the measured f allow if FE errors only lower f:\n"
             "4.1 g, and only if packed into the back plate at r = 15-25 mm (+53-79 % density there)", fontsize=7.2, loc="left")
ax.text(0.01, 0.97, "(a)", transform=ax.transAxes, va="top", fontweight="bold")
bot = np.zeros(len(d))
cols = {"front": "C0", "back_main": "C3", "hub": "C4", "vanes": "C1", "collar": "C2"}
for k in REG:
    bx.bar(x, d[k], bottom=bot, color=cols[k], width=0.6, label=k.replace("_main", " plate").replace("front", "front plate").replace("collar", "inlet collar"))
    bot += d[k]
bx.set_ylabel("FE kinetic-energy share")
bx.set_ylim(0, 1.0)
bx.legend(fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.42))
bx.set_xticks(x)
bx.set_xticklabels([f"{n}\n{fm:.0f} Hz" for n, fm in zip(d.name, d.f_meas)], fontsize=6.3, rotation=40, ha="right")
bx.text(0.01, 0.97, "(b)", transform=bx.transAxes, va="top", fontweight="bold", color="w")
fig.tight_layout()
fig.savefig(OUT + "s2c_mass.png", dpi=170)
print(d[["name", "err"] + REG].round(3).to_string())
for name, dist in hyp.items():
    print(name, np.round(pred(dist).to_numpy(), 2))
