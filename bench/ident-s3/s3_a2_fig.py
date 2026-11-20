"""그림 a2.png(47번): A2(취급 대조, 11-16 S12)를 A1-6v(11-11 S9)와 맞댄다.
(a) 모드별 A2 − A1-6v [%]. 채운 점 = 잘 정해진 모드(두 상태 모두 묶음 적합·6곳 이상·위치 흩어짐 < 0.3 %), 빈 점 = 약한 모드.
    가로선 = 잘 정해진 모드의 중앙(공통 이동 c), 회색 띠 = c ± 0.10 %(40번 §5 '멈출 때' 1의 문턱), 점선 = 기온 차로 기대되는 이동.
(b) 질량 시험의 닫힘: 잘 정해진 모드의 A1-6v → M120 → M240 → A2 변화 [%](A1-6v 대비). 조각을 떼면 c 근처로 돌아온다.
(c) 평균 FRF 크기(두 상태에 다 있는 위치) 3.0–3.8 kHz: A1-6v, A2, 그리고 A2의 주파수 축을 (1 + c)로 나눈 것.
(d) 감쇠비(위치별 묶음 적합의 중앙): 10-28·10-30(s2j_poles.csv), A1-6v, M120, M240, A2.
(e) 힘에 따른 주파수 변화(힘 2배당 [%], 위치마다 평균을 빼고 타격별 주파수를 log2(힘)에 회귀): A1-6v·M120·M240·A2와
    10-30 값(groove_lib 기울기).
입력: groove-test/out/{A1-6v,M120,M240,A2}.json·_frf.npz(groove_state.py), fe-nominal/s2j_poles.csv.
출력: groove-test/a2.png, 화면 표."""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import groove_lib as gl  # noqa: E402
from collections import defaultdict  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
GT = os.path.join(HERE, "..", "groove-test")
OUT = os.path.join(GT, "out")
ST = ("A1-6v", "M120", "M240", "A2")
S = {s: json.load(open(os.path.join(OUT, f"{s}.json"))) for s in ST}
COL = {"disk": "C0", "panel": "C1", "mixed": "C4", "back": "C2", "hub": "C3", "cand": "0.55", "none": "0.55"}
DT_COEF = -0.019                                                  # %/°C(groove_compare와 같은 재료값 어림)


def comb(St, n):
    c = St["modes"].get(n, {}).get("comb", {})
    return (c.get("f", np.nan), c.get("se", np.nan), c.get("method"), len(c.get("locs") or []),
            c.get("loc_spread_pct", np.nan))


def good(n):
    a, b = comb(S["A1-6v"], n), comb(S["A2"], n)
    return all(x[2] == "pool" and x[3] >= 6 and x[4] < 0.3 for x in (a, b))


def temps(St):
    q = list(St["qc"].values())
    return np.nanmean([x["T0"] for x in q]), np.nanmean([x["T1"] for x in q])


names = [m.name for m in gl.MODES]
d = {}
for n in names:
    (f1, s1, *_), (f2, s2, *_) = comb(S["A1-6v"], n), comb(S["A2"], n)
    d[n] = (100 * (f2 / f1 - 1), 100 * np.hypot(s1 / f1, s2 / f2)) if np.isfinite(f1) and np.isfinite(f2) else (np.nan, np.nan)
G = [n for n in names if good(n) and np.isfinite(d[n][0])]
c = float(np.median([d[n][0] for n in G]))
res = np.array([d[n][0] - c for n in G])
dT = np.mean(temps(S["A2"])) - np.mean(temps(S["A1-6v"]))
h = lambda St: comb(St, "2142")[0] - comb(St, "2134")[0]

print(f"잘 정해진 모드 {len(G)}개: {', '.join(G)}")
print(f"공통 이동 c = {c:+.3f} %(중앙), 뺀 뒤 rms {np.sqrt(np.mean(res ** 2)):.3f} %, 최대 {np.abs(res).max():.3f} %; "
      f"기온 차 {dT:+.2f} °C → {DT_COEF * dT:+.3f} %")
print(f"h2 갈라짐 [Hz]: " + " → ".join(f"{s} {h(S[s]):.2f}" for s in ST))
print("모드 | 종류 | 잘 정해짐 | A2−A1 [%] | −c | M120 | M240")
for n in names:
    dm = [100 * (comb(S[s], n)[0] / comb(S["A1-6v"], n)[0] - 1) for s in ("M120", "M240")]
    print(f"{n:>5} | {gl.MODE[n].kind:5s} | {'예' if n in G else '  '} | {d[n][0]:+.3f} | {d[n][0] - c:+.3f} | "
          f"{dm[0]:+.3f} | {dm[1]:+.3f}")

fig = plt.figure(figsize=(13, 12.0))
ax = fig.add_axes([0.06, 0.72, 0.9, 0.25])
bx = fig.add_axes([0.06, 0.39, 0.40, 0.25])
cx = fig.add_axes([0.55, 0.39, 0.41, 0.25])
dx = fig.add_axes([0.06, 0.05, 0.40, 0.25])
ex = fig.add_axes([0.55, 0.05, 0.41, 0.25])

# (a)
x = np.arange(len(names))
ax.axhspan(c - 0.10, c + 0.10, color="0.9", zorder=0)
ax.axhline(c, color="k", lw=1.0)
ax.axhline(0, color="0.5", lw=0.6)
ax.axhline(DT_COEF * dT, color="r", lw=1.0, ls=":")
for i, n in enumerate(names):
    v, s = d[n]
    if not np.isfinite(v):
        continue
    col = COL[gl.MODE[n].kind]
    ax.errorbar(i, v, yerr=s, fmt="o", ms=7, color=col, mfc=col if n in G else "none", mew=1.4, capsize=2)
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=8, rotation=45)
ax.set_xlim(-0.6, len(names) - 0.4)
ax.set_ylim(-0.45, 0.6)
ax.set_ylabel("A2 − A1-6v [%]", fontsize=9)
ax.text(len(names) - 0.5, c - 0.13, f"common shift c = {c:+.2f} % (median of {len(G)} well-determined modes); "
        f"grey = c ± 0.10 %", ha="right", va="top", fontsize=8)
ax.text(len(names) - 0.5, DT_COEF * dT + 0.02, f"expected from air temperature ({dT:+.1f} °C)", ha="right",
        va="bottom", fontsize=8, color="r")
for k, lab in (("disk", "disk"), ("panel", "panel"), ("mixed", "mixed"), ("back", "back plate"), ("hub", "hub"),
               ("cand", "candidate / unassigned")):
    ax.plot([], [], "o", color=COL[k], label=lab)
ax.plot([], [], "o", mfc="none", mec="k", label="weak in either state")
ax.legend(fontsize=7.5, ncol=7, loc="upper left", frameon=False)
ax.set_title(f"(a) handling control A2 (11-16, pieces removed, mass unchanged) against A1-6v (11-11): "
             f"after removing c, rms {np.sqrt(np.mean(res ** 2)):.3f} %, max {np.abs(res).max():.3f} %",
             fontsize=9, loc="left")

# (b)
for n in G:
    y = [0.0] + [100 * (comb(S[s], n)[0] / comb(S["A1-6v"], n)[0] - 1) for s in ST[1:]]
    bx.plot(range(4), y, "-o", ms=4, lw=1.0, color=COL[gl.MODE[n].kind])
    bx.text(3.08, y[-1], "", fontsize=6)
bx.axhline(c, color="k", lw=0.8, ls="--")
bx.axhline(0, color="0.5", lw=0.6)
bx.set_xticks(range(4))
bx.set_xticklabels(["A1-6v\n11-11", "M120\n11-12", "M240\n11-12", "A2\n11-16"], fontsize=8)
bx.set_ylabel("change from A1-6v [%]", fontsize=9)
bx.set_title("(b) mass test closes: −0.3 to −3.6 % with the pieces,\nback to the common shift c (dashed) without them",
             fontsize=9, loc="left")

# (c)
F = {s: np.load(os.path.join(OUT, f"{s}_frf.npz")) for s in ("A1-6v", "A2")}
keys = [k for k in F["A1-6v"].files if k.startswith("H_") and k in F["A2"].files]
mag = {s: np.mean([np.abs(F[s][k]) for k in keys], axis=0) for s in F}
fr = F["A1-6v"]["fr"]
sel = (fr > 3000) & (fr < 3800)
db = lambda m: 20 * np.log10(m / np.nanmax(mag["A1-6v"][sel]))
cx.plot(fr[sel], db(mag["A1-6v"][sel]), color="k", lw=1.0, label="A1-6v (11-11)")
fr2 = F["A2"]["fr"]
sel2 = (fr2 > 3000) & (fr2 < 3800)
cx.plot(fr2[sel2], db(mag["A2"][sel2]), color="C3", lw=0.9, label="A2 (11-16)")
cx.plot(fr2[sel2] / (1 + c / 100), db(mag["A2"][sel2]) - 12, color="C3", lw=0.9, ls="--",
        label=f"A2, frequency ÷ (1 {c:+.2f} %), −12 dB")
cx.plot(fr[sel], db(mag["A1-6v"][sel]) - 12, color="k", lw=0.6, alpha=0.5)
cx.set_xlim(3000, 3800)
cx.set_xlabel("frequency [Hz]", fontsize=9)
cx.set_ylabel("mean |H| [dB]", fontsize=9)
cx.legend(fontsize=7.5, loc="lower left", frameon=False)
cx.set_title(f"(c) mean FRF over {len(keys)} common positions: one frequency scale\n"
             f"brings A2 back onto A1-6v", fontsize=9, loc="left")

# (d) 감쇠비
DM = ["2134", "2142", "3164", "3459", "3589", "3639", "3748", "3804", "4225", "4967", "5745", "7093", "8321"]


def zeta(St, n):
    z = [v["pool"]["z"] for v in St["modes"].get(n, {}).get("loc", {}).values()
         if v.get("pool") and v["pool"].get("z") is not None and np.isfinite(v["pool"]["z"]) and v["pool"].get("se", 9) < 2.0]
    return 100 * float(np.median(z)) if z else np.nan


P = pd.read_csv(os.path.join(HERE, "..", "fe-nominal", "s2j_poles.csv"))
z1030 = {n: float(P.iloc[np.argmin(np.abs(P.f - gl.MODE[n].f))].zeta) for n in DM}
z1028 = {n: float(P.iloc[np.argmin(np.abs(P.f - gl.MODE[n].f))].zeta_1028) for n in DM}
xd = np.arange(len(DM))
for k, (lab, vals, col, mk) in enumerate((("10-28", z1028, "0.6", "s"), ("10-30", z1030, "0.35", "s"),
                                           ("A1-6v 11-11", {n: zeta(S["A1-6v"], n) for n in DM}, "k", "o"),
                                           ("M120 11-12", {n: zeta(S["M120"], n) for n in DM}, "C0", "^"),
                                           ("M240 11-12", {n: zeta(S["M240"], n) for n in DM}, "C9", "v"),
                                           ("A2 11-16", {n: zeta(S["A2"], n) for n in DM}, "C3", "D"))):
    dx.plot(xd + (k - 2.5) * 0.11, [vals[n] for n in DM], mk, color=col, ms=5, label=lab, mfc=col if k >= 2 else "none")
dx.set_xticks(xd)
dx.set_xticklabels(DM, fontsize=8, rotation=45)
dx.set_ylim(0, 0.6)
dx.set_ylabel("damping ratio ζ [%]", fontsize=9)
dx.legend(fontsize=7.5, ncol=3, loc="upper left", frameon=False)
zr = np.nanmedian([zeta(S["A2"], n) / zeta(S["A1-6v"], n) for n in DM])
dx.set_title(f"(d) damping: A2 is {zr:.2f} × A1-6v (median), lower than every earlier series", fontsize=9, loc="left")


# (e) 힘에 따른 주파수 변화
def soften(St, n):
    by = defaultdict(list)
    for hh in St["hits"]:
        v = (hh.get("f") or {}).get(n)
        if not v or v[0] is None or not np.isfinite(v[0]) or (v[2] is not None and v[2] > 0.25) or hh["N"] < 5:
            continue
        by[hh["loc"]].append((np.log2(hh["N"]), v[0]))
    X, Y = [], []
    for v in by.values():
        if len(v) < 4:
            continue
        v = np.array(v)
        X += list(v[:, 0] - v[:, 0].mean())
        y = 100 * (v[:, 1] / np.median(v[:, 1]) - 1)
        Y += list(y - y.mean())
    X, Y = np.array(X), np.array(Y)
    if len(X) < 10 or X.std() < 0.1:
        return np.nan, np.nan
    b = (X * Y).sum() / (X * X).sum()
    return b, np.sqrt(((Y - b * X) ** 2).sum() / (len(X) - 1) / (X * X).sum())


SM = ["2142", "3164", "3459", "3639", "3748", "4225", "5745"]
xs = np.arange(len(SM))
ex.plot(xs - 0.3, [100 * gl.MODE[n].slope if gl.MODE[n].slope else np.nan for n in SM], "s", color="0.35", mfc="none",
        ms=6, label="10-30 (groove_lib)")
for k, (st, col, mk) in enumerate((("A1-6v", "k", "o"), ("M120", "C0", "^"), ("M240", "C9", "v"), ("A2", "C3", "D"))):
    v = np.array([soften(S[st], n) for n in SM])
    ex.errorbar(xs - 0.18 + 0.12 * k, v[:, 0], yerr=v[:, 1], fmt=mk, color=col, ms=5, capsize=2, label=st)
ex.axhline(0, color="0.5", lw=0.6)
ex.set_xticks(xs)
ex.set_xticklabels(SM, fontsize=8)
ex.set_ylim(-0.2, 0.15)
ex.set_ylabel("frequency change per doubling of force [%]", fontsize=9)
ex.legend(fontsize=7.5, ncol=5, loc="lower left", frameon=False)
ex.set_title("(e) amplitude softening: −0.03 to −0.10 % per doubling before, nearly zero in A2", fontsize=9, loc="left")
fig.savefig(os.path.join(GT, "a2.png"), dpi=130)
print("→", os.path.join(GT, "a2.png"))
