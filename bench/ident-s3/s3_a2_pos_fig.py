"""그림 a2-positions.png(47번 §3.1): A2의 변화가 치는 자리(IPA 세척 뒤 다시 그린 표시) 탓인가.
(a) 위치 12곳마다 A2 − A1-6v 주파수 변화 [%](잘 정해진 모드). 회색 = A1-6v 안에서 위치마다 평균과의 차(자리 자체가 주파수를 얼마나 바꾸나).
(b) 위치마다 감쇠비 ζ(빈 = A1-6v, 채운 = A2).
(c) 위치마다 응답 크기(평균 FRF의 봉우리 높이, 그 상태의 위치 평균 대비 [dB]) — 자리가 바뀌었으면 모양이 달라진다.
입력: groove-test/out/{A1-6v,A2}.json·_frf.npz. 출력: groove-test/a2-positions.png, 화면 표."""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
GT = os.path.join(HERE, "..", "groove-test")
OUT = os.path.join(GT, "out")
S = {s: json.load(open(os.path.join(OUT, f"{s}.json"))) for s in ("A1-6v", "A2")}
F = {s: np.load(os.path.join(OUT, f"{s}_frf.npz")) for s in S}
G = ["2142", "3164", "3459", "3589", "3639", "3748", "3804", "4225", "7093"]
LOCS = [f"back_vane{k}_xi{x}" for k in range(1, 7) for x in ("0.3", "0.6")]
LAB = [f"v{k} {x}" for k in range(1, 7) for x in ("0.3", "0.6")]


def lp(St, n, loc):
    v = St["modes"][n]["loc"].get(loc, {}).get("pool") or {}
    f, se, z = v.get("f"), v.get("se"), v.get("z")
    ok = f is not None and se is not None and np.isfinite(f) and np.isfinite(se) and se < 2.0
    return (f, 100 * z) if ok else (np.nan, np.nan)


fig, (ax, bx, cx) = plt.subplots(3, 1, figsize=(12, 11.5))
x = np.arange(len(LOCS))
cols = plt.cm.tab10(np.linspace(0, 1, 10))
allA, allD = [], []
print("모드 | A1-6v 안 위치 흩어짐 SD [%] | 위치마다 A2−A1 평균 ± SD [%] | ζ 비(위치 중앙)")
for i, n in enumerate(G):
    a = np.array([lp(S["A1-6v"], n, l)[0] for l in LOCS])
    b = np.array([lp(S["A2"], n, l)[0] for l in LOCS])
    d = 100 * (b / a - 1)
    ax.plot(x, d, "-o", color=cols[i], ms=4, lw=0.9, label=n)
    ra = 100 * (a / np.nanmean(a) - 1)
    ax.plot(x, ra, "s", color="0.6", ms=3)
    allA += list(ra[np.isfinite(ra)]); allD += list(d[np.isfinite(d)])
    za = np.array([lp(S["A1-6v"], n, l)[1] for l in LOCS]); zb = np.array([lp(S["A2"], n, l)[1] for l in LOCS])
    if n in ("3164", "3459", "3748", "4225"):
        bx.plot(x, za, "o", mfc="none", mec=cols[i], ms=6, mew=1.3)
        bx.plot(x, zb, "o", color=cols[i], ms=5, label=n)
    dd = d[np.isfinite(d)]
    print(f"{n:>5} | {np.nanstd(ra):.3f} | {dd.mean():+.3f} ± {dd.std():.3f} (n {dd.size}) | {np.nanmedian(zb / za):.2f}")
ax.axhline(0, color="0.5", lw=0.6)
ax.axhline(np.median(allD), color="k", lw=1.0, ls="--")
ax.plot([], [], "s", color="0.6", ms=3, label="A1-6v: position minus its mean")
ax.set_xticks(x); ax.set_xticklabels(LAB, fontsize=8)
ax.set_ylabel("frequency change [%]", fontsize=9)
ax.set_ylim(-0.35, 0.15)
ax.legend(fontsize=7.5, ncol=6, loc="lower left", frameon=False)
ax.set_title(f"(a) A2 − A1-6v at each hit position: {np.mean(allD):+.2f} ± {np.std(allD):.2f} % at every position; "
             f"moving the hit 60° or 6.5 mm within A1-6v changes the frequency by only ±{np.std(allA):.2f} % (grey)",
             fontsize=9, loc="left")
bx.set_xticks(x); bx.set_xticklabels(LAB, fontsize=8)
bx.set_ylabel("damping ratio ζ [%]", fontsize=9)
bx.set_ylim(0, 0.35)
bx.plot([], [], "o", mfc="none", mec="k", label="A1-6v (open)")
bx.plot([], [], "o", color="k", label="A2 (filled)")
bx.legend(fontsize=7.5, ncol=6, loc="upper left", frameon=False)
bx.set_title("(b) damping at each hit position: lower in A2 at every position, by more than its spread over positions",
             fontsize=9, loc="left")

# (c) 봉우리 높이 모양
print("\n봉우리 높이 모양(위치 평균 대비 dB) — A1-6v / A2, 차 rms")
for i, n in enumerate(("3164", "3459", "3748", "4225")):
    pat = {}
    for s in S:
        f0 = S[s]["modes"][n]["comb"]["f"]
        fr = F[s]["fr"]
        sel = (fr > f0 * 0.997) & (fr < f0 * 1.003)
        v = np.array([np.abs(F[s][f"H_{l}"][sel]).max() if f"H_{l}" in F[s].files else np.nan for l in LOCS])
        pat[s] = 20 * np.log10(v / np.nanmean(v))
    ok = np.isfinite(pat["A1-6v"]) & np.isfinite(pat["A2"])
    off = np.nanmean(pat["A2"][ok] - pat["A1-6v"][ok])
    cx.plot(x, pat["A1-6v"], "o", mfc="none", mec=cols[G.index(n)], ms=6, mew=1.3)
    cx.plot(x, pat["A2"] - off, "o-", color=cols[G.index(n)], ms=4, lw=0.8, label=n)
    rms = np.sqrt(np.mean((pat["A2"][ok] - off - pat["A1-6v"][ok]) ** 2))
    print(f"{n}: " + " ".join(f"{a:+5.1f}/{b - off:+5.1f}" for a, b in zip(pat["A1-6v"], pat["A2"])) + f"  rms {rms:.1f} dB")
cx.set_xticks(x); cx.set_xticklabels(LAB, fontsize=8)
cx.set_ylabel("peak |H| relative to the position mean [dB]", fontsize=9)
cx.plot([], [], "o", mfc="none", mec="k", label="A1-6v (open)")
cx.plot([], [], "o-", color="k", label="A2 (filled)")
cx.legend(fontsize=7.5, ncol=6, loc="lower left", frameon=False)
cx.set_title("(c) response pattern over the hit positions (peak height of the mean FRF): mostly the same, "
             "so the redrawn marks are close to the old ones", fontsize=9, loc="left")
fig.tight_layout()
fig.savefig(os.path.join(GT, "a2-positions.png"), dpi=130)
print("→", os.path.join(GT, "a2-positions.png"))
