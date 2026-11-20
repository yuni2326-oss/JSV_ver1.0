"""2번 시편 L1(11-19 S16, 메모 s2_L34) 깃 1–6 × ξ0.3·ξ0.6 대 A2(11-16 S12, 같은 12자리) + 1번 깃 닫음 점검(18:08–18:09).

- 깎은 뒤 봉우리 자리(평균 스펙트럼, scratchpad l1s3/avg6.py)를 보고 대역을 손으로 정했다. A2에만 있는 봉우리,
  L1에만 있는 봉우리는 따로 대역을 두고, 옮겨 간 봉우리 짝은 이름에 '→'로 적었다(추정).
- 자리마다 s1p_check.compare(날짜마다 따로 둔 기울기, 15 N)로 A2 → L1 차를 내고, 모드마다 12자리의 중앙과 흩어짐을 낸다.
  A2 → A2-rh(11-19 S14, 1번 깃)는 0.03 % rms였으므로(49번) A2를 깃 2–6의 기준으로 쓴다.
- 닫음 점검: 1번 깃 17:38–17:40(15 + 15타) 대 18:08–18:09(4 + 4타)의 중앙값 차.
출력: groove-test/out/L1_six_check.csv·L1_six_check.png
사용: python s3_l1_six.py"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s1p_check as sp  # noqa: E402

OUT = os.path.join(HERE, "..", "groove-test", "out")
LOC12 = [f"back_vane{v}_xi{x}" for v in range(1, 7) for x in ("0.3", "0.6")]
V1 = ["back_vane1_xi0.3", "back_vane1_xi0.6"]
GROUPS = {"A2": (sp.S2F, [12], "20261116", LOC12),
          "L1v1": (sp.S2F, [16], "20261119", V1, None, "17:50"),          # 1번 깃 첫 무리(17:38–17:40)
          "L1v26": (sp.S2F, [16], "20261119", LOC12[2:]),                 # 깃 2–6(17:54–18:08)
          "L1end": (sp.S2F, [16], "20261119", V1, "18:05")}               # 1번 깃 닫음(18:08–18:09)


def l1_group(loc):
    return "L1v1" if loc in V1 else "L1v26"
# (이름, 아래, 위, 극점 수, 종류) — A2 봉우리 → L1 봉우리(평균 스펙트럼, 12자리)
BANDS = [("2134/2142 h2", 2115, 2150, 2, "disk"), ("3164→3113", 3100, 3170, 1, "panel"), ("3250", 3210, 3245, 1, "mixed"),
         ("3459", 3440, 3470, 1, "panel"), ("3496", 3485, 3505, 1, "?"), ("3514(new)", 3506, 3525, 1, "?"),
         ("3589", 3570, 3595, 1, "none"), ("3639", 3620, 3650, 1, "panel"), ("3748→3727", 3715, 3750, 1, "disk"),
         ("3804→3770", 3755, 3800, 1, "hub"), ("4225", 4190, 4225, 1, "hub"), ("4916(new)", 4905, 4930, 1, "back"),
         ("4967", 4935, 4970, 1, "back"), ("5617", 5555, 5595, 1, "back"), ("5677→5650", 5635, 5690, 1, "?"),
         ("5745", 5700, 5730, 1, "back"), ("6437(new)", 6425, 6450, 1, "?"), ("6439(6489)", 6478, 6500, 1, "panel"),
         ("6558", 6535, 6560, 1, "cand"), ("6817", 6770, 6805, 1, "cand"), ("7093→7030", 7015, 7085, 1, "cand"),
         ("7398(new)", 7385, 7412, 1, "?"), ("7462", 7440, 7475, 1, "none"), ("8321→8260", 8240, 8320, 1, "panel")]
KIND = {}
for b in BANDS:
    for n in sp.band_modes(b[0], b[3]):
        KIND[n] = b[4]


def main():
    D, _ = sp.per_hit(GROUPS, [b[:4] for b in BANDS])
    rows = []
    for name, lo, hi, k, kind in BANDS:
        for n in sp.band_modes(name, k):
            for loc in LOC12:
                gl1 = l1_group(loc)
                v = sp.compare(D, "A2", gl1, loc, n, common_slope=False)
                cnt = {g: int(((D.group == g) & (D["loc"] == loc) & (D["mode"] == n) & np.isfinite(D.f)).sum()) for g in ("A2", gl1)}
                rows.append(dict(mode=n, kind=kind, loc=loc, vane=int(loc[9]), xi=loc[-3:], n_ok_A2=cnt["A2"], n_ok_L1=cnt[gl1],
                                 **(v or {})))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, "L1_six_check.csv"), index=False, float_format="%.5g")

    # ---------------------------------------------------------------- 요약(모드마다 12자리)
    print("\n[A2 → L1, 12자리] 모드 | 잘 맞은 자리 수 | 중앙 차 [%] | 자리 사이 범위 | 감쇠 비 중앙 | A2·L1 주파수 중앙 [Hz]")
    S = []
    for name, lo, hi, k, kind in BANDS:
        for n in sp.band_modes(name, k):
            x = R[(R["mode"] == n) & np.isfinite(R.get("d_pct", np.nan)) & (R.se_pct < 0.08) & (R.n_a >= 5) & (R.n_b >= 5)]
            ya = R[(R["mode"] == n)]
            na, nl = int((ya.n_ok_A2 >= 5).sum()), int((ya.n_ok_L1 >= 5).sum())
            if len(x):
                S.append(dict(mode=n, kind=kind, n_pos=len(x), d_med=np.median(x.d_pct), d_lo=x.d_pct.min(), d_hi=x.d_pct.max(),
                              z_ratio=np.median(x.z_ratio), f_A2=np.median(x.f_a), f_L1=np.median(x.f_b)))
                print(f"  {n:14s} {kind:6s} {len(x):2d} | {np.median(x.d_pct):+.3f} | {x.d_pct.min():+.3f} ~ {x.d_pct.max():+.3f} | "
                      f"{np.median(x.z_ratio):.2f} | {np.median(x.f_a):.1f} → {np.median(x.f_b):.1f}")
            else:
                print(f"  {n:14s} {kind:6s}  0 | (A2에서 잘 맞은 자리 {na}, L1 {nl})")
    pd.DataFrame(S).to_csv(os.path.join(OUT, "L1_six_summary.csv"), index=False, float_format="%.5g")

    # ---------------------------------------------------------------- 닫음 점검(1번 깃)
    print("\n[닫음 점검] 1번 깃 17:38–17:40 → 18:08–18:09, 두 자리 합침, 타격별 중앙(15 N 환산 없음 — 연화 ~0)")
    for name, lo, hi, k, kind in BANDS:
        for n in sp.band_modes(name, k):
            a = D[(D.group == "L1v1") & (D["mode"] == n) & np.isfinite(D.f)]
            b = D[(D.group == "L1end") & (D["mode"] == n) & np.isfinite(D.f)]
            if len(a) >= 5 and len(b) >= 3:
                d = 100 * (np.median(b.f) / np.median(a.f) - 1)
                se = 100 * 1.2533 * np.hypot(a.f.std() / np.sqrt(len(a)), b.f.std() / np.sqrt(len(b))) / np.median(a.f)
                print(f"  {n:14s} {np.median(a.f):9.2f} → {np.median(b.f):9.2f} Hz  {d:+.3f} ± {se:.3f} %  (타격 {len(a)} → {len(b)})")
    T = D.groupby("group")[["T0", "T1"]].mean()
    print("\n기온 ai0/ai1:", {g: f"{T.loc[g, 'T0']:.2f}/{T.loc[g, 'T1']:.2f}" for g in T.index})
    figure(R)


def figure(R):
    order = [n for b in BANDS for n in sp.band_modes(b[0], b[3])]
    xs = {n: i for i, n in enumerate(order)}
    fig, ax = plt.subplots(2, 1, figsize=(14, 8.5), sharex=True)
    cmap = plt.get_cmap("viridis")
    for _, r in R.iterrows():
        if not np.isfinite(r.get("d_pct", np.nan)) or r.se_pct >= 0.08 or r.n_a < 5 or r.n_b < 5:
            continue
        c = cmap((r.vane - 1) / 5)
        mk = "o" if r.xi == "0.3" else "s"
        ax[0].plot(xs[r["mode"]] + (r.vane - 3.5) * 0.08, r.d_pct, mk, color=c, ms=5, mfc=c if r.xi == "0.6" else "white")
        ax[1].plot(xs[r["mode"]] + (r.vane - 3.5) * 0.08, r.z_ratio, mk, color=c, ms=5, mfc=c if r.xi == "0.6" else "white")
    for v in range(1, 7):
        ax[0].plot([], [], "s", color=cmap((v - 1) / 5), label=f"vane {v}")
    ax[0].legend(fontsize=8, ncol=6, loc="lower right")
    ax[0].axhspan(-0.05, 0.05, color="0.85", zorder=0); ax[0].axhline(0, color="k", lw=0.6); ax[0].grid(alpha=0.3)
    ax[0].set_ylabel("frequency change A2 → L1 [%]")
    ax[0].set_title("(a) specimen 2, all 12 points (vane 1–6 × ξ0.3 open / ξ0.6 filled), 15 N. Modes with no point: peak absent "
                    "in A2 or in L1 (new or vanished)", fontsize=8.5, loc="left")
    ax[1].axhline(1, color="k", lw=0.6); ax[1].set_ylabel("damping ratio L1 / A2"); ax[1].grid(alpha=0.3); ax[1].set_ylim(0, 3)
    ax[1].set_xticks(range(len(order))); ax[1].set_xticklabels([f"{n}\n{KIND[n]}" for n in order], fontsize=7, rotation=60)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "L1_six_check.png"), dpi=110)


if __name__ == "__main__":
    main()
