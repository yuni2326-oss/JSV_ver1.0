"""국부 손상 시험 사전 계산 정리(42번) — s3_local.py 결과 → 표·그림·summary_local.json.

(1) 대조: 같은 띠 360° 고리가 s3_km nominal(41번)과 맞는지
(2) 국부 포켓(B·C·M, 각폭 20·30·60°, 깃 기준 위치 0·15·30·45°): 쌍 평균·배 성분(c)·마디 쪽 성분(s)·갈라짐, 홑 모드, 깎은 질량
(3) 측정 짝에 맞춘 예측: s3_orient.py가 준 측정 성분의 배 자리(B: 아래 2134 = FEM θ_rel 14.4·44.4°, 위 2142 = 28.6·58.6°)
(4) 국부 질량(테이프, 테두리 ξ0.9–1.0): 10 mg당 성분별 변화
(5) 깃마다 같은 자리 6곳(주기 손상) = 자취 × 6, 갈라짐 0
단위: Δf/f [%] = 50·η. 깊이는 1차라 선형으로 늘린다(0.02 mm 계산 → 0.06 mm = ×3).
사용: python s3_local_report.py"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
D = HERE.parents[0] / "fe-nominal" / "s3_local"
KM = HERE.parents[0] / "fe-nominal" / "s3_km"
L = pd.concat([pd.read_csv(D / f"local_{t}.csv") for t in ("B", "C", "MR")], ignore_index=True)
NOM = pd.read_csv(KM / "nominal.csv")
EX = pd.read_csv(KM / "nominal_extra.csv")
#: 측정 극점 → (FEM 이름표, 성격)
MEAS = {"2134/2142": ("h2:m2n0", "원판형 짝"), "3748": ("h3:m3n1", "원판형"), "3164": ("h3:m3n0", "패널"),
        "3250": ("h0:m0n1", "섞임"), "3459": ("h1:m1n1", "패널 짝"), "3639": ("h2:m4n0", "패널 짝"),
        "4967": ("h2:m4n0#2", "뒤판 짝"), "5617": ("h0:m6n0", "뒤판"), "5745": ("h1:m5n0", "뒤판 짝"),
        "6439": ("h2:m2n1", "패널 짝"), "8321": ("h3:m3n1#2", "패널")}
P = 50.0
out = {}


def sel(band, w, rel):
    return L[(L.band == band) & (L.width_deg == w) & (L.rel_deg == rel)].set_index("label")


# (1) 대조
print("(1) 360° 고리 대조: 이 계산 / 41번 s3_km  [Δf/f %, 0.02 mm]")
for band, g, src in (("B", "xi0.9", NOM), ("C", "xi0.3", NOM), ("M", "xi0.6", NOM), ("R", "rim2", EX)):
    s = sel(band, 360.0, 0.0)
    ref = src[src.groove == g].set_index("label")
    lab = "h2:m2n0"
    print(f"  {band}: 깎은 질량 {s.dm_mg.iloc[0]:.1f} / {ref.dm_mg.iloc[0]:.1f} mg; h2 짝 K {P * s.loc[lab, 'eta_K']:+.3f} / "
          f"{P * ref.loc[lab, 'eta_K']:+.3f}, M {P * s.loc[lab, 'eta_M']:+.3f} / {P * ref.loc[lab, 'eta_M']:+.3f}, "
          f"갈라짐 {P * s.loc[lab, 'split']:.4f}")

# (2) 국부 포켓 표
print("\n(2) 국부 포켓 0.02 mm [Δf/f %]: 깎은 질량 | h2 짝: 평균 / c / s / 갈라짐 (c의 K·M) | 3748 | 패널 3164·3459평균·3639평균 | 뒤판 4967·5617")
rows = []
for band in ("B", "C", "M"):
    for w in sorted(L[(L.band == band) & (L.width_deg < 359)].width_deg.unique()):
        for rel in sorted(L[(L.band == band) & (L.width_deg == w)].rel_deg.unique()):
            s = sel(band, w, rel)
            h2 = s.loc["h2:m2n0"]
            r = {"band": band, "width": w, "rel": rel, "dm_mg": float(s.dm_mg.iloc[0]),
                 "h2_mean": P * h2.eta, "h2_c": P * h2.H_cc, "h2_s": P * h2.H_ss, "h2_split": P * h2.split,
                 "h2_cK": P * h2.K_cc, "h2_cM": P * h2.M_cc, "h2_sK": P * h2.K_ss, "h2_sM": P * h2.M_ss,
                 "h2_cs": P * h2.H_cs}
            for k, (lab, _) in MEAS.items():
                if k == "2134/2142":
                    continue
                o = s.loc[lab]
                r[k] = P * o.eta
                if o.deg == 2:
                    r[k + "_split"] = P * o.split
            rows.append(r)
            print(f"  {band} {w:>4.0f}° @{rel:>4.1f}°: {r['dm_mg']:6.1f} mg | {r['h2_mean']:+.3f} / {r['h2_c']:+.3f} / {r['h2_s']:+.3f} / "
                  f"{r['h2_split']:.3f} (c: K {r['h2_cK']:+.3f} M {r['h2_cM']:+.3f}; s: K {r['h2_sK']:+.3f} M {r['h2_sM']:+.3f}; cs {r['h2_cs']:+.3f}) | "
                  f"{r['3748']:+.3f} | {r['3164']:+.3f} {r['3459']:+.3f} {r['3639']:+.3f} | {r['4967']:+.3f} {r['5617']:+.3f}")
T = pd.DataFrame(rows)
T.to_csv(D / "local_table.csv", index=False)

# (3) 측정 짝에 맞춘 예측 — 깊이 0.06 mm(×3)
print("\n(3) 측정 짝 정렬 예측, 깊이 0.06 mm [Δf/f %]: 포켓을 한 성분의 배에 둔다(B: 아래 2134 = θ_rel 15·45°, 위 2142 = 0·30°)")
pred = []
for band in ("B", "C"):
    for w in (20.0, 30.0, 60.0):
        for rel in (0.0, 15.0, 30.0, 45.0):
            q = T[(T.band == band) & (T.width == w) & (T.rel == rel)].iloc[0]
            lower_at_antinode = rel in (15.0, 45.0)
            f_lo = 3 * (q.h2_c if lower_at_antinode else q.h2_s)
            f_hi = 3 * (q.h2_s if lower_at_antinode else q.h2_c)
            pred.append({"band": band, "width": w, "rel": rel, "dm_mg_006": 3 * q.dm_mg, "f2134": f_lo, "f2142": f_hi,
                         "split_change": f_hi - f_lo, "mean": 3 * q.h2_mean})
            print(f"  {band} {w:>4.0f}° @{rel:>4.1f}° ({3 * q.dm_mg:5.0f} mg): 2134 {f_lo:+.3f}, 2142 {f_hi:+.3f}, "
                  f"갈라짐 변화 {f_hi - f_lo:+.3f}, 평균 {3 * q.h2_mean:+.3f}")
Pr = pd.DataFrame(pred)
Pr.to_csv(D / "local_pred_pair.csv", index=False)
print("  * B의 측정 성분 배 자리는 FEM θ_rel 14.4/44.4°(아래)·28.6/58.6°(위)라 계산 자리(15·45·30·0°)와 1.4° 안이다. C(ξ0.3)는 51.7/21.7°·6.0/36.0°로 7–9° 어긋난다 → C는 근사.")

# (4) 국부 질량 테이프(테두리 ξ0.9–1.0, 질량 몫만, 부호 반대) — 10 mg당
print("\n(4) 테두리 테이프 10 mg당 [Δf/f %]: h2 짝 c / s | 3748 | 3164 3459 3639 | 4967 5617  (c = 테이프 자리에 배가 있는 성분)")
tape = []
for w in sorted(L[(L.band == "R") & (L.width_deg < 359)].width_deg.unique()):
    for rel in sorted(L[(L.band == "R") & (L.width_deg == w)].rel_deg.unique()):
        s = sel("R", w, rel)
        k = -10.0 / float(s.dm_mg.iloc[0])            # 깎은 질량 → 붙인 10 mg
        h2 = s.loc["h2:m2n0"]
        r = {"width": w, "rel": rel, "c": P * k * h2.M_cc, "s": P * k * h2.M_ss}
        for kk in ("3748", "3164", "3459", "3639", "4967", "5617"):
            o = s.loc[MEAS[kk][0]]
            r[kk] = P * k * o.eta_M
        tape.append(r)
        print(f"  {w:>3.0f}° @{rel:>4.1f}°: {r['c']:+.4f} / {r['s']:+.4f} | {r['3748']:+.4f} | {r['3164']:+.4f} {r['3459']:+.4f} "
              f"{r['3639']:+.4f} | {r['4967']:+.4f} {r['5617']:+.4f}")
s = sel("R", 360.0, 0.0)
k = -10.0 / float(s.dm_mg.iloc[0])
print(f"  360° 고리 10 mg: h2 짝 {P * k * s.loc['h2:m2n0', 'eta_M']:+.4f}, 3748 {P * k * s.loc['h3:m3n1', 'eta_M']:+.4f}")
Tp = pd.DataFrame(tape)
Tp.to_csv(D / "local_tape.csv", index=False)

# (5) 깃마다 같은 자리 6곳 = 자취 × 6
print("\n(5) 같은 포켓 6곳(깃마다 같은 자리, 0.02 mm): h2 짝 평균(갈라짐 0) / 같은 띠 360° 고리, 깎은 질량 대비")
for band in ("B", "C"):
    ring = sel(band, 360.0, 0.0)
    for w in (30.0, 60.0):
        for rel in (0.0, 30.0):
            q = T[(T.band == band) & (T.width == w) & (T.rel == rel)].iloc[0]
            print(f"  {band} {w:.0f}° @{rel:.0f}° × 6: 질량 {6 * q.dm_mg:.0f} mg, h2 짝 {6 * q.h2_mean:+.3f} "
                  f"(고리 {P * ring.loc['h2:m2n0', 'eta']:+.3f}, {ring.dm_mg.iloc[0]:.0f} mg), "
                  f"3748 {6 * q['3748']:+.3f} (고리 {P * ring.loc['h3:m3n1', 'eta']:+.3f})")

# (6) 빠진 질량(4.23 g) 위치 민감도 — 41번에서 건전 주파수가 측정과 맞은 두 분포(전체 균일·깃)
print("\n(6) 빠진 질량 위치 민감도, 깊이 0.06 mm [Δf/f %]: 공칭 / 전체 균일 / 깃")
VAR = {"nominal": L}
for w in ("all", "vanes"):
    f = D / f"local_KEY_{w}.csv"
    if f.exists():
        VAR[w] = pd.read_csv(f)
rob = {}


def g(df, band, w, rel, lab, col):
    q = df[(df.band == band) & (df.width_deg == w) & (df.rel_deg == rel) & (df.label == lab)]
    return float(q[col].iloc[0]) if len(q) else np.nan


lines = [("B 30° 깃 사이(30°): h2 c", "B", 30.0, 30.0, "h2:m2n0", "H_cc", 3),
         ("B 30° 깃 사이(30°): h2 s", "B", 30.0, 30.0, "h2:m2n0", "H_ss", 3),
         ("B 30° 깃 위(0°): h2 c", "B", 30.0, 0.0, "h2:m2n0", "H_cc", 3),
         ("B 30° 깃 위(0°): h2 s", "B", 30.0, 0.0, "h2:m2n0", "H_ss", 3),
         ("B 30° 15°: h2 c", "B", 30.0, 15.0, "h2:m2n0", "H_cc", 3),
         ("B 30° 15°: h2 s", "B", 30.0, 15.0, "h2:m2n0", "H_ss", 3),
         ("B 깃 사이 6곳: h2 평균", "B", 30.0, 30.0, "h2:m2n0", "eta", 18),
         ("B 깃 위 6곳: h2 평균", "B", 30.0, 0.0, "h2:m2n0", "eta", 18),
         ("B 360° 고리: h2 평균", "B", 360.0, 0.0, "h2:m2n0", "eta", 3),
         ("B 깃 사이 6곳: 3748", "B", 30.0, 30.0, "h3:m3n1", "eta", 18),
         ("B 깃 위 6곳: 3748", "B", 30.0, 0.0, "h3:m3n1", "eta", 18),
         ("B 깃 사이 6곳: 3164", "B", 30.0, 30.0, "h3:m3n0", "eta", 18),
         ("B 깃 사이 6곳: 3459", "B", 30.0, 30.0, "h1:m1n1", "eta", 18),
         ("B 깃 사이 6곳: 3639", "B", 30.0, 30.0, "h2:m4n0", "eta", 18),
         ("B 깃 위 6곳: 3164", "B", 30.0, 0.0, "h3:m3n0", "eta", 18),
         ("C 30° 깃 위: h2 평균", "C", 30.0, 0.0, "h2:m2n0", "eta", 3),
         ("C 30° 깃 사이: h2 평균", "C", 30.0, 30.0, "h2:m2n0", "eta", 3),
         ("C 360° 고리: h2 평균", "C", 360.0, 0.0, "h2:m2n0", "eta", 3),
         ("C 360° 고리: 3748", "C", 360.0, 0.0, "h3:m3n1", "eta", 3)]
for name, band, w, rel, lab, col, k in lines:
    vals = [P * k * g(df, band, w, rel, lab, col) for df in VAR.values()]
    rob[name] = vals
    print(f"  {name:<24s} " + " / ".join(f"{v:+.3f}" for v in vals))
print("  테이프 10 mg(테두리 8°): h2 c / s")
for rel in (0.0, 15.0, 30.0, 45.0):
    vals = []
    for df in VAR.values():
        q = df[(df.band == "R") & (df.width_deg == 8.0) & (df.rel_deg == rel) & (df.label == "h2:m2n0")]
        kk = -10.0 / float(q.dm_mg.iloc[0])
        vals.append((P * kk * float(q.M_cc.iloc[0]), P * kk * float(q.M_ss.iloc[0])))
    rob[f"tape {rel:.0f}"] = vals
    print(f"    @{rel:>4.1f}°: " + " / ".join(f"{c:+.4f},{s_:+.4f}" for c, s_ in vals))
print("  (열 순서:", " / ".join(VAR), ")")

out = {"table": T.to_dict(orient="records"), "pair_pred_006": Pr.to_dict(orient="records"),
       "tape_per10mg": Tp.to_dict(orient="records"), "mass_sensitivity_006": rob, "mass_columns": list(VAR)}
json.dump(out, open(D / "summary_local.json", "w"), ensure_ascii=False, indent=1)

# 그림: (a) B·C 창 하나 위치별 h2 짝 c·s, (b) B 창 하나의 강성·질량 몫, (c) B 창 6개(깃 사이·깃 위)와 고리,
#       (d) 테이프 10 mg, (e) 2번 시편 측정 짝 정렬 예측, (f) 빠진 질량 위치 민감도
fig, ax = plt.subplots(2, 3, figsize=(17, 8.6))
for band, col in (("B", "C3"), ("C", "C0")):
    g = T[(T.band == band) & (T.width == 30.0)].sort_values("rel")
    ax[0, 0].plot(g.rel, 3 * g.h2_c, "o-", color=col, label=f"{band}: member with antinode at window (c)")
    ax[0, 0].plot(g.rel, 3 * g.h2_s, "s--", color=col, label=f"{band}: other member (s)")
    ax[0, 0].plot(g.rel, 3 * g.h2_mean, ":", color=col, label=f"{band}: pair mean")
ax[0, 0].axhline(0, color="k", lw=0.6)
ax[0, 0].axhspan(-0.15, 0.15, color="0.88", zorder=0, label="±3σ (σ = 0.05 %)")
ax[0, 0].set_xlabel("window centre from vane [deg] (FEM; 0 = over vane, 30 = mid-bay)")
ax[0, 0].set_ylabel("h2 pair Δf/f [%]")
ax[0, 0].set_title("(a) one 30° window, 0.06 mm", fontsize=9)
ax[0, 0].legend(fontsize=6.5)
g = T[(T.band == "B") & (T.width == 30.0)].sort_values("rel")
x = np.arange(len(g))
for j, (c, lab, col) in enumerate((("h2_cK", "c stiffness", "C0"), ("h2_cM", "c mass", "C1"), ("h2_sK", "s stiffness", "C2"),
                                   ("h2_sM", "s mass", "C4"))):
    ax[0, 1].bar(x + (j - 1.5) * 0.2, 3 * g[c], width=0.2, color=col, label=lab)
ax[0, 1].plot(x, 3 * g.h2_c, "k^", label="c total")
ax[0, 1].plot(x, 3 * g.h2_s, "kv", label="s total")
ax[0, 1].axhline(0, color="k", lw=0.6)
ax[0, 1].set_xticks(x); ax[0, 1].set_xticklabels([f"{r:.0f}°" for r in g.rel])
ax[0, 1].set_title("(b) B one 30° window, 0.06 mm: stiffness vs mass parts", fontsize=9)
ax[0, 1].set_ylabel("Δf/f [%]"); ax[0, 1].legend(fontsize=7, ncol=2)
q30 = T[(T.band == "B") & (T.width == 30.0) & (T.rel == 30.0)].iloc[0]
q0 = T[(T.band == "B") & (T.width == 30.0) & (T.rel == 0.0)].iloc[0]
ringB = sel("B", 360.0, 0.0)
keys = [("h2 pair", "h2_mean", "h2:m2n0"), ("3748", "3748", "h3:m3n1"), ("3164", "3164", "h3:m3n0"),
        ("3459", "3459", "h1:m1n1"), ("3639", "3639", "h2:m4n0")]
x = np.arange(len(keys))
ax[0, 2].bar(x - 0.27, [18 * q30[k] for _, k, _ in keys], width=0.27, color="#fdae6b", ec="k", lw=0.4,
             label=f"6 mid-bay windows ({6 * 3 * q30.dm_mg:.0f} mg)")
ax[0, 2].bar(x, [18 * q0[k] for _, k, _ in keys], width=0.27, color="#9ecae1", ec="k", lw=0.4,
             label=f"6 over-vane windows ({6 * 3 * q0.dm_mg:.0f} mg)")
ax[0, 2].bar(x + 0.27, [150 * ringB.loc[l, "eta"] for _, _, l in keys], width=0.27, color="0.6", ec="k", lw=0.4,
             label=f"360° ring = sum ({3 * ringB.dm_mg.iloc[0]:.0f} mg)")
ax[0, 2].axhline(0, color="k", lw=0.6)
ax[0, 2].set_xticks(x); ax[0, 2].set_xticklabels([k for k, _, _ in keys])
ax[0, 2].set_title("(c) B, 0.06 mm: sign depends on where between the vanes", fontsize=9)
ax[0, 2].set_ylabel("Δf/f [%]"); ax[0, 2].legend(fontsize=7)
g = Tp[Tp.width == Tp.width.min()].sort_values("rel")
x = np.arange(len(g))
for j, (c, lab) in enumerate((("c", "h2 member c"), ("s", "h2 member s"), ("3748", "3748"), ("3164", "3164"),
                              ("3459", "3459"), ("3639", "3639"))):
    ax[1, 0].bar(x + (j - 2.5) * 0.13, g[c], width=0.13, label=lab)
ax[1, 0].axhline(0, color="k", lw=0.6)
ax[1, 0].set_xticks(x); ax[1, 0].set_xticklabels([f"{r:.0f}°" for r in g.rel])
ax[1, 0].set_title(f"(d) tape 10 mg at rim (ξ0.9–1.0, {g.width.iloc[0]:.0f}° wide)", fontsize=9)
ax[1, 0].set_xlabel("tape centre from vane [deg] (FEM)"); ax[1, 0].set_ylabel("Δf/f [%]"); ax[1, 0].legend(fontsize=7, ncol=3)
g = Pr[(Pr.band == "B") & (Pr.width == 30.0)].sort_values("rel")
x = np.arange(len(g))
ax[1, 1].bar(x - 0.2, g.f2134, width=0.4, label="2134 (lower)")
ax[1, 1].bar(x + 0.2, g.f2142, width=0.4, label="2142 (upper)")
ax[1, 1].axhline(0, color="k", lw=0.6)
ax[1, 1].axhspan(-0.15, 0.15, color="0.88", zorder=0)
ax[1, 1].set_xticks(x)
ax[1, 1].set_xticklabels([f"{r:.0f}°\n({'2134' if r in (15.0, 45.0) else '2142'} antinode)" for r in g.rel], fontsize=8)
ax[1, 1].set_title("(e) specimen 2: one B 30° window, 0.06 mm, aligned to a measured member", fontsize=9)
ax[1, 1].set_ylabel("Δf/f [%]"); ax[1, 1].legend(fontsize=7)
names = [("B 30° 깃 사이(30°): h2 c", "1 window\n(mid-bay): 2142"), ("B 30° 깃 사이(30°): h2 s", "1 window\n(mid-bay): 2134"),
         ("B 깃 사이 6곳: h2 평균", "6 mid-bay:\nh2 mean"), ("B 깃 위 6곳: h2 평균", "6 over-vane:\nh2 mean"),
         ("B 360° 고리: h2 평균", "ring:\nh2 mean"), ("C 360° 고리: h2 평균", "C ring:\nh2 mean")]
x = np.arange(len(names))
for j, (w, col) in enumerate(zip(list(VAR), ("k", "C0", "C2"))):
    ax[1, 2].bar(x + (j - 1) * 0.27, [rob[n][j] for n, _ in names], width=0.27, color=col,
                 label={"nominal": "nominal G3", "all": "missing 4.23 g uniform", "vanes": "missing 4.23 g on vanes"}[w])
ax[1, 2].axhline(0, color="k", lw=0.6)
ax[1, 2].set_xticks(x); ax[1, 2].set_xticklabels([t for _, t in names], fontsize=7.5)
ax[1, 2].set_title("(f) robustness to where the missing mass sits (0.06 mm)", fontsize=9)
ax[1, 2].set_ylabel("Δf/f [%]"); ax[1, 2].legend(fontsize=7)
fig.suptitle("G3 free-free FEM (32×16), first order: localized front-shroud windows and local mass (depth scaled linearly from 0.02 mm)", fontsize=10)
fig.tight_layout()
fig.savefig(HERE.parents[0] / "groove-test" / "s3-local.png", dpi=130)
print("\n그림:", HERE.parents[0] / "groove-test" / "s3-local.png")
