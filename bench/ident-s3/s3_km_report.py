"""41번 표·그림 — s3_km_split.py 출력(fe-nominal/s3_km/)을 측정 극점(groove_lib.MODES)에 붙여 정리한다.

(1) 홈 띠별 강성 몫·질량 몫(0.02 mm, Δf/f ≈ η/2): ξ0.3·0.6·0.85·0.9와 테두리 띠 rim2(ξ0.9–1.0)·rim1(ξ0.95–1.0)
(2) 질량 차 4.23 g을 놓는 곳별 건전 주파수 대 측정(rms) — 측정과 맞는 곳만 '그럴듯함'으로 친다
(3) 곳별 띠 예측(0.02 mm당)과 부호 유지, 원판형 − 패널 대비
(4) 그림: 반지름 지도(깎은 1 mg당 변화) — 공칭 강성/질량/합, 곳별 합
사용: python s3_km_report.py → 화면(마크다운)·fe-nominal/s3_km/summary.json·groove-test/s3-km.png"""
import json
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
D = os.path.join(HERE, "..", "fe-nominal", "s3_km")
WHERE = ["all", "back", "vanes", "hub", "collar", "front", "rim"]
KO_W = {"nominal": "공칭 G3(질량 차 없음)", "all": "전체 균일", "front": "앞판", "back": "뒤판+허브", "hub": "허브",
        "vanes": "깃", "collar": "흡입구 테", "rim": "바깥 1 mm 고리"}
KO = {"disk": "원판형", "panel": "패널", "mixed": "섞임", "back": "뒤판", "hub": "허브"}
GROOVES = ["xi0.3", "xi0.6", "xi0.85", "xi0.9", "rim2", "rim1"]
GNAME = {"xi0.3": "C ξ0.2–0.4", "xi0.6": "ξ0.5–0.7", "xi0.85": "ξ0.75–0.95", "xi0.9": "B ξ0.8–1.0",
         "rim2": "ξ0.9–1.0", "rim1": "ξ0.95–1.0"}
RMS_OK = 3.0                                             # 측정과 맞는 곳: 평균 뺀 rms ≤ 3 %
pc = lambda eta: 100 * (np.sqrt(1 + eta) - 1)            # noqa: E731
JUDGED_FEM = []
for m in gl.MODES:                                       # h2 짝은 FEM에서 한 관측량
    if m.judge and m.fem not in [x.fem for x in JUDGED_FEM]:
        JUDGED_FEM.append(m)
DISK = [m.fem for m in JUDGED_FEM if m.kind == "disk"]
PANEL = [m.fem for m in JUDGED_FEM if m.kind == "panel"]
name = lambda m: "2134/2142" if m.fem == "h2:m2n0" else m.name   # noqa: E731


def load_bands(w):
    if w == "nominal":
        parts = [os.path.join(D, "nominal.csv"), os.path.join(D, "nominal_extra.csv")]
    else:
        parts = [os.path.join(D, f"mass_{w}.csv")]
    parts = [p for p in parts if os.path.exists(p)]
    return pd.concat([pd.read_csv(p) for p in parts], ignore_index=True) if parts else None


BANDS = {w: load_bands(w) for w in ["nominal"] + WHERE}
BANDS = {w: v for w, v in BANDS.items() if v is not None}
out = {}


def val(w, g, lab, col="eta"):
    B = BANDS[w]
    s = B[(B.groove == g) & (B.label == lab)]
    return float(s[col].iloc[0]) if len(s) else np.nan


# (1) 공칭 띠별 분해
gs = [g for g in GROOVES if g in set(BANDS["nominal"].groove)]
print("### (1) 공칭 G3 — 홈 띠별 강성 몫 / 질량 몫 / 합, 깊이 0.02 mm, 주파수 변화 [%]\n")
print("| 극점 | FEM | 종류 | " + " | ".join(GNAME[g] for g in gs) + " |")
print("|---|---|---|" + "---|" * len(gs))
for m in JUDGED_FEM:
    cells = [f"{50 * val('nominal', g, m.fem, 'eta_K'):+.2f} / {50 * val('nominal', g, m.fem, 'eta_M'):+.2f} / "
             f"**{pc(val('nominal', g, m.fem)):+.2f}**" for g in gs]
    print(f"| {name(m)} | {m.fem} | {KO.get(m.kind, m.kind)} | " + " | ".join(cells) + " |")
print("\n깎은 질량 [mg]: " + ", ".join(f"{GNAME[g]} {val('nominal', g, JUDGED_FEM[0].fem, 'dm_mg'):.1f}" for g in gs))
out["nominal"] = {g: {m.fem: [50 * val('nominal', g, m.fem, 'eta_K'), 50 * val('nominal', g, m.fem, 'eta_M'),
                              pc(val('nominal', g, m.fem))] for m in JUDGED_FEM} for g in gs}

# (2) 건전 주파수 대 측정
meas = {}
for m in gl.MODES:
    if m.fem and (m.judge or m.kind == "hub"):
        meas.setdefault(m.fem, []).append(m.f)
meas = {k: float(np.mean(v)) for k, v in meas.items()}
print("\n### (2) 질량 차 4.23 g을 놓는 곳별 건전 주파수 — FEM 대 측정 [%] (판정·허브 극점, 쌍은 평균)\n")
fit = {}
for w in BANDS:
    f = BANDS[w][BANDS[w].groove == "xi0.9"].set_index("label").f_Hz
    e = {k: 100 * (f[k] / v - 1) for k, v in meas.items() if k in f.index}
    med = float(np.median(list(e.values())))
    lost = sorted(k for k, v in e.items() if abs(v - med) > 15.0)        # 이름표가 다른 모드로 옮겨 붙음(허브 질량 등)
    ev = np.array([v for k, v in e.items() if k not in lost])
    fit[w] = {"mean": float(ev.mean()), "sd": float(ev.std()), "n": len(ev), "lost": lost,
              "missing": len(meas) - len(e), "worst": max(((k, v) for k, v in e.items() if k not in lost),
                                                          key=lambda kv: abs(kv[1] - ev.mean()))}
    fit[w]["plausible"] = bool(w != "nominal" and fit[w]["sd"] <= RMS_OK and len(lost) == 0)
    print(f"- {KO_W[w]}: 평균 {fit[w]['mean']:+.2f} %, 평균 뺀 rms {fit[w]['sd']:.2f} % ({fit[w]['n']}개), "
          f"가장 어긋난 극점 {fit[w]['worst'][0]} {fit[w]['worst'][1]:+.1f} %"
          + (f", 이름표 어긋남 {', '.join(lost)}" if lost else "")
          + (" → **측정과 맞음**" if fit[w]["plausible"] else ""))
out["freq_fit"] = fit
PLAUS = [w for w in BANDS if fit[w]["plausible"]]

# (3) 곳별 예측과 부호
print(f"\n### (3) 곳별 예측(0.02 mm당 %) — 측정과 맞는 곳({', '.join(KO_W[w] for w in PLAUS)})과 전체\n")
sens = {}
for g in ("xi0.9", "rim2", "rim1", "xi0.3"):
    if not all(g in set(BANDS[w].groove) for w in PLAUS + ["nominal"]):
        continue
    print(f"**{GNAME[g]}**\n")
    print("| 극점 | 종류 | 공칭 | 측정과 맞는 곳 범위 | 모든 곳 범위 | 부호(맞는 곳) |")
    print("|---|---|---|---|---|---|")
    for m in JUDGED_FEM:
        nom = pc(val("nominal", g, m.fem))
        vp = np.array([pc(val(w, g, m.fem)) for w in PLAUS])
        va = np.array([pc(val(w, g, m.fem)) for w in BANDS if w != "nominal" and g in set(BANDS[w].groove)])
        keep = bool(np.all(np.sign(vp[np.isfinite(vp)]) == np.sign(nom)))
        sens[(g, m.fem)] = (nom, float(np.nanmin(vp)), float(np.nanmax(vp)), float(np.nanmin(va)), float(np.nanmax(va)), keep)
        print(f"| {name(m)} | {KO.get(m.kind, m.kind)} | {nom:+.3f} | {np.nanmin(vp):+.3f} – {np.nanmax(vp):+.3f} | "
              f"{np.nanmin(va):+.3f} – {np.nanmax(va):+.3f} | {'유지' if keep else '**바뀜**'} |")
    print()
    print("원판형 평균 − 패널 평균 [%p]: " + ", ".join(
        f"{KO_W[w]} {np.nanmean([pc(val(w, g, l)) for l in DISK]) - np.nanmean([pc(val(w, g, l)) for l in PANEL]):+.3f}"
        for w in ["nominal"] + PLAUS))
    print()
out["sens"] = {f"{k[0]}|{k[1]}": v for k, v in sens.items()}

# (4) 그림
RM = {}
for w in ["nominal"] + WHERE:
    p = os.path.join(D, "ring_map.csv" if w == "nominal" else f"ring_map_{w}.csv")
    if os.path.exists(p):
        R = pd.read_csv(p)
        R["per_mg_K"] = 50 * R.eta_K / R.dm_mg
        R["per_mg_M"] = 50 * R.eta_M / R.dm_mg
        R["per_mg"] = 50 * R.eta / R.dm_mg
        RM[w] = R
xi_s = {}
for w, R in RM.items():
    for lab in ("h2:m2n0", "h3:m3n1"):
        g = R[R.label == lab].sort_values("xi")
        s = np.sign(g.per_mg.values)
        k = np.nonzero(np.diff(s) > 0)[0]
        if len(k):
            i = k[-1]
            x0, x1, y0, y1 = g.xi.values[i], g.xi.values[i + 1], g.per_mg.values[i], g.per_mg.values[i + 1]
            xi_s[(w, lab)] = float(x0 - y0 * (x1 - x0) / (y1 - y0))
print("부호가 바뀌는 반지름(깎은 1 mg당 변화 = 0) ξ: " + "; ".join(
    f"{KO_W[w]} " + ", ".join(f"{lab} {xi_s[(w, lab)]:.3f}" for lab in ("h2:m2n0", "h3:m3n1") if (w, lab) in xi_s)
    for w in RM))
out["xi_sign_change"] = {f"{k[0]}|{k[1]}": v for k, v in xi_s.items()}

fig, axs = plt.subplots(2, 3, figsize=(16, 9))
COLW = {"nominal": "k", "all": "C0", "back": "C1", "vanes": "C2", "hub": "C4", "collar": "C5", "front": "C3", "rim": "C7"}
for col, (lab, ttl) in enumerate((("h2:m2n0", "h2 pair 2134/2142 (disk-type)"), ("h3:m3n1", "3748 h3:m3n1 (disk-type)"))):
    ax = axs[0, col]
    if "nominal" in RM:
        g = RM["nominal"][RM["nominal"].label == lab].sort_values("xi")
        ax.axvspan(0.8, 1.0, color="#d6604d", alpha=0.12); ax.axvspan(0.2, 0.4, color="#4393c3", alpha=0.12)
        ax.plot(g.xi, g.per_mg_K, "C0--", label="stiffness only")
        ax.plot(g.xi, g.per_mg_M, "C1--", label="mass only")
        ax.plot(g.xi, g.per_mg, "k-", lw=1.8, label="coupled")
        ax.axhline(0, color="0.5", lw=0.6)
        ax.set_title(f"({'ab'[col]}) {ttl}, nominal G3", fontsize=9, loc="left")
        ax.set_xlabel("ξ = (r − a)/(b − a)"); ax.set_ylabel("Δf/f per mg removed [%/mg]"); ax.legend(fontsize=7.5)
    ax = axs[1, col]
    ax.axvspan(0.8, 1.0, color="#d6604d", alpha=0.12); ax.axvspan(0.2, 0.4, color="#4393c3", alpha=0.12)
    for w, R in RM.items():
        g = R[R.label == lab].sort_values("xi")
        pl = w == "nominal" or fit.get(w, {}).get("plausible", False)
        ax.plot(g.xi, g.per_mg, color=COLW[w], lw=1.6 if pl else 0.9, ls="-" if pl else ":",
                label=f"{w}{'' if pl else ' (freq. mismatch)'}")
    ax.axhline(0, color="0.5", lw=0.6)
    ax.set_xlim(0.55, 1.0)
    yv = np.concatenate([R[(R.label == lab) & (R.xi >= 0.55)].per_mg.values for R in RM.values()])
    if len(yv):
        pad = 0.1 * (yv.max() - yv.min() + 1e-9)
        ax.set_ylim(yv.min() - pad, yv.max() + pad)
    ax.set_title(f"({'de'[col]}) {ttl}: coupled, missing 4.23 g placed at ...", fontsize=9, loc="left")
    ax.set_xlabel("ξ"); ax.set_ylabel("Δf/f per mg removed [%/mg]"); ax.legend(fontsize=7)
ax = axs[0, 2]
if "nominal" in RM:
    ax.axvspan(0.8, 1.0, color="#d6604d", alpha=0.12); ax.axvspan(0.2, 0.4, color="#4393c3", alpha=0.12)
    for m in JUDGED_FEM:
        g = RM["nominal"][RM["nominal"].label == m.fem].sort_values("xi")
        st = {"disk": "-", "panel": "--", "back": ":", "mixed": "-."}[m.kind]
        ax.plot(g.xi, g.per_mg, st, lw=1.2, label=f"{name(m)} {m.kind}")
    ax.axhline(0, color="0.5", lw=0.6)
    ax.set_title("(c) all judged poles, coupled, nominal G3", fontsize=9, loc="left")
    ax.set_xlabel("ξ"); ax.set_ylabel("Δf/f per mg removed [%/mg]"); ax.legend(fontsize=6.5, ncol=2)
ax = axs[1, 2]
ws = [w for w in BANDS]
x = np.arange(len(ws))
for k, (g, c) in enumerate((("xi0.9", "#d6604d"), ("rim2", "#b2182b"), ("xi0.3", "#4393c3"))):
    ys = [pc(val(w, g, "h2:m2n0")) if g in set(BANDS[w].groove) else np.nan for w in ws]
    ax.bar(x + (k - 1) * 0.27, ys, width=0.27, color=c, label={"xi0.9": "B ξ0.8–1.0", "rim2": "rim ξ0.9–1.0",
                                                                 "xi0.3": "C ξ0.2–0.4"}[g])
ax.axhline(0, color="0.5", lw=0.6)
ax.set_xticks(x); ax.set_xticklabels([f"{w}\n{'ok' if fit[w]['plausible'] else ('' if w == 'nominal' else 'mismatch')}"
                                      for w in ws], fontsize=7)
ax.set_ylabel("h2 pair Δf/f per 0.02 mm [%]")
ax.set_title("(f) h2 pair per band vs where the missing mass sits", fontsize=9, loc="left"); ax.legend(fontsize=7)
fig.suptitle("G3 FEM free-free, first-order: removing front-plate outer-face material (0.02 mm)", fontsize=10)
fig.tight_layout()
png = os.path.join(HERE, "..", "groove-test", "s3-km.png")
fig.savefig(png, dpi=125)
# (5) 형상 불일치만의 위치 편향: 같은 조립체 FEM 안에서 홈 띠 예측(1차)을 원고의 Gaussian 감육(폭 w = 3 mm)으로 역산.
#     Gaussian 깊이 = 반지름 지도 고리의 가중합(1차, 작은 심각도에서 깊이 ∝ 강성손실). 판정 극점(쌍은 하나)을 같은 무게로.
if "nominal" in RM:
    W_G = 3.0
    R = RM["nominal"]
    labs = [m.fem for m in JUDGED_FEM]
    rr = np.sort(R.r_mm.unique())
    V = np.array([[float(R[(R.r_mm == r) & (R.label == l)].eta.iloc[0]) for l in labs] for r in rr])   # (고리, 극점)
    a_mm, b_mm = 14.89, 36.465
    grid = np.linspace(0.05, 1.0, 381)
    print("\n### (5) 형상 불일치만의 편향 — 홈 띠(1차, 계획 깊이)를 Gaussian 감육으로 역산(같은 G3 조립체 FEM)\n")
    SIG_ETA = 2 * 0.05e-2                       # 주파수 0.05 %(다시 매달기 바닥) → 고유값 0.1 %
    inv = {}
    def invert(y):
        mk = np.isfinite(y)                          # 이름표를 잃은 극점은 빼고 역산
        y, Vm = y[mk], V[:, mk]
        chi = []
        for xd in grid:
            rd = a_mm + xd * (b_mm - a_mm)
            G = np.exp(-((rr - rd) / W_G) ** 2) @ Vm
            A = float(G @ y / (G @ G))
            chi.append(float(np.sum(((y - A * G) / SIG_ETA) ** 2)))
        chi = np.array(chi)
        i0 = int(np.argmin(chi))
        ok = grid[chi <= chi[i0] + 1.0]
        return float(grid[i0]), float(ok.min()), float(ok.max()), float(chi[i0] / (int(mk.sum()) - 2))

    # 응답을 만든 모델(공칭 = 형상 불일치만, 측정과 맞는 질량 분포 = 형상 + 모델 불일치)을 공칭 Gaussian 커널로 역산
    for g, k, true_xi in (("xi0.9", 3, "0.8–1.0"), ("xi0.3", 2, "0.2–0.4"), ("rim2", 3, "0.9–1.0")):
        for w in ["nominal"] + PLAUS:
            if g not in set(BANDS[w].groove):
                continue
            y = k * np.array([val(w, g, l) for l in labs])
            xh, lo, hi, c2 = invert(y)
            inv[f"{g}|{w}"] = {"xi_hat": xh, "xi_1sigma": [lo, hi], "chi2_per_dof": c2, "depth_factor": k}
            print(f"- {GNAME[g]}(참 ξ {true_xi}, 깊이 {0.02 * k:.2f} mm), 응답 모델 {KO_W[w]}: ξ̂ = {xh:.3f} "
                  f"(Δχ² ≤ 1: {lo:.3f}–{hi:.3f}), χ²/자유도 = {c2:.1f}")
    print(f"  (σ = 주파수 0.05 %, 판정 극점 {len(labs)}개, 역산 커널은 공칭 G3의 Gaussian w = 3 mm)")
    out["gauss_inverse"] = inv

# (6) 질량 더하기 시험 예측: B 띠(또는 테두리 띠)에 질량 m을 고르게 더하면 Δf/f = −(그 띠 질량 몫) × m / (띠가 깎일 때 질량)
print("\n### (6) 질량 더하기 시험 예측 — 띠에 150 mg를 고르게 붙일 때 [%] (질량 몫의 부호를 뒤집어 비례)\n")
mt = {}
for g in ("xi0.9", "rim2"):
    cols = [w for w in ["nominal"] + PLAUS if g in set(BANDS[w].groove)]
    if not cols:
        continue
    print(f"**{GNAME[g]}**: " + "; ".join(
        f"{name(m)} " + "/".join(f"{-50 * val(w, g, m.fem, 'eta_M') * 150 / val(w, g, m.fem, 'dm_mg'):+.2f}" for w in cols)
        for m in JUDGED_FEM) + f"  (순서: {', '.join(KO_W[w] for w in cols)})\n")
    mt[g] = {w: {m.fem: -50 * val(w, g, m.fem, "eta_M") * 150 / val(w, g, m.fem, "dm_mg") for m in JUDGED_FEM} for w in cols}
out["mass_test_150mg"] = mt
json.dump(out, open(os.path.join(D, "summary.json"), "w"), indent=1, ensure_ascii=False, default=str)
print(f"\n→ {png}")
