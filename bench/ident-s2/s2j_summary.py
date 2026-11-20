"""2번 시편 실측 그래프 정리(34번) — 그림 4장과 극점 표.
M1 FRF(10-28 ξ0.3·10-30 ξ0.6, 깃 6곳) + 극점 표시, M2 원주 차수·반지름 비·감쇠, M3 반복성(다시 매달기·무리 안 흩어짐),
M4 측정 대 FEM(표준 두께 0.40/0.60, 간격 3.00 mm, 질량 41.32 g 맞춤).
입력: out/s2/s2_avgs.npz·s2b_avgs.npz(깃별 평균 H), s2b_joint.json(공동 적합 극점·잔차), s2d_rehang_summary.json·s2d_h2.json
(다시 매달기), s2h_nominal_nom_ch300_32x16.json(FEM 짝)."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/tmp/claude-1000/-home-super-project/efb54ce3-bcf9-473d-9dc8-eb835c653517/scratchpad/impact1022/out/s2/"
HC = {0: "C0", 1: "C1", 2: "C2", 3: "C3"}
J = json.load(open(OUT + "s2b_joint.json"))
FEM = json.load(open(OUT + "s2h_nominal_nom_ch300_32x16.json"))
fe = {e["label"]: e for e in FEM["fe"]}
k6 = np.arange(6)
coef = lambda v, h, s=1: (v * np.exp(-1j * s * h * np.radians(60.0) * k6)).mean()


def hshare(v):
    A = np.array([abs(coef(v, h)) ** 2 for h in range(6)])
    P = np.array([A[0], A[1] + A[5], A[2] + A[4], A[3]])
    return P / P.sum()


# 짝: 확정(28번) + 후보(32번), 간격 3.00 mm FEM 이름
PAIR = {2133.80: ("h2:m2n0", 1), 2142.28: ("h2:m2n0", 1), 3163.65: ("h3:m3n0", 1), 3249.89: ("h0:m0n1", 1), 3458.89: ("h1:m1n1", 1),
        3588.78: (None, 0), 3638.99: ("h2:m4n0", 1), 3747.87: ("h3:m3n1", 1), 3804.07: ("h0:m6n1", 1), 4225.49: ("h1:m1n1#2", 1),
        4966.99: ("h2:m4n0#2", 1), 5617.22: ("h0:m6n0", 1), 5744.69: ("h1:m5n0", 1), 6438.86: ("h2:m2n1", 1), 6558.41: ("h2:m2n1", 2),
        6817.09: ("h1:m7n0", 2), 7093.13: ("h3:m3n2", 2), 7462.21: (None, 0), 7574.17: ("h2:m8n0", 2), 8321.24: ("h3:m3n1#2", 1),
        8824.79: (None, 0), 9961.03: (None, 0), 10098.27: (None, 0)}
meas_rho = {round(r["f"], 2): r for r in FEM["meas"]}
rows = []
for A, B in zip(J["A"], J["B"]):
    RA = np.array([complex(*x) for x in A["R"]]); RB = np.array([complex(*x) for x in B["R"]])
    P3, P6 = hshare(RB[:6]), hshare(RA[:6])
    f = A["f"]
    key = min(PAIR, key=lambda k: abs(k - f))
    lab, st = PAIR[key]
    m = meas_rho[min(meas_rho, key=lambda k: abs(k - f))]
    # 반지름 비(28번 s2b_radial과 같은 계산): ξ0.3 = 10-28 잔차 × 닫음 비 D(1번 깃, SNR ≥ 5일 때), ξ0.6 = 10-30 잔차, 거울(c_−h).
    # 차수는 FEM 짝이 있으면 그 짝의 차수(섞인 무리 3639 Hz는 h2 성분), 없으면 주된 차수.
    D = RA[6] / RB[0]
    sn = min(abs(RA[6]) / A["Rsd"][6], abs(RB[0]) / B["Rsd"][0])
    Du = D if sn >= 5 and abs(RB[0]) > 10 ** (-0.5) * np.abs(RB[:6]).max() else 1.0
    h = m["h"]
    hu = fe[lab]["h"] if lab else h
    rho = coef(RA[:6], hu, -1) / coef(RB[:6] * Du, hu, -1)
    Ab = np.array([[complex(*x) for x in row] for row in A["Rb"]]); Bb = np.array([[complex(*x) for x in row] for row in B["Rb"]])
    rb = np.array([coef(Ab[i, :6], hu, -1) / coef(Bb[i, :6] * Du, hu, -1) for i in range(min(len(Ab), len(Bb)))])
    dbw = np.percentile(np.abs(20 * np.log10(np.abs(rb / rho))), 90)
    row = dict(f=f, f_1028=B["f"], zeta=100 * A["z"], zeta_1028=100 * B["z"], h=h, h_used=hu, P3=P3, P6=P6,
               rho_dB=20 * np.log10(abs(rho)), rho_deg=np.degrees(np.angle(rho)), rho_w=dbw, label=lab, status=st)
    if lab:
        e = fe[lab]
        row.update(f_fe=e["f"], f_fm=e["fm"], err=100 * (f / e["fm"] - 1), fe_rho_dB=e["rho_dB"], fe_rho_deg=e["rho_deg"], T=e["T"])
    rows.append(row)
P = pd.DataFrame(rows)

# ---------------------------------------------------------------- M1 FRF
z28 = np.load(OUT + "s2_avgs.npz", allow_pickle=False)
z30 = np.load(OUT + "s2b_avgs.npz", allow_pickle=False)
fr = z28["fr"]
fig, axs = plt.subplots(2, 1, figsize=(10, 7.0), sharex=True)
vc = plt.cm.viridis(np.linspace(0, 0.92, 6))
band = (fr > 1900) & (fr < 10400)
ymax = max(np.max(20 * np.log10(np.abs(z[k][band]) + 1e-12)) for z in (z28, z30) for k in z.keys() if k.endswith("_H"))
for ax, z, keyf, ttl in [(axs[0], z28, "v{}_H", "(a) 10-28, back plate, vane line xi = 0.3 (r = 21.4 mm), six vanes"),
                         (axs[1], z30, "v{}x0.6_H", "(b) 10-30, back plate, vane line xi = 0.6 (r = 27.9 mm), six vanes")]:
    for v in range(1, 7):
        H = z[keyf.format(v)]
        ax.plot(fr / 1e3, 20 * np.log10(np.abs(H) + 1e-12), lw=0.7, color=vc[v - 1], label=f"vane {v}")
    for r in P.itertuples():
        ax.axvline(r.f / 1e3, color=HC[r.h], lw=0.6, alpha=0.5, ls=":" if r.status != 1 else "-")
    ax.set_ylabel("|H| [dB re 1 Pa/N]")
    ax.text(0.005, 0.97, ttl, transform=ax.transAxes, fontsize=8.5, ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.9))
    ax.set_ylim(-48, ymax + 4)
axs[0].legend(fontsize=7, ncol=6, loc="lower right")
for i, r in enumerate(P.itertuples()):
    axs[0].text(r.f / 1e3, ymax + 5 + (i % 2) * 7, f"{r.f:.0f}", rotation=90, fontsize=6, color=HC[r.h], ha="center", va="bottom", clip_on=False)
axs[1].set_xlabel("frequency [kHz]")
axs[1].set_xlim(1.9, 10.4)
for h in range(4):
    axs[1].plot([], [], color=HC[h], label=f"pole, h{h}")
axs[1].plot([], [], color="0.4", ls=":", label="pole without confirmed FEM pair")
axs[1].legend(fontsize=7, ncol=5, loc="lower right")
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(OUT + "s2j_frf.png", dpi=170)
plt.close(fig)

# ---------------------------------------------------------------- M2 차수·반지름 비·감쇠
Q = P[P.f < 8500].reset_index(drop=True)
x = np.arange(len(Q))
fig, axs = plt.subplots(3, 1, figsize=(10, 7.4), sharex=True, gridspec_kw={"height_ratios": [1.2, 1, 0.8]})
for k, (col, off) in enumerate([("P3", -0.19), ("P6", 0.19)]):
    bot = np.zeros(len(Q))
    for h in range(4):
        v = np.array([p[h] for p in Q[col]])
        axs[0].bar(x + off, v, bottom=bot, width=0.36, color=HC[h], edgecolor="white", lw=0.3, label=f"h{h}" if k == 0 else None)
        bot += v
axs[0].set_ylabel("circumferential\nshare")
axs[0].set_title("(a) circumferential order from six vanes: left bar xi = 0.3 (10-28), right bar xi = 0.6 (10-30)", fontsize=9, loc="left")
axs[0].legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.28))
axs[0].set_ylim(0, 1.0)
axs[1].errorbar(x, Q.rho_dB, yerr=Q.rho_w, fmt="o", color="k", ms=4, capsize=2, label="measured (90 % bootstrap)")
has = Q.label.notna()
axs[1].scatter(x[has], Q.fe_rho_dB[has], s=40, facecolors="none", edgecolors=[HC[h] for h in Q.h_used[has]], lw=1.4, label="FEM pair (x = candidate pair)", zorder=3)
for i in np.nonzero(has.to_numpy() & (Q.status == 2).to_numpy())[0]:
    axs[1].scatter(x[i], Q.fe_rho_dB[i], s=40, marker="x", color=HC[Q.h_used[i]], zorder=4)
axs[1].axhline(0, color="0.6", lw=0.6)
axs[1].set_ylabel("|rho| = xi0.6 / xi0.3\n[dB]")
axs[1].set_title("(b) radial ratio of the paired order (dominant order if unpaired; sign and phase in report 28)", fontsize=9, loc="left")
axs[1].legend(fontsize=7, loc="upper right")
axs[1].set_ylim(-33, 33)
axs[2].plot(x, Q.zeta, "o", color="k", ms=4, label="10-30 (xi = 0.6 fit)")
axs[2].plot(x, Q.zeta_1028, "o", mfc="none", color="0.45", ms=4, label="10-28 (xi = 0.3 fit)")
axs[2].set_ylabel("damping ratio [%]")
axs[2].set_ylim(0, 0.8)
axs[2].set_title("(c) modal damping", fontsize=9, loc="left")
axs[2].legend(fontsize=7, loc="upper left")
axs[2].set_xticks(x)
axs[2].set_xticklabels([f"{r.f:.0f}\n{r.label if isinstance(r.label, str) else '(none)'}" for r in Q.itertuples()], fontsize=6.2, rotation=55, ha="right")
fig.tight_layout()
fig.savefig(OUT + "s2j_modes.png", dpi=170)
plt.close(fig)

# ---------------------------------------------------------------- M3 반복성
R = json.load(open(OUT + "s2d_rehang_summary.json"))
H2 = json.load(open(OUT + "s2d_h2.json"))
G = ["10-28 17:08", "10-30 19:07 직후", "10-30 19:17 +10분", "10-30 19:28 +21분"]
GL = {"10-28 17:08": "10-28 (just after swap)", "10-30 19:07 직후": "re-hung, 0 min", "10-30 19:17 +10분": "re-hung, +10 min",
      "10-30 19:28 +21분": "re-hung, +21 min"}
GC = {"10-28 17:08": "0.55", "10-30 19:07 직후": "C0", "10-30 19:17 +10분": "C1", "10-30 19:28 +21분": "C2"}
modes = []
base2 = H2["10-30 15:55 전 ξ0.6|2"]
for j, nm in enumerate(["2134", "2142"]):
    d = {}
    for g, k in [("10-28 17:08", "10-28 17:08|2"), ("10-30 19:07 직후", "10-30 19:07 직후|2"), ("10-30 19:17 +10분", "10-30 19:17 +10분|2")]:
        e = H2[k]
        se = 100 * np.hypot(e["sd"][j] / np.sqrt(e["n"]), base2["sd"][j] / np.sqrt(base2["n"])) / base2["f"][j]
        d[g] = (100 * (e["f"][j] / base2["f"][j] - 1), se, 100 * e["sd"][j] / e["f"][j])
    d["10-30 19:28 +21분"] = (np.nan, np.nan, np.nan)
    modes.append((nm, d, 100 * base2["sd"][j] / base2["f"][j]))
for nm, dd in R.items():
    d = {g: (dd[g]["d_pct"], dd[g]["se_pct"], dd[g]["sd"]) for g in G if g in dd}
    modes.append((nm, d, dd["10-30 16:21 전"]["sd"]))
fig, axs = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True, gridspec_kw={"height_ratios": [1.4, 1]})
x = np.arange(len(modes))
for gi, g in enumerate(G):
    y = np.array([m[1].get(g, (np.nan,) * 3)[0] for m in modes]); s = np.array([m[1].get(g, (np.nan,) * 3)[1] for m in modes])
    axs[0].errorbar(x + (gi - 1.5) * 0.17, y, yerr=s, fmt="o", ms=4, capsize=2, color=GC[g], label=GL[g])
    sd = np.array([m[1].get(g, (np.nan,) * 3)[2] for m in modes])
    axs[1].plot(x + (gi - 1.5) * 0.17, sd, "o", ms=4, color=GC[g])
axs[1].plot(x, [m[2] for m in modes], "k_", ms=12, mew=2, label="before re-hanging (10-30 16:21; h2 pair: xi0.6 15:55)")
axs[0].axhspan(-0.05, 0.05, color="0.92", zorder=0)
axs[0].axhline(0, color="k", lw=0.6)
axs[0].set_ylabel("shift vs before\nre-hanging [%]")
axs[0].set_title("(a) frequency shift relative to 10-30 before re-hanging (20 N equivalent, +/- s.e.); grey band = +/-0.05 %", fontsize=9, loc="left")
axs[0].legend(fontsize=7, ncol=4, loc="lower left")
axs[0].set_ylim(-0.42, 0.22)
axs[1].set_ylabel("hit-to-hit sd [%]")
axs[1].set_title("(b) scatter between hits inside each group", fontsize=9, loc="left")
axs[1].legend(fontsize=7, loc="upper right")
axs[1].set_ylim(0, 0.3)
axs[1].set_xticks(x)
axs[1].set_xticklabels([m[0] + " Hz" for m in modes], fontsize=7, rotation=40, ha="right")
fig.tight_layout()
fig.savefig(OUT + "s2j_repeat.png", dpi=170)
plt.close(fig)

# ---------------------------------------------------------------- M4 측정 대 FEM
Q = P[P.label.notna()].reset_index(drop=True)
x = np.arange(len(Q))
fig, (ax, bx) = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True, gridspec_kw={"height_ratios": [1.5, 1]})
ax.axhspan(-2.5, 2.5, color="0.93", zorder=0)
for i, r in enumerate(Q.itertuples()):
    ax.plot(i, 100 * (r.f / r.f_fe - 1), "x", color="0.55", ms=5)
    hh = fe[r.label]["h"]
    ax.plot(i, r.err, "o", ms=7, mfc=HC[hh] if r.status == 1 else "white", mec=HC[hh], mew=1.5)
for h in range(4):
    ax.plot([], [], "o", color=HC[h], label=f"FEM h{h}")
ax.plot([], [], "o", mfc="white", mec="k", label="candidate pair")
ax.plot([], [], "x", color="0.55", label=f"FEM without missing mass ({FEM['mass']:.2f} g)")
ax.axhline(0, color="k", lw=0.6)
ax.set_ylabel("measured / FEM - 1 [%]")
ax.set_ylim(-9, 7)
ax.legend(fontsize=7, ncol=3, loc="upper right")
ax.set_title(f"(a) FEM: front 0.40, back 0.60, gap 3.00, vane 1.0 mm; missing {41.3186 - FEM['mass']:.2f} g spread like the FEM mass "
             "(x 0.9473); grey band = +/-2.5 %", fontsize=8.5, loc="left")
bot = np.zeros(len(Q))
for j, (k, c) in enumerate([("front plate", "C0"), ("back plate", "C3"), ("hub", "C4"), ("vanes", "C1"), ("inlet collar", "C2")]):
    v = np.array([t[j] for t in Q["T"]])
    bx.bar(x, v, bottom=bot, color=c, width=0.6, label=k)
    bot += v
bx.set_ylabel("FEM kinetic share")
bx.set_ylim(0, 1.28)
bx.legend(fontsize=7, ncol=5, loc="upper center")
bx.set_title("(b) FEM kinetic-energy share by part", fontsize=8.5, loc="left")
bx.set_xticks(x)
bx.set_xticklabels([f"{r.f:.0f}\n{r.label}" for r in Q.itertuples()], fontsize=6.3, rotation=50, ha="right")
fig.tight_layout()
fig.savefig(OUT + "s2j_fem.png", dpi=170)
plt.close(fig)

# ---------------------------------------------------------------- 표
T = P.copy()
for h in range(4):
    T[f"h{h}_x03"] = [p[h] for p in P.P3]
    T[f"h{h}_x06"] = [p[h] for p in P.P6]
T = T.drop(columns=["P3", "P6", "T"])
T.to_csv(OUT + "s2j_poles.csv", index=False)
q = P[P.label.notna() & (P.status == 1)]
print(T[["f", "f_1028", "zeta", "zeta_1028", "h", "rho_dB", "rho_w", "label", "status", "err"]].round(3).to_string())
print("확정 짝 측정/FEM−1: 범위 %.1f~%.1f %%" % (q.err.min(), q.err.max()))
