"""그림: (a) 타격별 20 N 주파수 대 시각(10-28 ξ0.3, 10-30 ξ0.6 + 닫음 점검), (b) 깃 6곳 패턴 ξ0.3·ξ0.6(같은 회전),
(c) 반지름 비 ρ = c_h(ξ0.6)/c_h(ξ0.3) 측정 대 FE, (d) FE 대비 주파수 차 대 FE 앞판 에너지 비."""
import sys, os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT, FED, PNG = sys.argv[1], sys.argv[2], sys.argv[3]
reg = pd.read_csv(os.path.join(FED, "rev1_n6_modes_ff32x16h6w62c40.csv")).set_index("label")
Phi = np.load(os.path.join(OUT, "s2", "s2b_fe_phi.npy"))
J = json.load(open(os.path.join(OUT, "s2", "s2b_joint.json")))
new = json.load(open(os.path.join(OUT, "s2", "s2b_track.json"))); old = json.load(open(os.path.join(OUT, "s2", "s2track.json")))
SL = {"3458": -0.00032, "3638": -0.00051, "4221": -0.00050, "7453": -0.00071}
k6 = np.arange(6); coef = lambda v, h: (v * np.exp(-1j * h * np.radians(60.0) * k6)).mean()
coefm = lambda v, h: (v * np.exp(1j * h * np.radians(60.0) * k6)).mean()
fig = plt.figure(figsize=(12.5, 9.2))
gs = fig.add_gridspec(2, 4, hspace=0.38, wspace=0.42)
# (a)
axL = fig.add_subplot(gs[0, 0]); axR = fig.add_subplot(gs[0, 1], sharey=axL)
cols = dict(zip(SL, ["C0", "C1", "C2", "C3"]))
for name, sl in SL.items():
    ref = np.median([r["f"][name][0] * (1 - sl * np.log2(r["N"] / 20)) for r in new if r["f"].get(name)])
    for rows, ax, t0 in ((old, axL, 17 * 3600), (new, axR, 15 * 3600 + 50 * 60)):
        t = np.array([(r["t"] - t0) / 60 for r in rows if r["f"].get(name)])
        y = np.array([100 * (r["f"][name][0] * (1 - sl * np.log2(r["N"] / 20)) / ref - 1) for r in rows if r["f"].get(name)])
        x3 = np.array([r.get("xi", 0.3) == 0.3 for r in rows if r["f"].get(name)])
        ok = np.abs(y) < 0.6
        ax.plot(t[ok & ~x3], y[ok & ~x3], "o", ms=2.5, color=cols[name], alpha=0.6, label=name + " Hz" if ax is axL else None)
        ax.plot(t[ok & x3], y[ok & x3], "s" if ax is axR else "o", ms=2.5 if ax is axL else 4, color=cols[name], alpha=0.6, mfc="none" if ax is axR else None)
for ax, rows, t0, lab in ((axL, old, 17 * 3600, "10-28 17:xx, xi0.3 vanes 1-6"), (axR, new, 15 * 3600 + 50 * 60, "10-30 from 15:50, xi0.6 vanes 1-6 + v1 xi0.3")):
    vv = sorted({(r["v"], r.get("xi", 0.3)) for r in rows}, key=lambda k: min(r["t"] for r in rows if (r["v"], r.get("xi", 0.3)) == k))
    for v, x in vv:
        ts = [(r["t"] - t0) / 60 for r in rows if (r["v"], r.get("xi", 0.3)) == (v, x)]
        ax.axvspan(min(ts), max(ts), color="0.92" if v % 2 else "0.97", zorder=0)
        ax.text(0.5 * (min(ts) + max(ts)), -0.385, f"v{v}" + ("*" if x == 0.3 and ax is axR else ""), ha="center", fontsize=7)
    ax.axhline(0, color="k", lw=0.6); ax.set_xlabel("minutes"); ax.set_title(lab, fontsize=8)
axL.set_ylim(-0.4, 0.38); axL.set_ylabel("f (20 N) rel. to 10-30 median [%]"); axL.legend(fontsize=6.5, loc="lower right")
axL.set_xlabel("minutes after 17:00"); axR.set_xlabel("minutes after 15:50  (* = closure v1 xi0.3)")
axL.text(0.02, 0.97, "(a)", transform=axL.transAxes, va="top", fontweight="bold")
axL.text(0.03, 0.80, "air ai1 22.9 C\nai0 21.9 C", transform=axL.transAxes, fontsize=6.5, bbox=dict(fc="w", ec="none", alpha=0.8))
axR.text(0.03, 0.80, "air ai1 20.0-20.5 C\nai0 21.4 C", transform=axR.transAxes, fontsize=6.5, bbox=dict(fc="w", ec="none", alpha=0.8))
# (b) 패턴
ax = fig.add_subplot(gs[0, 2:])
SHOW = [(2133.80, "h2 2134"), (3163.65, "h3 3164"), (3249.89, "h0 3250"), (3804.07, "h0 3804"), (4225.49, "h1 4225")]
for i, (fm, lab) in enumerate(SHOW):
    A = min(J["A"], key=lambda a: abs(a["f"] - fm)); B = J["B"][J["A"].index(A)]
    RA = np.array([complex(*x) for x in A["R"]]); RB = np.array([complex(*x) for x in B["R"]])
    D = RA[6] / RB[0]; sn = min(abs(RA[6]) / A["Rsd"][6], abs(RB[0]) / B["Rsd"][0]); D = D if sn >= 5 and abs(RB[0]) > 10 ** (-0.5) * np.abs(RB[:6]).max() else 1.0
    x = np.concatenate([RB[:6] * D, RA[:6]])
    lam, vec = np.linalg.eigh(np.array([[x.real @ x.real, x.real @ x.imag], [x.real @ x.imag, x.imag @ x.imag]]))
    al = np.arctan2(vec[1, 1], vec[0, 1]); a = (x * np.exp(-1j * al)).real; a = a / np.abs(a).max() * np.sign(a[np.argmax(np.abs(a))])
    xs = np.arange(6) + 7 * i
    ax.plot(xs, a[:6], "o-", color="C0", ms=4, label="xi0.3 (10-28, x closure D)" if i == 0 else None)
    ax.plot(xs, a[6:], "s--", color="C3", ms=4, label="xi0.6 (10-30)" if i == 0 else None)
    ax.text(xs.mean(), 1.12, lab, ha="center", fontsize=8)
ax.axhline(0, color="k", lw=0.6); ax.set_ylim(-1.15, 1.25); ax.set_xticks(np.concatenate([np.arange(6) + 7 * i for i in range(len(SHOW))]))
ax.set_xticklabels([str(v) for _ in SHOW for v in range(1, 7)], fontsize=6.5); ax.set_xlabel("vane"); ax.set_ylabel("residual (common rotation, max = 1)")
ax.legend(fontsize=7, loc="lower right"); ax.text(0.01, 0.03, "(b)", transform=ax.transAxes, va="bottom", fontweight="bold")
# (c) 반지름 비
PAIRS = [(2133.80, "h2:m2n0", 2), (3163.65, "h3:m3n0", 3), (3249.89, "h0:m0n1", 0), (3458.89, "h1:m1n1", 1), (3638.99, "h2:m4n0", 2), (3747.87, "h3:m3n1", 3),
         (3804.07, "h0:m6n1", 0), (4225.49, "h1:m1n1#2", 1), (4966.99, "h2:m4n1", 2), (5617.22, "h0:m6n0", 0), (5744.69, "h1:m5n1", 1), (6438.86, "h2:m2n1", 2),
         (7462.21, "h2:m8n2", 2), (8321.24, "h3:m3n1#2", 3)]
labs = list(reg.index); mem = {l: [int(i) for i in str(reg.loc[l, "members"]).split("|")] for l in labs}
axc = fig.add_subplot(gs[1, 0:2]); axd = fig.add_subplot(gs[1, 2:])
res = []
for fm, lab, h in PAIRS:
    A = min(J["A"], key=lambda a: abs(a["f"] - fm)); B = J["B"][J["A"].index(A)]
    RA = np.array([complex(*x) for x in A["R"]]); RB = np.array([complex(*x) for x in B["R"]])
    D = RA[6] / RB[0]; sn = min(abs(RA[6]) / A["Rsd"][6], abs(RB[0]) / B["Rsd"][0]); D = D if sn >= 5 and abs(RB[0]) > 10 ** (-0.5) * np.abs(RB[:6]).max() else 1.0
    rm = coefm(RA[:6], h) / coefm(RB[:6] * D, h)                      # 깃 번호가 FE −θ 방향(사진: 번호·깃 휨 모두 뒤판에서 시계 방향)
    rf = coef(Phi[mem[lab][0], 6:], h) / coef(Phi[mem[lab][0], :6], h)
    sm, sf = np.cos(np.angle(rm)) > 0, np.cos(np.angle(rf)) > 0
    res.append((fm, lab, h, rm, rf, reg.loc[lab, "f_Hz"], reg.loc[lab, "Efrac_front"]))
    c = "C2" if sm == sf else "C3"
    axc.plot(20 * np.log10(abs(rf)), 20 * np.log10(abs(rm)), "o" if sm else "^", color=c, ms=7, mfc=c if sm else "none")
    axc.annotate(f"{fm:.0f}", (20 * np.log10(abs(rf)), 20 * np.log10(abs(rm))), fontsize=6.5, xytext=(3, 3), textcoords="offset points")
axc.plot([-30, 30], [-30, 30], "k:", lw=0.8); axc.set_xlim(-28, 22); axc.set_ylim(-28, 30)
axc.set_xlabel("FE |rho| [dB]  (rho = c_h(xi0.6)/c_h(xi0.3), vane order = FE -theta)"); axc.set_ylabel("measured |rho| [dB]")
axc.plot([], [], "o", color="C2", label="same sign (meas.), FE agrees"); axc.plot([], [], "^", color="C2", mfc="none", label="opposite sign (meas.), FE agrees")
axc.plot([], [], "o", color="C3", label="sign disagrees with FE"); axc.legend(fontsize=7, loc="upper left")
axc.text(0.98, 0.03, "(c)", transform=axc.transAxes, ha="right", fontweight="bold")
for fm, lab, h, rm, rf, fF, ef in res:
    e = 100 * (fm / fF - 1)
    axd.plot(ef, e, "o", color=f"C{h}", ms=7, mfc="none" if lab in ("h2:m4n0",) else f"C{h}")
    axd.annotate(f"{lab} {fm:.0f}", (ef, e), fontsize=6.3, xytext=(3, 2), textcoords="offset points")
for h in range(4):
    axd.plot([], [], "o", color=f"C{h}", label=f"h{h}")
axd.axhline(0, color="k", lw=0.6); axd.set_xlabel("FE strain-energy share in front plate"); axd.set_ylabel("measured / FE - 1 [%]")
axd.legend(fontsize=7, loc="lower right", ncol=4); axd.text(0.02, 0.97, "(d)", transform=axd.transAxes, va="top", fontweight="bold")
fig.savefig(PNG, dpi=150, bbox_inches="tight")
ef = np.array([r[6] for r in res]); er = np.array([100 * (r[0] / r[5] - 1) for r in res])
print("corr(front share, err) =", np.corrcoef(ef, er)[0, 1].round(2), "n =", len(res))
for fm, lab, h, rm, rf, fF, e_ in res:
    print(f" {fm:8.2f} {lab:10s} FE {fF:7.1f} err {100*(fm/fF-1):+5.1f}% | rho meas {20*np.log10(abs(rm)):+5.1f} dB {np.degrees(np.angle(rm)):+5.0f} | FE {20*np.log10(abs(rf)):+5.1f} dB {np.degrees(np.angle(rf)):+5.0f} | front {e_:.2f}")
