"""그림: 측정 구성 변경 전후 스펙트럼과 최저 짝 주파수의 시간 변화."""
import sys, os, json
sys.path.insert(0, sys.argv[1])
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from w22joint import vane_avg

OUT = sys.argv[2]
O = np.load(os.path.join(OUT, "w22_spec.npz")); T = np.load(os.path.join(OUT, "w28_spec.npz")); Z = np.load(os.path.join(OUT, "s2", "s2_spec.npz"))
fr = Z["fr"]


def vsum(D, mask):
    H = vane_avg(D["F"][mask].astype(complex), D["X"][mask].astype(complex), D["vane"][mask])
    return sum(np.abs(H[v]) ** 2 for v in range(1, 7) if np.isfinite(H[v]).all())


Po = vsum(O, np.ones(O["N"].size, bool))
mt = (~T["late"]) & (T["N"] >= 5); Pt = vsum(T, mt)
mz = (~Z["late"]) & (Z["N"] >= 5); Pz = vsum(Z, mz)
fig = plt.figure(figsize=(13, 9))
ax1 = fig.add_subplot(2, 2, (1, 2)); ax2 = fig.add_subplot(2, 2, 3); ax3 = fig.add_subplot(2, 2, 4)
for P, lab, c, off in ((Po, "10-26 20:00–20:29 far mic, vanes 1–6 ξ0.3", "C1", 1e2), (Pt, "10-28 16:34–16:52 far mic, vane 1 ξ0.3·0.6, vane 2 ξ0.6", "C2", 1e1),
                       (Pz, "10-28 17:08–17:41 mic at the back plate, vanes 1–6 ξ0.3", "C0", 1.0)):
    s = (fr >= 1900) & (fr <= 4600)
    ax1.semilogy(fr[s], P[s] / P[s].max() * off, c, lw=0.8, label=lab)
    s2 = (fr >= 2000) & (fr <= 2200)
    ax2.semilogy(fr[s2], P[s2] / P[s].max() * off, c, lw=0.9)
for f0, t in ((2041, "2041"), (2062, "2062"), (2131.5, "2131.5"), (2140.4, "2140.4")):
    ax2.axvline(f0, color="k", lw=0.5, ls=":"); ax2.text(f0, 3e2, t, rotation=90, fontsize=7, va="top")
ax1.set_xlim(1900, 4600); ax1.set_xlabel("frequency [Hz]"); ax1.set_ylabel("Σ|H1|² over vanes (normalised, offset)"); ax1.legend(fontsize=8, loc="lower right")
ax1.set_title("(a) vane-summed FRF: the 10-26 and 16:34–16:52 sets share their peaks; the 17:08 set does not", fontsize=9)
ax2.set_xlim(2000, 2200); ax2.set_xlabel("frequency [Hz]"); ax2.set_title("(b) lowest pair: 2041/2062 Hz → 2131.5/2140.4 Hz (+4 %)", fontsize=9)
# (c) 최저 짝 타격별 주파수(20 N 환산 없이 원시) — 세 세션
old = json.load(open(os.path.join(OUT, "w22d_hits.json"))); t28 = json.load(open(os.path.join(OUT, "w28b_hits.json"))); s2 = json.load(open(os.path.join(OUT, "s2", "s2track.json")))
sec = lambda s: int(s[:2]) * 60 + int(s[3:5]) + int(s[6:8]) / 60
xs = []
for r in old:
    for k, mk in (("2041 h2", "o"), ("2062 h2", "s")):
        if r["f"][k] and r["v"] != 2:
            ax3.plot(sec(r["ts"]) - 20 * 60, r["f"][k], mk, ms=3, color="C1", alpha=0.7)
for r in t28:
    for k, mk in (("2041 h2", "o"), ("2062 h2", "s")):
        if r["f"][k]:
            ax3.plot(sec(r["ts"]) - 16 * 60 - 30 + 40, r["f"][k], mk, ms=3, color="C2", alpha=0.7)
for r in s2:
    for k, mk in (("2132", "o"), ("2141", "s")):
        if r["f"][k]:
            ax3.plot(sec(r["ts"]) - 16 * 60 - 30 + 40, r["f"][k][0], mk, ms=3, color="C0", alpha=0.7)
ax3.axvline(36, color="k", lw=0.6); ax3.text(37, 2150, "10-28", fontsize=8); ax3.text(1, 2150, "10-26", fontsize=8); ax3.axvline(70, color="k", lw=0.6, ls="--"); ax3.text(71, 2100, "configuration\nchanged", fontsize=7)
ax3.set_xticks([0, 10, 20, 30, 44, 62, 78, 90, 101])
ax3.set_xticklabels(["20:00", "20:10", "20:20", "20:30", "16:34", "16:52", "17:08", "17:20", "17:31"], fontsize=7)
ax3.set_xlim(-2, 115); ax3.set_ylim(2030, 2155); ax3.set_ylabel("per-hit pole [Hz]")
ax3.set_title("(c) lowest pair per hit: 10-26 (orange), 10-28 far mic (green), 10-28 mic at plate (blue)", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "s2", "s2_change.png"), dpi=120)
print("saved")
