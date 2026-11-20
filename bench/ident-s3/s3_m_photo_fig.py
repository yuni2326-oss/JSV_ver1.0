"""그림 m-photo.png(46번 §7): 땜납 조각 실제 자리와 FEM 비교.
(a) 사진 1721(M240, 종이 위)을 핀홀 보정으로 판 위에서 본 모습으로 펴고, 그린 깃 선(빨강)·조각(노랑)·도안 자리(파랑 20°, 초록 40°)를 겹친다.
(b) 측정/FEM 비 — 도안 자리(빈 표시)와 사진 자리 + 공통 어긋남 ε(채운 표시), 1단계(M120)·2단계(M240 증분).
입력: groove-test/photos/*.jpg, groove-test/out/m_photo.json(s3_m_photo.py), m_piece_fit.json(s3_m_piece_scan.py), m_angle_fit.json."""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s3_m_photo as mp  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
GT = os.path.join(HERE, "..", "groove-test")
NAME = "20261112_1721_M240_on_paper"
META = json.load(open(os.path.join(GT, "out", "m_photo.json")))[NAME]
FIT = json.load(open(os.path.join(GT, "out", "m_piece_fit.json")))
AF = json.load(open(os.path.join(GT, "out", "m_angle_fit.json")))
pose = np.array(META["pose"])
import pandas as pd  # noqa: E402
POS = pd.read_csv(os.path.join(GT, "m_piece_positions.csv"))

# (a) 판 위에서 본 모습(0.05 mm 격자)
im = np.asarray(ImageOps.exif_transpose(Image.open(os.path.join(GT, "photos", f"{NAME}.jpg"))).convert("RGB"), float)
h = 0.05
xs = np.arange(-37.5, 37.5, h)
X, Y = np.meshgrid(xs, -xs)
u, v = mp.plate_to_img(X.ravel(), Y.ravel(), pose)
rect = np.stack([ndimage.map_coordinates(im[:, :, c], [v, u], order=1, cval=255).reshape(X.shape) for c in range(3)], -1) / 255
fig = plt.figure(figsize=(13, 6.6))
ax = fig.add_axes([0.02, 0.06, 0.44, 0.86])
bx = fig.add_axes([0.53, 0.12, 0.45, 0.78])
ax.imshow(np.clip(rect, 0, 1), extent=[xs[0], xs[-1], -xs[-1], -xs[0]], origin="upper")
rs = np.linspace(26.0, 36.4, 30)
for vn, L in META["lines"].items():
    th = np.radians(L["th34"] + L["slope"] * (rs - 34.0))
    ax.plot(rs * np.cos(th), rs * np.sin(th), color="r", lw=1.6)
    t34 = L["th34"]
    ax.text(37.6 * np.cos(np.radians(t34 + 6)), 37.6 * np.sin(np.radians(t34 + 6)), f"vane {vn}", color="r", fontsize=7.5,
            ha="center", va="center")
    for a_, col in ((20.0, "C0"), (40.0, "C2")):
        t = np.radians(t34 + a_)
        ax.plot(34.3 * np.cos(t), 34.3 * np.sin(t), "o", mfc="none", mec=col, ms=9, mew=1.6)
d = POS[POS.photo == NAME]
for _, r in d.iterrows():
    L = META["lines"][r.bay.split("-")[0]]
    t = np.radians(L["th34"] + L["slope"] * (r.r_mm - 34.0) + r.ang_from_k)
    ax.plot(r.r_mm * np.cos(t), r.r_mm * np.sin(t), "o", color="gold", mec="k", mew=0.6, ms=6)
tt = np.linspace(0, 2 * np.pi, 300)
for rr_, ls in ((36.46, "-"), (34.3, ":")):
    ax.plot(rr_ * np.cos(tt), rr_ * np.sin(tt), color="m" if ls == "-" else "0.4", lw=0.8, ls=ls)
ax.set_xlim(-39, 39); ax.set_ylim(-39, 39); ax.set_aspect("equal"); ax.axis("off")
ax.plot([], [], "o", color="gold", mec="k", label="solder piece (measured)")
ax.plot([], [], "o", mfc="none", mec="C0", label="template 20° (M120)")
ax.plot([], [], "o", mfc="none", mec="C2", label="template 40° (M240)")
ax.plot([], [], color="r", label="drawn vane line (outer part, fitted)")
ax.legend(fontsize=7, loc="lower left", bbox_to_anchor=(0.0, -0.05), ncol=2, frameon=True)
p120 = np.mean([a for a, _ in FIT["positions"]["M120"]]); p240 = np.mean([a for a, _ in FIT["positions"]["M240"]])
ax.set_title(f"(a) photo 17:21 (M240) rectified to the plate plane: pieces sit at {p120:.1f}° and {p240:.1f}°\n"
             f"from the drawn line of vane k (template 20° and 40°), r ≈ 34 mm", fontsize=8.5, loc="left")

# (b) 비
KEYS = AF["keys"]
lab = {"h2": "h2 pair"}
x = np.arange(len(KEYS))
T25 = FIT["cases"]["template|25.0"]; E = FIT["eps"]
for j, (meas, tp, ph, col, nm) in enumerate(((AF["M120"], T25["pred_M120"], E["pred_M120"], "C3", "step 1 (M120)"),
                                             (AF["INC"], T25["pred_inc"], E["pred_inc"], "C0", "step 2 (M240 − M120)"))):
    rt = [meas[k] / tp[k] if abs(tp[k]) >= 0.3 else np.nan for k in KEYS]
    rp = [meas[k] / ph[k] if abs(ph[k]) >= 0.3 else np.nan for k in KEYS]
    off = (j - 0.5) * 0.3
    bx.plot(x + off, rt, "o", mfc="none", mec=col, ms=7, mew=1.4, label=f"{nm}: FEM at template positions")
    bx.plot(x + off, rp, "o", color=col, ms=6, label=f"{nm}: FEM at photo positions (ε {E['best']:+.1f}°)")
    for xi, a_, b_ in zip(x + off, rt, rp):
        if np.isfinite(a_) and np.isfinite(b_):
            bx.annotate("", xy=(xi, b_), xytext=(xi, a_), arrowprops=dict(arrowstyle="->", color=col, lw=0.8, alpha=0.6))
bx.axhspan(0.7, 1.3, color="0.92", zorder=0)
bx.axhline(1, color="k", lw=0.6)
bx.set_xticks(x); bx.set_xticklabels([lab.get(k, k) for k in KEYS], fontsize=8, rotation=45)
bx.set_ylim(0, 2.0)
bx.set_ylabel("measured / FEM", fontsize=8.5)
bx.legend(fontsize=7, loc="upper left", frameon=False)
c_t = T25["chi_M120"] + T25["chi_inc"]; c_p = E["chi_M120"] + E["chi_inc"]
bx.set_title(f"(b) measured / FEM per mode (solder piece 4 × 1 mm, 20 mg; |FEM| ≥ 0.3 % only)\n"
             f"χ² over 11 modes × 2 steps: template {c_t:.0f} → photo positions {c_p:.0f}; grey = 0.7–1.3", fontsize=8.5, loc="left")
fig.savefig(os.path.join(GT, "m-photo.png"), dpi=130)
print("saved")
