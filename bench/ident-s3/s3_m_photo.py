"""질량 띠(땜납 조각) 자리를 사진에서 잰다(46번 §7, 11-12).

사진(groove-test/photos): 1640 = M120 상태로 매단 것(조각 6개), 1721 = M240 상태로 종이 위(조각 12개).
둘 다 갤럭시 S24 울트라 초광각(EXIF 35 mm 환산 13 mm, 약 5 cm 거리)이라 원근이 강하다 → 핀홀 모형으로 판 평면을 편다.
1) 앞판 테두리: 중심에서 반지름 방향 밝기 경계 → 강건한 타원 맞춤(잔차 큰 점 제외).
   종이 위 사진은 가까운 쪽(아래)에 판 옆면이 보여 그쪽 점을 뺀다.
2) 판 자세: 초점 f = 13·5000/43.27 px, 주점 = 사진 가운데. 판 위 반지름 36.46 mm 원이 테두리 점에 맞도록 기울기·방향·거리·중심의 상을 맞춘다.
3) 조각: 손으로 고른 근사 위치 둘레(반지름 80 px)에서 주변 중앙값과 30 이상 다른 화소의 가장 큰 덩어리 → 판 좌표 무게중심·주축.
4) 그린 깃 선(사용자가 용접 점을 잇고 테두리까지 그은 검은 선): 바깥 부분에 손으로 고른 점 3개를 잉크 중심선으로 다듬고
   θ(r) 1차식으로 맞춘다. 조각 각은 그 조각 반지름에서 잰다.
   안쪽 용접 점은 기준으로 쓰지 않는다 — 판이 조금만 접시 모양이어도 안쪽 점이 평면 가정에서 벗어난다(6회 대칭 잔차 약 6°).
출력: groove-test/m_piece_positions.csv(조각마다), groove-test/photos/<사진>_measure.png(확인 그림), groove-test/out/m_photo.json
사용: python s3_m_photo.py"""
import json
import os

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
from scipy import ndimage, optimize

HERE = os.path.dirname(os.path.abspath(__file__))
GT = os.path.join(HERE, "..", "groove-test")
R_RIM = 36.46
F_PX = 13.0 * 5000 / 43.27                      # EXIF 35 mm 환산 13 mm, 대각 5000 px
PP = np.array([1500.0, 2000.0])                 # 바로 세운 3000×4000 사진의 가운데
PHOTOS = {
    "20261112_1640_M120_hanging": dict(
        state="M120", off=(640, 820), c0=(1686.0, 1880.0), r0=1014.0, edge="abs", near_side=None,
        pieces=[(515, 45), (925, 290), (880, 822), (440, 1000), (65, 690), (115, 262)],
        lines={6: [(240, 177), (200, 172), (165, 168)], 5: [(668, 130), (655, 95), (642, 60)], 4: [(912, 495), (940, 458), (962, 428)],
               3: [(700, 893), (750, 905), (800, 913)], 2: [(288, 880), (305, 930), (322, 985)], 1: [(95, 520), (72, 548), (50, 575)]}),
    "20261112_1721_M240_on_paper": dict(
        state="M240", off=(440, 1240), c0=(1482.0, 2258.0), r0=1010.0, edge="rise", near_side="bottom",
        pieces=[(432, 58), (275, 108), (675, 60), (818, 147), (985, 385), (990, 580), (835, 855), (665, 952), (305, 925),
                (170, 810), (55, 495), (85, 335)],
        lines={3: [(575, 80), (548, 58), (527, 42)], 2: [(880, 285), (893, 240), (903, 200)], 1: [(870, 703), (915, 695), (950, 688)],
               6: [(475, 905), (500, 940), (530, 980)], 5: [(122, 600), (100, 640), (80, 680)], 4: [(230, 215), (190, 225), (160, 232)]}),
}


def rot(alpha, beta):
    a = np.array([np.cos(beta), np.sin(beta), 0.0])
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(alpha) * K + (1 - np.cos(alpha)) * K @ K


def img_to_plate(uv, p):
    """사진 점 → 판 좌표(mm, 사진에서 보이는 그대로 반시계 = +각)."""
    alpha, beta, D, u0, v0 = p
    R = rot(alpha, beta)
    t = np.array([(u0 - PP[0]) / F_PX, (v0 - PP[1]) / F_PX, 1.0]); T = D * t / np.linalg.norm(t)
    n = R[:, 2]
    uv = np.atleast_2d(np.asarray(uv, float))
    d = np.column_stack([(uv[:, 0] - PP[0]) / F_PX, (uv[:, 1] - PP[1]) / F_PX, np.ones(len(uv))])
    s = (n @ T) / (d @ n)
    P = (d * s[:, None] - T) @ R
    return P[:, 0], -P[:, 1]


def plate_to_img(x, y, p):
    alpha, beta, D, u0, v0 = p
    R = rot(alpha, beta)
    t = np.array([(u0 - PP[0]) / F_PX, (v0 - PP[1]) / F_PX, 1.0]); T = D * t / np.linalg.norm(t)
    Xc = np.column_stack([x, -np.asarray(y), np.zeros(len(x))]) @ R.T + T
    return Xc[:, 0] / Xc[:, 2] * F_PX + PP[0], Xc[:, 1] / Xc[:, 2] * F_PX + PP[1]


def rim_points(g, c0, r0, edge):
    sm = ndimage.gaussian_filter(g, 2.5)
    H, W = g.shape
    pts = []
    for a in np.radians(np.arange(0, 360, 2.5)):
        rr = np.arange(int(0.82 * r0), int(1.12 * r0))
        x = c0[0] + rr * np.cos(a); y = c0[1] - rr * np.sin(a)
        ok = (x > 1) & (x < W - 2) & (y > 1) & (y < H - 2)
        v = ndimage.map_coordinates(sm, [y[ok], x[ok]], order=1)
        gr = np.gradient(v)
        i = int(np.argmax(gr if edge == "rise" else np.abs(gr)))
        pts.append((float(x[ok][i]), float(y[ok][i])))
    return np.array(pts)


def ellipse_inliers(P):
    keep = np.ones(len(P), bool)
    for _ in range(8):
        x, y = P[keep, 0], P[keep, 1]
        Dm = np.column_stack([x * x, x * y, y * y, x, y, np.ones_like(x)])
        A, B, C, Dd, E, F = np.linalg.svd(Dm)[2][-1]
        M0 = np.array([[F, Dd / 2, E / 2], [Dd / 2, A, B / 2], [E / 2, B / 2, C]]); M = np.array([[A, B / 2], [B / 2, C]])
        cx = (B * E - 2 * C * Dd) / (4 * A * C - B * B); cy = (B * Dd - 2 * A * E) / (4 * A * C - B * B)
        ev, evec = np.linalg.eigh(M)
        ax = np.sqrt((-np.linalg.det(M0) / np.linalg.det(M)) / ev)
        q = (np.column_stack([P[:, 0] - cx, P[:, 1] - cy]) @ evec) / ax
        r = np.sqrt((q ** 2).sum(1)) - 1
        s = 1.4826 * np.median(np.abs(r[keep] - np.median(r[keep])))
        keep = np.abs(r - np.median(r[keep])) < max(3 * s, 0.004)
    return keep, (cx, cy)


def fit_pose(rim):
    c0 = rim.mean(0)
    best = None
    for b0 in np.radians(np.arange(0, 360, 20)):
        for a0 in (0.1, 0.25, 0.4):
            q0 = [a0, b0, F_PX * R_RIM / np.median(np.hypot(*(rim - c0).T)), c0[0], c0[1]]
            sol = optimize.least_squares(lambda q: np.hypot(*img_to_plate(rim, q)) - R_RIM, q0, x_scale=[0.1, 0.5, 5, 20, 20])
            if best is None or sol.cost < best.cost:
                best = sol
    return best.x, float(np.sqrt(2 * best.cost / len(rim)))


def measure(name, cfg):
    im = ImageOps.exif_transpose(Image.open(os.path.join(GT, "photos", f"{name}.jpg")))
    g = ndimage.gaussian_filter(np.asarray(im.convert("L"), float), 1.0)
    P = rim_points(g, cfg["c0"], cfg["r0"], cfg["edge"])
    keep, (ecx, ecy) = ellipse_inliers(P)
    if cfg["near_side"] == "bottom":
        keep &= P[:, 1] < ecy + 200
    pose, rim_rms = fit_pose(P[keep])
    ox, oy = cfg["off"]
    # 조각
    pieces = []
    for sx, sy in cfg["pieces"]:
        x0, y0 = ox + 2 * sx, oy + 2 * sy
        win = g[y0 - 110:y0 + 111, x0 - 110:x0 + 111]
        yy, xx = np.mgrid[-110:111, -110:111]
        X, Y = img_to_plate(np.column_stack([(xx + x0).ravel(), (yy + y0).ravel()]), pose)
        inside = (np.hypot(X, Y) < 36.0).reshape(win.shape)
        m = (np.abs(win - np.median(win[inside])) > 30) & (xx ** 2 + yy ** 2 <= 80 ** 2) & inside
        lab, n = ndimage.label(m)
        k = int(np.argmax(ndimage.sum(m, lab, range(1, n + 1)))) + 1
        ys, xs = np.nonzero(lab == k)
        pX, pY = img_to_plate(np.column_stack([xs - 110 + x0, ys - 110 + y0]), pose)
        mu = np.array([pX.mean(), pY.mean()])
        w, V = np.linalg.eigh(np.cov(np.vstack([pX - mu[0], pY - mu[1]])))
        r_ = float(np.hypot(*mu))
        pieces.append(dict(r=r_, th=float(np.degrees(np.arctan2(mu[1], mu[0]))), len_mm=float(4 * np.sqrt(w[-1])),
                           tilt=float(np.degrees(np.arccos(min(1.0, abs(V[:, -1] @ (mu / r_)))))),
                           u=float(np.mean(xs - 110 + x0)), v=float(np.mean(ys - 110 + y0))))
    # 그린 깃 선
    lines = {}
    for vn, ps in cfg["lines"].items():
        pts = []
        for sx, sy in ps:
            c = np.array([ox + 2 * sx, oy + 2 * sy], float)
            for _ in range(3):
                x0, y0 = int(c[0]), int(c[1])
                w_ = g[y0 - 30:y0 + 31, x0 - 30:x0 + 31]
                yy, xx = np.mgrid[-30:31, -30:31]
                m = (w_ < 70) & (xx ** 2 + yy ** 2 <= 28 ** 2)
                if m.sum() < 20:
                    break
                c = np.array([x0 + xx[m].mean(), y0 + yy[m].mean()])
            pts.append(c)
        X, Y = img_to_plate(np.array(pts), pose)
        r = np.hypot(X, Y); t = np.degrees(np.arctan2(Y, X)); t = t[0] + (t - t[0] + 180) % 360 - 180
        coef, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(r)), r - 34.0]), t, rcond=None)
        lines[vn] = dict(th34=float(coef[0]), slope=float(coef[1]), r=r.tolist(), uv=[p.tolist() for p in pts])
    rows = []
    for i, p in enumerate(pieces):
        ang = {v: (L["th34"] + L["slope"] * (p["r"] - 34.0)) % 360 for v, L in lines.items()}
        rel = {v: (p["th"] - a) % 360 for v, a in ang.items()}
        k = min(rel, key=rel.get); kn = k % 6 + 1
        wd = (ang[kn] - ang[k]) % 360
        rows.append(dict(photo=name, state=cfg["state"], piece=i, bay=f"{k}-{kn}", r_mm=round(p["r"], 2), ang_from_k=round(rel[k], 1),
                         ang_to_k1=round(wd - rel[k], 1), bay_width=round(wd, 1), tilt_from_radial=round(p["tilt"]), len_mm=round(p["len_mm"], 1)))
    # 확인 그림
    dr_im = im.convert("RGB"); dr = ImageDraw.Draw(dr_im)
    rs = np.linspace(26, 36.4, 40)
    for vn, L in lines.items():
        x, y = plate_to_img(rs * np.cos(np.radians(L["th34"] + L["slope"] * (rs - 34))), rs * np.sin(np.radians(L["th34"] + L["slope"] * (rs - 34))), pose)
        dr.line(list(zip(x, y)), fill=(255, 0, 0), width=5)
        for u in L["uv"]:
            dr.ellipse([u[0] - 9, u[1] - 9, u[0] + 9, u[1] + 9], fill=(255, 0, 0))
        for a_, col in ((20.0, (0, 90, 255)), (40.0, (0, 170, 0))):            # 도안 자리(같은 반지름에서 그린 선 + 20°·40°)
            th = L["th34"] + a_
            x, y = plate_to_img(np.array([34.3 * np.cos(np.radians(th))]), np.array([34.3 * np.sin(np.radians(th))]), pose)
            dr.ellipse([x[0] - 16, y[0] - 16, x[0] + 16, y[0] + 16], outline=col, width=5)
    for p in pieces:
        dr.ellipse([p["u"] - 10, p["v"] - 10, p["u"] + 10, p["v"] + 10], fill=(255, 220, 0))
    t = np.linspace(0, 2 * np.pi, 360)
    x, y = plate_to_img(R_RIM * np.cos(t), R_RIM * np.sin(t), pose)
    dr.line(list(zip(x, y)), fill=(255, 0, 255), width=3)
    c = (ox, oy, ox + 2140, oy + 2140)
    dr_im.crop(c).resize((1070, 1070)).save(os.path.join(GT, "photos", f"{name}_measure.png"))
    print(f"[{name}] 자세: 기울기 {np.degrees(pose[0]):.1f}°, 거리 {pose[2]:.1f} mm, 테두리 잔차 rms {rim_rms:.3f} mm ({keep.sum()}점)")
    return rows, dict(pose=pose.tolist(), rim_rms_mm=rim_rms, lines=lines)


if __name__ == "__main__":
    allrows, meta = [], {}
    for name, cfg in PHOTOS.items():
        rows, m = measure(name, cfg)
        allrows += rows; meta[name] = m
    T = pd.DataFrame(allrows)
    T.to_csv(os.path.join(GT, "m_piece_positions.csv"), index=False)
    json.dump(meta, open(os.path.join(GT, "out", "m_photo.json"), "w"), indent=1)
    print(T.to_string(index=False))
    for name in PHOTOS:
        d = T[T.photo == name]
        first = d[d.ang_from_k < d.bay_width / 2]; second = d[d.ang_from_k >= d.bay_width / 2]
        print(f"{name}: 깃 k 가까운 조각 {len(first)}개 {first.ang_from_k.mean():.1f} ± {first.ang_from_k.std():.1f}° (r {first.r_mm.mean():.2f})"
              + (f" | 깃 k+1 가까운 조각 {len(second)}개: 깃 k+1까지 {second.ang_to_k1.mean():.1f} ± {second.ang_to_k1.std():.1f}° "
                 f"(r {second.r_mm.mean():.2f})" if len(second) else ""))
    print(f"→ {os.path.join(GT, 'm_piece_positions.csv')}")
