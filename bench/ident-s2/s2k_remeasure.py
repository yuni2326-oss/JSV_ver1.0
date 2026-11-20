"""선택 재측정 위치 고르기 — G3 자유-자유 FEM(32×16, 간격 3.00 mm)의 모드형으로, 대응이 불명확한 극점 근처 모드가
뒤판·앞판의 깃 선 위 여러 반지름에서 얼마나 크게 보이는지(원주 계수 |c_h|)와 두 반지름 사이 비(ρ)를 낸다.
측정 점 = 깃 6곳(깃 선 = 격자 loc 0·1 평균) × 반지름 ξ, 면 = 뒤판 바깥면(z = 0) 또는 앞판 바깥면(z = 판 맨 위)."""
import json
import numpy as np
import pandas as pd

D = "/home/super/project/docs/_generated_g3/data/paper3/"
OUT = "/tmp/claude-1000/-home-super-project/efb54ce3-bcf9-473d-9dc8-eb835c653517/scratchpad/impact1022/out/s2/"
KEY = "ff32x16h6w62c50"
z = np.load(D + f"rev1_n6_shapes_{KEY}.npz", allow_pickle=False)
meta = json.load(open(D + f"rev1_n6_modes_{KEY}.json"))
reg = pd.read_csv(D + f"rev1_n6_modes_{KEY}.csv").sort_values("f_Hz").reset_index(drop=True)
X, gi, V = z["coors"], z["grid_idx"], z["vshape"].astype(float)
a, b = meta["a"], meta["b"]
z_front = meta["t_back"] + meta["channel"] + meta["t_front"]
per = 1 + 16
rr = np.hypot(X[:, 0], X[:, 1])
loc, sec = gi[:, 1] % per, gi[:, 1] // per
scale = np.sqrt(meta["mass_model_g"] / meta["mass_measured_g"])


def point(face, k, xi):
    zt = 0.0 if face == "back" else z_front
    sel = np.isclose(X[:, 2], zt, atol=1e-9)
    r_t = a + xi * (b - a)
    W = []
    for l in (0, 1):
        c = np.nonzero(sel & (loc == l) & (sec == k))[0]
        c = c[np.argsort(rr[c])]
        i = np.searchsorted(rr[c], r_t)
        lo_, hi_ = c[i - 1], c[i]
        t = (r_t - rr[lo_]) / (rr[hi_] - rr[lo_])
        W += [(lo_, 0.5 * (1 - t)), (hi_, 0.5 * t)]
    return W


LOCS = [("back", 0.3), ("back", 0.6), ("back", 0.85), ("front", 0.6), ("front", 0.85)]
PTS = {L: [point(L[0], k, L[1]) for k in range(6)] for L in LOCS}
k6 = np.arange(6)


def uz(m, W):
    return sum(w * V[m, 3 * n + 2] for n, w in W)


rows = []
for r in reg.itertuples():
    if r.f_Hz * scale > 9000:
        break
    mem = [int(x) for x in str(r.members).split("|")]
    h = int(r.h)
    row = {"label": r.label, "h": h, "f_fm": r.f_Hz * scale, "Efront": r.Efrac_front, "Eback": r.Efrac_back_main}
    for L in LOCS:
        # 짝이면 두 멤버의 원주 계수 크기를 제곱평균(방위 무관)
        c = np.sqrt(np.mean([abs((np.array([uz(m, W) for W in PTS[L]]) * np.exp(-1j * h * np.radians(60) * k6)).mean()) ** 2
                             for m in mem]))
        row[f"{L[0]}{L[1]}"] = c
    rows.append(row)
T = pd.DataFrame(rows)
cols = [f"{f}{x}" for f, x in LOCS]
ref = T[cols].max().max()
for c in cols:
    T[c + "_dB"] = 20 * np.log10(T[c] / ref + 1e-30)
T["rho_back_06_03_dB"] = 20 * np.log10(T["back0.6"] / T["back0.3"])
T["rho_back_085_06_dB"] = 20 * np.log10(T["back0.85"] / T["back0.6"])
T["rho_front_085_06_dB"] = 20 * np.log10(T["front0.85"] / T["front0.6"])
T["front_vs_back06_dB"] = 20 * np.log10(T["front0.6"] / T["back0.6"])
show = ["label", "h", "f_fm", "Efront", "Eback"] + [c + "_dB" for c in cols] + ["rho_back_06_03_dB", "rho_back_085_06_dB",
                                                                               "rho_front_085_06_dB", "front_vs_back06_dB"]
txt = T[show].round(2).to_string(index=False)
print(txt)
open(OUT + "s2k_remeasure.txt", "w").write(txt + "\n")
T.to_csv(OUT + "s2k_remeasure.csv", index=False)
