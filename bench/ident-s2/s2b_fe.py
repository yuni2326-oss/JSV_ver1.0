"""측정 극점 잔차(깃 6곳 × ξ0.3·ξ0.6) 대 FE(ff32x16h6w62c40) 자유-자유 모드형 MAC.
FE 점 = 뒤판 바깥면(z = 0) 깃 선(격자 loc 0·1 평균) r = a + ξ(b − a)에서 u_z(반경 선형 보간).
짝 모드(h1·h2)는 두 멤버가 펼치는 부분공간에 사영한 MAC를 쓴다. 깃 번호 방향은 FE +θ(같음)와 거울(반대) 둘 다 본다.
측정 12점 = [D·잔차 10-28 ξ0.3 깃 1–6, 잔차 10-30 ξ0.6 깃 1–6], D = 1번 깃 ξ0.3 10-30/10-28(닫음 점검)."""
import sys, os, json
import numpy as np, pandas as pd

OUT, FED = sys.argv[1], sys.argv[2]
key = "ff32x16h6w62c40"
z = np.load(os.path.join(FED, f"rev1_n6_shapes_{key}.npz"), allow_pickle=False)
reg = pd.read_csv(os.path.join(FED, f"rev1_n6_modes_{key}.csv")).sort_values("f_Hz").reset_index(drop=True)
meta = json.load(open(os.path.join(FED, f"rev1_n6_modes_{key}.json")))
coors, gi, vsh, freqs = z["coors"], z["grid_idx"], z["vshape"].astype(float), z["freqs"]
a, b = meta["a"], meta["b"]
per = 1 + 16
rr = np.hypot(coors[:, 0], coors[:, 1]); th = np.degrees(np.arctan2(coors[:, 1], coors[:, 0]))
bot = np.isclose(coors[:, 2], 0.0, atol=1e-9)
loc = gi[:, 1] % per; sec = gi[:, 1] // per


def point(k, xi):
    """섹터 k, ξ → (가중 절점 목록, 각도[°])."""
    r_t = a + xi * (b - a); W = []; ang = []
    for l in (0, 1):
        c = np.nonzero(bot & (loc == l) & (sec == k))[0]
        c = c[np.argsort(rr[c])]
        i = np.searchsorted(rr[c], r_t); lo_, hi_ = c[i - 1], c[i]
        t = (r_t - rr[lo_]) / (rr[hi_] - rr[lo_])
        W += [(lo_, 0.5 * (1 - t)), (hi_, 0.5 * t)]
        ang.append((1 - t) * th[lo_] + t * th[hi_])
    return W, float(np.mean(ang))


PTS = [point(k, xi) for xi in (0.3, 0.6) for k in range(6)]
Phi = np.array([[sum(w * vsh[m, 3 * n + 2] for n, w in W) for W, _ in PTS] for m in range(vsh.shape[0])])   # (80, 12)
print("FE points: angle [deg] of sector 0..5 at xi0.3 / xi0.6:", [round(p[1], 1) for p in PTS])
members = [[int(i) for i in str(m).split("|")] for m in reg["members"]]
assert np.allclose([freqs[m[0]] for m in members], reg["f_Hz"], rtol=1e-6)

J = json.load(open(os.path.join(OUT, "s2", "s2b_joint.json")))


def macsub(x, B):
    """복소 측정 벡터 x를 실수 FE 부분공간 B(점 × 멤버)에 사영한 MAC."""
    Q, _ = np.linalg.qr(B)
    p = Q.T @ x
    return float((np.abs(p) ** 2).sum() / (np.abs(x) ** 2).sum())


def order(v, mirror):
    """측정 깃 1–6 → FE 섹터 순서로 다시 놓을 색인."""
    s = [(-(i)) % 6 if mirror else i for i in range(6)]
    o = np.empty(6, int); o[s] = np.arange(6)
    return v[o]


rows = []
print("\nmeasured pole (10-30) | D closure | best FE (MAC) for: xi0.3 6-pt (10-28) / xi0.6 6-pt (10-30) / 12-pt same-dir / 12-pt mirror")
for A, B in zip(J["A"], J["B"]):
    RA = np.array([complex(*x) for x in A["R"]]); RB = np.array([complex(*x) for x in B["R"]])
    D = RA[6] / RB[0]
    snA, snB = abs(RA[6]) / A["Rsd"][6], abs(RB[0]) / B["Rsd"][0]
    Duse = D if min(snA, snB) >= 5 and abs(RB[0]) > 10 ** (-0.5) * np.abs(RB[:6]).max() else 1.0      # 1번 깃 ξ0.3이 마디 근처면 닫음 비를 믿지 않는다
    out = {}
    for lab, mir in (("same", False), ("mirror", True)):
        x3 = order(RB[:6] * Duse, mir); x6 = order(RA[:6], mir); x12 = np.concatenate([x3, x6])
        M = []
        for g, mem in enumerate(members):
            if not (0.75 * A["f"] < reg["f_Hz"][g] < 1.30 * A["f"]):
                continue
            Bm = Phi[mem].T
            M.append((g, macsub(x3, Bm[:6]), macsub(x6, Bm[6:]), macsub(x12, Bm)))
        out[lab] = M
    def best(M, c):
        s = sorted(M, key=lambda t: -t[c])[:2]
        return " ".join(f"{reg['label'][g]}({reg['f_Hz'][g]:.0f}) {t[c]:.2f}" for g, *_ in s for t in [next(m for m in M if m[0] == g)])
    print(f" {A['f']:8.2f} | {20*np.log10(abs(D)):+5.1f} dB {np.degrees(np.angle(D)):+5.0f} (SNR {min(snA, snB):4.1f}{'' if Duse is D else ' -> D=1'})"
          f" | 0.3: {best(out['same'], 1)} | 0.6: {best(out['same'], 2)} | 12 same: {best(out['same'], 3)} | 12 mirror: {best(out['mirror'], 3)}")
    rows.append(dict(f=A["f"], fB=B["f"], D=[float(D.real), float(D.imag)], same=[list(map(float, m)) for m in out["same"]], mirror=[list(map(float, m)) for m in out["mirror"]]))
json.dump(dict(labels=list(reg["label"]), fFE=list(map(float, reg["f_Hz"])), rows=rows), open(os.path.join(OUT, "s2", "s2b_fe.json"), "w"))
np.save(os.path.join(OUT, "s2", "s2b_fe_phi.npy"), Phi)
