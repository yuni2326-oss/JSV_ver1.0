"""s2g_fit.py·s2g_check.py 공용: 측정 11쌍, 기준 FE(2번 시편 치수 32×16) 모드 무리, 부분공간 MAC 짝짓기."""
import numpy as np
import pandas as pd

D1 = "/tmp/claude-1000/-home-super-project/efb54ce3-bcf9-473d-9dc8-eb835c653517/scratchpad/impact1022/out/fe_s2/"
mem = lambda s: [int(x) for x in str(s).split("|")]
PAIRS = [("h2:m2n0", ("h2:m2n0",), 2138.04), ("h3:m3n0+h0:m0n1", ("h3:m3n0", "h0:m0n1"), (3163.65 + 3249.89) / 2),
         ("h1:m1n1", ("h1:m1n1",), 3458.89), ("h3:m3n1", ("h3:m3n1",), 3747.87), ("h0:m6n1", ("h0:m6n1",), 3804.07),
         ("h1:m1n1#2", ("h1:m1n1#2",), 4225.49), ("h2:m4n1", ("h2:m4n0#2",), 4966.99), ("h0:m6n0", ("h0:m6n0",), 5617.22),
         ("h1:m5n1", ("h1:m5n1",), 5744.69), ("h2:m2n1", ("h2:m2n1",), 6438.86), ("h3:m3n1#2", ("h3:m3n1#2",), 8321.24)]
fmeas = np.array([p[2] for p in PAIRS])
names = [p[0] for p in PAIRS]


def subspace(V, ids):
    Q, _ = np.linalg.qr(V[ids].T.astype(float))
    return Q


def load(npz, csv):
    z = np.load(npz, allow_pickle=False)
    t = pd.read_csv(csv)
    V = z["vshape"].astype(float)
    return z, t, V, [(r.label, r.f_Hz, subspace(V, mem(r.members))) for r in t.itertuples()]


zB, tB, VB, GB = load(D1 + "shapes_tf400_32x16.npz", D1 + "modes_tf400_32x16.csv")
QB = {lab: Q for lab, f, Q in GB}
fB_lab = {lab: f for lab, f, Q in GB}


def matched(G):
    """기준 FE의 각 짝 이름 → (G에서 MAC 최대인 무리의 주파수, MAC)."""
    out = {}
    for lab in {l for p in PAIRS for l in p[1]}:
        best = max(((np.linalg.svd(QB[lab].T @ Q, compute_uv=False) ** 2).mean(), f) for l2, f, Q in G
                   if abs(f / fB_lab[lab] - 1) < 0.25)
        out[lab] = (best[1], best[0])
    return out


def pair_freqs(mt):
    return np.array([np.mean([mt[l][0] for l in p[1]]) for p in PAIRS]), min(mt[l][1] for p in PAIRS for l in p[1])
