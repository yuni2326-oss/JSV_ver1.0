"""깃 높이 테이퍼(s2m_taper.py) 대 간격 일정 3.00 mm — E2-I 자유-자유 FEM 모드 주파수·질량·측정 대비(s2i_gap.py와 같은 짝·묶음).
사용: s2m_compare.py [변형 꼬리표 ...]  (기본: taper319_32x16 nom_ch319_32x16)"""
import sys, json
import numpy as np
import pandas as pd

D1 = "/tmp/claude-1000/-home-super-project/efb54ce3-bcf9-473d-9dc8-eb835c653517/scratchpad/impact1022/out/fe_s2/"
OUT = "/tmp/claude-1000/-home-super-project/efb54ce3-bcf9-473d-9dc8-eb835c653517/scratchpad/impact1022/out/taper/"
REF = "nom_ch300_32x16"
VARS = sys.argv[1:] or ["taper319_32x16", "nom_ch319_32x16"]
M_MEAS = 41.3186
mem = lambda s: [int(x) for x in str(s).split("|")]


def subspace(V, ids):
    Q, _ = np.linalg.qr(V[ids].T.astype(float))
    return Q


def groups(tag):
    z = np.load(D1 + f"shapes_{tag}.npz", allow_pickle=False)
    t = pd.read_csv(D1 + f"modes_{tag}.csv").sort_values("f_Hz").reset_index(drop=True)
    V = z["vshape"].astype(float)
    return t, [subspace(V, mem(m)) for m in t.members], json.load(open(D1 + f"meta_{tag}.json"))


PAIRS = [(2133.80, "h2:m2n0"), (2142.28, "h2:m2n0"), (3163.65, "h3:m3n0"), (3249.89, "h0:m0n1"), (3458.89, "h1:m1n1"),
         (3638.99, "h2:m4n0"), (3747.87, "h3:m3n1"), (3804.07, "h0:m6n1"), (4225.49, "h1:m1n1#2"), (4966.99, "h2:m4n0#2"),
         (5617.22, "h0:m6n0"), (5744.69, "h1:m5n0"), (6438.86, "h2:m2n1"), (6558.41, "h2:m2n1"), (6817.09, "h1:m7n0"),
         (7093.13, "h3:m3n2"), (7574.17, "h2:m8n0"), (8321.24, "h3:m3n1#2")]
S11 = [(2138.04, ["h2:m2n0"]), ((3163.65 + 3249.89) / 2, ["h3:m3n0", "h0:m0n1"]), (3458.89, ["h1:m1n1"]),
       (3747.87, ["h3:m3n1"]), (3804.07, ["h0:m6n1"]), (4225.49, ["h1:m1n1#2"]), (4966.99, ["h2:m4n0#2"]),
       (5617.22, ["h0:m6n0"]), (5744.69, ["h1:m5n0"]), (6438.86, ["h2:m2n1"]), (8321.24, ["h3:m3n1#2"])]

tA, QA, mA = groups(REF)
sA = np.sqrt(mA["mass_g"] / M_MEAS)
lines = [f"기준 {REF}: 간격 3.00 mm 일정, 질량 {mA['mass_g']:.3f} g, 41.32 g 맞춤 배율 {sA:.4f}"]
res = {"ref": REF, "variants": {}}
fA = dict(zip(tA.label, tA.f_Hz))
for tag in VARS:
    tB, QB, mB = groups(tag)
    sB = np.sqrt(mB["mass_g"] / M_MEAS)
    lines += ["", f"=== {tag}: {mB['params_mm']}", f"질량 {mB['mass_g']:.3f} g ({mB['mass_g'] - mA['mass_g']:+.3f} g), 맞춤 배율 {sB:.4f}",
              "모드별(9.5 kHz까지): 이름, 3.00 → 변형 [Hz], 변화(질량 그대로) %, 변화(각자 41.32 g 맞춤) %, MAC, 변형 쪽 이름"]
    chg = {}
    for i in range(len(tA)):
        if tA.f_Hz[i] > 9500:
            break
        mac, j = max(((np.linalg.svd(QA[i].T @ QB[j], compute_uv=False) ** 2).mean(), j) for j in range(len(tB))
                     if abs(tB.f_Hz[j] / tA.f_Hz[i] - 1) < 0.1)
        d_raw = 100 * (tB.f_Hz[j] / tA.f_Hz[i] - 1)
        d_m = 100 * (tB.f_Hz[j] * sB / (tA.f_Hz[i] * sA) - 1)
        chg[tA.label[i]] = (float(tA.f_Hz[i]), float(tB.f_Hz[j]), d_raw, d_m, float(mac), tB.label[j])
        lines.append(f"  {tA.label[i]:11s} {tA.f_Hz[i]:8.1f} → {tB.f_Hz[j]:8.1f}  {d_raw:+6.2f}  {d_m:+6.2f}  {mac:.3f}  {tB.label[j]}")
    lines += ["", "측정/FEM − 1 [%] (41.32 g 맞춤): 간격 3.00 → 변형"]
    rows = []
    for f, lab in PAIRS:
        ea, eb = 100 * (f / (chg[lab][0] * sA) - 1), 100 * (f / (chg[lab][1] * sB) - 1)
        rows.append(dict(f=f, label=lab, e300=ea, evar=eb))
        lines.append(f"  {f:8.2f} {lab:11s} {ea:+6.2f} → {eb:+6.2f}")
    stats = {}
    for nm_, k, s in (("3.00", 0, sA), (tag, 1, sB)):
        e = np.array([100 * (f / np.mean([chg[l][k] * s for l in Ls]) - 1) for f, Ls in S11])
        stats[nm_] = dict(mean=float(e.mean()), rms=float(np.sqrt((e ** 2).mean())), lo=float(e.min()), hi=float(e.max()),
                          per=e.tolist())
        lines.append(f"11쌍(31번 묶음) {nm_}: 평균 {e.mean():+.2f} %, rms {np.sqrt((e ** 2).mean()):.2f} %, 범위 {e.min():+.1f}~{e.max():+.1f} %")
    e_raw = np.array([100 * (f / np.mean([chg[l][1] for l in Ls]) - 1) for f, Ls in S11])
    lines.append(f"11쌍 질량 맞춤 없이({tag}): 평균 {e_raw.mean():+.2f} %, 범위 {e_raw.min():+.1f}~{e_raw.max():+.1f} %")
    res["variants"][tag] = dict(meta=mB, modes=chg, pairs=rows, s11=stats, s11_raw=e_raw.tolist())
txt = "\n".join(lines)
print(txt)
open(OUT + "s2m_compare.txt", "w").write(txt + "\n")
json.dump(res, open(OUT + "s2m_compare.json", "w"), ensure_ascii=False, indent=1)
