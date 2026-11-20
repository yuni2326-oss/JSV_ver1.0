"""판 사이 간격 3.05 → 3.00 mm의 영향 — (1) 원고 쪽 두 면판 식: 앞판 감육의 결합비 ζ와 레일 부호전환 반경 ξ*(a10과 같은 계산),
(2) E2-I 기준 FEM(표준 두께 0.40/0.60): 모드 주파수·질량·측정 대비."""
import sys, json
sys.path.insert(0, "/home/super/project-paper3")
import numpy as np
import pandas as pd
import importlib.util
from impeller_fingerprint import forward as fwd, figures as figs, kernels as ker, geometry as geo
from impeller_fingerprint.rev1 import g1_section as g1

OUT = "/tmp/claude-1000/-home-super-project/efb54ce3-bcf9-473d-9dc8-eb835c653517/scratchpad/impact1022/out/s2/"
D1 = "/tmp/claude-1000/-home-super-project/efb54ce3-bcf9-473d-9dc8-eb835c653517/scratchpad/impact1022/out/fe_s2/"
lines = []

# (1) 두 면판 식
GEOMS = [("r4.1 정본(1번 시편)", 0.375e-3, 0.59e-3, 3.18e-3), ("표준 두께, 간격 3.18", 0.40e-3, 0.60e-3, 3.18e-3),
         ("표준 두께, 간격 3.05", 0.40e-3, 0.60e-3, 3.05e-3), ("표준 두께, 간격 3.00", 0.40e-3, 0.60e-3, 3.00e-3)]
pool = [ker.mode_kernel(geo.DISK, m=m, n=n, n_grid=2001) for m, n in figs.MODES]
lines.append("두 면판 식(부록 A): 앞판 감육 ζ0 = d_M/d_K(작은 깊이), 뒤판 감육 ζ0, 레일 부호전환 반경 ξ*(m = 0–3, S̄ 3 %)")
res = []
for name, td, to, gap in GEOMS:
    z_f = g1.zeta0(td, to, gap)
    z_b = g1.zeta0(to, td, gap)
    d_sep = gap + 0.5 * (td + to)

    def law(d_K, td=td, to=to, gap=gap):
        p = fwd.two_face_depth_from_dk(d_K, td, to, gap)
        return p * td / (td + to)
    fwd.MASS_LAWS["_tmp"] = law
    _, _, loci = figs.null_loci(pool, coupling="_tmp")
    res.append(dict(name=name, zeta_front=z_f, zeta_back=z_b, d_sep_mm=d_sep * 1e3, xi_star=loci.tolist()))
    lines.append(f"  {name:20s} 면 중심 간격 {d_sep * 1e3:.3f} mm, ζ0 앞판 {z_f:.4f}, 뒤판 {z_b:.4f}, ξ* " + " / ".join(f"{x:.3f}" for x in loci))
fwd.MASS_LAWS.pop("_tmp", None)
lines.append("")

# (2) E2-I FEM
_s = importlib.util.spec_from_file_location("lib", "/tmp/claude-1000/-home-super-project/efb54ce3-bcf9-473d-9dc8-eb835c653517/scratchpad/impact1022/scripts/s2g_fit_lib.py")
L = importlib.util.module_from_spec(_s); _s.loader.exec_module(L)
mem = L.mem


def groups(tag):
    z = np.load(D1 + f"shapes_{tag}.npz", allow_pickle=False)
    t = pd.read_csv(D1 + f"modes_{tag}.csv").sort_values("f_Hz").reset_index(drop=True)
    V = z["vshape"].astype(float)
    return t, [L.subspace(V, mem(m)) for m in t.members], json.load(open(D1 + f"meta_{tag}.json"))


tA, QA, mA = groups("nom_32x16")
tB, QB, mB = groups("nom_ch300_32x16")
M_MEAS = 41.3186
sA, sB = np.sqrt(mA["mass_g"] / M_MEAS), np.sqrt(mB["mass_g"] / M_MEAS)
lines.append(f"E2-I FEM 질량: 간격 3.05 {mA['mass_g']:.3f} g → 3.00 {mB['mass_g']:.3f} g ({mB['mass_g'] - mA['mass_g']:+.3f} g); 질량 맞춤 배율 {sA:.4f} → {sB:.4f}")
match = []
for i in range(len(tA)):
    if tA.f_Hz[i] > 9500:
        break
    best = max(((np.linalg.svd(QA[i].T @ QB[j], compute_uv=False) ** 2).mean(), j) for j in range(len(tB))
               if abs(tB.f_Hz[j] / tA.f_Hz[i] - 1) < 0.1)
    match.append((i, best[1], best[0]))
lines.append("모드별(9.5 kHz까지): 이름, 3.05 → 3.00 [Hz], 변화(질량 그대로) %, 변화(각자 41.32 g 맞춤) %, MAC")
chg = {}
for i, j, mac in match:
    d_raw = 100 * (tB.f_Hz[j] / tA.f_Hz[i] - 1)
    d_m = 100 * (tB.f_Hz[j] * sB / (tA.f_Hz[i] * sA) - 1)
    chg[tA.label[i]] = (tA.f_Hz[i], tB.f_Hz[j], d_raw, d_m, mac)
    lines.append(f"  {tA.label[i]:11s} {tA.f_Hz[i]:8.1f} → {tB.f_Hz[j]:8.1f}  {d_raw:+6.2f}  {d_m:+6.2f}  {mac:.3f}")
# 측정 대비(32번 짝)
PAIRS = [(2133.80, "h2:m2n0"), (2142.28, "h2:m2n0"), (3163.65, "h3:m3n0"), (3249.89, "h0:m0n1"), (3458.89, "h1:m1n1"), (3638.99, "h2:m4n0"),
         (3747.87, "h3:m3n1"), (3804.07, "h0:m6n1"), (4225.49, "h1:m1n1#2"), (4966.99, "h2:m4n0#2"), (5617.22, "h0:m6n0"),
         (5744.69, "h1:m5n0"), (6438.86, "h2:m2n1"), (6558.41, "h2:m2n1"), (6817.09, "h1:m7n0"), (7093.13, "h3:m3n2"),
         (7574.17, "h2:m8n0"), (8321.24, "h3:m3n1#2")]
lines.append("")
lines.append("측정/FEM − 1 [%] (41.32 g 맞춤): 간격 3.05 → 3.00")
rows = []
for f, lab in PAIRS:
    fa, fb = chg[lab][0] * sA, chg[lab][1] * sB
    rows.append(dict(f=f, label=lab, e305=100 * (f / fa - 1), e300=100 * (f / fb - 1)))
    lines.append(f"  {f:8.2f} {lab:11s} {100 * (f / fa - 1):+6.2f} → {100 * (f / fb - 1):+6.2f}")
S11 = [((2133.80 + 2142.28) / 2 * 0 + 2138.04, ["h2:m2n0"]), ((3163.65 + 3249.89) / 2, ["h3:m3n0", "h0:m0n1"]), (3458.89, ["h1:m1n1"]),
       (3747.87, ["h3:m3n1"]), (3804.07, ["h0:m6n1"]), (4225.49, ["h1:m1n1#2"]), (4966.99, ["h2:m4n0#2"]), (5617.22, ["h0:m6n0"]),
       (5744.69, ["h1:m5n0"]), (6438.86, ["h2:m2n1"]), (8321.24, ["h3:m3n1#2"])]
for nm, k, s in (("3.05", 0, sA), ("3.00", 1, sB)):
    e = np.array([100 * (f / np.mean([chg[l][k] * s for l in Ls]) - 1) for f, Ls in S11])
    lines.append(f"11쌍(31번 묶음) 간격 {nm}: 평균 {e.mean():+.2f} %, rms {np.sqrt((e ** 2).mean()):.2f} %, 범위 {e.min():+.1f}~{e.max():+.1f} %")
txt = "\n".join(lines)
print(txt)
open(OUT + "s2i_gap.txt", "w").write(txt + "\n")
json.dump(dict(section=res, fem={k: list(v) for k, v in chg.items()}, pairs=rows), open(OUT + "s2i_gap.json", "w"), ensure_ascii=False, indent=1)
