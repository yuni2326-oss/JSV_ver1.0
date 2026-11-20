"""s2g_confirm.py 실행(치수·질량을 넣은 FE)을 측정 11쌍과 비교하고, s2g_fit.py 선형 예측과 맞대 본다."""
import json
import numpy as np
import pandas as pd

D1 = "/tmp/claude-1000/-home-super-project/efb54ce3-bcf9-473d-9dc8-eb835c653517/scratchpad/impact1022/out/fe_s2/"
OUT = "/tmp/claude-1000/-home-super-project/efb54ce3-bcf9-473d-9dc8-eb835c653517/scratchpad/impact1022/out/s2/"
import importlib.util
SCR = "/tmp/claude-1000/-home-super-project/efb54ce3-bcf9-473d-9dc8-eb835c653517/scratchpad/impact1022/scripts/"
spec = importlib.util.spec_from_file_location("lib", SCR + "s2g_fit_lib.py")
F = importlib.util.module_from_spec(spec); spec.loader.exec_module(F)
spec2 = importlib.util.spec_from_file_location("fit", SCR + "s2g_fit.py")      # 선형 모형(실행하면 적합 결과도 다시 쓴다)
FIT = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(FIT)

CASES = ["c_unif_fit", "c_free_fit", "c_s1thick_unif", "c_nominal_unif"]
lines, rows = [], {}
for tag in CASES:
    z, t, V, G = F.load(D1 + f"shapes_{tag}.npz", D1 + f"modes_{tag}.csv")
    f, mac = F.pair_freqs(F.matched(G))
    meta = json.load(open(D1 + f"meta_{tag}.json"))
    e = 100 * (F.fmeas / f - 1)
    rows[tag] = e
    lines.append(f"[{tag}] 치수 {meta['params_mm']}, 모형 질량 {meta['mass_model_g']:.2f} g → 합계 {meta['mass_total_g']:.2f} g, 최소 MAC {mac:.3f}")
    lines.append(f"   측정/FE − 1: 평균 {e.mean():+.2f} %, rms {np.sqrt((e ** 2).mean()):.2f} %, 범위 {e.min():+.1f}~{e.max():+.1f} %")
    lines.append("   모드별 " + ", ".join(f"{n} {x:+.1f}" for n, x in zip(F.names, e)))
    q = np.zeros(FIT.NP + 6)
    q[:FIT.NP] = [meta["params_mm"][k] - FIT.BASE[k] for k in FIT.PN]
    q[FIT.NP + 1:] = [meta["added_g"][r] for r in FIT.REGS]
    lnf = FIT.model(q, "uniform" if meta["uniform"] else "free", False)[0]
    d = e - 100 * (np.exp(np.log(F.fmeas) - lnf) - 1)
    lines.append("   FE − 선형 예측 [%p] " + ", ".join(f"{x:+.2f}" for x in d) + f" (최대 |{np.abs(d).max():.2f}|)")
txt = "\n".join(lines)
print(txt)
open(OUT + "s2g_check.txt", "w").write(txt + "\n")
pd.DataFrame(rows, index=F.names).to_csv(OUT + "s2g_check.csv")
