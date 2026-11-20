"""2번 시편 깃 6곳 공동 적합 극점(깨끗한 극점) ↔ FE(ff32x16h6w62c40) 같은 원주 차수 후보, 1번 시편(10-26 공동 적합) 대응 후보.
대응은 같은 h 안에서 가장 가까운 FE 모드(±8 %)를 후보로 적을 뿐이고, 모양(MAC) 확인 전이다."""
import json, csv, os, sys
import numpy as np
OUT = sys.argv[1]
S2 = json.load(open(os.path.join(OUT, "s2", "s2joint_slope-0.001.json")))
S1 = json.load(open(os.path.join(OUT, "w22c_slope0.0.json")))
FE = [r for r in csv.DictReader(open("/home/super/project-paper3/docs/_generated/data/paper3/rev1_n6_modes_ff32x16h6w62c40.csv")) if 1800 < float(r["f_Hz"]) < 10500]
print("specimen 2 pole | zeta % | MPC | h (energy) [boot] | theta0 | FE same-h candidates within 8 % | specimen-1 same-h candidates within 6 %")
rows = []
for p in S2:
    clean = p["mpc"] >= 0.75 and max(p["P"]) >= 0.6 and p["share"] >= 0.9
    h = p["h"]
    fe = sorted([(abs(float(r["f_Hz"]) / p["f"] - 1), r["label"], float(r["f_Hz"])) for r in FE if int(r["h"]) == h])
    fe = [x for x in fe if x[0] < 0.08][:2]
    s1 = sorted([(abs(q["f"] / p["f"] - 1), q["f"], q["h"], q["mpc"]) for q in S1 if q["h"] == h and q["mpc"] >= 0.6 and max(q["P"]) >= 0.5])
    s1 = [x for x in s1 if x[0] < 0.06][:2]
    fes = "; ".join(f"{l} {f:.0f} ({100*(p['f']/f-1):+.1f} %)" for _, l, f in fe) or "-"
    s1s = "; ".join(f"{f:.1f} ({100*(p['f']/f-1):+.1f} %)" for _, f, hh, m in s1) or "-"
    th = f"{p['th0']:.0f}" if p["th0"] is not None else "-"
    print(f"  {'*' if clean else ' '} {p['f']:8.1f} | {p['z']*100:.2f} | {p['mpc']:.2f} | h{h} ({max(p['P']):.2f}) [{p['share']:.2f}] | {th:>4s} | {fes} | {s1s}")
    rows.append(dict(f=p["f"], z=p["z"], mpc=p["mpc"], h=h, Ph=max(p["P"]), share=p["share"], th0=p["th0"], clean=clean, fe=[(l, f) for _, l, f in fe], s1=[f for _, f, _, _ in s1]))
json.dump(rows, open(os.path.join(OUT, "s2", "s2fe.json"), "w"), indent=1)
