"""흠집 시험 분석 도구 점검 — DAQ 형식의 합성 타격으로 groove_state·groove_compare가 알려진 변화를 되찾는지 본다.

합성: 20개 모드(기준 주파수 = 10-30 표, h2 짝 포함), 위치(ξ0.3·ξ0.6)마다 다른 모드 크기(뒤판 모드는 ξ0.3에서 약하게),
힘 10–25 N과 연화, 일부 타격에 재접촉, 마이크 잡음 20 mPa. 상태 A1·A2(다시 매달기 흩어짐 0.03 %),
B2(s3_pred_states.csv 예측 + 공통 오프셋 −0.05 % + 흩어짐), B3·C1(같은 방식, 오프셋 0) — C1은 B3 기준 증분으로도 비교한다.
42번 채택안 점검(s3_pred_poles.csv가 있으면): L1(깃 3–4 창 하나 — 2142만 오름, 성분 방향 모르는 짝은 한 성분 값을 참값으로)과
T-a(같은 매달기 테이프 — 흩어짐 0.005 %, --sig-h 0 --thr 0.06)를 더해, 성분별 판정·예측 갈라짐·'판정 밖' 표시를 본다.
사용: python check_groove_tools.py [작업 폴더]   (약 2–4분, 임시 자료 약 230 MB는 끝나면 지운다)
기대: 판정 모드의 측정 변화가 참값과 3σ 안(90 % 이상), 기울기 a 0.8–1.2, 오프셋 c가 −0.05 ± 0.03 %."""
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import groove_compare  # noqa: E402
import groove_lib as gl  # noqa: E402
import groove_state  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = sys.argv[1] if len(sys.argv) > 1 else "/tmp/claude-1000/-home-super-project/efb54ce3-bcf9-473d-9dc8-eb835c653517/scratchpad/groove_synth"
FS, TI, REC = 204800.0, 4096, 0.7
OFFSET = -0.05
rng = np.random.default_rng(2026)

# 상태별 재해석 예측(FEM 짝별)
SP = pd.read_csv(os.path.join(HERE, "..", "fe-nominal", "s3_pred_states.csv"))
PREDS = {st: {m.name: float(g.set_index("label").loc[m.fem, "df_pct_res"]) if m.fem in set(g.label) else 0.0 for m in gl.MODES}
         for st, g in SP.groupby("state")}
PRED = PREDS["B2"]
OFFS = {"A1": 0.0, "A2": 0.0, "B2": OFFSET, "B3": 0.0, "C1": 0.0}
POLES = os.path.join(HERE, "..", "fe-nominal", "s3_pred_poles.csv")
if os.path.exists(POLES):                                   # 42번: 극점별 예측(국부 상태는 h2 두 성분이 다르다)
    PP = pd.read_csv(POLES)
    for st in ("L1", "T-a"):
        g = PP[PP.state == st].assign(mode=lambda d: d["mode"].astype(str)).set_index("mode")
        # 성분 방향을 모르는 짝은 쌍 평균 + 반 갈라짐(한 성분)을 참값으로 둔다
        PREDS[st] = {m.name: (float(g.loc[m.name, "pred_pct"]) + (0.0 if bool(g.loc[m.name, "member_known"])
                                                                  else float(g.loc[m.name, "half_split_pct"])))
                     if m.name in g.index and np.isfinite(g.loc[m.name, "pred_pct"]) else 0.0 for m in gl.MODES}
    OFFS.update({"L1": 0.03, "T-a": 0.0})
ZETA = {m.name: (0.0015 if m.name in gl.PAIR else 0.0025) for m in gl.MODES}
AMP = {}
for m in gl.MODES:
    a06 = rng.uniform(0.04, 0.25)
    a03 = a06 * (0.15 if m.kind == "back" else rng.uniform(0.4, 1.6))
    AMP[m.name] = {"back_vane1_xi0.3": a03, "back_vane1_xi0.6": a06}
HANG = {s: {m.name: rng.normal(0, 0.03) for m in gl.MODES} for s in OFFS}
if "T-a" in HANG:                                           # T-a는 A1과 같은 매달기 — 매달기 흩어짐을 함께 갖고 0.005 %만 더한다
    HANG["T-a"] = {n: v + rng.normal(0, 0.005) for n, v in HANG["A1"].items()}
TRUE_F = {s: {m.name: m.f * (1 + (HANG[s][m.name] + PREDS.get(s, {}).get(m.name, 0.0) + OFFS[s]) / 100) for m in gl.MODES}
          for s in OFFS}


def write_state(folder, state, session, log):
    n_rec = TI + int(REC * FS)
    t = np.arange(n_rec - TI) / FS
    for loc in ("back_vane1_xi0.3", "back_vane1_xi0.6"):
        for hit in range(1, 17):
            N = rng.uniform(10, 25)
            d = np.zeros((4, n_rec), np.float32)
            w = int(0.00015 * FS)
            d[0, TI:TI + w] = N * np.sin(np.pi * np.arange(w) / w)
            if rng.random() < 0.2:                                   # 재접촉(0.8 ms 뒤 작은 펄스)
                k0 = TI + int(0.0008 * FS); d[0, k0:k0 + w] += 0.15 * N * np.sin(np.pi * np.arange(w) / w)
            x = np.zeros(n_rec - TI)
            for m in gl.MODES:
                f = TRUE_F[state][m.name] * (1 + m.slope * np.log2(N / 20))
                z = ZETA[m.name]; wd = 2 * np.pi * f * np.sqrt(1 - z * z)
                x += AMP[m.name][loc] * N * np.exp(-z * 2 * np.pi * f * t) * np.sin(wd * t + rng.uniform(0, 2 * np.pi))
            d[1, TI:] = x
            d[1] += rng.normal(0, 0.02, n_rec)
            ts = f"2026-11-03T10:{session:02d}:{hit:02d}.000"
            fn = f"20261103_10{session:02d}{hit:02d}_000_S{session}_{loc}_M1_H{hit:03d}.npz"
            np.savez(os.path.join(folder, fn), data=d, sample_rate=FS, trigger_index=TI, session=session, location=loc,
                     notes=state, timestamp=ts, peak_force_N=N, temperature_C=np.array([21.0, 21.0]), quality="ok",
                     channel_names=np.array(["CH7_hammer", "CH6_mic", "CH4_acc", "CH5_acc"]),
                     physical_channels=np.array(["PXI1Slot5/ai3", "PXI1Slot5/ai2", "PXI1Slot5/ai0", "PXI1Slot5/ai1"]))
            log.append({"file": fn, "status": "accepted", "resp_peak": float(np.abs(x[:2000]).max())})


def check_mass_chain(work):
    """질량 비례 조정: 줄인 B(11-16, A → L6 → R → C1, L1a·L1 건너뜀)와 원래 사슬(A → L1a → L1 → L6 → R). 표만 쓴다(1초)."""
    P = groove_compare.pred_table(POLES)
    fem = {s: P[s]["removed_mg"] for s in ("L1A", "L1", "L6", "R", "C1")}
    ok = True
    for name, mg in (("줄인 B", {"L6": 200.0, "R": 420.0, "C1": 510.0}),
                     ("원래", {"L1A": 18.0, "L1": 40.0, "L6": 230.0, "R": 450.0})):
        path = os.path.join(work, "mass_chain.csv")
        pd.DataFrame({"state": ["A"] + list(mg), "mass_g": [41.3] + [41.3 - v / 1e3 for v in mg.values()]}).to_csv(path, index=False)
        _, k = groove_compare.mass_ratios(path, P, tape_csv=None)
        seq = [s for s in ("L1A", "L1", "L6", "R", "C1") if s in mg]
        want, prev = {}, None
        for s in seq:                                        # 잰 상태 사이 증분 비, 건너뛴 상태는 다음 잰 상태의 비
            want[s] = (mg[s] - (mg[prev] if prev else 0.0)) / (fem[s] - (fem[prev] if prev else 0.0))
            prev = s
        for s in ("L1A", "L1"):
            want.setdefault(s, want["L6"])
        okk = all(abs(k.get(s, np.nan) - v) < 1e-9 for s, v in want.items()) and not any(s in k for s in ("B1", "B2", "B3"))
        # 예측: 줄인 B의 L6 = 표 L6 × (잰 질량 / FEM 질량), R = 거기에 R 증분 × R 증분 비
        p6 = groove_compare.scaled_pred(P, "L6", k); pR = groove_compare.scaled_pred(P, "R", k)
        n = "3164"
        t6 = P["L6"]["df"][n] * want["L6"] if name == "줄인 B" else (
            P["L1A"]["df"][n] * want["L1A"] + (P["L1"]["df"][n] - P["L1A"]["df"][n]) * want["L1"]
            + (P["L6"]["df"][n] - P["L1"]["df"][n]) * want["L6"])
        tR = t6 + (P["R"]["df"][n] - P["L6"]["df"][n]) * want["R"]
        okp = abs(p6[n] - t6) < 1e-9 and abs(pR[n] - tR) < 1e-9
        print(f"질량 비례({name}): 인자 {'맞음' if okk else '틀림'} {{{', '.join(f'{s} {k.get(s, np.nan):.3f}' for s in want)}}}, "
              f"{n} 예측 L6 {p6[n]:+.3f} %(표 {P['L6']['df'][n]:+.3f})·R {pR[n]:+.3f} % {'맞음' if okp else '틀림'}")
        ok = ok and okk and okp
    return ok


def main():
    folder = os.path.join(WORK, "daq"); out = os.path.join(WORK, "out")
    shutil.rmtree(WORK, ignore_errors=True); os.makedirs(folder); os.makedirs(out)
    log = []
    okm = check_mass_chain(WORK) if os.path.exists(POLES) else True
    SES = {"A1": 1, "A2": 2, "B2": 3, "B3": 4, "C1": 5}
    for s, ses in SES.items():
        write_state(folder, s, ses, log)
    pd.DataFrame(log).to_csv(os.path.join(folder, "impact_log.csv"), index=False)
    pred_csv = os.path.join(HERE, "..", "fe-nominal", "s3_pred_states.csv")
    for s, ses in SES.items():
        groove_state.main(["--state", s, "--folder", folder, "--sessions", str(ses), "--pred", pred_csv,
                           "--nboot", "30", "--out", out])
    groove_compare.main(["--base", "A1", "A2", "--states", "B2", "--dir", out, "--pred", pred_csv, "--name", "synth"])
    groove_compare.main(["--base", "B3", "--states", "C1", "--dir", out, "--pred", pred_csv, "--name", "synth_C"])
    T = pd.read_csv(os.path.join(out, "synth.csv"))
    J = T[T.judge.astype(bool)].copy()
    base_true = {n: 0.5 * (TRUE_F["A1"][n] + TRUE_F["A2"][n]) for n in TRUE_F["A1"]}
    J["true"] = [100 * (TRUE_F["B2"][n] / base_true[n] - 1) for n in J["mode"].astype(str)]
    J["z"] = (J.d_pct - J["true"]) / J.sig_pct
    print(J[["mode", "kind", "d_pct", "true", "sig_pct", "z", "pred_pct", "verdict"]].round(3).to_string(index=False))
    S = json.load(open(os.path.join(out, "B2.json")))
    ok = (J.z.abs() < 3).mean()
    rep = open(os.path.join(out, "synth.md")).read()
    row = [l for l in rep.splitlines() if l.startswith("| B2 |")][0].split("|")
    a = float(row[8].split("±")[0]); c = float(row[9].split("±")[0])
    print(f"\n3σ 안 {100 * ok:.0f} % (판정 모드 {len(J)}개), 기울기 a {a:.2f}, 오프셋 c {c:+.3f} % (넣은 값 {OFFSET:+.2f} %)")
    print("교차 경고:", {n: v["cross"] for n, v in S["modes"].items() if v.get("cross")})
    passed = ok >= 0.9 and 0.8 <= a <= 1.2 and abs(c - OFFSET) <= 0.03
    # 안쪽 홈 증분(B3 기준): 참값 = C1/B3 − 1
    TC = pd.read_csv(os.path.join(out, "synth_C.csv"))
    JC = TC[TC.judge.astype(bool)].copy()
    JC["true"] = [100 * (TRUE_F["C1"][n] / TRUE_F["B3"][n] - 1) for n in JC["mode"].astype(str)]
    JC["z"] = (JC.d_pct - JC["true"]) / JC.sig_pct
    JC["pred_ok"] = [abs(p - (100 * ((1 + PREDS["C1"][n] / 100) / (1 + PREDS["B3"][n] / 100) - 1))) < 1e-9
                     for p, n in zip(JC.pred_pct, JC["mode"].astype(str))]
    print(JC[["mode", "d_pct", "true", "sig_pct", "z", "pred_pct", "verdict"]].round(3).to_string(index=False))
    okC = (JC.z.abs() < 3).mean()
    print(f"C1(B3 기준): 3σ 안 {100 * okC:.0f} % (판정 모드 {len(JC)}개), 예측 증분 계산 {'맞음' if JC.pred_ok.all() else '틀림'}")
    passed = passed and okC >= 0.9 and bool(JC.pred_ok.all())
    if "L1" in OFFS:
        passed = check_local(folder, out, log) and passed
    passed = passed and okm
    print("통과" if passed else "실패")
    shutil.rmtree(folder, ignore_errors=True)
    return 0 if passed else 1


def check_local(folder, out, log):
    """42번 채택안: 창 하나(L1)와 테이프 한 점(T-a) — 성분별 판정과 '판정 밖' 표시."""
    for s, ses in (("L1", 6), ("T-a", 7)):
        write_state(folder, s, ses, log)
    pd.DataFrame(log).to_csv(os.path.join(folder, "impact_log.csv"), index=False)
    for s, ses in (("L1", 6), ("T-a", 7)):
        groove_state.main(["--state", s, "--folder", folder, "--sessions", str(ses), "--pred", POLES, "--nboot", "30",
                           "--out", out])
    groove_compare.main(["--base", "A1", "A2", "--states", "L1", "--dir", out, "--pred", POLES, "--name", "synth_L"])
    groove_compare.main(["--base", "A1", "--states", "T-a", "--dir", out, "--pred", POLES, "--name", "synth_T",
                         "--sig-h", "0", "--thr", "0.06"])
    ok = True
    for name, st, base in (("synth_L", "L1", ("A1", "A2")), ("synth_T", "T-a", ("A1",))):
        T = pd.read_csv(os.path.join(out, f"{name}.csv"))
        bt = {n: np.mean([TRUE_F[b][n] for b in base]) for n in TRUE_F["A1"]}
        T["true"] = [100 * (TRUE_F[st][n] / bt[n] - 1) for n in T["mode"].astype(str)]
        T["z"] = (T.d_pct - T["true"]) / T.sig_pct
        J = T[T.judge.astype(bool)]
        print(T[["mode", "d_pct", "true", "sig_pct", "z", "pred_pct", "verdict"]].round(3).to_string(index=False))
        v = T.set_index(T["mode"].astype(str)).verdict
        want = {"L1": {"2142": "예측대로", "2134": "작음(맞음)"}, "T-a": {"2142": "예측대로", "2134": "작음(맞음)"}}[st]
        okv = all(v.get(n) == w for n, w in want.items())
        unk = T[T.unknown.astype(bool) & T["mode"].astype(str).map(lambda n: gl.MODE[n].judge)]   # 판정 극점 가운데
        oku = st != "L1" or (len(unk) > 0 and unk.verdict.str.startswith("판정 밖(짝 성분").all())
        z3 = (J.z.abs() < 3).mean() if len(J) else 0.0
        rep = open(os.path.join(out, f"{name}.md")).read()
        prow = [l for l in rep.splitlines() if l.startswith(f"| {st} |") and "→" in l][0].split("|")
        sp, sps, spp = float(prow[6].split("±")[0]), float(prow[6].split("±")[1]), float(prow[7])
        oks = abs(sp - spp) <= 3 * sps
        print(f"{st}: 3σ 안 {100 * z3:.0f} % (판정 {len(J)}개), 성분 판정 {'맞음' if okv else '틀림'} {dict(v[list(want)])}, "
              f"짝 성분 방향 모름 표시 {'맞음' if oku else '틀림'}({len(unk)}개), 증분 갈라짐 {sp:+.3f} ± {sps:.3f} 대 예측 {spp:+.3f}")
        ok = ok and z3 >= 0.9 and okv and oku and oks
    return ok


if __name__ == "__main__":
    sys.exit(main())
