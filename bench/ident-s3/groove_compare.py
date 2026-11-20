"""흠집(홈) 시험 — 상태 사이 주파수 변화를 FEM 예측과 비교하고 판정한다.

사용 예:
  python groove_compare.py --base A1 A2 --states B1 B2 B3 C1 C2 [--mass ../groove-test/mass_log.csv]
  python groove_compare.py --base B3 --states C1 C2 --name groove_report_C   # 안쪽 홈만의 증분(예측도 B3 대비)
- 상태 JSON은 groove_state.py 출력(<dir>/<state>.json).
- 기준은 --base 상태들의 역분산 가중 평균. 기준이 둘 이상이면 그 차이(다시 매달기)로 '매달기 바닥' σ_h를 정한다
  (판정 모드의 차 rms / √2, 최소 0.03 %). 기준이 하나면 0.05 %(30번 다시 매달기 점검).
- 예측: fe-nominal/s3_pred_poles.csv(극점별, 42번 — 국부 상태에서 h2 짝 두 성분이 다르다). 옛 FEM 짝별 표
  (s3_pred_states.csv)도 읽는다. --mass를 주면 실제 깎은 질량으로 예측을 비례 조정한다: 단계 사슬(창 L1a → L1 → L6 → R →
  C1 → C2, 또는 고리 B1 → B2 → B3 → C1 → C2)의 증분마다 (실제 증분 질량 / FEM 증분 질량)을 곱해 더한다.
  건너뛴 상태(예: 줄인 B의 A → L6)는 다음에 잰 상태의 증분 비를 같이 쓴다. 기준 질량은 mass_log.csv의 A 행(가공 직전)이다.
  테이프 상태(T-a·T-b·T-c·M75·M150·M120·M240)는 --tape(tape_log.csv)의 조각 질량 합으로 비례 조정한다.
- 국부 상태에서 성분 방향을 모르는 짝(h2 밖의 짝, 반 갈라짐 ≥ 0.05 %)은 쌍 평균만 예측하므로 '판정 밖'으로 둔다.
  표의 judge_state가 거짓인 극점(B 상태의 3748 — 41·42번 부호 미정)도 판정에서 뺀다.
- 같은 매달기 안의 비교(예: --base A1 --states T-a T-b T-c T0)는 --sig-h 0으로 매달기 바닥을 뺀다.
- 판정(35번 §2·§4): 문턱 ±0.15 %. 원판형(h2 짝·3748)과 패널(3164·3459·3639·6439·8321)의 부호,
  그리고 기준 모드(뒤판 4967·5617·5745)와의 대비(공통 오프셋 — 온도·매달기 — 에 영향받지 않는다).
- 출력: <dir>/groove_report.md, groove_report.csv, groove_report.png, groove_frf.png
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import groove_lib as gl  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PRED = os.path.join(HERE, "..", "fe-nominal", "s3_pred_poles.csv")
if not os.path.exists(PRED):
    PRED = os.path.join(HERE, "..", "fe-nominal", "s3_pred_states.csv")
TAPE = os.path.join(HERE, "..", "groove-test", "tape_log.csv")
CHAINS = {"window": ["L1A", "L1", "L6", "R", "C1", "C2"], "ring": ["B1", "B2", "B3", "C1", "C2"]}
TAPE_STATES = ("T-A", "T-B", "T-C", "M75", "M150", "M120", "M240")
OUT = os.path.join(HERE, "..", "groove-test", "out")
DISK = ["2134", "2142", "3748"]
PANEL = ["3164", "3459", "3639", "6439", "8321"]
BACK = ["4967", "5617", "5745"]
KIND_KO = {"disk": "원판형", "panel": "패널", "mixed": "섞임", "back": "뒤판", "hub": "허브", "cand": "후보", "none": "—"}


def load(d, s):
    with open(os.path.join(d, f"{s}.json")) as fh:
        return json.load(fh)


def comb(S, n):
    c = S["modes"].get(n, {}).get("comb", {})
    return c.get("f", np.nan), c.get("se", np.nan)


def pooled(S, n):
    """묶음 적합으로 낸 값인가(타격별 값으로 물러난 모드는 약해서 판정에 쓰지 않는다)."""
    return S["modes"].get(n, {}).get("comb", {}).get("method") == "pool"


def temps(S):
    q = list(S["qc"].values())
    return float(np.nanmean([x["T0"] for x in q])), float(np.nanmean([x["T1"] for x in q]))


def pred_table(path):
    """→ {상태(대문자): dict(df={극점: %}, removed_mg, half={극점: 반 갈라짐 %}, known={극점: 성분 방향을 아는가},
    judge={극점: 이 상태에서 판정하는가})}."""
    p = pd.read_csv(path)
    out = {}
    if "mode" in p.columns:                                          # 극점별 표(s3_pred_poles.csv)
        for st, g in p.groupby("state"):
            g = g.assign(mode=g["mode"].astype(str)).set_index("mode")
            out[str(st).upper()] = dict(df={n: float(v) for n, v in g.pred_pct.items() if np.isfinite(v)},
                                        removed_mg=float(g.removed_mg.iloc[0]),
                                        half={n: float(v) for n, v in g.half_split_pct.items()},
                                        known={n: bool(v) for n, v in g.member_known.items()},
                                        judge={n: bool(v) for n, v in g.judge_state.items()})
        return out
    for st, g in p.groupby("state"):                                 # 옛 FEM 짝별 표(s3_pred_states.csv)
        col = g.set_index("label")
        v = {}
        for m in gl.MODES:
            if m.fem and m.fem in col.index:
                x = col.loc[m.fem, "df_pct_res"]
                v[m.name] = float(x if np.isfinite(x) else col.loc[m.fem, "df_pct_lin"])
        out[str(st).upper()] = dict(df=v, removed_mg=float(g.removed_mg.iloc[0]), half={}, known={}, judge={})
    return out


def mass_ratios(mass_csv, P, tape_csv=None):
    """실제 질량 → 상태별 비례 인자. 반환: (깎은/붙인 질량 {상태: mg}, 인자 {상태: 증분 비}).

    mass_log.csv(state, mass_g, …)의 A 행 평균이 기준이다(A2 등 A로 시작하는 다른 상태는 기준에 넣지 않는다).
    사슬 상태는 증분 비(실제 증분 / FEM 증분), 테이프 상태는 (조각 질량 합 / FEM 붙인 질량)이다.
    사슬에서 건너뛴 상태(11-16 줄인 B: L1a·L1 없이 A → L6)는 다음에 잰 상태의 증분 비를 같이 쓴다.
    사슬 고유 상태(창 L1a–R, 고리 B1–B3)를 하나도 재지 않은 사슬은 보지 않는다(C1·C2는 실제로 깎은 사슬 몫)."""
    removed, r = {}, {}
    if mass_csv and os.path.exists(mass_csv):
        m = pd.read_csv(mass_csv, comment="#").dropna(subset=["mass_g"])
        m = m.assign(state=m.state.astype(str).str.upper()).groupby("state").mass_g.mean()
        if "A" in m.index:
            removed = {s: 1e3 * (m["A"] - v) for s, v in m.items() if not s.startswith("A")}
        else:
            print(f"[경고] {mass_csv}에 A(가공 전) 질량이 없어 예측을 질량으로 비례 조정하지 않는다")
        for chain in CHAINS.values():
            own = [s for s in chain if sum(s in c for c in CHAINS.values()) == 1]
            if not any(s in removed for s in own):
                continue
            last, skipped = None, []
            for s in chain:
                if s not in P:
                    break
                if s not in removed:
                    skipped.append(s)
                    continue
                fem_inc = P[s]["removed_mg"] - (P[last]["removed_mg"] if last else 0.0)
                if abs(fem_inc) > 1e-9:
                    k = (removed[s] - (removed[last] if last else 0.0)) / fem_inc
                    for x in skipped + [s]:
                        r.setdefault(x, k)
                last, skipped = s, []
    if tape_csv and os.path.exists(tape_csv):
        t = pd.read_csv(tape_csv, comment="#").dropna(subset=["mass_mg"])
        t = t.assign(state=t.state.astype(str).str.upper())
        tot = t.groupby("state").mass_mg.sum().to_dict()
        if "M150" in tot:
            tot["M150"] += tot.get("M75", 0.0)                     # M150 = M75 조각 + 더한 조각
        if "M240" in tot:
            tot["M240"] += tot.get("M120", 0.0)                    # M240 = M120 띠 + 더한 띠(11-12부터 띠 20 mg)
        for s, mg in tot.items():
            if s in P and abs(P[s]["removed_mg"]) > 1e-9:
                removed[s] = -mg
                r[s] = -mg / P[s]["removed_mg"]
    return removed, r


def scaled_pred(P, st, ratio):
    """비례 조정한 예측 {극점: %}. 사슬 상태는 증분마다 인자를 곱해 더한다(인자가 없으면 1 → 표 값 그대로)."""
    S = st.upper()
    if S.endswith("-RH"):
        S = S[:-3]
    if S not in P:
        return {}
    cands = [c for c in CHAINS.values() if S in c and all(x in P for x in c[:c.index(S) + 1])]
    if not cands:
        return {n: v * ratio.get(S, 1.0) for n, v in P[S]["df"].items()}
    chain = cands[0]
    if len(cands) > 1:                                               # C1·C2: 실제 질량이 있는 쪽 사슬(창 또는 고리)
        withm = [c for c in cands if any(x in ratio for x in c[:c.index("C1")])]
        chain = withm[0] if withm else cands[0]
    out, prev = {}, {}
    for c in chain[:chain.index(S) + 1]:
        k = ratio.get(c, 1.0)
        for n, v in P[c]["df"].items():
            out[n] = out.get(n, 0.0) + (v - prev.get(n, 0.0)) * k
        prev = P[c]["df"]
    return out


def wls(x, y, s):
    """y = a·x + c 가중 최소제곱."""
    A = np.column_stack([x, np.ones_like(x)]) / s[:, None]
    b = y / s
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    cov = np.linalg.inv(A.T @ A)
    r = b - A @ coef
    chi2 = float(r @ r / max(len(y) - 2, 1))
    return coef, np.sqrt(np.diag(cov)), chi2


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", nargs="+", required=True)
    ap.add_argument("--states", nargs="+", required=True)
    ap.add_argument("--dir", default=OUT)
    ap.add_argument("--pred", default=PRED)
    ap.add_argument("--mass")
    ap.add_argument("--tape", default=TAPE, help="테이프 조각 질량 기록(tape_log.csv)")
    ap.add_argument("--sig-h", type=float, help="매달기 바닥 σ_h [%%] 직접 지정(같은 매달기 안 비교는 0)")
    ap.add_argument("--thr", type=float, default=0.15)
    ap.add_argument("--name", default="groove_report")
    a = ap.parse_args(argv)
    B = [load(a.dir, s) for s in a.base]
    Ss = [load(a.dir, s) for s in a.states]
    P = pred_table(a.pred) if os.path.exists(a.pred) else {}
    removed, ratio = mass_ratios(a.mass, P, a.tape)
    names = [m.name for m in gl.MODES]
    judged = [m.name for m in gl.MODES if m.judge]

    # ------------------------------------------------ 기준
    base = {}
    for n in names:
        v = [(comb(S, n)) for S in B]
        v = [(f, s) for f, s in v if np.isfinite(f) and np.isfinite(s) and s > 0]
        if v:
            w = np.array([1 / s ** 2 for _, s in v]); f = np.array([f for f, _ in v])
            base[n] = (float((w * f).sum() / w.sum()), float(1 / np.sqrt(w.sum())), all(pooled(S, n) for S in B))
    if len(B) >= 2:
        d = []
        for n in judged:
            f1, _ = comb(B[0], n); f2, _ = comb(B[-1], n)
            if np.isfinite(f1) and np.isfinite(f2) and pooled(B[0], n) and pooled(B[-1], n):
                d.append(100 * (f2 / f1 - 1))
        sig_h = max(0.03, float(np.sqrt(np.mean(np.square(d))) / np.sqrt(2))) if d else 0.05
        hang = {n: 100 * (comb(B[-1], n)[0] / comb(B[0], n)[0] - 1) for n in names
                if np.isfinite(comb(B[0], n)[0]) and np.isfinite(comb(B[-1], n)[0])}
    else:
        sig_h, hang = 0.05, {}
    if a.sig_h is not None:
        sig_h = a.sig_h
    Tb = np.nanmean([temps(S) for S in B], axis=0)

    # ------------------------------------------------ 상태별
    rows, summ = [], []
    base_key = a.base[0].upper()
    base_pred = scaled_pred(P, base_key, ratio) if base_key in P else {}    # 기준이 손상 상태(예: B3)면 예측은 그 대비 증분
    for st, S in zip(a.states, Ss):
        pr = scaled_pred(P, st, ratio)
        Sk = st.upper()[:-3] if st.upper().endswith("-RH") else st.upper()
        info = P.get(Sk, {})
        half, known, jst = info.get("half", {}), info.get("known", {}), info.get("judge", {})
        if base_pred:
            pr = {n: 100 * ((1 + v / 100) / (1 + base_pred.get(n, 0.0) / 100) - 1) for n, v in pr.items()}
        Ts = temps(S); dT = Ts - Tb
        rec = []
        for n in names:
            f, se = comb(S, n)
            if n not in base or not np.isfinite(f):
                continue
            fb, seb, pool_b = base[n]
            dpct = 100 * (f / fb - 1)
            sig = float(np.sqrt((100 * se / fb) ** 2 + (100 * seb / fb) ** 2 + sig_h ** 2))
            strong = pool_b and pooled(S, n)
            crossed = bool(S["modes"][n].get("cross"))
            hs = float(half.get(n, 0.0))
            unknown = (not known.get(n, True)) and hs >= 0.05           # 짝 성분 방향 모름(국부 상태의 h2 밖 짝)
            excl = not jst.get(n, True)                                  # 이 상태에서 판정하지 않는 극점(예: B의 3748)
            rec.append(dict(state=st, mode=n, kind=gl.MODE[n].kind, fem=gl.MODE[n].fem or "",
                            judge=gl.MODE[n].judge and strong and not crossed and not unknown and not excl,
                            weak=not strong, crossed=crossed, unknown=unknown, excl=excl, half_split=hs,
                            f_base=fb, f_state=f, d_pct=dpct, sig_pct=sig, pred_pct=pr.get(n, np.nan),
                            cross=";".join(S["modes"][n].get("cross") or [])))
        R = pd.DataFrame(rec)
        J = R[R.judge & np.isfinite(R.pred_pct)]
        if len(J) >= 4:
            (sc, off), (se_sc, se_off), chi2 = wls(J.pred_pct.to_numpy(), J.d_pct.to_numpy(), J.sig_pct.to_numpy())
        else:
            sc = off = se_sc = se_off = chi2 = np.nan
        R["d_minus_off"] = R.d_pct - (off if np.isfinite(off) else 0.0)

        def verdict(r):
            if r.crossed and gl.MODE[r["mode"]].judge:
                return "교차 — 판정 보류(6깃 패턴으로 가름)"
            if r.weak and gl.MODE[r["mode"]].judge:
                return "약함(묶음 적합 안 됨)"
            if r.unknown and gl.MODE[r["mode"]].judge:
                return f"판정 밖(짝 성분 방향 모름, 반 갈라짐 {r.half_split:.2f} %)"
            if r.excl and gl.MODE[r["mode"]].judge:
                return "판정 밖(이 상태에서 부호 미정)"
            if not r.judge or not np.isfinite(r.pred_pct):
                return "판정 밖"
            d, p = r.d_minus_off, r.pred_pct           # 공통 오프셋(온도·매달기)을 뺀 값으로 판정
            if abs(p) >= a.thr:
                if d * p > 0 and abs(d) >= a.thr:
                    return "예측대로"
                if d * p > 0:
                    return "같은 부호·문턱 아래"
                return "반대 부호" if abs(d) >= a.thr else "문턱 안"
            return "작음(맞음)" if abs(d) < a.thr else "작음 예측·크게 움직임"
        R["verdict"] = R.apply(verdict, axis=1)
        rows.append(R)
        # h2 짝: 쌍 평균 이동(방위 무관)과 증분 갈라짐(원고 규약). 축대칭 홈이면 FEM 갈라짐 예측은 0이다.
        Rp = R.set_index("mode")
        if set(gl.PAIR) <= set(Rp.index):
            p1, p2 = Rp.loc[gl.PAIR[0]], Rp.loc[gl.PAIR[1]]
            sp0 = 100 * (p2.f_base - p1.f_base) / (0.5 * (p1.f_base + p2.f_base))
            sp1 = 100 * (p2.f_state - p1.f_state) / (0.5 * (p1.f_state + p2.f_state))
            pair = dict(pair_mean=0.5 * (p1.d_pct + p2.d_pct), pair_mean_sig=0.5 * (p1.sig_pct + p2.sig_pct),
                        pair_mean_off=0.5 * (p1.d_minus_off + p2.d_minus_off),
                        pair_mean_pred=0.5 * (p1.pred_pct + p2.pred_pct), split0_Hz=p2.f_base - p1.f_base,
                        split_Hz=p2.f_state - p1.f_state, split_inc=sp1 - sp0,
                        split_inc_pred=float(p2.pred_pct - p1.pred_pct),
                        split_inc_sig=float(np.hypot(p1.sig_pct, p2.sig_pct)), pair_ok=bool(p1.judge and p2.judge))
        else:
            pair = dict(pair_mean=np.nan, pair_mean_sig=np.nan, pair_mean_off=np.nan, pair_mean_pred=np.nan,
                        split0_Hz=np.nan, split_Hz=np.nan, split_inc=np.nan, split_inc_pred=np.nan, split_inc_sig=np.nan,
                        pair_ok=False)

        def grp(ns, col):
            x = R[R["mode"].isin(ns) & R.judge]
            return (float(x[col].mean()) if len(x) else np.nan), len(x)
        dm, nd = grp(DISK, "d_pct"); pm, npn = grp(PANEL, "d_pct"); bm, nb = grp(BACK, "d_pct")
        dpd, _ = grp(DISK, "pred_pct"); ppd, _ = grp(PANEL, "pred_pct"); bpd, _ = grp(BACK, "pred_pct")
        summ.append(dict(state=st, dT0=float(dT[0]), dT1=float(dT[1]), scale=sc, se_scale=se_sc, offset=off,
                         se_offset=se_off, chi2=chi2, disk=dm, disk_pred=dpd, panel=pm, panel_pred=ppd, back=bm,
                         back_pred=bpd, disk_vs_back=dm - bm, disk_vs_back_pred=dpd - bpd,
                         panel_vs_back=pm - bm, panel_vs_back_pred=ppd - bpd,
                         n_ok=int((R.verdict == "예측대로").sum()), n_judge=int((R.judge & np.isfinite(R.pred_pct)
                                                                               & (R.pred_pct.abs() >= a.thr)).sum()),
                         n_opp=int((R.verdict == "반대 부호").sum()), removed_mg=removed.get(st.upper(), np.nan), **pair))
    T = pd.concat(rows, ignore_index=True)
    T.to_csv(os.path.join(a.dir, f"{a.name}.csv"), index=False)
    Sm = pd.DataFrame(summ)

    # ------------------------------------------------ 보고서
    L = [f"# 흠집(홈) 시험 비교 — 기준 {'+'.join(a.base)}", ""]
    if base_pred:
        L.append(f"- 기준이 손상 상태({base_key})라서 예측은 {base_key} 대비 증분이다.")
    L.append(f"- 기준 온도(ai0/ai1) {Tb[0]:.2f}/{Tb[1]:.2f} °C. 매달기 바닥 σ_h = {sig_h:.3f} %"
             + (f"(기준 {a.base[0]}→{a.base[-1]} 차 rms/√2)" if len(B) >= 2 else "(기본값)") + f". 문턱 ±{a.thr} %.")
    if hang:
        L.append("- 기준끼리 차(다시 매달기) [%]: " + ", ".join(f"{n} {v:+.3f}" for n, v in hang.items() if n in judged))
    if removed:
        L.append("- 깎은 질량(측정) [mg]: " + ", ".join(f"{s} {v:.1f}" for s, v in removed.items())
                 + " — 예측은 FEM 질량 대비로 비례 조정")
    L.append("")
    L.append("## 상태별 요약")
    L.append("")
    L.append("| 상태 | ΔT ai0/ai1 [°C] | 원판형 평균 측정/예측 [%] | 패널 평균 | 뒤판 기준 | 원판형 − 뒤판 | 패널 − 뒤판 | 기울기 a | 오프셋 c [%] | 예측대로 / 반대 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in summ:
        L.append(f"| {r['state']} | {r['dT0']:+.2f}/{r['dT1']:+.2f} | {r['disk']:+.3f} / {r['disk_pred']:+.3f} | "
                 f"{r['panel']:+.3f} / {r['panel_pred']:+.3f} | {r['back']:+.3f} / {r['back_pred']:+.3f} | "
                 f"{r['disk_vs_back']:+.3f} / {r['disk_vs_back_pred']:+.3f} | {r['panel_vs_back']:+.3f} / {r['panel_vs_back_pred']:+.3f} | "
                 f"{r['scale']:.2f} ± {r['se_scale']:.2f} | {r['offset']:+.3f} ± {r['se_offset']:.3f} | "
                 f"{r['n_ok']}/{r['n_judge']} · {r['n_opp']} |")
    L.append("")
    L.append("- 기울기 a = 측정/예측 비(1이면 FEM 크기 그대로), 오프셋 c = 모든 모드에 같은 변화(온도·매달기).")
    L.append("- 모드별 판정은 오프셋 c를 뺀 변화로 한다: 예측 |Δ| ≥ 문턱이면 같은 부호·문턱 넘음 = '예측대로'.")
    L.append(f"- 온도 1 °C는 약 {gl.DF_T:+.3f} %(재료값 어림).")
    L += ["", f"## h2 짝({'/'.join(gl.PAIR)}) — 쌍 평균 이동과 증분 갈라짐(원고 규약)", "",
          "| 상태 | 쌍 평균 [%] ± σ | 오프셋 뺀 쌍 평균 | 예측 [%] | 갈라짐 기준 → 상태 [Hz] | 증분 갈라짐 [% of f̄] ± σ | 예측 증분 갈라짐 | 두 극점 묶음 적합 |",
          "|---|---|---|---|---|---|---|---|"]
    for r in summ:
        L.append(f"| {r['state']} | {r['pair_mean']:+.3f} ± {r['pair_mean_sig']:.3f} | {r['pair_mean_off']:+.3f} | "
                 f"{r['pair_mean_pred']:+.3f} | {r['split0_Hz']:.2f} → {r['split_Hz']:.2f} | "
                 f"{r['split_inc']:+.3f} ± {r['split_inc_sig']:.3f} | {r['split_inc_pred']:+.3f} | {'예' if r['pair_ok'] else '아니오'} |")
    L.append("")
    L.append("- 쌍 평균은 두 극점 변화의 평균이다. 증분 갈라짐 = (상태 갈라짐 − 기준 갈라짐)/평균 주파수다.")
    L.append("- 6깃 대칭인 상태(고리·깃 사이 여섯 L6·R·고리 테이프)는 예측 증분 갈라짐이 0이다. 국부 상태(L1a·L1·T)는 "
             "한 성분만 움직여 갈라짐이 변한다(42번). 측정과 예측의 차가 2σ를 넘으면 창 자리·짝 방향을 다시 본다.")
    for st in a.states:
        R = T[T.state == st]
        L += ["", f"## {st}", "", "| 모드 | 종류 | FEM | 기준 [Hz] | 상태 [Hz] | 변화 [%] | 오프셋 뺀 변화 | ±σ | 예측 [%] | 판정 | 교차 |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in R.itertuples():
            L.append(f"| {r.mode} | {KIND_KO.get(r.kind, r.kind)} | {r.fem} | {r.f_base:.2f} | {r.f_state:.2f} | {r.d_pct:+.3f} | "
                     f"{r.d_minus_off:+.3f} | {r.sig_pct:.3f} | {r.pred_pct:+.3f} | {r.verdict} | {r.cross} |")
    txt = "\n".join(L) + "\n"
    with open(os.path.join(a.dir, f"{a.name}.md"), "w") as fh:
        fh.write(txt)
    print(txt)
    figure(T, Sm, a, os.path.join(a.dir, f"{a.name}.png"))
    frf_figure(a, os.path.join(a.dir, f"{a.name}_frf.png"))
    print(f"→ {os.path.join(a.dir, a.name)}.md·.csv·.png·_frf.png")


def figure(T, Sm, a, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(a.states)
    fig, ax = plt.subplots(2, n, figsize=(4.2 * n + 1, 7.5), squeeze=False)
    col = {"disk": "C3", "panel": "C0", "mixed": "C4", "back": "C2", "hub": "0.6", "cand": "0.6", "none": "0.6"}
    for i, st in enumerate(a.states):
        R = T[T.state == st].reset_index(drop=True)
        x = np.arange(len(R))
        axu = ax[0, i]
        axu.axhspan(-a.thr, a.thr, color="0.93", zorder=0)
        for j, r in R.iterrows():
            c = col.get(r.kind, "0.5")
            axu.errorbar(j, r.d_pct, yerr=2 * r.sig_pct, fmt="o", color=c, ms=5, mfc=c if r.judge else "white")
            if np.isfinite(r.pred_pct):
                axu.plot(j, r.pred_pct, "_", color="k", ms=12, mew=2)
        axu.axhline(0, color="k", lw=0.6)
        axu.set_xticks(x); axu.set_xticklabels(R["mode"], rotation=90, fontsize=6.5)
        axu.set_title(f"{st}: measured (dots, ±2σ) vs FEM (black bars)", fontsize=8.5, loc="left")
        if i == 0:
            axu.set_ylabel("frequency change vs baseline [%]")
        axl = ax[1, i]
        J = R[R.judge & np.isfinite(R.pred_pct)]
        for r in J.itertuples():
            axl.errorbar(r.pred_pct, r.d_pct, yerr=2 * r.sig_pct, fmt="o", color=col.get(r.kind, "0.5"), ms=5)
            axl.annotate(r.mode, (r.pred_pct, r.d_pct), (3, 2), textcoords="offset points", fontsize=6)
        lim = max(0.3, float(np.nanmax(np.abs(np.r_[J.pred_pct, J.d_pct]))) * 1.15) if len(J) else 1
        axl.plot([-lim, lim], [-lim, lim], "k--", lw=0.7)
        s = Sm[Sm.state == st].iloc[0]
        if np.isfinite(s.scale):
            xx = np.array([-lim, lim]); axl.plot(xx, s.scale * xx + s.offset, color="C1", lw=1,
                                                 label=f"a {s.scale:.2f}, c {s.offset:+.3f} %")
            axl.legend(fontsize=7, loc="upper left")
        axl.axhline(0, color="0.7", lw=0.5); axl.axvline(0, color="0.7", lw=0.5)
        axl.set_xlim(-lim, lim); axl.set_ylim(-lim, lim)
        axl.set_xlabel("FEM prediction [%]")
        if i == 0:
            axl.set_ylabel("measured [%]")
    import textwrap                                                 # 상태 하나면 그림이 좁다 — 제목을 폭에 맞춰 줄바꿈
    fig.suptitle(textwrap.fill("groove test: red = disk-type (h2 pair, 3748), blue = front-shroud panel, green = back-shroud "
                               "reference, violet = mixed; open = not judged", width=int(13 * (4.2 * n + 1))), fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def frf_figure(a, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bands = [(2.10, 2.17), (3.05, 3.30), (3.33, 3.70), (3.68, 3.86), (6.15, 6.62), (8.05, 8.40)]
    fig, ax = plt.subplots(1, len(bands), figsize=(3.2 * len(bands), 3.6))
    for st in list(a.base) + list(a.states):
        fz = os.path.join(a.dir, f"{st}_frf.npz")
        if not os.path.exists(fz):
            continue
        Z = np.load(fz)
        keys = [k for k in Z.files if k.startswith("H_") and "xi0.6" in k] or [k for k in Z.files if k.startswith("H_")]
        if not keys:
            continue
        H = np.abs(Z[keys[0]]); fr = Z["fr"] / 1e3
        for i, (lo, hi) in enumerate(bands):
            s = (fr >= lo) & (fr <= hi)
            ax[i].semilogy(fr[s], H[s], lw=0.9, label=st, ls="--" if st in a.base else "-")
    for i, (lo, hi) in enumerate(bands):
        ax[i].set_xlim(lo, hi); ax[i].set_xlabel("kHz")
    ax[0].set_ylabel("|H1| [Pa/N] (xi0.6 if measured)")
    ax[0].legend(fontsize=7)
    fig.suptitle("mean FRF by state (dashed = baseline)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
