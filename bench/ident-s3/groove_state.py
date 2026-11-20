"""흠집(홈) 시험 — 한 상태(A1·A2·B1…)의 타격을 읽어 품질을 점검하고 모드 주파수(20 N 환산)를 낸다.

사용 예:
  python groove_state.py --state A1 --folder /mnt/nas/data/GY_DATA/impact_groove/impact_groove --sessions 1 \\
         --locations back_vane1_xi0.3 back_vane1_xi0.6
  python groove_state.py --state B2 --folder ... --sessions 4 --ref ../groove-test/out/A1.json
- 대역 중심 = 기준 주파수(10-30 표, 또는 --ref 상태의 측정값) × (1 + 그 상태의 FEM 예측 이동).
  예측은 fe-nominal/s3_pred_poles.csv(극점별: 고리 B1–B3·C1·C2, 창 L1a·L1·L6·R, 테이프 T-a·T-b·T-c·M75·M150 — 42번).
  A·T0 상태는 이동 0, '-rh'(다시 매달기)는 본 상태의 예측.
- 위치(ξ0.3·ξ0.6)마다 타격별 적합과 묶음 적합을 하고, 위치를 합친 값은 묶음 적합의 역분산 가중 평균이다.
- 출력: <out>/<state>.json, <state>_frf.npz(위치별 평균 H1·코히어런스), <state>_qc.png. 화면에 품질 경고를 찍는다.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import groove_lib as gl  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PRED = os.path.join(HERE, "..", "fe-nominal", "s3_pred_poles.csv")
if not os.path.exists(PRED):
    PRED = os.path.join(HERE, "..", "fe-nominal", "s3_pred_states.csv")
OUT = os.path.join(HERE, "..", "groove-test", "out")
FRF_NFFT = 1 << 17
FRF_BAND = (1000.0, 10000.0)
COH_BAND = (1900.0, 9000.0)


def load_shifts(pred_path: str, state: str) -> dict:
    """그 상태의 FEM 예측 이동[%]을 측정 극점 이름으로.

    극점별 표(s3_pred_poles.csv, 열 mode)면 그대로 쓴다 — 국부 상태(L1a·L1·T-a…)에서 h2 짝 두 성분이 다르다.
    FEM 짝별 표(s3_pred_states.csv, 열 label)면 같은 FEM 짝을 가진 극점은 같은 값이다.
    'L1-rh'처럼 다시 매달기 상태는 본 상태(L1)의 예측을, A·T0(테이프 뗀 뒤)는 0을 쓴다."""
    st = state.upper()
    if st.endswith("-RH"):
        st = st[:-3]
    if st.startswith("A") or st == "T0" or not os.path.exists(pred_path):
        return {}
    p = pd.read_csv(pred_path)
    s = p[p.state.str.upper() == st]
    if s.empty:
        print(f"[경고] 예측 표에 {state}가 없다 — 대역 중심을 옮기지 않는다")
        return {}
    if "mode" in s.columns:
        return {str(r["mode"]): float(r.pred_pct) for _, r in s.iterrows() if np.isfinite(r.pred_pct)}
    col = s.set_index("label")
    out = {}
    for m in gl.MODES:
        if m.fem and m.fem in col.index:
            v = col.loc[m.fem, "df_pct_res"]
            out[m.name] = float(v if np.isfinite(v) else col.loc[m.fem, "df_pct_lin"])
    return out


def robust(v):
    v = np.asarray(v, float)
    if v.size < 2:
        return np.nan, np.nan, v.size, np.nan
    md = np.median(v)
    res = np.abs(v / md - 1)
    keep = res < max(3 * 1.4826 * np.median(res), 2e-4)
    v = v[keep]
    if v.size < 2:
        return float(md), np.nan, int(v.size), np.nan
    sd = np.std(v, ddof=1)
    return float(np.median(v)), float(1.2533 * sd / np.sqrt(v.size)), int(v.size), float(100 * sd / np.median(v))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", required=True, help="A1·A2·B1·B2·B3·C1·C2 등")
    ap.add_argument("--folder", required=True, help="DAQ 저장 폴더(.npz와 impact_log.csv)")
    ap.add_argument("--sessions", type=int, nargs="*")
    ap.add_argument("--locations", nargs="*", default=["back_vane1_xi0.3", "back_vane1_xi0.6"])
    ap.add_argument("--date", help="YYYYMMDD(파일 이름 앞부분)")
    ap.add_argument("--from", dest="t_from", help="HH:MM")
    ap.add_argument("--to", dest="t_to", help="HH:MM")
    ap.add_argument("--notes", help="DAQ '메모' 칸이 이 값인 타격만")
    ap.add_argument("--pred", default=PRED)
    ap.add_argument("--expect", help="대역을 잡을 예측 상태(기본 = --state)")
    ap.add_argument("--ref", help="기준 상태 JSON — 그 측정값을 대역 중심의 기준으로 쓴다")
    ap.add_argument("--modes", nargs="*", help="이 모드만(기본 전부)")
    ap.add_argument("--nboot", type=int, default=60)
    ap.add_argument("--min-force", type=float, default=5.0)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--force", action="store_true", help="채널 점검에 걸려도 계속")
    a = ap.parse_args(argv)
    t_start = time.time()
    os.makedirs(a.out, exist_ok=True)

    hits = gl.list_hits(a.folder, a.sessions, a.locations, a.date, a.t_from, a.t_to, a.notes)
    if not hits:
        raise SystemExit("고른 타격이 없다 — 폴더·세션·위치·날짜를 확인")
    bad = gl.check_channels(hits)
    if bad and not a.force:
        raise SystemExit("채널 점검 실패(마이크 CH6 = PXI1Slot5/ai2, 해머 CH7이어야 한다):\n  " + "\n  ".join(bad[:5]))

    # ---------------------------------------------------------------- 대역
    shifts = load_shifts(a.pred, a.expect or a.state)
    base = {m.name: m.f for m in gl.MODES}
    if a.ref:
        R = json.load(open(a.ref))
        for n, v in R["modes"].items():
            if v.get("comb") and np.isfinite(v["comb"].get("f", np.nan)):
                base[n] = v["comb"]["f"]
    names = a.modes or [m.name for m in gl.MODES]
    # 기준값을 바꾼 대역 잡기: plan_bands는 MODE.f를 쓰므로 비율로 옮긴다
    shift_eff = {n: 100 * (base[n] * (1 + shifts.get(n, 0.0) / 100) / gl.MODE[n].f - 1) for n in names}
    bands = gl.plan_bands(shift_eff, names)

    # ---------------------------------------------------------------- 타격별
    locs = sorted({h["location"] for h in hits}, key=lambda s: (len(s), s))
    rows, spec, frf = [], {}, {}
    for h in hits:
        N = float(h["peak_force_N"])
        f, m, ti, fs = gl.load_signals(h)
        t0, late = gl.free_decay_start(f, N, ti, fs)
        pre = float(m[:ti - 100].std())
        T = np.asarray(h.get("temperature_C") or [np.nan, np.nan], float)
        r = dict(file=h["file"], loc=h["location"], session=int(h["session"]), ts=str(h["timestamp"]), N=N, late=late,
                 t0=t0, pre=pre, T0=float(T[0]), T1=float(T[1]) if T.size > 1 else np.nan, status=h["status"],
                 resp_peak=h["resp_peak"], quality=str(h.get("quality", "")), f={})
        if N >= a.min_force and h["status"] != "rejected":        # DAQ '직전 타격 제외'는 쓰지 않는다
            X = gl.decay_spectrum(m, ti, fs, t0)
            fr = np.fft.rfftfreq(gl.NFFT, 1 / fs)
            for bi, b in enumerate(bands):
                lo, hi, k = b["lo"], b["hi"], len(b["modes"])
                sel = (fr >= lo - 30) & (fr <= hi + 30)
                spec[(h["file"], bi)] = (fr[sel], X[sel])
                inn = (fr[sel] >= lo - 8) & (fr[sel] <= hi + 8)
                fk, zk, amp, res, err = gl.band_fit(fr[sel][inn], X[sel][inn], lo, hi, k)
                ok = err < 0.3 and np.all((fk > lo) & (fk < hi)) and (k == 1 or np.min(np.diff(fk)) > 3)
                for j, n in enumerate(b["modes"]):
                    r["f"][n] = [float(fk[j]), float(zk[j]), float(err), float(np.abs(res[j]))] if ok else None
            if not late:                                          # 평균 FRF·코히어런스(늦은 재타격 없는 타격)
                fw = np.zeros_like(f); i1 = ti + int(0.005 * fs)
                fw[ti - 200:i1] = f[ti - 200:i1]
                seg = slice(ti - 200, ti - 200 + FRF_NFFT)
                xm = m[seg] * np.exp(np.log(0.01) * np.arange(FRF_NFFT) / FRF_NFFT)
                Fk = np.fft.rfft(fw[seg], FRF_NFFT); Xk = np.fft.rfft(xm, FRF_NFFT)
                acc = frf.setdefault(h["location"], [0, 0, 0, 0])
                acc[0] = acc[0] + np.conj(Fk) * Xk; acc[1] = acc[1] + np.abs(Fk) ** 2
                acc[2] = acc[2] + np.abs(Xk) ** 2; acc[3] += 1
        rows.append(r)
    print(f"[{a.state}] 타격 {len(rows)}개, 위치 {locs} — 타격별 적합 끝 ({time.time() - t_start:.0f}s)", flush=True)

    # ---------------------------------------------------------------- 품질 점검
    qc, warns = {}, []
    for loc in locs:
        R = [r for r in rows if r["loc"] == loc]
        use = [r for r in R if r["N"] >= a.min_force and r["status"] != "rejected"]
        q = dict(hits=len(R), used=len(use), late=sum(r["late"] for r in use),
                 flagged=sum(r["status"] == "flagged" for r in R), rejected=sum(r["status"] == "rejected" for r in R),
                 force_med=float(np.median([r["N"] for r in R])), force_min=float(min(r["N"] for r in R)),
                 force_max=float(max(r["N"] for r in R)), resp_med=float(np.nanmedian([r["resp_peak"] for r in R])),
                 pre_mPa=float(np.median([r["pre"] for r in R]) * 1e3),
                 T0=float(np.nanmean([r["T0"] for r in R])), T1=float(np.nanmean([r["T1"] for r in R])),
                 T0_span=float(np.nanmax([r["T0"] for r in R]) - np.nanmin([r["T0"] for r in R])),
                 sessions=sorted({r["session"] for r in R}), first=R[0]["ts"], last=R[-1]["ts"])
        if loc in frf and frf[loc][3] >= 2:
            S, Sff, Sxx, n = frf[loc]
            fr2 = np.fft.rfftfreq(FRF_NFFT, 1 / gl.FS_EXPECTED)
            coh = np.abs(S) ** 2 / np.maximum(Sff * Sxx, 1e-300)
            cb = (fr2 >= COH_BAND[0]) & (fr2 <= COH_BAND[1])
            q["coh_mean"] = float(coh[cb].mean()); q["frf_hits"] = int(n)
        qc[loc] = q
        if q["used"] < 12:
            warns.append(f"{loc}: 쓸 타격 {q['used']}개 — 위치당 15개 이상 권장")
        if not (8 <= q["force_med"] <= 30):
            warns.append(f"{loc}: 힘 중앙 {q['force_med']:.1f} N — 12–20 N 권장")
        if q["resp_med"] < 0.3:
            warns.append(f"{loc}: 마이크 응답 피크 중앙 {q['resp_med']:.2f} Pa — 마이크 채널·위치 확인(지난번 4–8 Pa)")
        if q["T0_span"] > 1.0:
            warns.append(f"{loc}: 온도(ai0)가 측정 중 {q['T0_span']:.1f} °C 움직임")
        if q.get("coh_mean", 1) < 0.7:
            warns.append(f"{loc}: 코히어런스 평균 {q['coh_mean']:.2f}(1.9–9 kHz) — 설치·접촉 확인(지난번 0.75–0.89)")

    # ---------------------------------------------------------------- 위치별 요약(타격별 중앙값·묶음 적합)
    rng = np.random.default_rng(11)
    modes = {n: {"loc": {}, "pred_shift_pct": shifts.get(n, 0.0), "center": None} for n in names}
    for bi, b in enumerate(bands):
        for j, n in enumerate(b["modes"]):
            modes[n]["center"] = b["centers"][j]
            modes[n]["band"] = [b["lo"], b["hi"]]
            modes[n]["cross"] = b["cross"]
    for loc in locs:
        use = [r for r in rows if r["loc"] == loc and r["N"] >= a.min_force and r["status"] != "rejected"]
        for n in names:
            sl = gl.MODE[n].slope
            v = [r["f"][n][0] * (1 - sl * np.log2(r["N"] / 20)) for r in use if r["f"].get(n)]
            med, se, nn, sd = robust(v)
            modes[n]["loc"][loc] = {"hit": dict(f=med, se=se, n=nn, sd_pct=sd, ok_frac=len(v) / max(len(use), 1),
                                                amp=float(np.median([r["f"][n][3] for r in use if r["f"].get(n)]))
                                                if v else np.nan)}
        for bi, b in enumerate(bands):
            k = len(b["modes"])
            ok_frac = min(modes[n]["loc"][loc]["hit"]["ok_frac"] for n in b["modes"])
            # h2 짝은 두 극점이 붙으면(T-a: 갈라짐 6 → 2.6 Hz, 11-11) 타격별 적합이 실패해도 묶음 적합은 된다.
            # 그래서 짝 대역은 타격별 성공률과 무관하게 묶음 적합을 하고, 적합 오차가 작을 때만 쓴다.
            weak_pair = set(gl.PAIR) <= set(b["modes"]) and ok_frac < 0.5
            if len(use) < 5 or (ok_frac < 0.5 and not weak_pair):
                continue
            slope = float(np.mean([gl.MODE[n].slope for n in b["modes"]]))
            Ys = []
            for r in use:
                frs, Xs = spec[(r["file"], bi)]
                g, Y = gl.stretched(Xs, frs, r["N"], b["lo"], b["hi"], slope)
                Ys.append(Y)
            p, err = gl.joint_fit(g, Ys, b["lo"], b["hi"], k)
            if weak_pair and (err > 0.3 or np.min(np.diff(p[:k])) < 1.0):
                continue
            bs = []
            for _ in range(a.nboot):
                idx = rng.choice(len(Ys), len(Ys), replace=True)
                pb, _ = gl.joint_fit(g, [Ys[i] for i in idx], b["lo"], b["hi"], k, p0=p)
                bs.append(pb[:k])
            sd = np.array(bs).std(0, ddof=1)
            for j, n in enumerate(b["modes"]):
                modes[n]["loc"][loc]["pool"] = dict(f=float(p[j]), se=float(sd[j]), z=float(p[k + j]), err=float(err), n=len(Ys),
                                                    weak_pair=bool(weak_pair))
    print(f"[{a.state}] 묶음 적합 끝 ({time.time() - t_start:.0f}s)", flush=True)

    # ---------------------------------------------------------------- 위치 합치기
    for n in names:
        est = []
        for loc in locs:
            q = modes[n]["loc"].get(loc, {})
            pq = q.get("pool")
            if pq and np.isfinite(pq["se"]) and pq["se"] > 0:
                est.append((pq["f"], max(pq["se"], 5e-5 * pq["f"]), loc))
        conflict = False
        if len(est) == 2:
            (f1, s1, l1), (f2, s2, l2) = est
            if abs(f1 - f2) > max(3 * np.hypot(s1, s2), 5e-4 * f1):
                # 두 위치가 어긋나면(약한 쪽의 적합이 이웃 모드·잡음에 끌린 경우) 잘 보이는 위치 하나만 쓴다
                vis = {l: modes[n]["loc"][l]["hit"]["ok_frac"] * np.nan_to_num(modes[n]["loc"][l]["hit"]["amp"]) for l in (l1, l2)}
                keep = max(vis, key=vis.get)
                est = [e for e in est if e[2] == keep]
                conflict = True
        if est:
            w = np.array([1 / s ** 2 for _, s, _ in est]); fv = np.array([f for f, _, _ in est])
            fm = float((w * fv).sum() / w.sum()); se = float(1 / np.sqrt(w.sum()))
            spread = float(100 * (fv.max() / fv.min() - 1)) if len(fv) > 1 else 0.0
            modes[n]["comb"] = dict(f=fm, se=se, method="pool", locs=[l for _, _, l in est], loc_spread_pct=spread,
                                    loc_conflict=conflict)
        else:
            hs = [(q["hit"]["f"], q["hit"]["se"]) for q in modes[n]["loc"].values() if np.isfinite(q["hit"].get("se", np.nan))]
            if hs:
                w = np.array([1 / s ** 2 for _, s in hs]); fv = np.array([f for f, _ in hs])
                modes[n]["comb"] = dict(f=float((w * fv).sum() / w.sum()), se=float(1 / np.sqrt(w.sum())), method="hit")
            else:
                modes[n]["comb"] = dict(f=np.nan, se=np.nan, method="none")

    out = dict(state=a.state, folder=os.path.abspath(a.folder), args=vars(a), created=time.strftime("%Y-%m-%d %H:%M"),
               qc=qc, warnings=warns, bands=bands, modes=modes,
               hits=rows)
    js = os.path.join(a.out, f"{a.state}.json")
    with open(js, "w") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=float)
    if frf:
        fr2 = np.fft.rfftfreq(FRF_NFFT, 1 / gl.FS_EXPECTED); sel = (fr2 >= FRF_BAND[0]) & (fr2 <= FRF_BAND[1])
        np.savez_compressed(os.path.join(a.out, f"{a.state}_frf.npz"), fr=fr2[sel],
                            **{f"H_{l}": (frf[l][0] / np.maximum(frf[l][1], 1e-300))[sel].astype(np.complex64) for l in frf},
                            **{f"coh_{l}": (np.abs(frf[l][0]) ** 2 / np.maximum(frf[l][1] * frf[l][2], 1e-300))[sel].astype(np.float32)
                               for l in frf})
    qc_figure(out, os.path.join(a.out, f"{a.state}_qc.png"))

    # ---------------------------------------------------------------- 화면 요약
    print(f"\n[{a.state}] 품질")
    for loc, q in qc.items():
        print(f"  {loc}: 타격 {q['hits']}(쓴 {q['used']}, 늦은 재타격 {q['late']}, 사람 제외 {q['rejected']}), 힘 중앙 "
              f"{q['force_med']:.1f} N({q['force_min']:.0f}–{q['force_max']:.0f}), 마이크 피크 {q['resp_med']:.2f} Pa, "
              f"잡음 {q['pre_mPa']:.0f} mPa, 온도 {q['T0']:.2f}/{q['T1']:.2f} °C, 코히어런스 {q.get('coh_mean', np.nan):.2f}, "
              f"세션 {q['sessions']}, {q['first'][11:19]}–{q['last'][11:19]}")
    for w in warns:
        print("  [경고]", w)
    print(f"\n[{a.state}] 모드(20 N 환산) — 합친 값 [Hz] ± 표준오차, 위치별 묶음/타격별")
    for n in names:
        c = modes[n]["comb"]
        parts = []
        for loc in locs:
            q = modes[n]["loc"].get(loc, {})
            pq, hq = q.get("pool"), q.get("hit", {})
            parts.append(f"{loc[-5:]}: " + (f"{pq['f']:.2f}±{pq['se']:.2f}" if pq else "—") +
                         f" / {hq.get('f', np.nan):.2f}(n{hq.get('n', 0)}, {100 * hq.get('ok_frac', 0):.0f} %)")
        cross = f"  ⚠ 교차 {modes[n]['cross']}" if modes[n].get("cross") else ""
        if c.get("loc_conflict"):
            cross += f"  ⚠ 위치 어긋남 → {c['locs'][0]}만 씀"
        print(f"  {n:>5s} 예측 {modes[n]['pred_shift_pct']:+.2f} %: {c['f']:9.2f} ± {c['se']:.2f} ({c['method']})  | "
              + " | ".join(parts) + cross)
    print(f"\n→ {js} ({time.time() - t_start:.0f}s)")


def qc_figure(S, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    hits = S["hits"]
    t = np.array([int(h["ts"][11:13]) * 3600 + int(h["ts"][14:16]) * 60 + float(h["ts"][17:]) for h in hits])
    t = (t - t.min()) / 60
    locs = sorted({h["loc"] for h in hits})
    for i, loc in enumerate(locs):
        sel = [j for j, h in enumerate(hits) if h["loc"] == loc]
        ax[0].plot(t[sel], [hits[j]["N"] for j in sel], "o", ms=4, label=loc)
    ax[0].axhspan(12, 20, color="0.92", zorder=0)
    ax[0].set_xlabel("time [min]"); ax[0].set_ylabel("peak force [N]"); ax[0].legend(fontsize=7)
    ax[0].set_title("(a) force (grey = 12-20 N target)", fontsize=9, loc="left")
    for n, col in (("2134", "C0"), ("2142", "C1"), ("3164", "C2"), ("3459", "C3"), ("3748", "C4"), ("6439", "C5")):
        c = S["modes"].get(n, {}).get("comb", {}).get("f", np.nan)
        if not np.isfinite(c):
            continue
        sl = {m.name: m.slope for m in gl.MODES}[n]
        xs, ys = [], []
        for j, h in enumerate(hits):
            v = h["f"].get(n)
            if v and h["N"] >= 5:
                xs.append(t[j]); ys.append(100 * (v[0] * (1 - sl * np.log2(h["N"] / 20)) / c - 1))
        ax[1].plot(xs, ys, ".", color=col, ms=5, label=n)
    ax[1].axhline(0, color="k", lw=0.6); ax[1].set_ylim(-0.4, 0.4)
    ax[1].set_xlabel("time [min]"); ax[1].set_ylabel("per-hit f(20 N) / state value - 1 [%]"); ax[1].legend(fontsize=7, ncol=3)
    ax[1].set_title("(b) per-hit frequencies", fontsize=9, loc="left")
    fz = os.path.splitext(path)[0].replace("_qc", "_frf") + ".npz"
    if os.path.exists(fz):
        Z = np.load(fz)
        for loc in locs:
            if f"H_{loc}" in Z.files:
                ax[2].semilogy(Z["fr"] / 1e3, np.abs(Z[f"H_{loc}"]), lw=0.7, label=loc)
    for n, m in S["modes"].items():
        if m.get("center"):
            ax[2].axvline(m["center"] / 1e3, color="0.7", lw=0.5)
    ax[2].set_xlim(1.9, 9); ax[2].set_xlabel("frequency [kHz]"); ax[2].set_ylabel("|H1| [Pa/N]"); ax[2].legend(fontsize=7)
    ax[2].set_title("(c) mean FRF; grey = expected band centres", fontsize=9, loc="left")
    fig.suptitle(f"state {S['state']} — quality check", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
