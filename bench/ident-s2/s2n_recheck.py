"""2번 시편 11-11 재측정(1번 깃 ξ0.3)을 기존 1번 깃 ξ0.3 무리와 비교한다(43번).
- 11-11 S7(16:55–16:56, 9타) → 걸이를 다시 걸고(사용자, 물기 없음) S8(17:12–17:15, 22타).

방법은 30번 다시 매달기 점검(s2d_rehang·s2d_pool)과 같다.
- 타격별: 마지막 힘 접촉 3 ms 뒤부터 0.4 s 자유 감쇠의 극점(groove_lib.band_fit) → 20 N 환산(연화 기울기) → 무리 중앙값·표준오차.
- 묶음: 한 무리의 타격이 극점을 공유하고 잔차만 다른 적합(groove_lib.joint_fit), 부트스트랩 표준편차.
- 무리: 11-11 S8·S7 · 10-30 S3–S5(다시 매단 뒤 19:07–19:30, 장갑) · 10-30 S2(다시 매달기 전 16:21) ·
  10-28 S1(시편 교체 직후 17:08). 맨 앞 무리(--today 첫 번호)를 나머지 모두와 비교한다.
- 11-11 S1·S6(16:54)은 ξ0.6 타격이 ξ0.3 이름으로 섞여 있어 뺀다(사용자 11-11). S8 H021은 EXCLUDE.
- 대역은 모든 무리에 같게 둔다(30번 s2d_rehang 대역에 뒤판 5617·5745를 더함). groove_state 기본 대역(10-30 값 ± 0.35 %)은
  오늘 h2 짝이 한 타도 맞지 않고 3250 Hz가 대역 아래 끝에 붙어 쓰지 않는다.
사용: python s2n_recheck.py <out> [--today 8 7]
출력: <out>/s2n_recheck.json·.txt, s2n_frf.npz
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ident-s3"))
import groove_lib as gl  # noqa: E402

SRC = "/mnt/nas/data/GY_DATA/impact_testw2s2/impact_testw2s2"
LOC = "back_vane1_xi0.3"
OLD = [("10-30 S3–5", "20261030", [3, 4, 5]), ("10-30 S2", "20261030", [2]), ("10-28 S1", "20261028", [1])]
REF = "10-30 S3–5"
# 뺄 타격(이유) — DAQ 판정은 그대로 두고 여기서만 뺀다
EXCLUDE = {"20261111_171455_962_S8_back_vane1_xi0.3_M1_H021.npz": "타격 89 ms 뒤 34 Pa 응답(해머 밖 접촉), 늦은 재타격"}
BANDS = [(("2134", "2142"), 2110, 2165), (("3164",), 3148, 3182), (("3250",), 3225, 3268), (("3459",), 3440, 3478),
         (("3589",), 3572, 3606), (("3639",), 3620, 3656), (("3748",), 3730, 3766), (("3804",), 3780, 3825),
         (("4225",), 4180, 4265), (("4967",), 4940, 4990), (("5617",), 5585, 5640), (("5745",), 5710, 5780),
         (("6439",), 6415, 6462), (("7462",), 7420, 7500), (("8321",), 8285, 8355)]
NAMES = [n for ms, _, _ in BANDS for n in ms]
FRF_NFFT = 1 << 17
NBOOT = 100


def robust(v):
    """무리 중앙값·중앙값 표준오차·n·sd[%] — 중앙값에서 3 MAD(최소 0.02 %) 넘게 벗어난 타격은 뺀다(groove_state와 같다)."""
    v = np.asarray(v, float)
    if v.size < 2:
        return (float(v[0]) if v.size else np.nan), np.nan, int(v.size), np.nan
    md = np.median(v)
    res = np.abs(v / md - 1)
    v = v[res < max(3 * 1.4826 * np.median(res), 2e-4)]
    if v.size < 2:
        return float(md), np.nan, int(v.size), np.nan
    sd = np.std(v, ddof=1)
    return float(np.median(v)), float(1.2533 * sd / np.sqrt(v.size)), int(v.size), float(100 * sd / np.median(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default=os.path.join(HERE, "out_s2n"))
    ap.add_argument("--today", type=int, nargs="+", default=[8, 7], help="11-11 세션 번호(첫 번호 = 주 비교 무리)")
    a = ap.parse_args()
    OUT = a.out
    GROUPS = [(f"11-11 S{s}", "20261111", [s]) for s in a.today] + OLD
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(11)
    res, frfs = {}, {}
    for g, date, sess in GROUPS:
        hits = [h for h in gl.list_hits(SRC, sess, [LOC], date) if h["file"] not in EXCLUDE]
        bad = gl.check_channels(hits)
        assert not bad, bad[:3]
        rows, spec = [], {}
        S = [0, 0, 0, 0]
        for h in hits:
            N = float(h["peak_force_N"])
            f, m, ti, fs = gl.load_signals(h)
            t0, late = gl.free_decay_start(f, N, ti, fs)
            T = np.asarray(h.get("temperature_C"), float)
            r = dict(file=h["file"], ts=str(h["timestamp"])[11:19], N=N, late=late, t0=t0, status=h["status"],
                     pre_mPa=float(m[:ti - 100].std() * 1e3), T0=float(T[0]), T1=float(T[1]), f={})
            rows.append(r)
            if N < 5 or h["status"] == "rejected":
                continue
            X = gl.decay_spectrum(m, ti, fs, t0)
            fr = np.fft.rfftfreq(gl.NFFT, 1 / fs)
            for bi, (ms, lo, hi) in enumerate(BANDS):
                k = len(ms)
                sel = (fr >= lo - 30) & (fr <= hi + 30)
                spec[(r["file"], bi)] = (fr[sel], X[sel])
                inn = (fr[sel] >= lo - 8) & (fr[sel] <= hi + 8)
                fk, zk, amp, rr, err = gl.band_fit(fr[sel][inn], X[sel][inn], lo, hi, k)
                ok = err < 0.3 and np.all((fk > lo) & (fk < hi)) and (k == 1 or np.min(np.diff(fk)) > 3)
                for j, n in enumerate(ms):
                    r["f"][n] = [float(fk[j]), float(zk[j]), float(err)] if ok else None
            if not late:                         # 평균 H1(늦은 재타격 없는 타격, groove_state와 같은 창)
                fw = np.zeros_like(f)
                i1 = ti + int(0.005 * fs)
                fw[ti - 200:i1] = f[ti - 200:i1]
                seg = slice(ti - 200, ti - 200 + FRF_NFFT)
                xm = m[seg] * np.exp(np.log(0.01) * np.arange(FRF_NFFT) / FRF_NFFT)
                Fk, Xk = np.fft.rfft(fw[seg], FRF_NFFT), np.fft.rfft(xm, FRF_NFFT)
                S[0] = S[0] + np.conj(Fk) * Xk; S[1] = S[1] + np.abs(Fk) ** 2; S[2] = S[2] + np.abs(Xk) ** 2; S[3] += 1
        frfs[g] = S
        use = [r for r in rows if r["N"] >= 5 and r["status"] != "rejected"]
        modes = {}
        for n in NAMES:
            sl = gl.MODE[n].slope
            v = [r["f"][n][0] * (1 - sl * np.log2(r["N"] / 20)) for r in use if r["f"].get(n)]
            z = [r["f"][n][1] for r in use if r["f"].get(n)]
            md, se, nn, sd = robust(v)
            modes[n] = dict(hit=dict(f=md, se=se, n=nn, n_ok=len(v), n_use=len(use), sd_pct=sd,
                                     zeta=float(np.median(z)) if z else np.nan))
        for bi, (ms, lo, hi) in enumerate(BANDS):
            k = len(ms)
            if len(use) < 5 or min(modes[n]["hit"]["n_ok"] for n in ms) < 0.5 * len(use):
                continue
            slope = float(np.mean([gl.MODE[n].slope for n in ms]))
            Ys = []
            for r in use:
                frs, Xs = spec[(r["file"], bi)]
                gg, Y = gl.stretched(Xs, frs, r["N"], lo, hi, slope)
                Ys.append(Y)
            p, err = gl.joint_fit(gg, Ys, lo, hi, k)
            bs = []
            for _ in range(NBOOT):
                idx = rng.choice(len(Ys), len(Ys), replace=True)
                pb, _ = gl.joint_fit(gg, [Ys[i] for i in idx], lo, hi, k, p0=p)
                bs.append(pb)
            sd = np.array(bs).std(0, ddof=1)
            for j, n in enumerate(ms):
                modes[n]["pool"] = dict(f=float(p[j]), se=float(sd[j]), zeta=float(p[k + j]), zeta_se=float(sd[k + j]),
                                        err=float(err), n=len(Ys))
        q = dict(hits=len(rows), used=len(use), late=sum(r["late"] for r in use),
                 force_med=float(np.median([r["N"] for r in rows])), force_min=float(min(r["N"] for r in rows)),
                 force_max=float(max(r["N"] for r in rows)), pre_mPa=float(np.median([r["pre_mPa"] for r in rows])),
                 T0=float(np.mean([r["T0"] for r in rows])), T1=float(np.mean([r["T1"] for r in rows])),
                 first=rows[0]["ts"], last=rows[-1]["ts"], frf_hits=int(S[3]))
        if S[3] >= 2:
            fr2 = np.fft.rfftfreq(FRF_NFFT, 1 / gl.FS_EXPECTED)
            coh = np.abs(S[0]) ** 2 / np.maximum(S[1] * S[2], 1e-300)
            q["coh_mean"] = float(coh[(fr2 >= 1900) & (fr2 <= 9000)].mean())
        res[g] = dict(qc=q, modes=modes, hits=rows)
        print(f"{g}: 타격 {q['hits']}, 쓴 {q['used']}, 늦은 재타격 {q['late']}, 힘 중앙 {q['force_med']:.1f} N", flush=True)

    # ------------------------------------------------------------ 비교(오늘 대 기존 무리)
    def val(g, n):
        """묶음 적합 값이 있으면 그것, 없으면 타격별 중앙값."""
        M = res[g]["modes"][n]
        if "pool" in M and np.isfinite(M["pool"]["se"]):
            return M["pool"]["f"], max(M["pool"]["se"], 5e-5 * M["pool"]["f"]), "묶음"
        return M["hit"]["f"], M["hit"]["se"], "타격"

    today = GROUPS[0][0]
    comp = {}
    for n in NAMES:
        f1, s1, w1 = val(today, n)
        comp[n] = {}
        for g, _, _ in GROUPS[1:]:
            f0, s0, w0 = val(g, n)
            d = 100 * (f1 / f0 - 1) if np.isfinite(f1) and np.isfinite(f0) else np.nan
            se = 100 * np.hypot(s1, s0) / f0 if np.isfinite(s1) and np.isfinite(s0) else np.nan
            comp[n][g] = dict(d_pct=d, se_pct=se, how=f"{w1}/{w0}")
    # 다른 무리 − 기준(10-30 S3–5): 이전 매달기 사이 흔들림의 크기와 오늘 다른 세션
    prev = {}
    for n in NAMES:
        f0, s0, _ = val(REF, n)
        prev[n] = {}
        for g in [g for g, _, _ in GROUPS[1:] if g != REF]:
            f1, s1, _ = val(g, n)
            prev[n][g] = dict(d_pct=100 * (f1 / f0 - 1) if np.isfinite(f1) and np.isfinite(f0) else np.nan,
                              se_pct=100 * np.hypot(s1, s0) / f0 if np.isfinite(s1) and np.isfinite(s0) else np.nan)
    fr2 = np.fft.rfftfreq(FRF_NFFT, 1 / gl.FS_EXPECTED)
    sel = (fr2 >= 1000) & (fr2 <= 10000)
    H = {g: (S[0] / np.maximum(S[1], 1e-300))[sel] for g, S in frfs.items()}
    C = {g: (np.abs(S[0]) ** 2 / np.maximum(S[1] * S[2], 1e-300))[sel] for g, S in frfs.items()}
    np.savez_compressed(os.path.join(OUT, "s2n_frf.npz"), fr=fr2[sel], groups=np.array(list(H)),
                        **{f"H{i}": H[g].astype(np.complex64) for i, g in enumerate(H)},
                        **{f"C{i}": C[g].astype(np.float32) for i, g in enumerate(C)})
    fb = (fr2[sel] >= 1900) & (fr2[sel] <= 9000)

    def frac(a, b):
        x, y = H[a][fb], H[b][fb]
        return float(np.abs(np.vdot(x, y)) ** 2 / (np.vdot(x, x).real * np.vdot(y, y).real))
    fracs = {g: frac(today, g) for g in H if g != today}
    fracs.update({f"{REF}~{g}": frac(REF, g) for g in H if g not in (today, REF)})
    json.dump(dict(groups=[g for g, _, _ in GROUPS], today=today, ref=REF, bands=BANDS, exclude=EXCLUDE, res=res, comp=comp,
                   prev=prev, frac=fracs),
              open(os.path.join(OUT, "s2n_recheck.json"), "w"), ensure_ascii=False, indent=1, default=float)

    L = []
    P = lambda s="": L.append(s)
    P("무리: 타격 / 쓴(5 N 이상) / 늦은 재타격 / 힘 중앙 [N](범위) / 타격 전 잡음 [mPa] / ai0·ai1 [°C] / 코히어런스(1.9–9 kHz) / 시각")
    for g in res:
        q = res[g]["qc"]
        P(f"  {g:10s} {q['hits']:2d} / {q['used']:2d} / {q['late']:2d} / {q['force_med']:4.1f}({q['force_min']:.0f}–{q['force_max']:.0f})"
          f" / {q['pre_mPa']:3.0f} / {q['T0']:.2f}·{q['T1']:.2f} / {q.get('coh_mean', np.nan):.2f} / {q['first']}–{q['last']}")
    P()
    P("FRAC(1.9–9 kHz 평균 H1 모양): " + ", ".join(f"{k} {v:.2f}" for k, v in fracs.items()))
    P()
    P(f"20 N 환산 [Hz](방법, n/쓴 타격) · 오늘 대비 기존 무리 % ± 표준오차 · ζ[‰]")
    for n in NAMES:
        line = f"  {n} ({gl.MODE[n].fem or '—'}, {gl.MODE[n].kind}): "
        for g in res:
            f, s, w = val(g, n)
            M = res[g]["modes"][n]
            z = M.get("pool", {}).get("zeta", M["hit"]["zeta"])
            line += f"| {g[:5]} {f:8.2f}±{s:.2f}({w[0]},{M['hit']['n_ok']}/{M['hit']['n_use']}) ζ{1e3 * z:.2f} "
        P(line)
        P("      오늘 − " + "  ".join(f"{g}: {c['d_pct']:+.3f}±{c['se_pct']:.3f}" for g, c in comp[n].items())
          + "   | 다른 무리 − S3–5: " + ", ".join(f"{g} {v['d_pct']:+.3f}" for g, v in prev[n].items()))
    txt = "\n".join(L)
    print(txt)
    open(os.path.join(OUT, "s2n_recheck.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
