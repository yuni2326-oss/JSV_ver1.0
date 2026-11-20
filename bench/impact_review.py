"""E2-I 해머 시험 검토 — 타격 품질·배경·FRF 일관성·피크·모드 주파수 반복성.

사용: python -I impact_review.py <npz 폴더> <출력 폴더> [--fe FE모드.csv] [--last N] [--tag 이름] [--session S] [--location vane1_xi0.3]
- 입력은 DAQ 해머 프로그램(pxie4464_daq impact v1)의 타격별 .npz (data = [해머 N, 마이크 Pa, 가속도…]).
- 파일 이름의 시각 순으로 정렬하고 --last N이면 마지막 N개만 쓴다. 피크 5 N 미만(오타격)은 뺀다.
- 그림은 <출력 폴더>/<tag>-summary.png.

판정 기준(2026-10-22 1·2차 자료에서 정함):
- 재접촉 = 주 피크 뒤 0.05–0.5 ms 안에 피크의 10 %를 넘는 날카로운 봉우리. 실제로 다시 닿은 것이다.
- 혹 = 0.8–2.5 ms 평균 힘 / 피크. 이 해머(LC00KE)는 모든 타격에서 약 1.4 ms에 혹이 생긴다. 근접 마이크의
  저주파 압력이 따라오지 않아 임펠러에 실제로 걸린 힘이 아니다(해머 쪽 인공물) → 힘 창은 접촉 구간(≤ 0.45 ms)만 쓴다.
- 압력 부호 = 타격 순간(0–0.4 ms) 1.2 kHz 저역 마이크 압력의 부호. 임펠러가 마이크 쪽으로 밀리면 +다.
"""
import argparse
import csv
import glob
import os

import numpy as np
from scipy import optimize, signal, stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SHAFT_HZ = 58.26                     # 시험장 펌프 축 주파수(1차 자료의 가속도·마이크 배경)
F_END = 0.00045                      # 힘 창 끝 [s]: 접촉 구간만(혹 앞)
MODES = [(2030, 2075), (3680, 3720), (4280, 4330), (5170, 5230), (6670, 6720)]   # 펜슬 적합 대역 [Hz]
# 추적 모드: (대역 [Hz], 성분 수). 2.04/2.06 kHz는 두 피크로 분해되는 짝, 나머지는 단일 성분으로 안정하게 맞는 모드.
# 3.7·6.7 kHz 대역은 성분이 둘보다 많아(두 성분 적합이 불안정) 다점 식별 전까지 추적하지 않는다.
TRACK = [((2025, 2080), 2), ((3240, 3275), 1), ((4310, 4350), 1), ((6125, 6165), 1)]


def load(path):
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    for k, v in list(d.items()):
        if isinstance(v, np.ndarray) and v.ndim == 0:
            d[k] = v.item()
    d["tag"] = "S%d_H%03d" % (d["session"], d["hit"])
    return d


def demod(x, fs, fc, bw, fs_d):
    t = np.arange(x.size) / fs
    y = x * np.exp(-2j * np.pi * fc * t)
    sos = signal.butter(6, bw / 2, fs=fs, output="sos")
    y = signal.sosfiltfilt(sos, y.real) + 1j * signal.sosfiltfilt(sos, y.imag)
    q = int(round(fs / fs_d))
    return y[::q], fs / q


def pencil_dominant(x, fs, ti, lo, hi, t0=0.005, dur=0.15):
    """대역 [lo, hi]에서 행렬 펜슬로 감쇠 정현파를 맞추고 가장 큰 성분의 (주파수, 감쇠비)."""
    fc, bw = 0.5 * (lo + hi), hi - lo
    y, fsd = demod(x, fs, fc, bw * 1.3, bw * 6)
    y = y[int((ti / fs + t0) * fsd): int((ti / fs + t0) * fsd) + int(dur * fsd)]
    N = y.size; L = N // 3
    Y = np.array([y[i:i + L + 1] for i in range(N - L)])
    _, s, Vh = np.linalg.svd(Y, full_matrices=False)
    r = int(min(6, np.sum(s > 0.02 * s[0])))
    V = Vh.conj().T[:, :r]
    z = np.linalg.eigvals(np.linalg.pinv(V[:-1]) @ V[1:])
    a, *_ = np.linalg.lstsq(z[None, :] ** np.arange(N)[:, None], y, rcond=None)
    sp = np.log(z) * fsd
    f = fc + sp.imag / (2 * np.pi)
    zeta = -sp.real / np.abs(sp + 2j * np.pi * fc)
    k = (f > lo) & (f < hi) & (zeta > 0) & (zeta < 0.02)
    if not k.any():
        return np.nan, np.nan
    j = np.argmax(np.abs(a) * k)
    return f[j], zeta[j]


def spectral_fit(x, fs, ti, lo, hi, k=2, t0=0.005, dur=0.4, nfft=1 << 20, pad=8.0):
    """감쇠 응답(타격 뒤 t0 … t0+dur)의 복소 스펙트럼을 Σ r_j/(i2πf − s_j) + c0 + c1·(f − fc)로 맞춘다.

    시간 영역 필터를 쓰지 않으므로 좁은 대역 복조 필터의 과도응답 편향(행렬 펜슬에서 −3 ~ +2 Hz,
    합성 신호로 확인)이 없다. 합성 두 성분(2040/2063 Hz)을 0.001 Hz 안으로 되찾는다.
    반환: 주파수 오름차순 (f, ζ, 상대 진폭, 복소 잔차 r, 적합 상대오차).
    """
    seg = x[ti + int(t0 * fs): ti + int((t0 + dur) * fs)].astype(float)
    X = np.fft.rfft(seg, nfft) / fs
    fr = np.fft.rfftfreq(nfft, 1 / fs)
    sel = (fr >= lo - pad) & (fr <= hi + pad)
    f = fr[sel]; Xs = X[sel]; w = 2j * np.pi * f; fc = 0.5 * (lo + hi)
    mag = np.abs(Xs)
    pk = signal.find_peaks(mag)[0]
    pk = pk[(f[pk] >= lo) & (f[pk] <= hi)]
    if pk.size < k:
        pk = np.argsort(mag)[::-1][:k]
    f0 = np.sort(f[pk[np.argsort(mag[pk])[::-1][:k]]])
    z0 = np.full(k, 0.0015)

    def model(p):
        fk, zk = p[:k], p[k:2 * k]
        r = p[2 * k:4 * k:2] + 1j * p[2 * k + 1:4 * k:2]
        sk = -zk * 2 * np.pi * fk + 1j * 2 * np.pi * fk * np.sqrt(1 - zk ** 2)
        return (r[None, :] / (w[:, None] - sk[None, :])).sum(1) + (p[4 * k] + 1j * p[4 * k + 1]) + (p[4 * k + 2] + 1j * p[4 * k + 3]) * (f - fc)

    def resid(p):
        d = model(p) - Xs
        return np.concatenate([d.real, d.imag])

    sk0 = -z0 * 2 * np.pi * f0 + 1j * 2 * np.pi * f0
    coef, *_ = np.linalg.lstsq(np.column_stack([1 / (w - q) for q in sk0] + [np.ones_like(w), f - fc]), Xs, rcond=None)
    p0 = np.concatenate([f0, z0, np.column_stack([coef[:k].real, coef[:k].imag]).ravel(),
                         [coef[k].real, coef[k].imag, coef[k + 1].real, coef[k + 1].imag]])
    lb = np.concatenate([np.full(k, lo - pad), np.full(k, 1e-5), np.full(2 * k + 4, -np.inf)])
    ub = np.concatenate([np.full(k, hi + pad), np.full(k, 0.02), np.full(2 * k + 4, np.inf)])
    sol = optimize.least_squares(resid, np.clip(p0, lb + 1e-9, ub - 1e-9), bounds=(lb, ub), x_scale="jac", max_nfev=4000)
    p = sol.x
    fk, zk = p[:k], p[k:2 * k]
    r = p[2 * k:4 * k:2] + 1j * p[2 * k + 1:4 * k:2]
    o = np.argsort(fk)
    amp = np.abs(r)[o]
    err = np.linalg.norm(sol.fun) / np.linalg.norm(np.concatenate([Xs.real, Xs.imag]))
    return fk[o], zk[o], amp / amp.max(), r[o], err


def hit_metrics(d):
    fs, ti = d["sample_rate"], d["trigger_index"]
    f = d["data"][0].astype(float) - d["data"][0, :ti - 100].astype(float).mean()
    m = d["data"][1].astype(float) - d["data"][1, :ti - 100].astype(float).mean()
    ipk = ti - 20 + int(np.argmax(f[ti - 20: ti + 200])); pk = f[ipk]
    seg = f[ipk + int(0.00005 * fs): ti + int(0.0005 * fs)]
    p = signal.find_peaks(seg, prominence=0.08 * pk)[0]
    recontact = [float(seg[j]) for j in p if seg[j] > 0.1 * pk]
    I_main = np.trapezoid(f[ti - 40: ti + int(F_END * fs)]) / fs
    hump = f[ti + int(0.0008 * fs): ti + int(0.0025 * fs)].mean() / pk
    # 접촉 구간 힘 스펙트럼(−0.2 ms … F_END, 끝 0.05 ms 테이퍼), 기준 50–500 Hz
    nm = int((0.0002 + F_END) * fs); w = np.ones(nm); nt = int(0.00005 * fs)
    w[-nt:] = 0.5 * (1 + np.cos(np.pi * np.arange(nt) / nt))
    nfft = 1 << 15
    F = np.abs(np.fft.rfft(f[ti - int(0.0002 * fs): ti - int(0.0002 * fs) + nm] * w, nfft)); fr = np.fft.rfftfreq(nfft, 1 / fs)
    ref = np.median(F[(fr > 50) & (fr < 500)])
    bw = [fr[np.flatnonzero((fr > 500) & (F < ref * 10 ** (-x / 20)))[0]] if np.any((fr > 500) & (F < ref * 10 ** (-x / 20))) else fr[-1] for x in (10, 20)]
    ml = signal.sosfiltfilt(signal.butter(4, 1200, fs=fs, output="sos"), m)[ti - 40: ti + int(0.0004 * fs)]
    psign = ml[np.argmax(np.abs(ml))]
    # 0.1 s 뒤 재가진 사건(3–12 kHz 2 ms rms가 직전 50 ms 중앙값의 3배·배경의 2배 초과)
    mb = signal.sosfiltfilt(signal.butter(4, [3000, 12000], btype="band", fs=fs, output="sos"), m)
    nb = int(0.002 * fs); nw = (m.size - ti) // nb
    env = np.sqrt((mb[ti:ti + nw * nb].reshape(nw, nb) ** 2).mean(axis=1)); tt = (np.arange(nw) + 0.5) * nb / fs
    bgv = np.median(env[tt > 3.0]); ev = []
    for i in range(int(0.1 / 0.002), nw):
        if env[i] > 3 * np.median(env[max(0, i - 25):i - 2]) and env[i] > 2 * bgv and (not ev or tt[i] - ev[-1][0] > 0.03):
            ev.append((tt[i], float(np.abs(m[ti + int((tt[i] - 0.01) * fs): ti + int((tt[i] + 0.01) * fs)]).max())))
    bg = m[-int(2 * fs):].std()
    modes = [pencil_dominant(m, fs, ti, lo, hi) for lo, hi in MODES]
    pairs = [spectral_fit(m, fs, ti, lo, hi, k=k) for (lo, hi), k in TRACK]
    temp = np.asarray(d.get("temperature_C", []), float)
    return dict(tag=d["tag"], time=d["timestamp"][11:19], quality=d["quality"], F_pk=pk, I_main=I_main, recontact=recontact,
                pairs=pairs, temp=temp,
                hump=hump, bw10=bw[0], bw20=bw[1], psign=psign, mic_pk=np.abs(m).max(),
                snr_db=20 * np.log10(m[ti + int(0.002 * fs): ti + int(0.05 * fs)].std() / bg), events=ev,
                modes=modes, spec=(fr, F / ref))


def coherence(Fs, Xs):
    return np.abs((np.conj(Fs) * Xs).sum(0)) ** 2 / ((np.abs(Fs) ** 2).sum(0) * (np.abs(Xs) ** 2).sum(0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("out")
    ap.add_argument("--fe"); ap.add_argument("--last", type=int); ap.add_argument("--tag", default="review")
    ap.add_argument("--session", type=int, help="이 세션 번호만"); ap.add_argument("--location", help="위치 이름에 이 문자열이 든 타격만(예 vane1_xi0.3 — 깃 번호까지 넣는다)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(a.src, "*.npz")), key=lambda p: os.path.basename(p)[:19])
    if a.last:
        paths = paths[-a.last:]
    H = [d for d in (load(p) for p in paths) if d["peak_force_N"] >= 5.0
         and (a.session is None or d["session"] == a.session) and (a.location is None or a.location in d["location"])]
    fs, ti = H[0]["sample_rate"], H[0]["trigger_index"]
    M = [hit_metrics(d) for d in H]
    print(f"타격 {len(H)}개 ({M[0]['time']}–{M[-1]['time']}, fs {fs:.0f}, 기록 {H[0]['data'].shape[1]/fs:.2f} s)")
    print("tag       시각      DAQ판정     F_pk[N] I[mN·s] 재접촉  혹[%] −10dB −20dB[kHz] 압력부호[Pa] 마이크pk[Pa] SNR[dB] 재가진(>0.1 s)")
    for r in M:
        ev = "; ".join(f"{t:.2f}s {p:.2f}Pa" for t, p in r["events"][:3]) + (" …" if len(r["events"]) > 3 else "")
        print(f"{r['tag']:9s} {r['time']} {r['quality']:11s} {r['F_pk']:6.1f} {r['I_main']*1e3:6.2f}  {len(r['recontact']):3d}   {r['hump']*100:5.1f} {r['bw10']/1e3:5.1f} {r['bw20']/1e3:5.1f}"
              f"     {r['psign']:+7.2f}      {r['mic_pk']:7.1f}   {r['snr_db']:6.1f}  {ev or '-'}")
    single = [i for i, r in enumerate(M) if not r["recontact"]]
    print(f"\n한 번 접촉(재접촉 없음): {len(single)}/{len(M)} — " + ", ".join(M[i]["tag"] for i in single))
    ok = [r["tag"] for r in M if r["quality"] == "ok"]
    print(f"DAQ ok: {len(ok)}/{len(M)} — " + ", ".join(ok))

    # H1·코히어런스: 힘 = 접촉 구간(−0.2 ms … F_END), 응답 = 0–250 ms(끝 10 ms 테이퍼)
    n = int(0.3 * fs); fr = np.fft.rfftfreq(n, 1 / fs)
    Fs, Xs = [], []
    for d in H:
        f = d["data"][0].astype(float) - d["data"][0, :ti - 100].astype(float).mean()
        m = d["data"][1].astype(float) - d["data"][1, :ti - 100].astype(float).mean()
        wf = np.zeros(n); ne = 40 + int(F_END * fs); nt = int(0.00005 * fs)
        wf[:ne] = 1.0; wf[ne - nt:ne] = 0.5 * (1 + np.cos(np.pi * np.arange(nt) / nt))
        wr = np.zeros(n); nr = int(0.25 * fs); ntr = int(0.01 * fs)
        wr[:nr] = 1.0; wr[nr - ntr:nr] = 0.5 * (1 + np.cos(np.pi * np.arange(ntr) / ntr))
        Fs.append(np.fft.rfft(f[ti - 40: ti - 40 + n] * wf)); Xs.append(np.fft.rfft(m[ti - 40: ti - 40 + n] * wr))
    Fs, Xs = np.array(Fs), np.array(Xs)
    bands = ((1500, 3000), (3000, 5000), (5000, 7500), (7500, 10000), (10000, 15000), (15000, 20000))
    daq_ok = [i for i, r in enumerate(M) if r["quality"] == "ok"]
    cohs = {"전체": coherence(Fs, Xs)}
    if 2 <= len(single) < len(M):
        cohs["한 번 접촉"] = coherence(Fs[single], Xs[single])
    if 2 <= len(daq_ok) < len(M) and daq_ok != single:
        cohs["DAQ ok"] = coherence(Fs[daq_ok], Xs[daq_ok])
    for k, g in cohs.items():
        print(f"H1 코히어런스 중앙값 [{k}]: " + "  ".join(f"{lo/1e3:.1f}–{hi/1e3:.1f} kHz {np.median(g[(fr >= lo) & (fr < hi)]):.2f}" for lo, hi in bands))

    # 이른 창(5–105 ms) 대 늦은 창(2–4 s) PSD → 구조 피크와 배경 피크
    def per(x):
        return signal.periodogram(x, fs, window="hann", nfft=1 << 17)
    E, L = [], []
    for d in H:
        m = d["data"][1].astype(float)
        fq, Pe = per(m[ti + int(0.005 * fs): ti + int(0.105 * fs)])
        L.append(np.mean([per(m[ti + int(t * fs): ti + int((t + 0.1) * fs)])[1] for t in np.arange(2.0, 3.9, 0.1)], 0))
        E.append(Pe)
    Edb, Ldb = 10 * np.log10(np.mean(E, 0)), 10 * np.log10(np.mean(L, 0))
    sel = (fq > 1000) & (fq < 20000)
    pk = signal.find_peaks(Edb[sel], prominence=8, distance=int(20 / (fq[1] - fq[0])))[0]
    fp = fq[sel][pk]
    gap = lambda x: Edb[np.argmin(abs(fq - x))] - Ldb[np.argmin(abs(fq - x))]
    struct = [x for x in fp if gap(x) > 12]
    backg = [x for x in fp if gap(x) < 6]
    print(f"\n구조 피크(이른 창이 늦은 창보다 12 dB 넘게 큼) {len(struct)}개: " + ", ".join(f"{x:.0f}" for x in struct))
    print("배경 피크(6 dB 미만): " + (", ".join(f"{x:.0f}" + (f"(축×{x/SHAFT_HZ:.0f})" if abs(x / SHAFT_HZ - round(x / SHAFT_HZ)) < 0.03 else "") for x in backg) or "-"))
    bgall = np.concatenate([d["data"][1, -int(2 * fs):].astype(float) for d in H])
    fb, Pb = signal.welch(bgall, fs, nperseg=int(0.2 * fs))
    lvl = lambda lo, hi: 10 * np.log10(np.trapezoid(Pb[(fb >= lo) & (fb < hi)], fb[(fb >= lo) & (fb < hi)]) / 4e-10)
    print(f"배경 음압: 전대역 {lvl(10, 80000):.1f} dB, 2–5 kHz {lvl(2000, 5000):.1f} dB, 5–10 kHz {lvl(5000, 10000):.1f} dB")
    acc = [np.mean([d["data"][r].astype(float).std() for d in H]) for r in range(2, H[0]["data"].shape[0])]
    print("가속도 채널 rms [m/s²]: " + ", ".join(f"{n_}: {v:.3f}" for n_, v in zip(list(H[0]["channel_names"])[2:], acc)))

    # 모드 주파수 반복성(행렬 펜슬 지배 성분, 5–155 ms)
    print("\n모드 주파수 반복성(대역별 행렬 펜슬의 가장 큰 성분, 5–155 ms):")
    for k, (lo, hi) in enumerate(MODES):
        fa = np.array([r["modes"][k][0] for r in M]); za = np.array([r["modes"][k][1] for r in M])
        med = np.nanmedian(fa); mad = 1.4826 * np.nanmedian(np.abs(fa - med))
        fsg = fa[single] if len(single) >= 2 else np.array([np.nan])
        fok = fa[[i for i, r in enumerate(M) if r["quality"] == "ok"]]
        fok = fok if fok.size >= 2 else np.array([np.nan])
        print(f"  {lo}–{hi} Hz: 중앙 {med:8.2f} Hz, ζ 중앙 {np.nanmedian(za)*100:.2f} % | 전체 sd {np.nanstd(fa, ddof=1)/med*100:.3f} %, MAD {mad/med*100:.3f} %"
              f" | 한 번 접촉 sd {np.nanstd(fsg, ddof=1)/np.nanmean(fsg)*100:.3f} % (n={len(single)})"
              f" | DAQ ok sd {np.nanstd(fok, ddof=1)/np.nanmean(fok)*100:.3f} % (n={fok.size})")

    # 추적 모드 반복성(주파수 영역 극점 적합, 5–405 ms) + 힘 의존(연화)
    groups = {"전체": list(range(len(M))), "DAQ ok": [i for i, r in enumerate(M) if r["quality"] == "ok"]}
    print("\n추적 모드(주파수 영역 극점 적합, 적합 오차 < 0.25만): 중앙 [Hz], sd·MAD [%], ζ 중앙 [%] / 힘 2배당 이동 [%] (p) / 20 N 환산 [Hz]")
    for t_, ((lo, hi), k) in enumerate(TRACK):
        for gname, idx in groups.items():
            rows = [(M[i]["F_pk"], M[i]["pairs"][t_]) for i in idx]
            rows = [(fp, q) for fp, q in rows if q[4] < 0.25 and np.all((q[0] > lo) & (q[0] < hi))]
            if len(rows) < 3:
                continue
            F = np.array([q[0] for _, q in rows]); Z = np.array([q[1] for _, q in rows]); P = np.array([fp for fp, _ in rows])
            med = np.median(F, axis=0); mad = 1.4826 * np.median(np.abs(F - med), axis=0); sd = F.std(axis=0, ddof=1)
            line = f"  {lo}–{hi} Hz k={k} [{gname:6s} n={len(F):2d}] " + " / ".join(
                f"{m_:8.2f} (sd {s_/m_*100:.3f}, MAD {a_/m_*100:.3f}, ζ {z_*100:.2f})" for m_, s_, a_, z_ in zip(med, sd, mad, np.median(Z, axis=0)))
            if k == 2:
                pm = F.mean(axis=1)
                line += f" | 쌍평균 sd {pm.std(ddof=1)/np.median(pm)*100:.3f} %, 분리 {np.median(F[:,1]-F[:,0]):.1f} Hz"
            if np.ptp(np.log2(P)) > 0.5:
                sl, ic, rr, pp, _ = stats.linregress(np.log2(P), F[:, 0])
                f20 = ic + sl * np.log2(20)
                line += f" | 힘 2배당 {sl/f20*100:+.3f} % (p {pp:.3f}), 20 N 환산 {f20:.2f}"
            print(line)

    # 설치 감시: 타격별 FRF 크기(3–7 kHz 중앙, 전체 중앙 대비)와 온도
    Hm = np.abs(Xs / Fs); refm = np.median(Hm, axis=0); sb = (fr > 3000) & (fr < 7000)
    lv = [np.median(20 * np.log10(Hm[i][sb] / refm[sb])) for i in range(len(M))]
    print("\n설치 감시 — FRF 크기(3–7 kHz, 전체 중앙 대비 dB) / 압력 부호 / 온도[°C]:")
    print("  " + "  ".join(f"{r['tag'][-4:]} {l:+.1f}/{r['psign']:+.2f}" + (f"/{r['temp'].mean():.2f}" if r['temp'].size else "") for r, l in zip(M, lv)))
    print(f"  FRF 크기 범위 {min(lv):+.1f} … {max(lv):+.1f} dB, 이웃 타격 사이 최대 계단 {max(abs(np.diff(lv))) if len(lv) > 1 else 0:.1f} dB")
    temps = np.array([r["temp"] for r in M if r["temp"].size])
    if temps.size:
        print(f"  온도: 채널별 평균 {', '.join(f'{x:.2f}' for x in temps.mean(axis=0))} °C, 측정 중 변화 {', '.join(f'{x:.2f}' for x in temps.max(axis=0) - temps.min(axis=0))} K")

    # 그림
    fig, ax = plt.subplots(2, 2, figsize=(16, 10))
    for i, d in enumerate(H):
        f = d["data"][0].astype(float)
        t = (np.arange(f.size) - ti) / fs * 1e3
        s = slice(ti - 40, ti + int(0.003 * fs))
        ax[0, 0].plot(t[s], f[s] / M[i]["F_pk"], color="C0" if i in single else "C3", lw=0.6, alpha=0.7)
    ax[0, 0].axhline(0, color="k", lw=0.5); ax[0, 0].set_ylim(-0.2, 1.05)
    ax[0, 0].set_xlabel("time after trigger [ms]"); ax[0, 0].set_ylabel("force / peak")
    ax[0, 0].set_title("(a) force pulses (blue: single contact, red: re-contact <0.5 ms); hump ~1.4 ms = hammer artifact")
    ax[0, 0].grid(alpha=0.3)
    for r in M:
        frr, Fr = r["spec"]
        ax[0, 1].plot(frr / 1e3, 20 * np.log10(Fr), lw=0.6)
    for y in (-10, -20):
        ax[0, 1].axhline(y, color="k", lw=0.6, ls="--")
    ax[0, 1].set_xlim(0, 25); ax[0, 1].set_ylim(-45, 5); ax[0, 1].grid(alpha=0.3)
    ax[0, 1].set_xlabel("kHz"); ax[0, 1].set_ylabel("dB re 50-500 Hz")
    ax[0, 1].set_title(f"(b) force spectrum of the contact window (-0.2 to {F_END*1e3:.2f} ms)")
    ax[1, 0].plot(fq / 1e3, Edb, "C0", lw=0.6, label="response 5-105 ms (mean)")
    ax[1, 0].plot(fq / 1e3, Ldb, "k", lw=0.6, label="background 2-4 s")
    if a.fe:
        for r in csv.DictReader(open(a.fe)):
            ax[1, 0].axvline(float(r["f_Hz"]) / 1e3, ymin=0.93, ymax=1.0, color="r", lw=0.8)
    ax[1, 0].set_xlim(1, 20); ax[1, 0].set_ylim(Ldb[sel].min() - 5, Edb[sel].max() + 8)
    ax[1, 0].set_xlabel("kHz"); ax[1, 0].set_ylabel("mic PSD [dB re 1 Pa²/Hz]")
    ax[1, 0].set_title("(c) mic: structural peaks vs background; red ticks = FE free-free (ff32x16)")
    ax[1, 0].legend(fontsize=8, loc="lower left"); ax[1, 0].grid(alpha=0.3)
    for (k, g), c in zip(cohs.items(), ("C3", "C0", "C2")):
        ax[1, 1].plot(fr / 1e3, g, color=c, lw=0.5, label={"전체": f"all {len(M)} hits", "한 번 접촉": f"single contact ({len(single)})",
                                                           "DAQ ok": f"DAQ ok ({len(daq_ok)})"}[k])
    ax[1, 1].axhline(0.9, color="k", lw=0.8, ls="--")
    ax[1, 1].set_xlim(1, 20); ax[1, 1].set_ylim(0, 1); ax[1, 1].grid(alpha=0.3); ax[1, 1].legend(fontsize=8)
    ax[1, 1].set_xlabel("kHz"); ax[1, 1].set_ylabel("coherence")
    ax[1, 1].set_title("(d) H1 coherence (force = contact window, response 0-250 ms)")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, f"{a.tag}-summary.png"), dpi=90)


if __name__ == "__main__":
    main()
