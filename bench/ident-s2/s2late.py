"""늦은 재타격(해머가 10–20 ms 뒤 다시 닿음) 타격을 살릴 수 있는지: 힘 창을 첫 접촉(0.45 ms)에 더해 재타격 구간까지 넓혀 H1을 만들고,
같은 깃 깨끗한 타격(재타격 없음, 5 N 이상) 평균과 FRAC으로 비교한다. 재타격 시각·크기 분포도 본다."""
import sys, glob, os
import numpy as np

SRC = sys.argv[1]
fs = 204800.0; ti = 4096
n = int(0.3 * fs); nfft = 4 * n
fr = np.fft.rfftfreq(nfft, 1 / fs); band = (fr >= 1800) & (fr <= 10500); sel = (fr[band] >= 1900) & (fr[band] <= 7500)
ph = np.exp(2j * np.pi * fr[band] * 40 / fs)
wr = np.zeros(n); nr = int(0.25 * fs); wr[:nr] = 1; nt = int(0.01 * fs); wr[nr - nt:nr] = 0.5 * (1 + np.cos(np.pi * np.arange(nt) / nt))


def frac(a, b):
    return np.abs(np.vdot(a, b)) ** 2 / (np.vdot(a, a).real * np.vdot(b, b).real)


R = []
for p in sorted(glob.glob(os.path.join(SRC, "*.npz"))):
    with np.load(p, allow_pickle=False) as z:
        d = z["data"].astype(float); N = float(z["peak_force_N"]); v = int(str(z["location"]).split("vane")[1][0]); hit = int(z["hit"])
    f = d[0] - d[0, :ti - 100].mean(); m = d[1] - d[1, :ti - 100].mean()
    seg = f[ti + int(0.003 * fs): ti + int(0.25 * fs)]
    thr = max(1.0, 0.05 * N)
    over = np.where(seg > thr)[0]
    wf = np.zeros(n); wf[:40 + int(0.00045 * fs)] = 1
    events = []
    if over.size:
        # 덩어리(2 ms 이상 떨어지면 새 덩어리)마다 −0.3 ~ +1.5 ms 창
        st = [over[0]] + [over[i] for i in range(1, over.size) if over[i] - over[i - 1] > int(0.002 * fs)]
        for s0 in st:
            a = s0 + int(0.003 * fs) + 40 - int(0.0003 * fs); b = s0 + int(0.003 * fs) + 40 + int(0.0015 * fs)
            wf[max(a, 0):min(b, n)] = 1
            pk = seg[s0:s0 + int(0.0015 * fs)].max()
            events.append(((s0 + int(0.003 * fs)) / fs * 1e3, pk / N))
    seg_f = f[ti - 40: ti - 40 + n]
    F1 = np.fft.rfft(seg_f * np.where(np.arange(n) < 40 + int(0.00045 * fs), 1.0, 0.0), nfft)[band] * ph
    F2 = np.fft.rfft(seg_f * wf, nfft)[band] * ph
    Xm = np.fft.rfft(m[ti - 40: ti - 40 + n] * wr, nfft)[band] * ph
    imp = np.sum(seg_f * wf) / fs; imp1 = np.sum(seg_f[:40 + int(0.00045 * fs)]) / fs
    R.append(dict(v=v, hit=hit, N=N, ev=events, F1=F1, F2=F2, X=Xm, impr=(imp - imp1) / imp1 if events else 0.0))
print("late re-hit timing/size: per vane, n late hits, first re-hit time ms (median, range), peak ratio (median), extra impulse ratio (median), n events/hit max")
for v in range(1, 7):
    L = [r for r in R if r["v"] == v and r["ev"]]
    if not L:
        continue
    t = [r["ev"][0][0] for r in L]; a = [r["ev"][0][1] for r in L]
    print(f"  vane {v}: {len(L)}/{sum(r['v']==v for r in R)} | t {np.median(t):.1f} ({min(t):.1f}-{max(t):.1f}) ms | peak ratio {np.median(a):.2f} | impulse ratio {np.median([r['impr'] for r in L]):.2f} | max events {max(len(r['ev']) for r in L)}")
print("\nFRAC 1.9-7.5 kHz of late-hit H1 vs mean of the vane's clean hits (>=5 N): standard force window -> extended window (median [min]); clean-hit LOO for reference")
for v in range(1, 7):
    C = [r for r in R if r["v"] == v and not r["ev"] and r["N"] >= 5]
    L = [r for r in R if r["v"] == v and r["ev"] and r["N"] >= 5]
    if len(C) < 2:
        print(f"  vane {v}: too few clean hits ({len(C)})"); continue
    Hc = sum(np.conj(r["F1"]) * r["X"] for r in C) / sum(np.abs(r["F1"]) ** 2 for r in C)
    f1 = np.array([frac((r["X"] / r["F1"])[sel], Hc[sel]) for r in L]); f2 = np.array([frac((r["X"] / r["F2"])[sel], Hc[sel]) for r in L])
    loo = np.array([frac((r["X"] / r["F1"])[sel], (sum(np.conj(q["F1"]) * q["X"] for q in C if q is not r) / sum(np.abs(q["F1"]) ** 2 for q in C if q is not r))[sel]) for r in C])
    # 늦은 타격 평균(넓힌 창) 대 깨끗한 평균
    Hl = sum(np.conj(r["F2"]) * r["X"] for r in L) / sum(np.abs(r["F2"]) ** 2 for r in L) if L else None
    lev = 10 * np.log10(np.mean(np.abs(Hl[sel]) ** 2) / np.mean(np.abs(Hc[sel]) ** 2)) if L else np.nan
    print(f"  vane {v}: clean {len(C)} LOO {np.median(loo):.2f} [{loo.min():.2f}] | late {len(L)}: standard {np.median(f1):.2f} [{f1.min():.2f}] -> extended {np.median(f2):.2f} [{f2.min():.2f}]"
          f" | mean(late, ext) vs mean(clean) FRAC {frac(Hl[sel], Hc[sel]):.3f}, level {lev:+.2f} dB")
