"""한 폴더의 모든 타격 힘·마이크 스펙트럼을 w22spec과 같은 격자(1.8–10.5 kHz, 0.25 s 응답 창, 4배 영 채움)로 저장.
타격마다 품질·재접촉·늦은 재타격(3–200 ms, max(1 N, 5 %))·응답 피크/힘·타격 전 잡음·온도를 남긴다.
사용: s2spec.py <폴더> <출력 npz> [파일 glob]"""
import sys, glob, os
import numpy as np

SRC, OUTF = sys.argv[1], sys.argv[2]
PAT = sys.argv[3] if len(sys.argv) > 3 else "*.npz"
fs = 204800.0; ti = 4096
n = int(0.3 * fs); nfft = 4 * n
fr = np.fft.rfftfreq(nfft, 1 / fs); band = (fr >= 1800) & (fr <= 10500)
ph = np.exp(2j * np.pi * fr[band] * 40 / fs)
wf = np.zeros(n); wf[:40 + int(0.00045 * fs)] = 1
wr = np.zeros(n); nr = int(0.25 * fs); wr[:nr] = 1; nt = int(0.01 * fs); wr[nr - nt:nr] = 0.5 * (1 + np.cos(np.pi * np.arange(nt) / nt))
Fs, Xs, meta = [], [], []
for p in sorted(glob.glob(os.path.join(SRC, PAT))):
    with np.load(p, allow_pickle=False) as z:
        assert str(z["channel_names"][1]) == "CH6_mic" and str(z["physical_channels"][1]) == "PXI1Slot5/ai2", p
        d = z["data"].astype(float); N = float(z["peak_force_N"]); loc = str(z["location"]); hit = int(z["hit"]); ts = str(z["timestamp"])[11:19]
        q = str(z["quality"]); kind = str(z["double_kind"]); ses = int(z["session"]); T = z["temperature_C"].astype(float)
        day = str(z["timestamp"])[:10]
    f = d[0] - d[0, :ti - 100].mean(); m = d[1] - d[1, :ti - 100].mean()
    seg = f[ti + int(0.003 * fs): ti + int(0.2 * fs)]
    late_i = np.where(seg > max(1.0, 0.05 * N))[0]
    tl = (late_i[0] + int(0.003 * fs)) / fs * 1e3 if late_i.size else -1.0
    Fs.append((np.fft.rfft(f[ti - 40: ti - 40 + n] * wf, nfft)[band] * ph).astype(np.complex64))
    Xs.append((np.fft.rfft(m[ti - 40: ti - 40 + n] * wr, nfft)[band] * ph).astype(np.complex64))
    meta.append(dict(vane=int(loc.split("vane")[1][0]), xi=float(loc.split("xi")[1]), ses=ses, hit=hit, N=N, ts=ts, day=day, q=q, kind=kind,
                     late=tl >= 0, tlate=tl, pkF=np.abs(m[ti:ti + int(0.05 * fs)]).max() / N, pre=m[:ti - 100].std(), bg=m[-int(2 * fs):].std(),
                     T0=T[0], T1=T[1], sat=float(np.abs(m).max())))
cols = {k: np.array([m_[k] for m_ in meta]) for k in meta[0]}
np.savez(OUTF, fr=fr[band], F=np.array(Fs), X=np.array(Xs), **cols)
print(len(meta), "hits saved ->", OUTF)
