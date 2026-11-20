"""10-28(impact_testw22 안 20261028_*) 모든 타격의 힘·마이크 스펙트럼을 w22spec과 같은 격자(1.8–10.5 kHz, 0.25 s 응답 창, 4배 영 채움)로 저장.
타격마다 품질·재접촉·늦은 재타격(3–200 ms, max(1 N, 5 %))·힘 −10 dB 대역도 남긴다."""
import sys, glob, os
import numpy as np

OUT = sys.argv[1]
fs = 204800.0; ti = 4096
SRC = "/mnt/nas/data/GY_DATA/impact_testw22/impact_testw22"
n = int(0.3 * fs); nfft = 4 * n
fr = np.fft.rfftfreq(nfft, 1 / fs); band = (fr >= 1800) & (fr <= 10500)
ph = np.exp(2j * np.pi * fr[band] * 40 / fs)
wf = np.zeros(n); wf[:40 + int(0.00045 * fs)] = 1
wr = np.zeros(n); nr = int(0.25 * fs); wr[:nr] = 1; nt = int(0.01 * fs); wr[nr - nt:nr] = 0.5 * (1 + np.cos(np.pi * np.arange(nt) / nt))
Fs, Xs, meta = [], [], []
for p in sorted(glob.glob(SRC + "/20261028_*.npz")):
    with np.load(p, allow_pickle=False) as z:
        assert str(z["channel_names"][1]) == "CH6_mic" and str(z["physical_channels"][1]) == "PXI1Slot5/ai2", p
        d = z["data"].astype(float); N = float(z["peak_force_N"]); loc = str(z["location"]); hit = int(z["hit"]); ts = str(z["timestamp"])[11:19]
        q = str(z["quality"]); kind = str(z["double_kind"]); ses = int(z["session"]); T = z["temperature_C"].astype(float)
    f = d[0] - d[0, :ti - 100].mean(); m = d[1] - d[1, :ti - 100].mean()
    late = bool((f[ti + int(0.003 * fs): ti + int(0.2 * fs)] > max(1.0, 0.05 * N)).any())
    F = np.fft.rfft(f[ti - 40: ti - 40 + n] * wf, nfft)[band] * ph
    Fs.append(F.astype(np.complex64))
    Xs.append((np.fft.rfft(m[ti - 40: ti - 40 + n] * wr, nfft)[band] * ph).astype(np.complex64))
    meta.append((int(loc.split("vane")[1][0]), float(loc.split("xi")[1]), ses, hit, N, ts, q, kind, late, np.abs(m[ti:ti + int(0.05 * fs)]).max() / N, T[0], T[1]))
cols = list(zip(*meta))
np.savez(os.path.join(OUT, "w28_spec.npz"), fr=fr[band], F=np.array(Fs), X=np.array(Xs), vane=np.array(cols[0]), xi=np.array(cols[1]), ses=np.array(cols[2]),
         hit=np.array(cols[3]), N=np.array(cols[4]), ts=np.array(cols[5]), q=np.array(cols[6]), kind=np.array(cols[7]), late=np.array(cols[8]),
         pkF=np.array(cols[9]), T0=np.array(cols[10]), T1=np.array(cols[11]))
print(len(meta), "hits saved")
for (v, x) in sorted(set((m_[0], m_[1]) for m_ in meta)):
    R = [m_ for m_ in meta if m_[0] == v and m_[1] == x]
    Ns = np.array([m_[4] for m_ in R]); late = np.array([m_[8] for m_ in R])
    print(f"  vane {v} xi{x}: {len(R)} hits {R[0][5]}-{R[-1][5]}; N quartiles {np.percentile(Ns,25):.1f}/{np.median(Ns):.1f}/{np.percentile(Ns,75):.1f}; "
          f"no-late >=10 N {((Ns>=10)&~late).sum()}, 5-10 N {((Ns>=5)&(Ns<10)&~late).sum()}, <5 N {(Ns<5).sum()}, late {late.sum()}")
