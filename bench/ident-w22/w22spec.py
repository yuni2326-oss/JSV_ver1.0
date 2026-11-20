"""impact_testw22 승인(ok)·10 N 이상 타격의 힘·마이크 스펙트럼(1.8–10.5 kHz, 0.25 s 응답 창, 4배 영 채움)을 저장한다."""
import sys, glob, os
import numpy as np

OUT = sys.argv[1]
fs = 204800.0; ti = 4096
NEW = "/mnt/nas/data/GY_DATA/impact_testw22/impact_testw22"
n = int(0.3 * fs); nfft = 4 * n
fr = np.fft.rfftfreq(nfft, 1 / fs); band = (fr >= 1800) & (fr <= 10500)
ph = np.exp(2j * np.pi * fr[band] * 40 / fs)
wf = np.zeros(n); wf[:40 + int(0.00045 * fs)] = 1
wr = np.zeros(n); nr = int(0.25 * fs); wr[:nr] = 1; nt = int(0.01 * fs); wr[nr - nt:nr] = 0.5 * (1 + np.cos(np.pi * np.arange(nt) / nt))
Fs, Xs, meta = [], [], []
for p in sorted(glob.glob(NEW + "/*.npz")):
    with np.load(p, allow_pickle=False) as z:
        if str(z["quality"]) != "ok" or float(z["peak_force_N"]) < 10:
            continue
        d = z["data"].astype(float); N = float(z["peak_force_N"]); loc = str(z["location"]); hit = int(z["hit"]); ts = str(z["timestamp"])[11:19]
    f = d[0] - d[0, :ti - 100].mean(); m = d[1] - d[1, :ti - 100].mean()
    Fs.append((np.fft.rfft(f[ti - 40: ti - 40 + n] * wf, nfft)[band] * ph).astype(np.complex64))
    Xs.append((np.fft.rfft(m[ti - 40: ti - 40 + n] * wr, nfft)[band] * ph).astype(np.complex64))
    meta.append((int(loc.split("vane")[1][0]), hit, N, ts))
np.savez(os.path.join(OUT, "w22_spec.npz"), fr=fr[band], F=np.array(Fs), X=np.array(Xs), vane=np.array([m[0] for m in meta]),
         hit=np.array([m[1] for m in meta]), N=np.array([m[2] for m in meta]), ts=np.array([m[3] for m in meta]))
print(len(meta), "hits saved;", {v: sum(m[0] == v for m in meta) for v in range(1, 7)})
