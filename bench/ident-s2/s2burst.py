"""타격 뒤 4 s 마이크 기록에서 다시 울림(임펠러가 흔들려 마이크·다른 것에 닿을 때)을 찾는다: 1.5–10 kHz 대역 포락선(10 ms 창)의 시간 변화."""
import sys, glob, os
import numpy as np
from scipy import signal
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT = sys.argv[1]
fs = 204800.0; ti = 4096
sos = signal.butter(4, [1500, 10000], btype="band", fs=fs, output="sos")
sets = [("10-28 16:34-16:52 far mic (v1 xi0.3/0.6, v2 xi0.6)", "/mnt/nas/data/GY_DATA/impact_testw22/impact_testw22/20261028_16[345]*.npz"),
        ("10-28 17:08-17:41 near mic (v1-v6 xi0.3)", "/mnt/nas/data/GY_DATA/impact_testw2s2/impact_testw2s2/*.npz")]
fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
for k, (lab, pat) in enumerate(sets):
    ps = sorted(glob.glob(pat)); nb = 0; nt = 0; times = []
    for p in ps:
        with np.load(p, allow_pickle=False) as z:
            m = z["data"][1].astype(float); N = float(z["peak_force_N"])
        if N < 5:
            continue
        m = m - m[:ti - 100].mean()
        y = signal.sosfiltfilt(sos, m)
        w = int(0.01 * fs); e = np.sqrt(np.convolve(y ** 2, np.ones(w) / w, mode="same"))[::w // 4]
        t = (np.arange(e.size) * (w // 4) - ti) / fs
        base = np.median(e[t < -0.005]) if (t < -0.005).any() else e.min()
        ax[k].semilogy(t, e / e[(t > 0) & (t < 0.02)].max(), lw=0.5, alpha=0.5)
        # 0.1 s 뒤에 포락선이 바로 앞 50 ms 최소보다 10배(20 dB) 넘게 다시 솟는 곳
        nt += 1
        idx = np.where(t > 0.1)[0]
        hit = False
        for i in idx:
            prev = e[max(i - int(0.05 * fs / (w // 4)), 0):i].min()
            if e[i] > 10 * prev and e[i] > 5 * base:
                times.append(t[i]); hit = True; break
        nb += hit
    ax[k].set_title(f"{lab}: re-burst (>20 dB rise after 0.1 s) in {nb}/{nt} hits; times {np.round(sorted(times)[:12], 2)}", fontsize=9)
    ax[k].set_ylim(1e-4, 2); ax[k].set_ylabel("1.5-10 kHz envelope (norm.)")
    print(lab, f"re-burst {nb}/{nt}", "median time", np.median(times) if times else None)
ax[1].set_xlabel("time after trigger [s]"); ax[1].set_xlim(-0.02, 4.0)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "s2", "s2_burst.png"), dpi=110)
