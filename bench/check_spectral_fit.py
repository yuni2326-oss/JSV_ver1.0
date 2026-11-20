"""합성 감쇠 정현파로 impact_review.spectral_fit의 편향을 확인한다(2026-10-26).
사용: python -I check_spectral_fit.py  (같은 폴더의 impact_review.py를 읽는다)
기대: 두 성분 2040/2063 Hz를 0.01 Hz 안으로, 가까운 세 번째 모드가 있어도 1 Hz 안으로 되찾는다."""
import importlib.util
import os
import numpy as np
_spec = importlib.util.spec_from_file_location("impact_review", os.path.join(os.path.dirname(os.path.abspath(__file__)), "impact_review.py"))
_ir = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_ir)
spectral_fit = _ir.spectral_fit
fs = 204800.0; ti = 4096; n = 823296
t = (np.arange(n) - ti) / fs
rng = np.random.default_rng(1)
def make(comps, noise=0.0):
    x = np.zeros(n)
    for f, z, a, ph in comps:
        w = 2 * np.pi * f
        x += np.where(t >= 0, a * np.exp(-z * w * t) * np.cos(w * np.sqrt(1 - z * z) * t + ph), 0)
    return x + noise * rng.standard_normal(n)
cases = {
    "2 comps 2040/2063 equal": [(2040, 0.001, 1.0, 0.3), (2063, 0.0012, 1.0, 1.1)],
    "ratio 0.3": [(2040, 0.001, 1.0, 0.3), (2063, 0.0012, 0.3, 1.1)],
    "+ 3rd at 2085 (0.4)": [(2040, 0.001, 1.0, 0.3), (2063, 0.0012, 0.5, 1.1), (2085, 0.0015, 0.4, 2.0)],
    "+ strong 1.99 kHz": [(2040, 0.001, 1.0, 0.3), (2063, 0.0012, 0.5, 1.1), (1990, 0.001, 3.0, 0.5)],
    "noisy (SNR low)": [(2040, 0.001, 1.0, 0.3), (2063, 0.0012, 0.6, 1.1)],
}
for name, comps in cases.items():
    x = make(comps, noise=0.002 if "noisy" not in name else 0.05)
    for band in ((2025, 2075), (2015, 2085)):
        f, z, a, r, e = spectral_fit(x, fs, ti, *band)
        print(f"{name:26s} band {band}: f {f[0]:8.3f} {f[1]:8.3f}  zeta {z[0]*100:.3f} {z[1]*100:.3f}  amp {a[0]:.2f} {a[1]:.2f}  err {e:.3f}")
