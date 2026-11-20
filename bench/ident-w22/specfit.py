"""주파수 영역 극점 적합: 감쇠 응답의 복소 스펙트럼을 Σ r_k/(j2πf − s_k) + c0 + c1·Δf 로 맞춘다.
시간 영역 필터를 쓰지 않으므로 필터 과도응답 편향이 없다."""
import numpy as np
from scipy import optimize, signal


def spectral_fit(x, fs, ti, lo, hi, k=2, t0=0.005, dur=0.4, nfft=1 << 20, pad=8.0):
    seg = x[ti + int(t0 * fs): ti + int((t0 + dur) * fs)].astype(float)
    X = np.fft.rfft(seg, nfft) / fs                     # 연속 변환 근사
    fr = np.fft.rfftfreq(nfft, 1 / fs)
    sel = (fr >= lo - pad) & (fr <= hi + pad)
    f = fr[sel]; Xs = X[sel]
    w = 2j * np.pi * f
    # 초기값: 대역 안 |X| 피크 상위 k개
    mag = np.abs(Xs)
    pk = signal.find_peaks(mag)[0]
    pk = pk[(f[pk] >= lo) & (f[pk] <= hi)]
    if pk.size < k:
        pk = np.argsort(mag)[::-1][:k]
    pk = pk[np.argsort(mag[pk])[::-1][:k]]
    f0 = np.sort(f[pk])
    z0 = np.full(k, 0.0015)
    fc = 0.5 * (lo + hi)

    def model(p):
        fk = p[:k]; zk = p[k:2 * k]
        r = p[2 * k:4 * k:2] + 1j * p[2 * k + 1:4 * k:2]
        c0 = p[4 * k] + 1j * p[4 * k + 1]; c1 = p[4 * k + 2] + 1j * p[4 * k + 3]
        sk = -zk * 2 * np.pi * fk + 1j * 2 * np.pi * fk * np.sqrt(1 - zk ** 2)
        Y = (r[None, :] / (w[:, None] - sk[None, :])).sum(1) + c0 + c1 * (f - fc)
        return Y

    def resid(p):
        d = model(p) - Xs
        return np.concatenate([d.real, d.imag])

    # 선형 초기 잔차(r, c) 추정
    sk0 = -z0 * 2 * np.pi * f0 + 1j * 2 * np.pi * f0
    Bm = np.column_stack([1 / (w - s) for s in sk0] + [np.ones_like(w), f - fc])
    coef, *_ = np.linalg.lstsq(Bm, Xs, rcond=None)
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
    fit_err = np.linalg.norm(sol.fun) / np.linalg.norm(np.concatenate([Xs.real, Xs.imag]))
    return fk[o], zk[o], amp / amp.max(), r[o], fit_err
