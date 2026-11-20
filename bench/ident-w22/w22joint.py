"""깃 6곳 공동 적합: 공통 극점(주파수·감쇠) + 깃별 잔차(variable projection), 깃 공통 정규화.
극점마다 잔차 6개의 위상 공선성(MPC)과 원주 차수(h = 0–3) 에너지 비율, 방향 θ0를 낸다."""
import numpy as np
from scipy import optimize, signal

TH = np.radians(60.0 * np.arange(6))          # 깃 1–6 각도(번호 방향을 + 로)


def vane_avg(F, X, vane, N=None, slope=0.0, fr=None):
    """깃별 평균 H1. slope ≠ 0이면 타격마다 힘으로 주파수 축을 20 N 기준으로 늘인다(연화 보정)."""
    H = {}
    for v in range(1, 7):
        s = np.where(vane == v)[0]
        if slope and N is not None:
            Sfx = 0; Sff = 0
            for i in s:
                sc = 1 + slope * np.log2(N[i] / 20.0)            # f_N = f_20 · sc
                fi = fr * sc
                Sfx = Sfx + np.interp(fi, fr, (np.conj(F[i]) * X[i]).real) + 1j * np.interp(fi, fr, (np.conj(F[i]) * X[i]).imag)
                Sff = Sff + np.interp(fi, fr, np.abs(F[i]) ** 2)
            H[v] = Sfx / Sff
        else:
            H[v] = (np.conj(F[s]) * X[s]).sum(0) / (np.abs(F[s]) ** 2).sum(0)
    return H


def joint_fit(fr, Hs, lo, hi, k, starts=None, zmax=0.02):
    sel = (fr >= lo) & (fr <= hi); f = fr[sel]; w = 2j * np.pi * f; fc = 0.5 * (lo + hi)
    Ys = [H[sel] for H in Hs]
    norm = max(np.abs(Y).max() for Y in Ys)
    def basis(p):
        fk, zk = p[:k], p[k:]
        sk = -zk * 2 * np.pi * fk + 1j * 2 * np.pi * fk * np.sqrt(1 - zk ** 2)
        return np.column_stack([1 / (w - s) for s in sk] + [np.ones_like(w), f - fc]), sk
    def resid(p):
        B, _ = basis(p); out = []
        for Y in Ys:
            c, *_ = np.linalg.lstsq(B, Y, rcond=None); d = (B @ c - Y) / norm
            out += [d.real, d.imag]
        return np.concatenate(out)
    mag = sum(np.abs(Y) ** 2 for Y in Ys)
    pk = signal.find_peaks(mag)[0]
    pk = pk[np.argsort(mag[pk])[::-1]]
    cand = []
    base = list(np.sort(f[pk[:k]])) if pk.size >= k else []
    if len(base) == k:
        cand.append(np.array(base))
    rng = np.random.default_rng(0)
    for _ in range(starts or 12):
        cand.append(np.sort(rng.uniform(lo + 0.1 * (hi - lo), hi - 0.1 * (hi - lo), k)))
    if pk.size:
        for j in range(min(4, pk.size)):
            c = np.sort(np.concatenate([[f[pk[j]]], rng.uniform(lo + 2, hi - 2, k - 1)]))
            cand.append(c)
    best = None
    lb = np.concatenate([np.full(k, lo), np.full(k, 2e-5)]); ub = np.concatenate([np.full(k, hi), np.full(k, zmax)])
    for f0 in cand:
        p0 = np.concatenate([f0, np.full(k, 0.002)])
        sol = optimize.least_squares(resid, np.clip(p0, lb + 1e-6, ub - 1e-6), bounds=(lb, ub), x_scale="jac", max_nfev=2000)
        if best is None or sol.cost < best.cost:
            best = sol
    B, sk = basis(best.x)
    R = np.array([np.linalg.lstsq(B, Y, rcond=None)[0][:k] for Y in Ys])          # (6, k)
    err = np.sqrt(2 * best.cost) / np.sqrt(sum((np.abs(Y / norm) ** 2).sum() for Y in Ys))
    o = np.argsort(best.x[:k])
    return best.x[:k][o], best.x[k:][o], R[:, o], err


def pattern(r):
    """잔차 6개(복소) → MPC, 주축으로 돌린 실수 패턴, h 에너지 비율(0–3), h1·h2 방향 θ0[°]."""
    x, y = r.real, r.imag
    Sxx, Syy, Sxy = x @ x, y @ y, x @ y
    lam, vec = np.linalg.eigh(np.array([[Sxx, Sxy], [Sxy, Syy]]))
    mpc = ((lam[1] - lam[0]) / (lam[1] + lam[0])) ** 2
    u = vec[:, 1]; alpha = np.arctan2(u[1], u[0])
    a = (r * np.exp(-1j * alpha)).real
    if a[np.argmax(np.abs(a))] < 0:
        a = -a; alpha += np.pi
    A = np.array([(a * np.exp(-1j * h * TH)).sum() for h in range(6)])
    P = np.array([abs(A[0]) ** 2, abs(A[1]) ** 2 + abs(A[5]) ** 2, abs(A[2]) ** 2 + abs(A[4]) ** 2, abs(A[3]) ** 2])
    P = P / P.sum()
    th0 = {h: (-np.degrees(np.angle(A[h])) / h) % (360 / h) for h in (1, 2)}   # a_v = cos h(θ_v − θ0) → A_h ∝ e^{−ihθ0}
    return mpc, a, P, th0, np.degrees(alpha)
