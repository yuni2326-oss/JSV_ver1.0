"""흠집(홈) 시험 분석 공통 — 타격 읽기·품질 점검·자유 감쇠 극점(타격별·묶음 적합)·상태별 대역 잡기.

자료: DAQ `pxie4464_daq impact v2`(PXIe-4464_Data `e3693ea`)가 쓰는 타격별 .npz와 impact_log.csv.
방법은 2번 시편 다시 매달기 점검(30번, ident-s2/s2d_rehang.py·s2d_pool_lib.py)과 같다.
- 자유 감쇠: 마지막 힘 접촉(0.5 N 또는 피크의 3 %를 넘는 마지막 표본) 3 ms 뒤부터 0.4 s, 2^20 영 채움.
- 타격별: 대역마다 주파수 영역 극점 적합(spectral_fit) → 20 N 환산(연화 기울기) → 무리 중앙값·표준오차.
- 묶음: 한 무리의 타격들이 극점을 공유하고 잔차만 다른 적합(VarPro) → 부트스트랩 표준편차.
"""
from __future__ import annotations

import csv
import glob
import os
from dataclasses import dataclass

import numpy as np
from scipy import optimize, signal

FS_EXPECTED = 204800.0
NFFT = 1 << 20
DUR = 0.4                         # 자유 감쇠 길이 [s]
MIC = ("CH6_mic", "PXI1Slot5/ai2")
HAMMER = "CH7_hammer"
DF_T = -0.019                     # 온도 1 °C당 주파수 변화 [%] — SUS304 탄성계수(−0.039 %/°C)와 열팽창으로 어림(참고용)


@dataclass(frozen=True)
class Mode:
    name: str            # 측정 극점 이름(10-30 주파수 반올림)
    f: float             # 10-30 2번 시편 ξ0.6 기준 [Hz] (s2j_poles.csv)
    fem: str | None      # G3 FEM 짝(32번·34번), 후보는 '?'를 붙이지 않고 kind로 구분
    kind: str            # disk·panel·mixed·back·hub·cand(후보)·none
    slope: float = 0.0   # 연화 기울기: 힘 2배당 상대 변화(s2b_track)
    judge: bool = True   # 판정에 쓰는가(허브 모드·후보는 뺀다)


MODES = [
    Mode("2134", 2133.80, "h2:m2n0", "disk", -0.00059), Mode("2142", 2142.28, "h2:m2n0", "disk", -0.00059),
    Mode("3164", 3163.65, "h3:m3n0", "panel"), Mode("3250", 3249.89, "h0:m0n1", "mixed"),
    Mode("3459", 3458.89, "h1:m1n1", "panel", -0.00032), Mode("3589", 3588.78, None, "none", 0.0, False),
    Mode("3639", 3638.99, "h2:m4n0", "panel", -0.00051), Mode("3748", 3747.87, "h3:m3n1", "disk"),
    Mode("3804", 3804.07, "h0:m6n1", "hub", -0.00036, False), Mode("4225", 4225.49, "h1:m1n1#2", "hub", -0.00050, False),
    Mode("4967", 4966.99, "h2:m4n0#2", "back"), Mode("5617", 5617.22, "h0:m6n0", "back"),
    Mode("5745", 5744.69, "h1:m5n0", "back"), Mode("6439", 6438.86, "h2:m2n1", "panel"),
    Mode("6558", 6558.41, "h2:m2n1", "cand", 0.0, False), Mode("6817", 6817.09, "h1:m7n0", "cand", 0.0, False),
    Mode("7093", 7093.13, "h3:m3n2", "cand", 0.0, False), Mode("7462", 7462.21, None, "none", -0.00071, False),
    Mode("7574", 7574.17, "h2:m8n0", "cand", 0.0, False), Mode("8321", 8321.24, "h3:m3n1#2", "panel", -0.00046),
]
MODE = {m.name: m for m in MODES}
PAIR = ("2134", "2142")           # h2 짝 — 늘 한 대역에서 극점 2개로 맞춘다


# ---------------------------------------------------------------- 타격 읽기와 품질 점검
def read_log(folder: str) -> dict:
    p = os.path.join(folder, "impact_log.csv")
    if not os.path.exists(p):
        return {}
    with open(p, newline="", encoding="utf-8") as fh:
        return {r["file"]: r for r in csv.DictReader(fh)}


def list_hits(folder: str, sessions=None, locations=None, date=None, t_from=None, t_to=None, notes=None) -> list[dict]:
    """폴더의 타격 메타(자료는 읽지 않음). 거르기: 세션 번호·위치·날짜(YYYYMMDD)·시각(HH:MM)·메모."""
    log = read_log(folder)
    out = []
    for p in sorted(glob.glob(os.path.join(folder, "*.npz"))):
        b = os.path.basename(p)
        if date and not b.startswith(str(date)):
            continue
        hhmm = f"{b[9:11]}:{b[11:13]}"
        if (t_from and hhmm < t_from) or (t_to and hhmm > t_to):
            continue
        with np.load(p, allow_pickle=False) as z:
            meta = {k: z[k].item() if z[k].ndim == 0 else z[k].tolist() for k in z.files if k != "data"}
        if sessions and int(meta["session"]) not in sessions:
            continue
        if locations and str(meta["location"]) not in locations:
            continue
        if notes and str(meta.get("notes", "")).strip() != notes:
            continue
        lg = log.get(b, {})
        meta.update(file=b, path=p, status=lg.get("status", ""), resp_peak=float(lg["resp_peak"]) if lg.get("resp_peak") else np.nan)
        out.append(meta)
    return out


def check_channels(hits: list[dict]) -> list[str]:
    """채널 이름·물리 채널 확인(25번: 마이크가 CH11 빈 입력으로 잡힌 적이 있다)."""
    bad = []
    for h in hits:
        names, phys = list(h["channel_names"]), list(h["physical_channels"])
        if len(names) < 2 or names[0] != HAMMER or names[1] != MIC[0] or phys[1] != MIC[1]:
            bad.append(f"{h['file']}: channel_names {names}, physical {phys}")
    return bad


def load_signals(h: dict):
    with np.load(h["path"], allow_pickle=False) as z:
        d = z["data"].astype(float)
        ti = int(z["trigger_index"]); fs = float(z["sample_rate"])
    f = d[0] - d[0, :ti - 100].mean()
    m = d[1] - d[1, :ti - 100].mean()
    return f, m, ti, fs


def free_decay_start(f: np.ndarray, N: float, ti: int, fs: float):
    """마지막 접촉 3 ms 뒤(최소 5 ms)와 늦은 재타격 여부(s2d_rehang과 같은 기준)."""
    seg = f[ti + int(0.003 * fs): ti + int(0.2 * fs)]
    o = np.where(seg > max(0.5, 0.03 * N))[0]
    late = bool(np.any(seg > max(1.0, 0.05 * N)))
    t0 = max(0.005, (o[-1] + int(0.003 * fs)) / fs + 0.003) if o.size else 0.005
    return t0, late


# ---------------------------------------------------------------- 극점 적합
def spectral_fit(x, fs, ti, lo, hi, k=2, t0=0.005, dur=DUR, nfft=NFFT, pad=8.0):
    """감쇠 응답의 복소 스펙트럼을 Σ r_j/(i2πf − s_j) + c0 + c1·(f − fc)로 맞춘다(ident-w22/specfit.py와 같다).
    반환: 주파수 오름차순 (f, ζ, 상대 진폭, 복소 잔차, 적합 상대오차)."""
    seg = x[ti + int(t0 * fs): ti + int((t0 + dur) * fs)].astype(float)
    X = np.fft.rfft(seg, nfft) / fs
    fr = np.fft.rfftfreq(nfft, 1 / fs)
    sel = (fr >= lo - pad) & (fr <= hi + pad)
    return band_fit(fr[sel], X[sel], lo, hi, k, pad)


def band_fit(f, Xs, lo, hi, k, pad=8.0):
    """spectral_fit의 본체 — 이미 잘라 둔 대역 스펙트럼(f, Xs)에 극점 k개를 맞춘다."""
    w = 2j * np.pi * f
    mag = np.abs(Xs)
    pk = signal.find_peaks(mag)[0]
    pk = pk[(f[pk] >= lo) & (f[pk] <= hi)]
    if pk.size < k:
        pk = np.argsort(mag)[::-1][:k]
    pk = pk[np.argsort(mag[pk])[::-1][:k]]
    f0 = np.sort(f[pk]); z0 = np.full(k, 0.0015); fc = 0.5 * (lo + hi)

    def model(p):
        fk = p[:k]; zk = p[k:2 * k]
        r = p[2 * k:4 * k:2] + 1j * p[2 * k + 1:4 * k:2]
        c0 = p[4 * k] + 1j * p[4 * k + 1]; c1 = p[4 * k + 2] + 1j * p[4 * k + 3]
        sk = -zk * 2 * np.pi * fk + 1j * 2 * np.pi * fk * np.sqrt(1 - zk ** 2)
        return (r[None, :] / (w[:, None] - sk[None, :])).sum(1) + c0 + c1 * (f - fc)

    def resid(p):
        d = model(p) - Xs
        return np.concatenate([d.real, d.imag])

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
    err = np.linalg.norm(sol.fun) / np.linalg.norm(np.concatenate([Xs.real, Xs.imag]))
    return fk[o], zk[o], amp / max(amp.max(), 1e-300), r[o], err


def decay_spectrum(m, ti, fs, t0):
    seg = m[ti + int(t0 * fs): ti + int((t0 + DUR) * fs)]
    return np.fft.rfft(seg, NFFT) / fs


def stretched(X, fr, N, lo, hi, slope, pad=8.0):
    """타격 스펙트럼 X를 20 N 기준 주파수 축으로 옮긴다(연화 보정)."""
    g = np.arange(lo - pad, hi + pad, fr[1] - fr[0])        # fr은 전체 축이든 대역 조각이든 된다
    sc = 1 + slope * np.log2(N / 20.0)
    sel = (fr >= (lo - pad) * sc - 2) & (fr <= (hi + pad) * sc + 2)
    x, y = fr[sel], X[sel]
    return g, np.interp(g * sc, x, y.real) + 1j * np.interp(g * sc, x, y.imag)


def joint_fit(f, Ys, lo, hi, k, p0=None, zmax=0.02):
    """여러 타격이 극점(f, ζ)을 공유하고 잔차만 다른 적합(s2d_pool_lib.jfit과 같다)."""
    w = 2j * np.pi * f; fc = 0.5 * (lo + hi)
    norm = max(np.abs(Y).max() for Y in Ys)
    Ym = np.column_stack(Ys)

    def basis(p):
        fk, zk = p[:k], p[k:]
        sk = -zk * 2 * np.pi * fk + 1j * 2 * np.pi * fk * np.sqrt(1 - zk ** 2)
        return np.column_stack([1 / (w - s) for s in sk] + [np.ones_like(w), f - fc])

    def resid(p):
        B = basis(p)
        c, *_ = np.linalg.lstsq(B, Ym, rcond=None)
        dd = ((B @ c - Ym) / norm).T.ravel()
        return np.concatenate([dd.real, dd.imag])

    lb = np.concatenate([np.full(k, lo), np.full(k, 2e-5)]); ub = np.concatenate([np.full(k, hi), np.full(k, zmax)])
    if p0 is not None:
        cands = [p0]
    else:
        mag = sum(np.abs(Y) ** 2 for Y in Ys); pk = signal.find_peaks(mag)[0]; pk = pk[(f[pk] >= lo) & (f[pk] <= hi)]
        pk = pk[np.argsort(mag[pk])[::-1]]
        rng = np.random.default_rng(0)
        f0s = ([np.sort(f[pk[:k]])] if pk.size >= k else []) + \
              [np.sort(rng.uniform(lo + 0.1 * (hi - lo), hi - 0.1 * (hi - lo), k)) for _ in range(10)]
        cands = [np.concatenate([f0, np.full(k, 0.0015)]) for f0 in f0s]
    best = None
    for c in cands:
        sol = optimize.least_squares(resid, np.clip(c, lb + 1e-6, ub - 1e-6), bounds=(lb, ub), x_scale="jac", max_nfev=3000)
        if best is None or sol.cost < best.cost:
            best = sol
    o = np.argsort(best.x[:k])
    err = np.sqrt(2 * best.cost) / np.sqrt(sum((np.abs(Y / norm) ** 2).sum() for Y in Ys))
    return np.concatenate([best.x[:k][o], best.x[k:][o]]), err


# ---------------------------------------------------------------- 상태별 대역
def plan_bands(shift_pct: dict, names=None, min_hw=0.0035, rel_err=0.4, extra=0.0025, cross=0.0025):
    """예측 이동으로 대역 중심을 옮기고(기준 × (1 + 예측)), 반폭 = max(0.35 %, 0.4 × |예측| + 0.25 %).
    겹치는 대역은 합쳐 극점 여럿으로 맞춘다. 예측 중심이 0.25 % 안으로 붙으면 '교차 위험'으로 표시한다.
    h2 짝(2134/2142)은 늘 한 대역에서 극점 둘."""
    names = names or [m.name for m in MODES]
    items = []
    for n in names:
        m = MODE[n]
        s = float(shift_pct.get(n, 0.0) or 0.0)
        c = m.f * (1 + s / 100)
        hw = m.f * max(min_hw, rel_err * abs(s) / 100 + extra)
        items.append([c - hw, c + hw, [(c, n)]])
    items.sort()
    merged = []
    for lo, hi, ms in items:
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi); merged[-1][2] += ms
        else:
            merged.append([lo, hi, list(ms)])
    bands = []
    for lo, hi, ms in merged:
        ms.sort()
        cs = [c for c, _ in ms]
        warn = [f"{ms[i][1]}·{ms[i + 1][1]}" for i in range(len(ms) - 1) if cs[i + 1] / cs[i] - 1 < cross
                and {ms[i][1], ms[i + 1][1]} != set(PAIR)]
        bands.append({"lo": float(lo), "hi": float(hi), "modes": [n for _, n in ms], "centers": [float(c) for c in cs],
                      "cross": warn})
    return bands
