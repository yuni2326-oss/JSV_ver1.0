"""N6 가진·응답점 — 자유-자유 실측 형상 모드형에서 응답 가시도와 해머·강구 타격 힘 스펙트럼 (E2-I 1단계).

가시도 V_k(x) = √(Σ_members φ_z(x)² / max_{앞판 윗면} Σ_members φ_z²): doublet은 두 성분 제곱합이라 방향과 무관하다.
고정 응답점(반복성 측정)은 낮은 관측량 모두의 V가 큰 곳, 가진점은 힘 스펙트럼과 V를 함께 본다.

타격 모의: 해머(유효질량 m_h, 강철 구면 팁 반경 R, 속도 v0)가 Hertz 접촉 F = k_H δ^{3/2}(같은 재질,
E* = E/(2(1 − ν²)), k_H = 4/3·E*·√R)로 구조의 한 절점을 판 법선 방향으로 친다. 구조 = 저장된 탄성 모드
(질량정규, ζ 0.1 %) + 강체 3자유도(z 병진, x·y축 흔들림; 질량·관성은 격자에서). 반암시 오일러, dt ≪ 접촉 주기.

경계: 모드 절단(80모드 ≈ 19 kHz) 위의 국부 유연도가 빠져 펄스가 실제보다 짧다(낙관). 비관 쪽 비교 =
무한판 점 임피던스 Z = 8√(Dρh) 대시팟. 강체 기준(Hertz만, 2 g·R 1.27 mm·0.5 m/s)은 −10 dB ≈ 29 kHz로
PCB 086E80 매뉴얼의 강철 팁 곡선(≈ 28 kHz)과 맞는다. 베인–판은 연속 접합이다(실물은 점 용접 → 용접점 위만 해당).
"""
from __future__ import annotations

import math

import numpy as np

HERTZ_DURATION = 2.9432      # t_c = 2.9432·δ_max/v0 (F = kδ^{3/2}, 강체 반공간)


def mass_props(coors: np.ndarray, conn: np.ndarray, rho: float) -> dict:
    """8절점 육면체 격자의 질량·질량중심 z·질량중심 지나는 x·y축 관성 [SI]."""
    e = coors[conn]
    V = np.zeros(len(e))
    C = np.zeros((len(e), 3))
    for t in ((0, 1, 3, 4), (1, 2, 3, 6), (1, 4, 5, 6), (3, 4, 6, 7), (1, 3, 4, 6)):
        p = e[:, list(t)]
        v = np.abs(np.einsum("ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]),
                             p[:, 3] - p[:, 0])) / 6.0
        V += v
        C += v[:, None] * p.mean(axis=1)
    C /= V[:, None]
    m = rho * V
    M = float(m.sum())
    zc = float((m * C[:, 2]).sum() / M)
    return {"M": M, "zc": zc,
            "Ixx": float((m * (C[:, 1] ** 2 + (C[:, 2] - zc) ** 2)).sum()),
            "Iyy": float((m * (C[:, 0] ** 2 + (C[:, 2] - zc) ** 2)).sum())}


def rigid_phi2(xy: np.ndarray, props: dict) -> float:
    """z 방향 점력에 대한 강체 이동도 계수 Σ φ_r² = 1/M + y²/I_xx + x²/I_yy (C6 → 관성곱 0)."""
    x, y = float(xy[0]), float(xy[1])
    return 1.0 / props["M"] + y ** 2 / props["Ixx"] + x ** 2 / props["Iyy"]


def hertz_k(E: float, nu: float, R: float) -> float:
    return 4.0 / 3.0 * E / (2.0 * (1.0 - nu ** 2)) * math.sqrt(R)


def plate_impedance(E: float, nu: float, rho: float, t: float) -> float:
    """무한 Kirchhoff 판 점 임피던스 Z = 8√(Dρh) [N·s/m]."""
    D = E * t ** 3 / (12.0 * (1.0 - nu ** 2))
    return 8.0 * math.sqrt(D * rho * t)


def impact(phi2: np.ndarray, omega: np.ndarray, m_h: float, *, E: float, nu: float, R: float = 1.27e-3,
           v0: float = 0.5, zeta: float = 1e-3, rigid: float = 0.0, Z: float | None = None,
           dt: float = 2e-8, T: float = 3e-3) -> np.ndarray:
    """접촉력 이력 F(t) [N] (t = 0부터 dt 간격). Z가 있으면 구조 = 대시팟.

    끝: 떨어져 있고, 서로 멀어지는 중이고, 틈이 탄성 진폭(지금까지 최대)의 두 배를 넘으면 0 하나를 붙이고 멈춘다
    — 그 뒤에는 다시 닿을 수 없다(강체 운동은 틈을 키우기만 한다). 아니면 T까지.
    """
    kH = hertz_k(E, nu, R)
    phi = np.sqrt(np.asarray(phi2, float))
    om = np.asarray(omega, float)
    q = np.zeros_like(phi)
    qd = np.zeros_like(phi)
    xh, vh, qr, qrd, wz = 0.0, v0, 0.0, 0.0, 0.0
    amp = 0.0
    out = []
    t, started = 0.0, False
    while t < T:
        w_el = float(phi @ q) if Z is None else 0.0
        amp = max(amp, abs(w_el))
        w = wz if Z is not None else w_el + qr
        dl = xh - w
        F = kH * dl ** 1.5 if dl > 0 else 0.0
        if F > 0:
            started = True
        elif started:
            w_dot = (float(phi @ qd) + qrd) if Z is None else 0.0
            if w_dot - vh > 0 and -dl > 2.0 * amp:
                out.append(0.0)
                break
        out.append(F)
        vh -= F / m_h * dt
        xh += vh * dt
        if Z is None:
            qd += (phi * F - 2 * zeta * om * qd - om ** 2 * q) * dt
            q += qd * dt
            qrd += rigid * F * dt
            qr += qrd * dt
        else:
            wz += F / Z * dt
        t += dt
    return np.asarray(out)


def n_contacts(F: np.ndarray) -> int:
    on = np.asarray(F) > 0
    return int(on[0]) + int(np.count_nonzero(on[1:] & ~on[:-1]))


def force_band(F: np.ndarray, dt: float, nfft: int = 1 << 22) -> dict:
    """힘 스펙트럼(직류 정규화) — 처음 −10·−20 dB 아래로 내려가는 주파수, 10·25 kHz 수준, 접촉 횟수·길이."""
    S = np.abs(np.fft.rfft(F, nfft))
    f = np.fft.rfftfreq(nfft, dt)
    dB = 20 * np.log10(np.maximum(S / S[0], 1e-12))

    def first_below(level):
        i = np.flatnonzero(dB < level)
        return float(f[i[0]]) if i.size else float("nan")

    return {"f_m10dB_Hz": first_below(-10.0), "f_m20dB_Hz": first_below(-20.0),
            "dB_10kHz": float(dB[np.searchsorted(f, 10e3)]), "dB_25kHz": float(dB[np.searchsorted(f, 25e3)]),
            "n_contact": n_contacts(F), "duration_us": float(np.count_nonzero(F > 0) * dt * 1e6),
            "span_us": float((np.flatnonzero(F > 0)[-1] + 1) * dt * 1e6), "F_peak_N": float(F.max())}


def visibility(vshape: np.ndarray, members: list[list[int]], ref_mask: np.ndarray) -> np.ndarray:
    """(관측량, 절점) 가시도 V = √(Σ_members u_z² / max_ref Σ u_z²)."""
    out = []
    for mem in members:
        uz2 = sum(vshape[k].reshape(-1, 3)[:, 2] ** 2 for k in mem)
        out.append(np.sqrt(uz2 / uz2[ref_mask].max()))
    return np.asarray(out)
