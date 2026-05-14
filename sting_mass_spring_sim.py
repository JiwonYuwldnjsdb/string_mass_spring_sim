import argparse
import math
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def positive_float(value: str) -> float:
    x = float(value)
    if x <= 0:
        raise argparse.ArgumentTypeError("양수여야 합니다.")
    return x


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Taut string with a point mass-spring defect at x=0"
    )

    # 물리 파라미터
    p.add_argument("--mu", type=positive_float, default=1.0,
                   help="현의 선밀도 μ")
    p.add_argument("--T", type=positive_float, default=1.0,
                   help="현의 장력 T")
    p.add_argument("--m", type=positive_float, default=1.0,
                   help="x=0에 붙은 점질량 m")
    p.add_argument("--ks", type=positive_float, default=2.0,
                   help="용수철 상수 k_s")

    # 입사 파동 묶음
    p.add_argument("--A", type=positive_float, default=0.35,
                   help="입사 파동 묶음의 진폭")
    p.add_argument("--omega", type=positive_float, default=1.414,
                   help="입사 파동 묶음의 중심 각진동수 ω")
    p.add_argument("--width", type=positive_float, default=8.0,
                   help="가우시안 파동 묶음의 폭")
    p.add_argument("--x0", type=float, default=-35.0,
                   help="초기 파동 묶음 중심 위치")

    # 수치 영역
    p.add_argument("--L", type=positive_float, default=80.0,
                   help="계산 영역 [-L, L]")
    p.add_argument("--dx", type=positive_float, default=0.08,
                   help="공간 격자 간격")
    p.add_argument("--safety", type=positive_float, default=0.45,
                   help="시간 간격 안정성 계수. 보통 0.2~0.6")
    p.add_argument("--tmax", type=positive_float, default=95.0,
                   help="시뮬레이션 총 시간")
    p.add_argument("--frames", type=int, default=650,
                   help="애니메이션 프레임 수")

    # 흡수 경계
    p.add_argument("--sponge-width", type=positive_float, default=14.0,
                   help="양끝 감쇠층 폭")
    p.add_argument("--sponge-strength", type=positive_float, default=2.2,
                   help="감쇠층 최대 감쇠율")

    # 출력
    p.add_argument("--fps", type=int, default=30,
                   help="저장 시 초당 프레임 수")
    p.add_argument("--save", type=str, default="",
                   help="파일로 저장하려면 sim.gif 또는 sim.mp4 같은 이름 입력")

    return p


def analytic_coefficients(omega: float, mu: float, T: float, m: float, ks: float):
    """
    연속계 해석해:
      r = Δ / (2 i T k - Δ)
      t = 1 + r
      Δ = k_s - m ω^2
      k = ω / c, c = sqrt(T/μ)

    반환: r, t, reflectance, transmittance
    """
    c = math.sqrt(T / mu)
    wave_number = omega / c
    delta = ks - m * omega**2
    denom = 2j * T * wave_number - delta
    r = delta / denom
    t = 1 + r
    R = abs(r)**2
    Tr = abs(t)**2
    return r, t, R, Tr


def make_sponge(x: np.ndarray, L: float, sponge_width: float, strength: float) -> np.ndarray:
    """
    영역 양끝에서 속도에 비례하는 감쇠 a_damp = -sigma(x) v 를 준다.
    """
    sigma = np.zeros_like(x)
    start = max(0.0, L - sponge_width)
    mask = np.abs(x) > start
    s = (np.abs(x[mask]) - start) / max(sponge_width, 1e-12)
    sigma[mask] = strength * s**2
    return sigma


def spring_polyline(q: float, y_anchor: float, x_center: float = 0.0,
                    coils: int = 9, amp: float = 0.35):
    """
    질점과 아래 고정점 사이를 잇는 시각화용 지그재그 용수철.
    실제 힘은 -k_s q 로 따로 계산한다.
    """
    n = 2 * coils + 3
    ys = np.linspace(y_anchor, q, n)
    xs = np.full(n, x_center)

    for i in range(1, n - 1):
        xs[i] += amp * (1 if i % 2 else -1)

    xs[0] = x_center
    xs[-1] = x_center
    return xs, ys


def main():
    args = build_parser().parse_args()

    if args.frames < 2:
        raise ValueError("--frames는 2 이상이어야 합니다.")

    mu = args.mu
    T = args.T
    m_point = args.m
    ks = args.ks
    c = math.sqrt(T / mu)
    omega = args.omega
    k_wave = omega / c

    # x=0이 정확히 격자점이 되도록 홀수 개 격자 구성
    n_half = int(round(args.L / args.dx))
    x = np.linspace(-n_half * args.dx, n_half * args.dx, 2 * n_half + 1)
    dx = x[1] - x[0]
    L_eff = abs(x[0])
    j0 = len(x) // 2

    # 각 격자점의 질량. 중앙에는 점질량을 추가한다.
    masses = np.full_like(x, mu * dx)
    masses[j0] += m_point

    # 현의 이산 스프링 상수. 인접 격자점 간 횡방향 힘 = (T/dx)(y_{i+1}-y_i)
    K_link = T / dx

    sigma = make_sponge(x, L_eff, args.sponge_width, args.sponge_strength)

    # 안정적인 시간 간격 자동 선택
    dt_wave = dx / c
    center_freq = math.sqrt((2 * K_link + ks) / masses[j0])
    dt_center = 2.0 / center_freq
    dt = args.safety * min(dt_wave, dt_center)

    nsteps = int(math.ceil(args.tmax / dt))
    steps_per_frame = max(1, nsteps // args.frames)
    actual_frames = int(math.ceil(nsteps / steps_per_frame))

    # 오른쪽으로 진행하는 가우시안 변조 코사인 파동 묶음
    y = args.A * np.exp(-((x - args.x0) / args.width) ** 2) * np.cos(k_wave * (x - args.x0))
    dy_dx = np.gradient(y, dx, edge_order=2)
    v = -c * dy_dx       # y(x,t)=f(x-ct) 이므로 y_t=-c f'(x)

    # 양 끝은 고정하되, 그 앞 감쇠층이 대부분의 반사를 흡수한다.
    y[0] = y[-1] = 0.0
    v[0] = v[-1] = 0.0

    def acceleration(y_now: np.ndarray, v_now: np.ndarray) -> np.ndarray:
        a = np.zeros_like(y_now)

        # 현의 장력에 의한 힘
        a[1:-1] = K_link * (y_now[2:] - 2 * y_now[1:-1] + y_now[:-2]) / masses[1:-1]

        # 중앙 점질량에 작용하는 용수철 복원력
        a[j0] += -ks * y_now[j0] / masses[j0]

        # 흡수 경계 감쇠
        a[1:-1] += -sigma[1:-1] * v_now[1:-1]
        return a

    def step():
        # semi-implicit Euler. dx와 dt를 충분히 작게 잡으면 이 문제 시각화에 안정적이다.
        a = acceleration(y, v)
        v[1:-1] += a[1:-1] * dt
        y[1:-1] += v[1:-1] * dt
        y[0] = y[-1] = 0.0
        v[0] = v[-1] = 0.0

    # 해석적 반사/투과 계수 출력
    r, tcoef, R, Tr = analytic_coefficients(omega, mu, T, m_point, ks)
    omega0 = math.sqrt(ks / m_point)

    print("=== Parameters ===")
    print(f"mu={mu:.6g}, T={T:.6g}, c=sqrt(T/mu)={c:.6g}")
    print(f"m={m_point:.6g}, ks={ks:.6g}, no-reflection omega0=sqrt(ks/m)={omega0:.6g}")
    print(f"carrier omega={omega:.6g}, wave number={k_wave:.6g}")
    print(f"domain=[{-L_eff:.3g}, {L_eff:.3g}], dx={dx:.6g}, dt={dt:.6g}, steps={nsteps}")
    print()
    print("=== Analytic scattering coefficient at carrier omega ===")
    print(f"r = {r.real:+.6f} {r.imag:+.6f} i")
    print(f"t = {tcoef.real:+.6f} {tcoef.imag:+.6f} i")
    print(f"|r|^2 = {R:.6f}, |t|^2 = {Tr:.6f}, sum = {R + Tr:.6f}")
    print()

    # 그림 구성
    y_limit = max(1.05, 3.2 * args.A)
    spring_anchor = -0.86 * y_limit

    fig, (ax, ax_spec) = plt.subplots(
        1, 2, figsize=(13, 5.6),
        gridspec_kw={"width_ratios": [2.35, 1.0]}
    )
    fig.suptitle("Wave scattering by a point mass-spring defect", fontsize=14)

    line, = ax.plot(x, y, lw=1.8, label="string displacement")
    mass_dot, = ax.plot([0.0], [y[j0]], "o", ms=10, label="point mass")
    sx, sy = spring_polyline(y[j0], spring_anchor)
    spring_line, = ax.plot(sx, sy, lw=1.3, label="spring visualization")
    ax.plot([0], [spring_anchor], "s", ms=8, label="fixed support")
    ax.axvline(0, lw=0.8, alpha=0.4)

    # 감쇠층 영역 표시
    ax.axvspan(-L_eff, -L_eff + args.sponge_width, alpha=0.08)
    ax.axvspan(L_eff - args.sponge_width, L_eff, alpha=0.08)

    ax.set_xlim(-L_eff, L_eff)
    ax.set_ylim(-y_limit, y_limit)
    ax.set_xlabel("x")
    ax.set_ylabel("transverse displacement y")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)

    info = ax.text(
        0.02, 0.95, "", transform=ax.transAxes,
        va="top", ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        fontsize=9
    )

    # 오른쪽: 해석적 반사율/투과율 스펙트럼
    om_max = max(3.0 * omega0, 2.7 * omega, 3.0)
    oms = np.linspace(0.02, om_max, 800)
    Rvals = np.zeros_like(oms)
    Tvals = np.zeros_like(oms)
    for i, om in enumerate(oms):
        _, _, Rvals[i], Tvals[i] = analytic_coefficients(om, mu, T, m_point, ks)

    ax_spec.plot(oms, Rvals, label=r"Reflectance $|r|^2$")
    ax_spec.plot(oms, Tvals, label=r"Transmittance $|t|^2$")
    ax_spec.axvline(omega, ls="-", lw=1.0, label=r"carrier $\omega$")
    ax_spec.axvline(omega0, ls="--", lw=1.0, label=r"no reflection $\sqrt{k_s/m}$")
    ax_spec.set_xlim(0, om_max)
    ax_spec.set_ylim(-0.03, 1.03)
    ax_spec.set_xlabel(r"angular frequency $\omega$")
    ax_spec.set_ylabel("energy ratio")
    ax_spec.grid(True, alpha=0.25)
    ax_spec.legend(fontsize=8, loc="center right")
    ax_spec.set_title("Analytic spectrum")

    state = {"t": 0.0, "step": 0}

    def update(_frame):
        if state["step"] < nsteps:
            for _ in range(steps_per_frame):
                if state["step"] >= nsteps:
                    break
                step()
                state["t"] += dt
                state["step"] += 1

        line.set_ydata(y)
        mass_dot.set_data([0.0], [y[j0]])
        sx, sy = spring_polyline(y[j0], spring_anchor)
        spring_line.set_data(sx, sy)

        info.set_text(
            f"t = {state['t']:.2f}\n"
            f"q(t)=y(0,t) = {y[j0]:+.3f}\n"
            f"ω = {omega:.3f},  ω0 = {omega0:.3f}\n"
            f"|r|² = {R:.3f},  |t|² = {Tr:.3f}"
        )
        return line, mass_dot, spring_line, info

    anim = FuncAnimation(
        fig, update, frames=actual_frames, interval=1000 / args.fps,
        blit=False, repeat=False
    )

    plt.tight_layout()

    if args.save:
        filename = args.save
        if filename.lower().endswith(".gif"):
            anim.save(filename, writer="pillow", fps=args.fps, dpi=120)
        else:
            # mp4는 시스템에 ffmpeg가 있어야 한다.
            anim.save(filename, fps=args.fps, dpi=140)
        print(f"saved: {filename}")
    else:
        plt.show()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
