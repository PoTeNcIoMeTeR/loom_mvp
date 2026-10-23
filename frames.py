
import numpy as np

SIZE = 31
DEG_PER_PIXEL = 5


def make_frame(radius, cx=SIZE // 2, cy=SIZE // 2):
    """Темний диск на світлому фоні з м'яким краєм."""
    if not np.isfinite(radius) or radius < 0:
        raise ValueError("radius must be a finite non-negative number")
    if not np.isfinite(cx) or not np.isfinite(cy):
        raise ValueError("cx and cy must be finite numbers")
    y, x = np.ogrid[:SIZE, :SIZE]
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    return np.clip(dist - radius + 0.5, 0, 1)


def looming_radii(rv_ms=20, dt_ms=5, radius_scale=1.0):
    if not np.isfinite(rv_ms) or rv_ms <= 0:
        raise ValueError("rv_ms must be a positive finite number")
    if not np.isfinite(dt_ms) or dt_ms <= 0:
        raise ValueError("dt_ms must be a positive finite number")
    if not np.isfinite(radius_scale) or radius_scale <= 0:
        raise ValueError("radius_scale must be a positive finite number")

    t_start = rv_ms / np.tan(np.radians(5))
    t = np.arange(-t_start, -rv_ms, dt_ms)
    if len(t) < 2:
        raise ValueError("dt_ms is too large for the requested looming stimulus")

    theta_deg = np.degrees(
        2 * np.arctan(rv_ms / np.abs(t))
    )

    radius_px = theta_deg / DEG_PER_PIXEL / 2
    radius_px *= radius_scale

    return t, theta_deg, radius_px



def stimulus_looming(rv_ms=20, dt_ms=5, radius_scale=1.0,
                     cx=SIZE // 2, cy=SIZE // 2):
    _, _, radius_px = looming_radii(rv_ms, dt_ms, radius_scale)
    return np.array([
        make_frame(r, cx=cx, cy=cy)
        for r in radius_px
    ])

def stimulus_looming_drift(
    rv_ms=20,
    dt_ms=5,
    radius_scale=1.0,
    drift_px_per_frame=0.5,
    direction=1,
    cx=SIZE // 2,
    cy=SIZE // 2,
):
    _, _, radius_px = looming_radii(rv_ms, dt_ms, radius_scale)

    frames = []
    n = len(radius_px)
    start_cx = cx - direction * drift_px_per_frame * (n - 1) / 2

    for i, radius in enumerate(radius_px):
        current_cx = start_cx + direction * drift_px_per_frame * i
        frame = make_frame(
            radius,
            cx=int(round(current_cx)),
            cy=cy,
        )
        frames.append(frame)

    return np.array(frames)



def stimulus_receding(rv_ms=20, dt_ms=5, radius_scale=1.0):
    return stimulus_looming(
        rv_ms=rv_ms,
        dt_ms=dt_ms,
        radius_scale=radius_scale,
    )[::-1]


def stimulus_translation(n_frames, radius=4):
    if not isinstance(n_frames, (int, np.integer)) or n_frames < 2:
        raise ValueError("n_frames must be an integer greater than or equal to 2")
    xs = np.linspace(7, SIZE - 8, n_frames)
    return np.array([
        make_frame(radius, cx=x) for x in xs
    ])


def stimulus_vertical_translation(n_frames, radius=4):
    if not isinstance(n_frames, (int, np.integer)) or n_frames < 2:
        raise ValueError("n_frames must be an integer greater than or equal to 2")
    ys = np.linspace(7, SIZE - 8, n_frames)
    return np.array([
        make_frame(radius, cy=y) for y in ys
    ])
