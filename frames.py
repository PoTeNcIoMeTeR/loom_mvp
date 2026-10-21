import numpy as np
import matplotlib.pyplot as plt

SIZE = 31
DEG_PER_PIXEL = 5


def make_frame(radius, cx=SIZE // 2, cy=SIZE // 2):
    """Темний диск (0) на світлому фоні (1) з м'яким краєм."""
    y, x = np.ogrid[:SIZE, :SIZE]
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    return np.clip(dist - radius + 0.5, 0, 1)


def looming_radii(rv_ms=20, dt_ms=5):
    t_start = rv_ms / np.tan(np.radians(5))
    t = np.arange(-t_start, -rv_ms, dt_ms)
    theta_deg = np.degrees(2 * np.arctan(rv_ms / np.abs(t)))
    radius_px = theta_deg / DEG_PER_PIXEL / 2
    return t, theta_deg, radius_px


def stimulus_looming(rv_ms=20, dt_ms=5):
    _, _, radius_px = looming_radii(rv_ms, dt_ms)
    return np.array([make_frame(r) for r in radius_px])


def stimulus_receding(rv_ms=20, dt_ms=5):
    return stimulus_looming(rv_ms, dt_ms)[::-1]  # той самий диск, але стискається


def stimulus_translation(n_frames, radius=4):
    xs = np.linspace(7, SIZE - 8, n_frames)
    return np.array([make_frame(radius, cx=x) for x in xs])

def stimulus_vertical_translation(n_frames, radius=4):
    ys = np.linspace(7, SIZE - 8, n_frames)
    return np.array([
        make_frame(radius, cy=y)
        for y in ys
    ])
if __name__ == "__main__":
    loom = stimulus_looming()
    recede = stimulus_receding()
    shift = stimulus_translation(len(loom))
    print(loom.shape, recede.shape, shift.shape)

    names = ["розширення", "стиснення", "зсув"]
    fig, axes = plt.subplots(3, 6)
    for row, (name, stim) in enumerate(zip(names, [loom, recede, shift])):
        idx = np.linspace(0, len(stim) - 1, 6).astype(int)
        for col, j in enumerate(idx):
            axes[row, col].imshow(stim[j], cmap="gray", vmin=0, vmax=1)
            axes[row, col].axis("off")
        axes[row, 0].set_title(name, loc="left")
    plt.show()