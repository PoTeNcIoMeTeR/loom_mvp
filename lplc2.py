import numpy as np
from frames import SIZE, stimulus_looming, stimulus_receding, stimulus_translation
from motion import motion_maps


def radial_masks(cx=SIZE // 2, cy=SIZE // 2):
    y, x = np.mgrid[:SIZE, :SIZE]
    return {
        "right": (x > cx).astype(float),
        "left": (x < cx).astype(float),
        "up": (y < cy).astype(float),
        "down": (y > cy).astype(float),
    }


def lplc2_drive(stack):
    """Збудження (рух назовні) мінус гальмування (рух досередини), по кадрах."""
    right, left, up, down = motion_maps(stack)
    m = radial_masks()
    excite = (right * m["right"] + left * m["left"] + up * m["up"] + down * m["down"]).sum(axis=(1, 2))
    inhibit = (left * m["right"] + right * m["left"] + down * m["up"] + up * m["down"]).sum(axis=(1, 2))
    return excite, inhibit


if __name__ == "__main__":
    loom = stimulus_looming()
    stimuli = {"розширення": loom, "стиснення": stimulus_receding(), "зсув": stimulus_translation(len(loom))}
    for name, s in stimuli.items():
        e, i = lplc2_drive(s)
        net = e - i
        print("net_abs_sum =", np.abs(net).sum())
        print(f"{name:12} E_sum={e.sum():7.2f} I_sum={i.sum():7.2f} net_sum={net.sum():7.2f}  net_max={net.max():6.2f} net_min={net.min():6.2f}")
        print("   net по кадрах (кожен 5-й):", np.round(net[::5], 2))