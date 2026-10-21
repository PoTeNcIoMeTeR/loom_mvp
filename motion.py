import numpy as np
from frames import stimulus_looming, stimulus_receding, stimulus_translation


def motion_maps(stack, delay=1):
    """stack: (час, y, x). Повертає (4, час-delay, y, x): праворуч, ліворуч, вгору, вниз."""
    D = 1 - stack              # "темність": диск = 1, фон = 0
    before = D[:-delay]        # кадри в момент t - delay
    now = D[delay:]            # кадри в момент t

    H = np.zeros_like(now)     # горизонтальний кореляційний детектор
    H[:, :, :-1] = before[:, :, :-1] * now[:, :, 1:] - now[:, :, :-1] * before[:, :, 1:]

    V = np.zeros_like(now)     # вертикальний (y росте вниз)
    V[:, :-1, :] = before[:, :-1, :] * now[:, 1:, :] - now[:, :-1, :] * before[:, 1:, :]

    right = np.maximum(H, 0)
    left = np.maximum(-H, 0)
    down = np.maximum(V, 0)
    up = np.maximum(-V, 0)
    return np.array([right, left, up, down])


if __name__ == "__main__":
    loom = stimulus_looming()
    stimuli = {
        "розширення": loom,
        "стиснення": stimulus_receding(),
        "зсув": stimulus_translation(len(loom)),
    }
    print(f"{'':12}{'праворуч':>10}{'ліворуч':>10}{'вгору':>10}{'вниз':>10}")
    for name, stim in stimuli.items():
        maps = motion_maps(stim)
        totals = maps.sum(axis=(1, 2, 3))
        print(f"{name:12}" + "".join(f"{v:10.2f}" for v in totals))