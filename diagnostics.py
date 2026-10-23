import numpy as np
import matplotlib.pyplot as plt

from frames import SIZE, stimulus_looming_drift
from lplc2 import radial_masks, sector_signals
from motion import motion_maps
from lif import DT


def show_radial_masks():
    masks = radial_masks((SIZE, SIZE))
    _, axes = plt.subplots(1, 4, figsize=(12, 3))

    for ax, (name, mask) in zip(axes, masks.items()):
        ax.imshow(mask, cmap="gray", vmin=0, vmax=1)
        ax.set_title(name)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def diagnose_masks():
    masks = radial_masks((SIZE, SIZE))
    total = np.zeros((SIZE, SIZE), dtype=int)

    print("\n=== Діагностика масок ===")
    for name, mask in masks.items():
        print(f"{name:5}: {int(mask.sum())} пікселів")
        total += mask.astype(int)

    print(f"Усього пікселів: {SIZE * SIZE}")
    print(f"Рівно в одному секторі: {np.sum(total == 1)}")
    print(f"Без сектора: {np.sum(total == 0)}")
    print(f"У кількох секторах: {np.sum(total > 1)}")


def diagnose_motion_maps():
    print("\n=== Діагностика карт руху ===")
    cases = [
        ("Без дрейфу", stimulus_looming_drift(drift_px_per_frame=0.0)),
        ("Дрейф вправо", stimulus_looming_drift(
            drift_px_per_frame=0.5,
            direction=1,
        )),
    ]

    for name, stack in cases:
        maps = motion_maps(stack)
        activity = maps.sum(axis=(0, 2, 3))
        peak_idx = int(np.argmax(activity))

        print(f"\n{name}:")
        print("Активність за часом:", np.round(activity, 4))
        print(f"Час максимуму: {peak_idx * DT:.1f} мс")
        print(
            "Активність напрямків у момент максимуму:",
            np.round(maps[:, peak_idx].sum(axis=(1, 2)), 4),
        )


def diagnose_sector_signals_over_time():
    print("\n=== Секторні сигнали в часі ===")
    cases = [
        ("Без дрейфу", stimulus_looming_drift(drift_px_per_frame=0.0)),
        ("Дрейф вправо", stimulus_looming_drift(
            drift_px_per_frame=0.5,
            direction=1,
        )),
    ]

    for name, stack in cases:
        signals = sector_signals(stack)
        print(f"\n--- {name} ---")
        print(
            f"{'Час (мс)':>10} {'right':>10} {'left':>10} "
            f"{'up':>10} {'down':>10}"
        )
        for t in (100, 150, 200):
            idx = min(round(t / DT), len(signals) - 1)
            print(
                f"{idx * DT:>10.1f} "
                f"{signals[idx, 0]:>10.4f} "
                f"{signals[idx, 1]:>10.4f} "
                f"{signals[idx, 2]:>10.4f} "
                f"{signals[idx, 3]:>10.4f}"
            )


if __name__ == "__main__":
    show_radial_masks()
    diagnose_masks()
    diagnose_motion_maps()
    diagnose_sector_signals_over_time()
