
import numpy as np
import matplotlib.pyplot as plt

from frames import (
    SIZE,
    make_frame,
    stimulus_looming,
    stimulus_receding,
    stimulus_translation,
    stimulus_vertical_translation,
)
from motion import motion_maps
from lif import lif_neuron, DT, V_THRESHOLD


def radial_masks(cx=SIZE // 2, cy=SIZE // 2):
    y, x = np.mgrid[:SIZE, :SIZE]
    dx = x - cx
    dy = cy - y
    angle = np.arctan2(dy, dx)

    return {
        "right": (
            (angle >= -np.pi / 4) &
            (angle < np.pi / 4)
        ).astype(float),

        "up": (
            (angle >= np.pi / 4) &
            (angle < 3 * np.pi / 4)
        ).astype(float),

        "left": (
            (angle >= 3 * np.pi / 4) |
            (angle < -3 * np.pi / 4)
        ).astype(float),

        "down": (
            (angle >= -3 * np.pi / 4) &
            (angle < -np.pi / 4)
        ).astype(float),
    }


def show_radial_masks():
    masks = radial_masks()
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))

    for ax, (name, mask) in zip(axes, masks.items()):
        ax.imshow(mask, cmap="gray", vmin=0, vmax=1)
        ax.set_title(name)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def diagnose_masks():
    masks = radial_masks()
    total = np.zeros((SIZE, SIZE), dtype=int)

    print("\nДіагностика радіальних масок:")

    for name, mask in masks.items():
        print(f"{name:5}: {int(mask.sum())} пікселів")
        total += mask.astype(int)

    print(f"\nУсього пікселів: {SIZE * SIZE}")
    print(f"Рівно в одному секторі: {np.sum(total == 1)}")
    print(f"Без сектора: {np.sum(total == 0)}")
    print(f"У кількох секторах: {np.sum(total > 1)}")

    plt.figure(figsize=(5, 5))
    plt.imshow(total, cmap="viridis", vmin=0, vmax=2)
    plt.colorbar(label="Кількість секторів на піксель")
    plt.title("Перетини радіальних масок")
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


def lplc2_drive(stack):
    right, left, up, down = motion_maps(stack)
    masks = radial_masks()

    excite = (
        right * masks["right"]
        + left * masks["left"]
        + up * masks["up"]
        + down * masks["down"]
    ).sum(axis=(1, 2))

    inhibit = (
        left * masks["right"]
        + right * masks["left"]
        + down * masks["up"]
        + up * masks["down"]
    ).sum(axis=(1, 2))

    return excite, inhibit


def diagnose_sector_contributions(stimulus, name):
    right, left, up, down = motion_maps(stimulus)
    masks = radial_masks()

    contributions = {
        "right": (
            (right * masks["right"]).sum(axis=(1, 2)),
            (left * masks["right"]).sum(axis=(1, 2)),
        ),
        "left": (
            (left * masks["left"]).sum(axis=(1, 2)),
            (right * masks["left"]).sum(axis=(1, 2)),
        ),
        "up": (
            (up * masks["up"]).sum(axis=(1, 2)),
            (down * masks["up"]).sum(axis=(1, 2)),
        ),
        "down": (
            (down * masks["down"]).sum(axis=(1, 2)),
            (up * masks["down"]).sum(axis=(1, 2)),
        ),
    }

    print(f"\nДіагностика секторів: {name}")

    for i in range(24, min(30, len(right))):
        print(f"\nКадр {i}, час {i * DT:.0f} мс")
        total_e = 0.0
        total_i = 0.0

        for sector, (e, inh) in contributions.items():
            total_e += e[i]
            total_i += inh[i]
            print(
                f"  {sector:5}: E={e[i]:.4f}, "
                f"I={inh[i]:.4f}, net={e[i] - inh[i]:.4f}"
            )

        print(
            f"  ЗАГАЛОМ: E={total_e:.4f}, "
            f"I={total_i:.4f}, net={total_e - total_i:.4f}"
        )


def diagnose_radial_selectivity(stimuli):
    masks = radial_masks()

    print("\n=== Діагностика радіальної вибірковості ===")

    for name, stimulus in stimuli.items():
        right, left, up, down = motion_maps(stimulus)

        channels = {
            "right": (
                (right * masks["right"]).sum(axis=(1, 2)),
                (left * masks["right"]).sum(axis=(1, 2)),
            ),
            "left": (
                (left * masks["left"]).sum(axis=(1, 2)),
                (right * masks["left"]).sum(axis=(1, 2)),
            ),
            "up": (
                (up * masks["up"]).sum(axis=(1, 2)),
                (down * masks["up"]).sum(axis=(1, 2)),
            ),
            "down": (
                (down * masks["down"]).sum(axis=(1, 2)),
                (up * masks["down"]).sum(axis=(1, 2)),
            ),
        }

        sector_net = np.stack(
            [e - i for e, i in channels.values()],
            axis=1,
        )

        positive = np.maximum(sector_net, 0)
        product = np.prod(positive, axis=1)

        frame = int(np.argmax(product))

        print(f"\n{name}:")
        print(f"Максимальний добуток: {product[frame]:.6f}")
        print(f"Час максимуму: {frame * DT:.0f} мс")
        print("Чисті сигнали секторів у цей момент:")

        for j, sector in enumerate(channels):
            print(f"  {sector:5}: {sector_net[frame, j]:.4f}")


def test_motion_direction():
    tests = {
        "праворуч": (10, 11, SIZE // 2, SIZE // 2),
        "ліворуч": (11, 10, SIZE // 2, SIZE // 2),
        "вниз": (SIZE // 2, SIZE // 2, 10, 11),
        "вгору": (SIZE // 2, SIZE // 2, 11, 10),
    }

    for name, (x1, x2, y1, y2) in tests.items():
        stack = np.array([
            make_frame(radius=4, cx=x1, cy=y1),
            make_frame(radius=4, cx=x2, cy=y2),
        ])

        right, left, up, down = motion_maps(stack)
        totals = [right.sum(), left.sum(), up.sum(), down.sum()]

        print(
            f"{name:10}: right={totals[0]:.3f}, "
            f"left={totals[1]:.3f}, "
            f"up={totals[2]:.3f}, "
            f"down={totals[3]:.3f}"
        )


def plot_lif_response(name, net):
    voltage, spikes, spike_voltage = lif_neuron(net)
    time = np.arange(len(voltage)) * DT

    print(f"{name}: спайків — {len(spikes)}")
    print(f"Моменти спайків (мс): {spikes * DT}")

    plt.figure(figsize=(9, 4))
    plt.plot(time, voltage, label="Мембранний потенціал")
    plt.axhline(
        V_THRESHOLD, color="red", linestyle="--", label="Поріг"
    )
    plt.scatter(
        spikes * DT, spike_voltage, marker="x", label="Спайки"
    )
    plt.xlabel("Час (мс)")
    plt.ylabel("Потенціал")
    plt.title(f"LIF — {name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def diagnose_horizontal_translation(stimulus, net):
    right, left, up, down = motion_maps(stimulus)
    voltage, spikes, _ = lif_neuron(net)

    print("\nПеревірка карт руху:")
    for i in range(27, min(30, len(right))):
        print(
            f"Кадр {i}, "
            f"right={right[i].sum():.3f}, "
            f"left={left[i].sum():.3f}, "
            f"up={up[i].sum():.3f}, "
            f"down={down[i].sum():.3f}"
        )

    print("\nПеревірка потенціалу LIF:")
    spike_set = set(spikes.tolist())

    for i in range(len(net)):
        if net[i] > 0 or i in spike_set:
            print(
                f"t={i * DT:3.0f} мс | "
                f"net={net[i]:.3f} | "
                f"V={voltage[i]:.3f}"
                f"{' | SPIKE' if i in spike_set else ''}"
            )


if __name__ == "__main__":
    show_radial_masks()
    test_motion_direction()
    diagnose_masks()

    loom = stimulus_looming()

    stimuli = {
        "Розширення": loom,
        "Стискання": stimulus_receding(),
        "Горизонтальний зсув": stimulus_translation(len(loom)),
        "Вертикальний зсув": stimulus_vertical_translation(len(loom)),
    }

    # Нова діагностика: чотири чисті сигнали та їхній добуток
    diagnose_radial_selectivity(stimuli)

    net_signals = {}

    for name, stimulus in stimuli.items():
        excite, inhibit = lplc2_drive(stimulus)
        net = excite - inhibit
        net_signals[name] = net

        if name in ("Горизонтальний зсув", "Вертикальний зсув"):
            diagnose_sector_contributions(stimulus, name)

        print(f"\n{name}:")
        print(f"Максимум net: {net.max():.4f}")
        print(f"Мінімум net:  {net.min():.4f}")
        print(f"Сума збудження: {excite.sum():.2f}")
        print(f"Сума гальмування: {inhibit.sum():.2f}")
        print(f"Сума net: {net.sum():.2f}")

        if name == "Горизонтальний зсув":
            diagnose_horizontal_translation(stimulus, net)

        plot_lif_response(name, net)

    plt.figure(figsize=(10, 5))

    for name, signal in net_signals.items():
        time = np.arange(len(signal)) * DT
        plt.plot(time, signal, label=name)

    plt.axhline(0, color="black", linewidth=0.8)
    plt.xlabel("Час (мс)")
    plt.ylabel("net = збудження − гальмування")
    plt.title("Вхідний сигнал LPLC2 для різних стимулів")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
