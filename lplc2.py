import numpy as np
import matplotlib.pyplot as plt

from frames import (
    SIZE,
    make_frame,
    stimulus_looming,
    stimulus_receding,
    stimulus_translation,
    stimulus_vertical_translation,
    stimulus_looming_drift,
)
from lif import lif_neuron, DT, V_THRESHOLD
from motion import motion_maps


SECTORS = ("right", "left", "up", "down")
DEFAULT_GAIN = 0.5


def radial_masks(shape, cx=None, cy=None):
    if (
        not isinstance(shape, (tuple, list)) or
        len(shape) != 2 or
        any(
            not isinstance(value, (int, np.integer)) or value < 2
            for value in shape
        )
    ):
        raise ValueError("shape must be a pair of integers greater than 1")

    height, width = shape
    if cx is None:
        cx = (width - 1) / 2
    if cy is None:
        cy = (height - 1) / 2
    if not np.isfinite(cx) or not np.isfinite(cy):
        raise ValueError("cx and cy must be finite numbers")

    y, x = np.mgrid[:height, :width]
    dx = x - cx
    dy = cy - y
    angle = np.arctan2(dy, dx)
    center = (dx == 0) & (dy == 0)

    return {
        "right": (
            (angle >= -np.pi / 4) &
            (angle < np.pi / 4) &
            ~center
        ).astype(float),
        "up": (
            (angle >= np.pi / 4) &
            (angle < 3 * np.pi / 4) &
            ~center
        ).astype(float),
        "left": (
            ((angle >= 3 * np.pi / 4) |
             (angle < -3 * np.pi / 4)) &
            ~center
        ).astype(float),
        "down": (
            (angle >= -3 * np.pi / 4) &
            (angle < -np.pi / 4) &
            ~center
        ).astype(float),
    }


def sector_signals(stack, cx=None, cy=None):
    """Return signed radial signals in right, left, up, down order."""
    right, left, up, down = motion_maps(stack)
    masks = radial_masks(stack.shape[1:], cx, cy)

    return np.stack([
        (right * masks["right"] - left * masks["right"]).sum(axis=(1, 2)),
        (left * masks["left"] - right * masks["left"]).sum(axis=(1, 2)),
        (up * masks["up"] - down * masks["up"]).sum(axis=(1, 2)),
        (down * masks["down"] - up * masks["down"]).sum(axis=(1, 2)),
    ], axis=1)


def lplc2_bio_drive(
    stack,
    cx=SIZE // 2,
    cy=SIZE // 2,
    saturation=1.0,
    inhibition_gain=0.5,
    coincidence_gain=1.0,
    coincidence_power=2.0,
):
    """Compute a local dendritic, bio-inspired looming drive."""
    if not np.isfinite(saturation) or saturation <= 0:
        raise ValueError("saturation must be a positive finite number")
    if not np.isfinite(inhibition_gain) or inhibition_gain < 0:
        raise ValueError("inhibition_gain must be finite and non-negative")
    if not np.isfinite(coincidence_gain) or coincidence_gain < 0:
        raise ValueError("coincidence_gain must be finite and non-negative")
    if not np.isfinite(coincidence_power) or coincidence_power <= 0:
        raise ValueError("coincidence_power must be positive and finite")

    signals = sector_signals(stack, cx, cy)
    excitation = np.maximum(signals, 0.0)
    inhibition = np.maximum(-signals, 0.0)

    saturated_excitation = (
        saturation * excitation / (saturation + excitation)
    )
    branch_sum = saturated_excitation.sum(axis=1)
    opponent_inhibition = inhibition.sum(axis=1)
    active_branches = np.count_nonzero(excitation > 0, axis=1)
    coincidence = active_branches / len(SECTORS)

    drive = (
        branch_sum * (
            1.0 + coincidence_gain * coincidence ** coincidence_power
        )
        - inhibition_gain * opponent_inhibition
    )
    return np.maximum(drive, 0.0), signals


def plot_lif_response(name, signal, gain=1.0, dt_ms=DT):
    voltage, spikes, spike_voltage = lif_neuron(gain * signal, dt_ms=dt_ms)
    time = np.arange(len(voltage)) * dt_ms

    print(f"{name}: спайків — {len(spikes)}")
    print(f"Моменти спайків (мс): {spikes * dt_ms}")

    plt.figure(figsize=(9, 4))
    plt.plot(time, voltage, label="Мембранний потенціал")
    plt.axhline(V_THRESHOLD, color="red", linestyle="--", label="Поріг")
    if len(spikes) > 0:
        plt.scatter(
            spikes * dt_ms,
            spike_voltage,
            marker="x",
            label="Спайки",
        )
    plt.xlabel("Час (мс)")
    plt.ylabel("Потенціал")
    plt.title(f"LIF — {name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def _diagonal_translation(n_frames, direction_x=1, direction_y=1):
    return np.array([
        make_frame(
            radius=4,
            cx=7 + direction_x * 15 * i / (n_frames - 1),
            cy=7 + direction_y * 15 * i / (n_frames - 1),
        )
        for i in range(n_frames)
    ])


def test_motion_direction():
    """Check the low-level direction convention of motion_maps."""
    tests = {
        "праворуч": (10, 11, SIZE // 2, SIZE // 2),
        "ліворуч": (11, 10, SIZE // 2, SIZE // 2),
        "вниз": (SIZE // 2, SIZE // 2, 10, 11),
        "вгору": (SIZE // 2, SIZE // 2, 11, 10),
    }

    print("\n=== Перевірка напрямків руху ===")
    for name, (x1, x2, y1, y2) in tests.items():
        stack = np.array([
            make_frame(radius=4, cx=x1, cy=y1),
            make_frame(radius=4, cx=x2, cy=y2),
        ])
        right, left, up, down = motion_maps(stack)
        print(
            f"{name:10}: right={right.sum():.3f}, "
            f"left={left.sum():.3f}, "
            f"up={up.sum():.3f}, "
            f"down={down.sum():.3f}"
        )


def test_motion_patterns(gain=DEFAULT_GAIN):
    """Acceptance test for looming and translational controls."""
    loom = stimulus_looming()
    patterns = {
        "looming": loom,
        "receding": stimulus_receding(),
        "right": stimulus_translation(len(loom)),
        "vertical": stimulus_vertical_translation(len(loom)),
        "diagonal": _diagonal_translation(len(loom)),
        "anti_diag": _diagonal_translation(len(loom), 1, -1),
        "looming_drift": stimulus_looming_drift(
            drift_px_per_frame=0.25,
        ),
    }
    expected_response = {
        "looming": True,
        "receding": False,
        "right": False,
        "vertical": False,
        "diagonal": False,
        "anti_diag": False,
    }

    print("\n=== Біоінспірована перевірка рухів ===")
    for name, stack in patterns.items():
        drive, signals = lplc2_bio_drive(stack)
        _, spikes, _ = lif_neuron(gain * drive)
        responds = len(spikes) > 0
        peak = int(np.argmax(drive))
        print(
            f"{name:14}: max={drive[peak]:.4f}, "
            f"spikes={len(spikes):2d}, "
            f"peak_signals={np.round(signals[peak], 3)}"
        )
        if name in expected_response and responds != expected_response[name]:
            raise AssertionError(
                f"{name} response mismatch: expected "
                f"{expected_response[name]}, got {responds}"
            )


if __name__ == "__main__":
    test_motion_direction()
    test_motion_patterns()

    looming = stimulus_looming()
    drive, _ = lplc2_bio_drive(looming)
    plot_lif_response(
        "Біоінспірований LPLC2 — looming",
        drive,
        gain=DEFAULT_GAIN,
    )
