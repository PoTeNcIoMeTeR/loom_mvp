import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_DATA_DIR = Path(r"C:\Users\gavaj\Downloads")
TRIALS = {
    3: "lateral_pass_right",
    10: "approach_then_right",
    345: "pass_two_poles",
}
SMOOTHING_WINDOW = 9
LATERAL_SPEED_THRESHOLD = 0.4


def _read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def _column(rows, name):
    return np.asarray([float(row[name]) for row in rows], dtype=float)


def _moving_average(values, window=SMOOTHING_WINDOW):
    if len(values) < window:
        return values.copy()
    kernel = np.ones(window, dtype=float) / window
    padding = window // 2
    padded = np.pad(values, (padding, padding), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _prepare_trial(data_dir, trial):
    log_rows = _read_csv(data_dir / f"realtime_log_{trial}.csv")
    optitrack_rows = _read_csv(data_dir / f"optitrack_{trial}.csv")

    log_time = _column(log_rows, "timestamp_ms")
    log_time -= log_time[0]
    drive = _column(log_rows, "drive")
    voltage = _column(log_rows, "voltage")
    spikes = _column(log_rows, "spike") > 0.5

    opti_time = _column(optitrack_rows, "time [ns]") / 1_000_000.0
    opti_time -= opti_time[0]
    x = _column(optitrack_rows, "x [m]")
    y = _column(optitrack_rows, "y [m]")
    z = _column(optitrack_rows, "z [m]")

    x_video = np.interp(log_time, opti_time, x)
    y_video = np.interp(log_time, opti_time, y)
    z_video = np.interp(log_time, opti_time, z)
    seconds = log_time / 1000.0

    x_smooth = _moving_average(x_video)
    y_smooth = _moving_average(y_video)
    z_smooth = _moving_average(z_video)
    vx = np.gradient(x_smooth, seconds)
    vy = np.gradient(y_smooth, seconds)
    vz = np.gradient(z_smooth, seconds)
    lateral_speed = np.abs(vx)
    speed = np.sqrt(vx * vx + vy * vy + vz * vz)

    return {
        "trial": trial,
        "scenario": TRIALS[trial],
        "time": seconds,
        "drive": drive,
        "voltage": voltage,
        "spikes": spikes,
        "x": x_video,
        "y": y_video,
        "z": z_video,
        "vx": vx,
        "vy": vy,
        "vz": vz,
        "lateral_speed": lateral_speed,
        "speed": speed,
    }


def _first_threshold_time(time, values, threshold):
    indices = np.flatnonzero(values >= threshold)
    return float(time[indices[0]]) if len(indices) else None


def _summary(trial_data):
    time = trial_data["time"]
    spikes = np.flatnonzero(trial_data["spikes"])
    first_spike = float(time[spikes[0]]) if len(spikes) else None
    first_lateral = _first_threshold_time(
        time,
        trial_data["lateral_speed"],
        LATERAL_SPEED_THRESHOLD,
    )
    lead_time = (
        first_lateral - first_spike
        if first_spike is not None and first_lateral is not None
        else None
    )
    return {
        "trial": trial_data["trial"],
        "scenario": trial_data["scenario"],
        "duration_s": time[-1],
        "spikes": int(trial_data["spikes"].sum()),
        "first_spike_s": first_spike,
        "max_drive": np.max(trial_data["drive"]),
        "mean_drive": np.mean(trial_data["drive"]),
        "max_voltage": np.max(trial_data["voltage"]),
        "max_lateral_speed_m_s": np.max(trial_data["lateral_speed"]),
        "max_speed_m_s": np.max(trial_data["speed"]),
        "first_lateral_speed_over_threshold_s": first_lateral,
        "spike_to_lateral_lead_s": lead_time,
    }


def _write_summary(path, summaries):
    fields = list(summaries[0])
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)


def _plot_trial(trial_data, output_path):
    time = trial_data["time"]
    figure, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    figure.suptitle(
        f"Trial {trial_data['trial']}: {trial_data['scenario']}",
    )

    axes[0].plot(time, trial_data["drive"], label="drive")
    axes[0].plot(time, trial_data["voltage"], label="voltage")
    spike_indices = np.flatnonzero(trial_data["spikes"])
    if len(spike_indices):
        axes[0].scatter(
            time[spike_indices],
            trial_data["drive"][spike_indices],
            color="red",
            marker="x",
            label="spike",
        )
    axes[0].set_ylabel("LPLC2 / LIF")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(time, trial_data["vx"], label="vx")
    axes[1].plot(time, trial_data["vy"], label="vy")
    axes[1].plot(time, trial_data["vz"], label="vz")
    axes[1].plot(
        time,
        trial_data["lateral_speed"],
        label="abs(vx)",
        linewidth=2,
    )
    axes[1].axhline(
        LATERAL_SPEED_THRESHOLD,
        color="gray",
        linestyle="--",
        label="lateral threshold",
    )
    axes[1].set_ylabel("OptiTrack speed (m/s)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(time, trial_data["x"], label="x")
    axes[2].plot(time, trial_data["y"], label="y")
    axes[2].plot(time, trial_data["z"], label="z")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Position (m)")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def analyze(data_dir=DEFAULT_DATA_DIR, output_dir=None):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir) if output_dir else data_dir / "trial_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    trial_data = []
    summaries = []
    for trial in TRIALS:
        data = _prepare_trial(data_dir, trial)
        trial_data.append(data)
        summaries.append(_summary(data))
        _plot_trial(data, output_dir / f"trial_{trial}.png")

    _write_summary(output_dir / "trial_summary.csv", summaries)
    return summaries, output_dir


if __name__ == "__main__":
    summaries, output_dir = analyze()
    print(f"Analysis saved to: {output_dir}")
    for summary in summaries:
        print(
            f"trial {summary['trial']}: "
            f"spikes={summary['spikes']}, "
            f"max_drive={summary['max_drive']:.3f}, "
            f"first_spike_s={summary['first_spike_s']}"
        )
