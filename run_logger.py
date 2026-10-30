import csv
from pathlib import Path

import cv2


LOG_FIELDS = [
    "frame_index", "timestamp_ms", "drive", "voltage",
    "refractory_left", "spike", "right_signal", "left_signal",
    "up_signal", "down_signal", "total_excitation",
    "dominant_sector", "horizontal_vertical_ratio",
    "left_right_symmetry", "up_down_symmetry", "dt_ms",
    "actual_fps", "active_center", "spike_neurons",
    "first_spike_frame", "first_spike_neuron",
]


class RunLogger:
    def __init__(self, log_path, snapshot_dir):
        self.log_file = open(log_path, "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.log_file)
        self.writer.writerow(LOG_FIELDS)
        self.snapshot_path = Path(snapshot_dir)
        self.snapshot_path.mkdir(parents=True, exist_ok=True)

    def write_frame(self, row):
        self.writer.writerow(row)
        self.log_file.flush()

    def save_first_spike(self, frame, dashboard, grid_dashboard,
                         frame_index, active_center):
        snapshot_frame = frame.copy()
        cv2.putText(
            snapshot_frame,
            f"FIRST SPIKE | frame {frame_index} | center {active_center}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (0, 0, 255), 2, cv2.LINE_AA,
        )
        files = (
            (f"first_spike_frame_{frame_index:06d}.png", snapshot_frame),
            (f"first_spike_dashboard_{frame_index:06d}.png", dashboard),
            (f"first_spike_grid_{frame_index:06d}.png", grid_dashboard),
        )
        for name, image in files:
            if not cv2.imwrite(str(self.snapshot_path / name), image):
                raise OSError(f"Could not save snapshot: {name}")

    def close(self):
        self.log_file.close()
