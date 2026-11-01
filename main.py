import cv2
import numpy as np
from collections import deque
from pathlib import Path
from time import perf_counter

from lif import DT, LIFNeuron, V_THRESHOLD
from lplc2 import lplc2_grid_bio_drive
from motion import motion_maps
from run_logger import RunLogger


FRAME_SIZE = (32, 32)
HISTORY_SIZE = 120
SECTOR_NAMES = ("right", "left", "up", "down")
GRID_SIZE = (3, 3)
WARMUP_FRAMES = 15


def _grid_centers(shape, grid_size=GRID_SIZE):
    height, width = shape
    rows, columns = grid_size
    if rows < 1 or columns < 1:
        raise ValueError("grid_size values must be positive")
    x_values = np.linspace((width - 1) / 4, 3 * (width - 1) / 4, columns)
    y_values = np.linspace((height - 1) / 4, 3 * (height - 1) / 4, rows)
    return tuple(
        (float(cx), float(cy))
        for cy in y_values
        for cx in x_values
    )


def _grid_regions(shape, grid_size=GRID_SIZE):
    height, width = shape
    rows, columns = grid_size
    y_edges = np.linspace(0, height, rows + 1, dtype=int)
    x_edges = np.linspace(0, width, columns + 1, dtype=int)
    return tuple(
        (int(y_edges[row]), int(y_edges[row + 1]),
         int(x_edges[column]), int(x_edges[column + 1]))
        for row in range(rows)
        for column in range(columns)
    )


def _grid_radii(shape, grid_size=GRID_SIZE):
    height, width = shape
    rows, columns = grid_size
    if rows < 1 or columns < 1:
        raise ValueError("grid_size values must be positive")
    return tuple(
        min(height / rows, width / columns) / 2.0
        for _ in range(rows * columns)
    )


def preprocess_frame(frame, size=FRAME_SIZE):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


def _heatmap(image, size):
    normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    colored = cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.resize(colored, size, interpolation=cv2.INTER_NEAREST)


def _plot_history(values, title, height=180, width=640, color=(0, 220, 0),
                  limit=None):
    panel = np.full((height, width, 3), 25, dtype=np.uint8)
    cv2.putText(panel, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (230, 230, 230), 1, cv2.LINE_AA)
    if len(values) < 2:
        return panel

    points = np.asarray(values, dtype=float)
    maximum = limit if limit is not None else max(float(points.max()), 1e-6)
    minimum = min(0.0, float(points.min()))
    span = max(maximum - minimum, 1e-6)
    coords = []
    for index, value in enumerate(points):
        x = int(index * (width - 20) / max(len(points) - 1, 1)) + 10
        y = height - 12 - int((value - minimum) / span * (height - 42))
        coords.append((x, max(30, min(height - 12, y))))
    cv2.polylines(panel, [np.asarray(coords)], False, color, 2)
    if limit is not None:
        threshold_y = height - 12 - int((limit - minimum) / span * (height - 42))
        cv2.line(panel, (10, threshold_y), (width - 10, threshold_y),
                 (0, 0, 255), 1)
    return panel


def _label_panel(image, title):
    panel = image.copy()
    cv2.putText(panel, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def _direction_metrics(signals):
    excitation = np.maximum(np.asarray(signals, dtype=float), 0.0)
    total_excitation = float(excitation.sum())
    if total_excitation > 0:
        dominant_sector = SECTOR_NAMES[int(np.argmax(excitation))]
    else:
        dominant_sector = "none"

    horizontal = abs(float(signals[0])) + abs(float(signals[1]))
    vertical = abs(float(signals[2])) + abs(float(signals[3]))
    horizontal_vertical_ratio = horizontal / max(vertical, 1e-6)

    left_right_total = abs(float(signals[0])) + abs(float(signals[1]))
    up_down_total = abs(float(signals[2])) + abs(float(signals[3]))
    left_right_symmetry = (
        1.0 - abs(abs(float(signals[0])) - abs(float(signals[1]))) /
        max(left_right_total, 1e-6)
    )
    up_down_symmetry = (
        1.0 - abs(abs(float(signals[2])) - abs(float(signals[3]))) /
        max(up_down_total, 1e-6)
    )
    return (
        total_excitation,
        dominant_sector,
        horizontal_vertical_ratio,
        left_right_symmetry,
        up_down_symmetry,
    )


def _dashboard(frame, processed, motion, drive, signals, neuron, spiked,
               active_center, drive_history,
               voltage_history, dt_ms, actual_fps):
    display_size = (320, 240)
    original = _label_panel(cv2.resize(frame, display_size), "Input video")
    processed_image = (processed * 255).clip(0, 255).astype(np.uint8)
    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)
    processed_image = _label_panel(
        cv2.resize(processed_image, display_size, interpolation=cv2.INTER_NEAREST),
        "Network input 32x32",
    )
    motion_image = _label_panel(
        _heatmap(np.abs(motion).sum(axis=0), display_size),
        "Motion activity",
    )
    (
        total_excitation,
        dominant_sector,
        horizontal_vertical_ratio,
        left_right_symmetry,
        up_down_symmetry,
    ) = _direction_metrics(signals)

    stats = np.full((240, 320, 3), 25, dtype=np.uint8)
    entries = [
        f"Drive: {drive:.4f}",
        f"Voltage: {neuron.v:.4f}",
        f"Threshold: {V_THRESHOLD:.4f}",
        f"Refractory: {neuron.refractory_left:.1f} ms",
        f"dt: {dt_ms:.2f} ms  FPS: {actual_fps:.2f}",
        f"Signals R/L/U/D: {signals[0]:.2f} {signals[1]:.2f}",
        f"             {signals[2]:.2f} {signals[3]:.2f}",
        f"Excitation: {total_excitation:.2f}",
        f"Dominant: {dominant_sector}",
        f"H/V ratio: {horizontal_vertical_ratio:.2f}",
        f"Sym L/R U/D: {left_right_symmetry:.2f} {up_down_symmetry:.2f}",
        f"Active center: {active_center}",
    ]
    for index, text in enumerate(entries):
        cv2.putText(stats, text, (10, 20 + index * 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (220, 220, 220), 1,
                    cv2.LINE_AA)

    top = np.hstack((original, processed_image, motion_image, stats))
    drive_plot = _plot_history(drive_history, "Drive history", limit=V_THRESHOLD)
    voltage_plot = _plot_history(voltage_history, "Voltage history",
                                  color=(255, 190, 0), limit=V_THRESHOLD)
    plots = np.hstack((drive_plot, voltage_plot))
    return np.vstack((top, plots))


def _grid_dashboard(
    spike_counts, spiked_indices, alarm_triggered, warming_up=False
):
    """Render the nine receptive-field neurons in a separate window."""
    rows, columns = GRID_SIZE
    cell_size = 120
    panel = np.full(
        (rows * cell_size, columns * cell_size, 3),
        (35, 35, 35),
        dtype=np.uint8,
    )
    current_spikes = set(spiked_indices)
    for index, count in enumerate(spike_counts):
        row, column = divmod(index, columns)
        x0, y0 = column * cell_size, row * cell_size
        if index in current_spikes:
            color = (0, 0, 220)
        elif count > 0:
            color = (0, 0, 130)
        else:
            color = (0, 150, 0)
        cv2.rectangle(
            panel, (x0 + 3, y0 + 3),
            (x0 + cell_size - 3, y0 + cell_size - 3),
            color, cv2.FILLED,
        )
        cv2.putText(
            panel, f"Neuron {index + 1}", (x0 + 10, y0 + 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel, f"spikes: {int(count)}", (x0 + 10, y0 + 72),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
            cv2.LINE_AA,
        )
    if warming_up:
        status, status_color = "WARMING UP", (0, 180, 220)
    else:
        status = "ALARM" if alarm_triggered else "NO ALARM"
        status_color = (0, 0, 255) if alarm_triggered else (0, 220, 0)
    cv2.putText(
        panel, status, (10, rows * cell_size - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA,
    )
    return panel


def run(
    source=0,
    size=FRAME_SIZE,
    dt_ms=DT,
    log_path=None,
    snapshot_dir=None,
):
    if log_path is None or snapshot_dir is None:
        source_name = (
            "webcam"
            if isinstance(source, (int, np.integer))
            else Path(str(source)).stem
        )
        output_dir = Path("runs") / source_name
        output_dir.mkdir(parents=True, exist_ok=True)
        if log_path is None:
            log_path = output_dir / "realtime_log.csv"
        if snapshot_dir is None:
            snapshot_dir = output_dir / "first_spike_frames"

    # Truncate the log before opening any source, including a failed source.
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")
    logger = RunLogger(log_path, snapshot_dir)

    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    is_camera = isinstance(source, (int, np.integer))
    nominal_dt_ms = (
        1000.0 / source_fps
        if not is_camera and np.isfinite(source_fps) and source_fps > 0
        else float(dt_ms)
    )
    centers = _grid_centers(size)
    radii = _grid_radii(size)
    neurons = [LIFNeuron(dt_ms=nominal_dt_ms) for _ in centers]
    buffer = []
    drive_history = deque(maxlen=HISTORY_SIZE)
    voltage_history = deque(maxlen=HISTORY_SIZE)
    frame_index = 0
    started_at = perf_counter()
    previous_frame_time = None
    alarm_triggered = False
    first_spike_frame = None
    first_spike_neuron = None
    spike_counts = np.zeros(len(neurons), dtype=int)

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            processed = preprocess_frame(frame, size)
            buffer.append(processed)
            if len(buffer) > 2:
                buffer.pop(0)

            now = perf_counter()
            if is_camera:
                current_dt_ms = (
                    float(dt_ms)
                    if previous_frame_time is None
                    else max((now - previous_frame_time) * 1000.0, 1e-3)
                )
            else:
                current_dt_ms = nominal_dt_ms
            previous_frame_time = now
            actual_fps = 1000.0 / current_dt_ms

            drive_value = 0.0
            signals = np.zeros(4)
            active_center = "none"
            active_neuron = neurons[0]
            motion = np.zeros((4, *processed.shape), dtype=float)
            spiked = False
            spiked_indices = []
            if len(buffer) == 2:
                stack = np.stack(buffer, axis=0)
                drive, signals_by_time = lplc2_grid_bio_drive(
                    stack, centers, radii=radii
                )
                motion = motion_maps(stack)[:, 0]
                current_drives = drive[-1]
                current_signals = signals_by_time[-1]
                drive_value = float(np.max(current_drives))
                active_index = int(np.argmax(current_drives))
                signals = current_signals[active_index]
                active_neuron = neurons[active_index]
                active_center = (
                    f"{centers[active_index][0]:.1f},"
                    f"{centers[active_index][1]:.1f}"
                )
                warming_up = frame_index < WARMUP_FRAMES
                spiked_indices = [
                    index
                    for index, (neuron, current) in enumerate(
                        zip(neurons, current_drives)
                    )
                    if not warming_up and neuron.step(
                        current, dt_ms=current_dt_ms
                    )
                ]
                spiked = bool(spiked_indices)
                if spiked:
                    spike_counts[spiked_indices] += 1
                if spiked and not alarm_triggered:
                    alarm_triggered = True
                    first_spike_frame = frame_index
                    first_spike_neuron = spiked_indices[0] + 1
                    active_index = spiked_indices[0]
                    active_neuron = neurons[active_index]
                    active_center = (
                        f"{centers[active_index][0]:.1f},"
                        f"{centers[active_index][1]:.1f}"
                    )

            (
                total_excitation,
                dominant_sector,
                horizontal_vertical_ratio,
                left_right_symmetry,
                up_down_symmetry,
            ) = _direction_metrics(signals)
            drive_history.append(drive_value)
            voltage_history.append(active_neuron.v)
            timestamp_ms = (perf_counter() - started_at) * 1000
            logger.write_frame([
                frame_index, f"{timestamp_ms:.3f}", f"{drive_value:.6f}",
                f"{active_neuron.v:.6f}", active_neuron.refractory_left,
                int(spiked),
                *[f"{value:.6f}" for value in signals],
                f"{total_excitation:.6f}", dominant_sector,
                f"{horizontal_vertical_ratio:.6f}",
                f"{left_right_symmetry:.6f}", f"{up_down_symmetry:.6f}",
                f"{current_dt_ms:.6f}", f"{actual_fps:.6f}",
                active_center,
                ",".join(str(index + 1) for index in spiked_indices),
                "" if first_spike_frame is None else first_spike_frame,
                "" if first_spike_neuron is None else first_spike_neuron,
            ])
            dashboard = _dashboard(
                frame, processed, motion, drive_value, signals, active_neuron,
                spiked, active_center, drive_history,
                voltage_history, current_dt_ms,
                actual_fps,
            )
            grid_dashboard = _grid_dashboard(
                spike_counts, spiked_indices, alarm_triggered,
                frame_index < WARMUP_FRAMES,
            )
            if spiked and first_spike_frame == frame_index:
                logger.save_first_spike(
                    frame, dashboard, grid_dashboard,
                    frame_index, active_center,
                )
            cv2.imshow("LPLC2 real-time dashboard", dashboard)
            cv2.imshow(
                "LPLC2 neuron grid",
                grid_dashboard,
            )
            frame_index += 1
            key = cv2.waitKey(10)
            if key != -1 and (key & 0xFF) in (ord("q"), ord("Q"), 27):
                break
    finally:
        logger.close()
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run(r"C:\Users\gavaj\Downloads\345.avi")
   # run(0)
