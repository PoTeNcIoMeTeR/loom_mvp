import cv2
import csv
import numpy as np
from collections import deque
from time import perf_counter

from lif import DT, LIFNeuron, V_THRESHOLD
from lplc2 import lplc2_bio_drive
from motion import motion_maps


FRAME_SIZE = (32, 32)
HISTORY_SIZE = 120
SECTOR_NAMES = ("right", "left", "up", "down")


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
               drive_history, voltage_history, dt_ms, actual_fps):
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
    ]
    for index, text in enumerate(entries):
        cv2.putText(stats, text, (10, 25 + index * 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1,
                    cv2.LINE_AA)
    status = "SPIKE" if spiked else "NO SPIKE"
    status_color = (0, 0, 255) if spiked else (0, 180, 0)
    cv2.putText(stats, status, (10, 232), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                status_color, 2, cv2.LINE_AA)

    top = np.hstack((original, processed_image, motion_image, stats))
    drive_plot = _plot_history(drive_history, "Drive history", limit=V_THRESHOLD)
    voltage_plot = _plot_history(voltage_history, "Voltage history",
                                  color=(255, 190, 0), limit=V_THRESHOLD)
    plots = np.hstack((drive_plot, voltage_plot))
    return np.vstack((top, plots))


def run(source=0, size=FRAME_SIZE, dt_ms=DT, log_path="realtime_log.csv"):
    # Truncate the log before opening any source, including a failed source.
    log_file = open(log_path, "w", newline="", encoding="utf-8")
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        log_file.close()
        raise RuntimeError(f"Could not open video source: {source}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    is_camera = isinstance(source, (int, np.integer))
    nominal_dt_ms = (
        1000.0 / source_fps
        if not is_camera and np.isfinite(source_fps) and source_fps > 0
        else float(dt_ms)
    )
    neuron = LIFNeuron(dt_ms=nominal_dt_ms)
    buffer = []
    drive_history = deque(maxlen=HISTORY_SIZE)
    voltage_history = deque(maxlen=HISTORY_SIZE)
    frame_index = 0
    started_at = perf_counter()
    previous_frame_time = None

    try:
        writer = csv.writer(log_file)
        writer.writerow([
            "frame_index", "timestamp_ms", "drive", "voltage",
            "refractory_left", "spike", "right_signal", "left_signal",
            "up_signal", "down_signal", "total_excitation",
            "dominant_sector", "horizontal_vertical_ratio",
            "left_right_symmetry", "up_down_symmetry", "dt_ms",
            "actual_fps",
        ])

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
            motion = np.zeros((4, *processed.shape), dtype=float)
            spiked = False
            if len(buffer) == 2:
                stack = np.stack(buffer, axis=0)
                drive, signals_by_time = lplc2_bio_drive(stack)
                motion = motion_maps(stack)[:, 0]
                drive_value = float(drive[-1])
                signals = signals_by_time[-1]
                spiked = neuron.step(drive_value, dt_ms=current_dt_ms)

            (
                total_excitation,
                dominant_sector,
                horizontal_vertical_ratio,
                left_right_symmetry,
                up_down_symmetry,
            ) = _direction_metrics(signals)
            drive_history.append(drive_value)
            voltage_history.append(neuron.v)
            timestamp_ms = (perf_counter() - started_at) * 1000
            writer.writerow([
                frame_index, f"{timestamp_ms:.3f}", f"{drive_value:.6f}",
                f"{neuron.v:.6f}", neuron.refractory_left, int(spiked),
                *[f"{value:.6f}" for value in signals],
                f"{total_excitation:.6f}", dominant_sector,
                f"{horizontal_vertical_ratio:.6f}",
                f"{left_right_symmetry:.6f}", f"{up_down_symmetry:.6f}",
                f"{current_dt_ms:.6f}", f"{actual_fps:.6f}",
            ])
            log_file.flush()

            dashboard = _dashboard(
                frame, processed, motion, drive_value, signals, neuron,
                spiked, drive_history, voltage_history, current_dt_ms,
                actual_fps,
            )
            cv2.imshow("LPLC2 real-time dashboard", dashboard)
            frame_index += 1
            key = cv2.waitKey(10)
            if key != -1 and (key & 0xFF) in (ord("q"), ord("Q"), 27):
                break
    finally:
        log_file.close()
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run(r"C:\Users\gavaj\Downloads\345.avi")
