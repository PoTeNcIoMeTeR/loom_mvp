import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

from main import FRAME_SIZE, _grid_centers, run


ROOT = Path("synthetic_runs")
FPS = 30.0
FRAME_COUNT = 120


def _looming_frames(cx, cy, reverse=False):
    radii = np.linspace(1.0, 7.5, FRAME_COUNT)
    if reverse:
        radii = radii[::-1]
    y, x = np.ogrid[:FRAME_SIZE[0], :FRAME_SIZE[1]]
    frames = []
    for radius in radii:
        disk = ((x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2)
        image = np.zeros(FRAME_SIZE, dtype=np.uint8)
        image[disk] = 255
        frames.append(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
    return frames


def _static_frames(cx, cy):
    return _looming_frames(cx, cy)[-1:] * FRAME_COUNT


def _translation_frames():
    y, x = np.ogrid[:FRAME_SIZE[0], :FRAME_SIZE[1]]
    frames = []
    for cx in np.linspace(4, 27, FRAME_COUNT):
        disk = (x - cx) ** 2 + (y - 16) ** 2 <= 4 ** 2
        image = np.zeros(FRAME_SIZE, dtype=np.uint8)
        image[disk] = 255
        frames.append(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
    return frames


def _write_video(path, frames):
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        FPS,
        (FRAME_SIZE[1], FRAME_SIZE[0]),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create synthetic video: {path}")
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


def _scenario_frames(name, centers):
    if name.startswith("loom_"):
        index = int(name.removeprefix("loom_")) - 1
        cx, cy = centers[index]
        return _looming_frames(round(cx), round(cy))
    if name == "static_center":
        cx, cy = centers[4]
        return _static_frames(round(cx), round(cy))
    if name == "translation":
        return _translation_frames()
    if name == "receding_center":
        cx, cy = centers[4]
        return _looming_frames(round(cx), round(cy), reverse=True)
    raise ValueError(f"Unknown scenario: {name}")


def run_scenario(name, centers):
    scenario_dir = ROOT / name
    if scenario_dir.exists():
        shutil.rmtree(scenario_dir)
    scenario_dir.mkdir(parents=True)

    video_path = scenario_dir / "input.avi"
    _write_video(video_path, _scenario_frames(name, centers))
    run(
        str(video_path),
        log_path=scenario_dir / "realtime_log.csv",
        snapshot_dir=scenario_dir / "first_spike_frames",
    )


def main():
    parser = argparse.ArgumentParser(description="Run synthetic LPLC2 tests")
    parser.add_argument(
        "--scenario",
        help="Run one scenario, for example loom_5 or static_center",
    )
    args = parser.parse_args()

    centers = _grid_centers(FRAME_SIZE)
    scenarios = [f"loom_{index}" for index in range(1, 10)]
    scenarios.extend(("static_center", "translation", "receding_center"))
    if args.scenario:
        if args.scenario not in scenarios:
            raise SystemExit(f"Unknown scenario. Choose from: {scenarios}")
        scenarios = [args.scenario]

    for scenario in scenarios:
        print(f"Running synthetic scenario: {scenario}")
        run_scenario(scenario, centers)


if __name__ == "__main__":
    main()
