# LPLC2 real-time prototype

This project is a real-time, bio-inspired looming and potential-collision
detector. It processes two consecutive grayscale frames, estimates directional
motion, computes an LPLC2-like drive, and feeds the result to LIF neurons.

## Current architecture

```text
two video frames
    -> directional motion maps
    -> 3x3 grid of LPLC2-like receptive-field centers
    -> nine independent LIF neurons
    -> first-spike alarm
```

The grid is numbered as follows:

```text
1 2 3
4 5 6
7 8 9
```

The alarm is triggered when any neuron produces a spike for the first time.
The neurons continue running after that so the dashboard and CSV can count
subsequent spikes for each neuron.

The 3x3 grid is currently an **engineering approximation**, not a biologically
exact implementation of the Drosophila visual system. The grid uses different
receptive-field centers, but fully local non-overlapping receptive-field
regions have not yet been implemented.

## Running

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run the configured video source:

```powershell
python main.py
```

To use the webcam, change the last line of `main.py` to:

```python
run(0)
```

## Development status

The current version includes real-time video processing, a 3x3 grid of
LPLC2-like neurons, per-neuron spike logging, and automatic first-spike
snapshots. The implementation is a working prototype and is still being
evaluated on recorded flight videos and controlled synthetic scenarios.

## Results

Results are grouped by source automatically:

```text
runs/
└── <source-name>/
    ├── realtime_log.csv
    └── first_spike_frames/
        ├── first_spike_frame_XXXXXX.png
        ├── first_spike_dashboard_XXXXXX.png
        └── first_spike_grid_XXXXXX.png
```

The CSV records frame timing, drive, LIF state, directional signals, active
center, neurons spiking on each frame, and the first-spike frame and neuron.

The snapshot directory stores the original frame, dashboard, and neuron grid
at the first spike.

## Limitations

- The model detects optical patterns associated with looming; it does not
  identify objects or guarantee that a collision will occur.
- Camera motion, foliage, texture, lighting changes, and other scene motion
  can cause false positives.
- The current grid does not provide a precise obstacle location.
- Parameters are experimentally configured and are not biological measurements.
- Real-time testing is currently based on available recorded flights and
  should not be interpreted as a statistically complete benchmark.
