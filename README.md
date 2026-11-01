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
exact implementation of the Drosophila visual system. Each neuron currently
uses a circular local field centered in its grid cell, with radial directional
masks applied inside that circle. The circles are sized from the frame and
grid dimensions and may overlap neighboring fields.

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

## Synthetic tests

The separate `synthetic_tests.py` script generates small AVI sequences with a
white disk on a black background and runs the same real-time pipeline on them.
Run all scenarios with:

```powershell
python synthetic_tests.py
```

Run one scenario, for example:

```powershell
python synthetic_tests.py --scenario loom_5
```

Results are stored in `synthetic_runs/<scenario-name>/`. Each scenario folder
contains the generated `input.avi`, `realtime_log.csv`, and
`first_spike_frames/`. The folder for a scenario is cleared before that
scenario is generated again.

The `loom_1` through `loom_9` scenarios test expansion in each grid region.
`static_center` checks for activity without motion, `translation` checks
constant-size horizontal motion, and `receding_center` checks the opposite of
looming.

The previous rectangular-field tests produced the expected first-spike neuron
in all nine looming positions:

```text
loom_1 -> neuron 1    loom_2 -> neuron 2    loom_3 -> neuron 3
loom_4 -> neuron 4    loom_5 -> neuron 5    loom_6 -> neuron 6
loom_7 -> neuron 7    loom_8 -> neuron 8    loom_9 -> neuron 9
```

The static and receding controls produced no spikes. Constant-size horizontal
translation produced spikes, showing that the current prototype detects
strong local motion but does not yet distinguish looming from every other
moving pattern.

The current implementation uses the same nine centers with circular fields.
These fields are the next experimental stage and must be rechecked against
both synthetic and real recordings before their parameters are considered
validated.

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
- The circular grid provides only a coarse obstacle location.
- Parameters are experimentally configured and are not biological measurements.
- Real-time testing is currently based on available recorded flights and
  should not be interpreted as a statistically complete benchmark.
