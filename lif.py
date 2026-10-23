import numpy as np

DT = 5.0
TAU_M = 20.0
V_REST = 0.0
V_RESET = 0.0
V_THRESHOLD = 1.0
T_REF = 10.0


def lif_neuron(current, dt_ms=DT):
    if not np.isfinite(dt_ms) or dt_ms <= 0:
        raise ValueError("dt_ms must be a positive finite number")
    current = np.asarray(current, dtype=float)
    if current.ndim != 1:
        raise ValueError("current must be a one-dimensional sequence")
    if not np.all(np.isfinite(current)):
        raise ValueError("current must contain only finite values")

    v = V_REST
    voltage = []
    spikes = []
    spike_voltage = []
    refractory_steps = max(1, int(np.ceil(T_REF / dt_ms)))
    refractory_left = 0

    for i in range(len(current)):
        if refractory_left > 0:
            v = V_RESET
            refractory_left -= 1
        else:
            dv = (-(v - V_REST) + current[i]) * dt_ms / TAU_M
            v += dv

            if v >= V_THRESHOLD:
                spikes.append(i)
                spike_voltage.append(v)
                v = V_RESET
                refractory_left = max(0, refractory_steps - 1)

        voltage.append(v)

    return (
        np.array(voltage),
        np.array(spikes),
        np.array(spike_voltage)
    )