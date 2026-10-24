import numpy as np

DT = 5.0
TAU_M = 20.0
V_REST = 0.0
V_RESET = 0.0
V_THRESHOLD = 1.0
T_REF = 10.0


class LIFNeuron:
    def __init__(self, dt_ms=DT):
        if not np.isfinite(dt_ms) or dt_ms <= 0:
            raise ValueError("dt_ms must be a positive finite number")

        self.dt_ms = float(dt_ms)
        self.v = V_REST
        self.refractory_left = 0
        self.last_spike_voltage = None
        self.refractory_steps = max(1, int(np.ceil(T_REF / self.dt_ms)))

    def step(self, current, dt_ms=None):
        if not np.isfinite(current):
            raise ValueError("current must be a finite number")
        if dt_ms is None:
            dt_ms = self.dt_ms
        if not np.isfinite(dt_ms) or dt_ms <= 0:
            raise ValueError("dt_ms must be a positive finite number")
        dt_ms = float(dt_ms)

        self.last_spike_voltage = None
        if self.refractory_left > 0:
            self.v = V_RESET
            self.refractory_left = max(0.0, self.refractory_left - dt_ms)
            return False

        decay = np.exp(-dt_ms / TAU_M)
        self.v = (
            V_REST +
            (self.v - V_REST) * decay +
            float(current) * (1.0 - decay)
        )

        if self.v >= V_THRESHOLD:
            self.last_spike_voltage = self.v
            self.v = V_RESET
            self.refractory_left = T_REF
            return True

        return False


def lif_neuron(current, dt_ms=DT):
    if not np.isfinite(dt_ms) or dt_ms <= 0:
        raise ValueError("dt_ms must be a positive finite number")
    current = np.asarray(current, dtype=float)
    if current.ndim != 1:
        raise ValueError("current must be a one-dimensional sequence")
    if not np.all(np.isfinite(current)):
        raise ValueError("current must contain only finite values")

    neuron = LIFNeuron(dt_ms=dt_ms)
    voltage = []
    spikes = []
    spike_voltage = []

    for i in range(len(current)):
        if neuron.step(current[i]):
            spikes.append(i)
            spike_voltage.append(neuron.last_spike_voltage)
        voltage.append(neuron.v)

    return (
        np.array(voltage),
        np.array(spikes),
        np.array(spike_voltage)
    )