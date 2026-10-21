import numpy as np

DT = 5.0
TAU_M = 20.0
V_REST = 0.0
V_RESET = 0.0
V_THRESHOLD = 1.0
T_REF = 10.0


def lif_neuron(current):
    v = V_REST
    voltage = []
    spikes = []
    spike_voltage = []
    refractory_steps = int(T_REF / DT)
    refractory_left = 0

    for i in range(len(current)):
        if refractory_left > 0:
            v = V_RESET
            refractory_left -= 1
        else:
            dv = (-(v - V_REST) + current[i]) * DT / TAU_M
            v += dv

            if v >= V_THRESHOLD:
                spikes.append(i)
                spike_voltage.append(v)
                v = V_RESET
                refractory_left = refractory_steps

        voltage.append(v)

    return (
        np.array(voltage),
        np.array(spikes),
        np.array(spike_voltage)
    )