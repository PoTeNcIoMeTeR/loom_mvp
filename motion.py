import numpy as np


def motion_maps(stack, delay=1):
   
    if not isinstance(delay, (int, np.integer)) or delay < 1:
        raise ValueError("delay must be a positive integer")
    stack = np.asarray(stack, dtype=float)
    if stack.ndim != 3:
        raise ValueError("stack must have shape (time, height, width)")
    if stack.shape[0] <= delay:
        raise ValueError("stack must contain more frames than delay")
    if not np.all(np.isfinite(stack)):
        raise ValueError("stack must contain only finite values")

    D = 1.0 - stack
    before = D[:-delay]
    now = D[delay:]

    H = (
        before[:, :, :-1] * now[:, :, 1:]
        - now[:, :, :-1] * before[:, :, 1:]
    )

    V = (
        before[:, :-1, :] * now[:, 1:, :]
        - now[:, :-1, :] * before[:, 1:, :]
    )

    H = np.pad(H, ((0, 0), (0, 0), (0, 1)))
    V = np.pad(V, ((0, 0), (0, 1), (0, 0)))

    right = np.maximum(H, 0)
    left = np.maximum(-H, 0)
    down = np.maximum(V, 0)
    up = np.maximum(-V, 0)

    return np.array([right, left, up, down])
