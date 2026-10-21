import numpy as np


def motion_maps(stack, delay=1):
    """
    stack: (час, y, x).

    Повертає карти руху в порядку:
    right, left, up, down.
    """
    D = 1.0 - stack
    before = D[:-delay]
    now = D[delay:]

    # Горизонтальний рух: порівнюємо сусідні стовпці
    H = (
        before[:, :, :-1] * now[:, :, 1:]
        - now[:, :, :-1] * before[:, :, 1:]
    )

    # Вертикальний рух: порівнюємо сусідні рядки
    V = (
        before[:, :-1, :] * now[:, 1:, :]
        - now[:, :-1, :] * before[:, 1:, :]
    )

    # Додаємо нульову межу, щоб розміри карт збігалися
    H = np.pad(H, ((0, 0), (0, 0), (0, 1)))
    V = np.pad(V, ((0, 0), (0, 1), (0, 0)))

    right = np.maximum(H, 0)
    left = np.maximum(-H, 0)
    down = np.maximum(V, 0)
    up = np.maximum(-V, 0)

    return np.array([right, left, up, down])
