import cv2
import numpy as np

from lif import DT, LIFNeuron
from lplc2 import lplc2_bio_drive


FRAME_SIZE = (32, 32)


def preprocess_frame(frame, size=FRAME_SIZE):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


def run(source=0, size=FRAME_SIZE, dt_ms=DT):
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    neuron = LIFNeuron(dt_ms=dt_ms)
    buffer = []

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            processed = preprocess_frame(frame, size)
            buffer.append(processed)
            if len(buffer) > 2:
                buffer.pop(0)

            if len(buffer) == 2:
                stack = np.stack(buffer, axis=0)
                drive, _ = lplc2_bio_drive(stack)
                spiked = neuron.step(float(drive[-1]))
                if spiked:
                    cv2.putText(
                        frame,
                        "SPIKE",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                    )

            cv2.imshow("LPLC2 real-time", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
