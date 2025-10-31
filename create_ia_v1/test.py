import cv2, threading
from ultralytics import YOLO
import pathlib, functions

# -----------------------------
# Thread de capture
# -----------------------------
class CameraStream:
    def __init__(self, index=0, width=640, height=480, fps=20):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.frame = None
        self.running = True
        t = threading.Thread(target=self.update, daemon=True)
        t.start()

    def update(self):
        while self.running:
            ok, f = self.cap.read()
            if ok:
                self.frame = f

    def read(self):
        return self.frame

    def release(self):
        self.running = False
        self.cap.release()

# -----------------------------
# YOLO
# -----------------------------
def get_model():
    model_path = pathlib.Path(__file__).resolve().parent.parent / "MODELE IA" / "yolo11s.pt"
    return YOLO(str(model_path))

def display_yolo_stream():
    model = get_model()
    cam = CameraStream(functions.get_camera_index())
    try:
        while True:
            frame = cam.read()
            if frame is None:
                continue
            results = model.predict(frame, verbose=False)
            for r in results:
                annotated = r.plot()
                cv2.imshow("YOLO Detection", annotated)
            if cv2.waitKey(1) == 27:
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()

# -----------------------------
if __name__ == "__main__":
    display_yolo_stream()