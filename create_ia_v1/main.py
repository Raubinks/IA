import functions
from ultralytics import YOLO
import cv2
import pathlib

PATH = functions.get_path()
image_path = PATH / "image/wine.png"


def get_ia_model():
    model_path = pathlib.Path(__file__).resolve().parent.parent / "MODELE IA"
    model_path = model_path / "ia_v4.pt"
    return model_path

def display_yolo_stream(stream):
    model_path = get_ia_model()
    model = YOLO(str(model_path))  # modèle préchargé
    for frame in stream:
        results = model.predict(frame, stream=True)
        for r in results:
            annotated = r.plot()
            cv2.imshow("YOLO Detection", annotated)
            if cv2.waitKey(1) == 27:  # Échap pour quitter
                break
    cv2.destroyAllWindows()

def yolo_analyse(stream):
    model_path = get_ia_model()
    model = YOLO(str(model_path))  # modèle préchargé
    for frame in stream:
        results = model.predict(frame, stream=True)
        for r in results:
            annotated = r.plot()
            cv2.imshow("YOLO Detection", annotated)
            if cv2.waitKey(1) == 27:  # Échap pour quitter
                break
    cv2.destroyAllWindows()

def image_result(image_path, modele_path):
    model = YOLO(str(modele_path))
    results = model.predict(image_path)
    for r in results:
        annotated = r.plot()
        cv2.imshow("YOLO Detection", annotated)
        cv2.waitKey(0)  # Attend une touche pour fermer
        cv2.destroyAllWindows()

#image_result(image_path)
display_yolo_stream(functions.rs_stream_intern_camera())