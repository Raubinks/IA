from ultralytics import YOLO
import pathlib

path = pathlib.Path(__file__).resolve().parent.parent
yaml_path = path / "dataset_v2/data.yaml"
print(yaml_path)
# Charger le modèle pré-entraîné
model = YOLO("best.pt")

# Entraînement
model.train(data=yaml_path, 
            epochs=100, 
            imgsz=640, 
            batch=15, 
            device=0
            )

# Validation
model.val()

# Prédiction sur un dossier
model.predict(source="samples", save=True)