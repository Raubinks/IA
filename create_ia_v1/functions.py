import pathlib
import json
import pathlib
import time
import numpy as np
import cv2, numpy as np, pyrealsense2 as rs
from ultralytics import YOLO
import json
from collections import Counter
import os
import nltk
from nltk.corpus import wordnet as wn



# --------------------Load config.json and get parameters-------------------

# Load config.json
def get_config():
    PATH = get_path()
    config_json = PATH / "config.json"
    return json.load(open(config_json))

#Get robot IP from config.json
def get_path():
    return pathlib.Path(__file__).resolve().parent

# Get robot IP from config
def get_ip():
    return get_config().get("robot_ip", "192.168.1.5")

# Get camera index from config
def get_camera_index():
    return int(get_config().get("camera_index", 0))

# Get yolo model from config
def get_yolo_model():
    return get_config().get("yolo_model", "yolov8n.pt")

#Get recognition frequency from config
def get_recognition_frequency():
    return float(get_config().get("frequency", 2.0))

def get_sort_model():
    return 


#-------------------------- Recognition -------------------------

# RealSense stream generator
def rs_stream():
    camera_index = get_camera_index() 
    size, fps = (640, 480), 30
    devs = list(rs.context().query_devices()); assert devs, "No RealSense camera"
    serials = [d.get_info(rs.camera_info.serial_number) for d in devs]
    assert 0 <= camera_index < len(serials), "camera_index out of range"
    pipe, conf = rs.pipeline(), rs.config()
    conf.enable_device(serials[camera_index])
    conf.enable_stream(rs.stream.color, *size, rs.format.bgr8, fps)
    pipe.start(conf)
    try:
        while True:
            f = pipe.wait_for_frames().get_color_frame()
            if f: yield np.asanyarray(f.get_data())
    finally:
        pipe.stop()

def rs_stream_intern_camera():
    camera_index = get_camera_index()
    cap = cv2.VideoCapture(camera_index)
    assert cap.isOpened(), "No webcam"

    # Choisis une résolution standard
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 20)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


# Show realtime stream with YOLO detections and export data in json file
def show_realtime(stream, path="recognized_data.json", on_frame=None):
    out_path = path / "recognized_data.json"
    out_path2 = path / "list_objects.json"
    win = "YOLO preview - q to quit"
    model = YOLO(get_yolo_model())
    last = None
    for i, img in enumerate(stream):
        res = model(img, verbose=False)[0]
        dets = []
        if res.boxes is not None:
            for cls, conf, xyxy in zip(res.boxes.cls.tolist(), res.boxes.conf.tolist(), res.boxes.xyxy.tolist()):
                dets.append({
                    "class_id": int(cls),
                    "class_name": res.names[int(cls)],
                    "confidence": float(conf),
                    "bbox_xyxy": [float(x) for x in xyxy]
                })
        last = {"frame_id": i, "detections": dets}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(last, f, ensure_ascii=False, indent=2)
        cv2.imshow(win, res.plot())
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if on_frame:
            on_frame(last)
    cv2.destroyAllWindows()
    return last

# ---------------------Deal with the datas-----------------------------

# Function to treat "recognized_data.json and output a "list_ojects.json" with the amount
def list_objects(data_path, list_objects_path, database_path, model):
    with open(data_path, 'r') as f:
        data = json.load(f)

    counts = Counter(d["class_name"] for d in data.get("detections", []))
    
    grouped_results = {
        "food": [],
        "others": []
    }

    for obj, n in counts.items():
        # Crée l'objet à stocker
        item = {"object": obj, "amount": n}
        
        # Ajoute l'objet à la bonne catégorie
        if is_food_list(obj,database_path):
            grouped_results["food"].append(item)
        elif is_food_ai(obj,model):
            grouped_results["food"].append(item)
        else:
            grouped_results["others"].append(item)

    with open(list_objects_path, "w") as f:
        json.dump(grouped_results, f, indent=2)

# Detect if an object is food based on ai
def is_food_ai(word: str, model) -> bool:    
    candidate_labels = ["food", "non-food object"]
    result = model(word, candidate_labels)
    return result['labels'][0] == 'food'



# Detect if an object is food based on a database
def is_food_list(word, json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    food_set = set(data["foods"])
    return word in food_set

    
# Verify target presence
def verify_target_presence(target, path, min_conf=0.0):
    try:
        dets = json.load(open(path, encoding="utf-8")).get("detections", [])
    except Exception:
        return False

    def match(d):
        conf = d.get("confidence", d.get("conf", 0.0))
        if conf < min_conf: return False
        cid  = d.get("class_id", d.get("cls_id"))
        cname = d.get("class_name", d.get("cls", ""))
        if callable(target): return bool(target({**d, "conf": conf, "cls_id": cid, "cls": cname}))
        if isinstance(target, int): return cid == target
        t = str(target).lower()
        return str(cid) == t or str(cname).lower() == t
    return any(map(match, dets))
