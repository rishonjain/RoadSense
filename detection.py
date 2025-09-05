import cv2
import torch
import pandas as pd
from ultralytics import YOLO

def run_detection(model_path, img_path, conf_threshold=0.5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(model_path).to(device)

    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = model.predict(img_rgb, conf=conf_threshold, device=0 if device=="cuda" else "cpu")
    annotated_img = results[0].plot()

    det_data = results[0].boxes.data.cpu().numpy()
    df = pd.DataFrame(det_data, columns=["x1", "y1", "x2", "y2", "confidence", "class"])
    df["class_name"] = df["class"].apply(lambda x: model.names[int(x)])
    df["bbox_area"] = (df["x2"] - df["x1"]) * (df["y2"] - df["y1"])

    return df, annotated_img
