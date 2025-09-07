from ultralytics import YOLO

def main():
    # Path to your YOLO model config (using a small model for speed)
    model = YOLO("yolov8s.pt")  # You can also try yolov8m.pt or yolov8l.pt

    # Train the model
    model.train(
        data="config/data.yaml",   # path to dataset config file
        epochs=50,                 # adjust as needed
        imgsz=640,                 # image size
        batch=16,                  # batch size
        device=0,                  # use GPU (set to "cpu" if no CUDA)
        project="runs",            # output folder
        name="train",              # run name
        pretrained=True            # use pretrained weights
    )

if __name__ == "__main__":
    main()
