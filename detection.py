import os
from ultralytics import YOLO

# --- UPDATED MODEL PATH ---
MODEL_PATH = r"C:\Users\risho\Documents\RoadSense\runs\yolo11n_finetuned\weights\best.pt"

# Load the YOLO model once when the module is imported
model = YOLO(MODEL_PATH)

def run_detection(video_path):
    """
    Runs YOLO detection on a video, returns the path of the annotated video,
    and the raw detection results for ALL frames.
    """
    # Run prediction with image size matching the training configuration
    results_list = model.predict(
        source=video_path,
        imgsz=416,  # <-- Added to match your new training
        save=True,
        save_txt=True,
        conf=0.25,
        project="runs/detect"
    )

    if not results_list:
        raise RuntimeError("Detection failed, no results were produced.")

    first_frame_results = results_list[0]
    output_dir = first_frame_results.save_dir
    
    video_basename = os.path.basename(video_path)
    output_path = os.path.join(output_dir, video_basename)

    # Fallback check for .avi format
    if not os.path.exists(output_path):
        base, _ = os.path.splitext(output_path)
        avi_path = base + ".avi"
        if os.path.exists(avi_path):
            output_path = avi_path
        else:
            raise FileNotFoundError(f"Annotated video not found in '{output_dir}'.")

    # Return the output path AND the ENTIRE list of frame results
    return output_path, results_list