# detection.py (Corrected)
import os
from ultralytics import YOLO

# Path to your trained weights
MODEL_PATH = r"C:\Users\risho\Documents\GitHub\RoadSense\runs\train5\weights\best.pt"

# Load the YOLO model once when the module is imported
model = YOLO(MODEL_PATH)

def run_detection(video_path):
    """
    Runs YOLO detection on a video, returns the path of the annotated video,
    and the raw detection results for ALL frames.
    """
    # Run prediction to get results for every frame
    results_list = model.predict(
        source=video_path,
        save=True,
        save_txt=True,
        conf=0.25,
        project="runs/detect"
    )

    # Check if results were produced
    if not results_list:
        raise RuntimeError("Detection failed, no results were produced.")

    # Get the results from the first frame to find the save directory
    first_frame_results = results_list[0]
    output_dir = first_frame_results.save_dir
    
    # Construct the full path to the annotated video
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

    # --- CRUCIAL FIX ---
    # Return the output path AND the ENTIRE list of frame results
    return output_path, results_list