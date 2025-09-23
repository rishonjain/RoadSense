import os
import streamlit as st
import pandas as pd
import traceback
import time
from moviepy.editor import VideoFileClip
from pathlib import Path
import detection

# --- CONFIG & SETUP --------------------------------------------------------
# --- CORRECTED MODEL PATH ---
MODEL_PATH = Path(r"C:\Users\risho\Documents\RoadSense\runs\train\yolo11n_finetuned\weights\best.pt")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="RoadSense - Road Damage Detection", layout="centered")
st.title("🚧 RoadSense – Road Damage Detection")
st.write("Upload an image (.jpg/.png) or an MP4 video to get an annotated output and a damage report.")

# --- HELPER FUNCTIONS ------------------------------------------------------
@st.cache_data
def convert_to_web_format(input_path):
    """Converts a video to a web-friendly MP4 H.264 format using moviepy."""
    st.write("Cache miss: converting video...")
    try:
        clip = VideoFileClip(str(input_path))
        # Define an output path in the same directory with a suffix
        output_path = input_path.parent / f"{input_path.stem}_web.mp4"
        clip.write_videofile(str(output_path), codec='libx264', audio_codec='aac', preset='veryfast', logger=None)
        clip.close()
        return output_path
    except Exception as e:
        st.warning(f"Video conversion failed: {e}. The video may not play correctly in the browser.")
        return None

def results_to_dataframe(results_list):
    """
    Converts a list of Ultralytics results objects (one per frame/image) 
    into a single pandas DataFrame.
    """
    rows = []
    if not results_list:
        return pd.DataFrame(columns=["Frame", "Damage Type", "Confidence", "Width", "Height", "Bbox"])
    
    for frame_index, frame_results in enumerate(results_list):
        if frame_results.boxes is not None:
            names = frame_results.names
            for box in frame_results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, xyxy)
                
                rows.append([
                    frame_index,
                    names[cls_id], 
                    round(conf, 3), 
                    x2 - x1, 
                    y2 - y1, 
                    (x1, y1, x2, y2)
                ])
    
    return pd.DataFrame(rows, columns=["Frame", "Damage Type", "Confidence", "Width", "Height", "Bbox"])


# --- MAIN APP LOGIC ---------------------------------------------------------
uploaded_file = st.file_uploader("Upload an image or mp4 video", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file is None:
    st.info("Upload a file to begin analysis.")
else:
    input_path = UPLOAD_DIR / uploaded_file.name
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    file_ext = input_path.suffix.lower()

    with st.spinner("Processing your file... this might take a moment."):
        try:
            annotated_path, raw_results = detection.run_detection(str(input_path))
            annotated_path = Path(annotated_path) # Ensure it's a Path object
            df = results_to_dataframe(raw_results)

        except Exception as e:
            st.error("An error occurred during processing.")
            st.code(traceback.format_exc())
            st.stop()

    # --- DISPLAY RESULTS ----------------------------------------------------
    st.success("Processing complete!")

    if annotated_path.exists():
        if file_ext in [".jpg", ".jpeg", ".png"]:
            st.subheader("🖼 Annotated Image")
            image_bytes = annotated_path.read_bytes()
            st.image(image_bytes, use_container_width=True)
            st.download_button(
                label="Download Annotated Image",
                data=image_bytes,
                file_name=annotated_path.name,
                mime=f"image/{file_ext.strip('.')}"
            )
        
        elif file_ext == ".mp4":
            st.subheader("📹 Annotated Video")
            web_video_path = convert_to_web_format(annotated_path)
            display_path = web_video_path if web_video_path and web_video_path.exists() else annotated_path
            
            video_bytes = display_path.read_bytes()
            st.video(video_bytes)
            st.download_button(
                label="Download Annotated Video",
                data=video_bytes,
                file_name=display_path.name,
                mime="video/mp4"
            )
    else:
        st.warning("Annotated output file not found.")

    if df is not None and not df.empty:
        st.subheader("📊 Detection Results")
        st.dataframe(df)
        
        # Convert dataframe to CSV for download
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name=f"{input_path.stem}_results.csv",
            mime="text/csv",
        )

        summary = df.groupby("Damage Type").agg(
            Count=("Damage Type", "count"),
            Avg_Confidence=("Confidence", "mean")
        ).reset_index()
        st.subheader("📋 Summary by Damage Type")
        st.table(summary)
    else:
        st.info("✅ No road damage was detected in the provided file.")