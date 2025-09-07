# app.py (Updated with the new Streamlit parameter)
import os
import streamlit as st
import pandas as pd
import traceback
import time
from moviepy import VideoFileClip

# Import your updated detection module
import detection

# --- CONFIG & SETUP --------------------------------------------------------
MODEL_PATH = r"C:\Users\risho\Documents\GitHub\RoadSense\runs\train5\weights\best.pt"

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.set_page_config(page_title="RoadSense - Road Damage Detection", layout="centered")
st.title("🚧 RoadSense – Road Damage Detection")
st.write("Upload an image (.jpg/.png) or an MP4 video to get an annotated output and a damage report.")

# --- HELPER FUNCTIONS ------------------------------------------------------
def convert_to_web_format(input_path):
    """Converts a video to a web-friendly MP4 H.24 format using moviepy."""
    try:
        output_filename = f"{os.path.splitext(os.path.basename(input_path))[0]}_web.mp4"
        output_path = os.path.join(os.path.dirname(input_path), output_filename)
        
        clip = VideoFileClip(input_path)
        clip.write_videofile(output_path, codec='libx264', audio_codec='aac', preset='veryfast', logger=None)
        clip.close()
        
        return output_path
    except Exception as e:
        st.warning(f"Video conversion failed: {e}. The video may not play correctly in the browser.")
        return None

def results_to_dataframe(results_list):
    """
    Converts a list of Ultracyclists results objects (one for each video frame) 
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
    input_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    file_ext = os.path.splitext(uploaded_file.name)[-1].lower()

    with st.spinner("Processing your file... this might take a moment for videos."):
        try:
            annotated_path, raw_results = detection.run_detection(input_path)
            df = results_to_dataframe(raw_results)

        except Exception as e:
            st.error("An error occurred during processing.")
            st.code(traceback.format_exc())
            st.stop()

    # --- DISPLAY RESULTS ----------------------------------------------------
    st.success("Processing complete!")

    if annotated_path and os.path.exists(annotated_path):
        if file_ext in [".jpg", ".jpeg", ".png"]:
            st.subheader("🖼 Annotated Image")
            # --- THIS IS THE FIXED LINE ---
            st.image(annotated_path, use_container_width=True)
        
        elif file_ext == ".mp4":
            st.subheader("📹 Annotated Video")
            
            with st.spinner("Converting video for browser playback..."):
                web_video_path = convert_to_web_format(annotated_path)

            if web_video_path:
                with open(web_video_path, "rb") as vf:
                    st.video(vf.read())
            else:
                st.warning("Could not convert video. Attempting to play original file (may fail).")
                with open(annotated_path, "rb") as vf:
                    st.video(vf.read())
    else:
        st.warning("Annotated output file not found.")

    if df is not None and not df.empty:
        st.subheader("📊 Detection Results")
        st.dataframe(df)

        summary = df.groupby("Damage Type").agg(
            Count=("Damage Type", "count"),
            Avg_Confidence=("Confidence", "mean")
        ).reset_index()
        st.subheader("📋 Summary by Damage Type")
        st.table(summary)
    else:
        st.info("✅ No road damage was detected in the provided file.")