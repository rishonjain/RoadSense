import streamlit as st
import cv2
import time
import tempfile
import os
from detection import run_detection
from severity import add_severity_labels
from report import create_severity_chart, generate_pdf_report

# -------- CONFIG --------
MODEL_PATH = "best.pt"  # Path to YOLOv8 trained model

st.set_page_config(page_title="Road Damage Detection", layout="wide")
st.title("🚧 Road Damage Detection System")
st.write("Upload a road image to detect damages, classify severity, and generate a detailed PDF report.")

# -------- FILE UPLOAD --------
uploaded_file = st.file_uploader("Upload a road image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    temp_dir = tempfile.mkdtemp()
    img_path = os.path.join(temp_dir, uploaded_file.name)
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(img_path, caption="Uploaded Image", use_column_width=True)

    if st.button("Run Detection"):
        with st.spinner("Detecting damages..."):
            start_time = time.time()
            df, annotated_img = run_detection(MODEL_PATH, img_path)
            processing_time = round(time.time() - start_time, 3)

            # Save annotated image
            annotated_img_path = os.path.join(temp_dir, "annotated_result.jpg")
            cv2.imwrite(annotated_img_path, annotated_img)

            # Add severity labels
            df = add_severity_labels(df)

            # Create severity chart
            chart_path = os.path.join(temp_dir, "severity_chart.png")
            create_severity_chart(df, chart_path)

            # Generate PDF
            pdf_path = os.path.join(temp_dir, "road_damage_report.pdf")
            generate_pdf_report(pdf_path, processing_time, df, annotated_img_path, chart_path)

        # -------- SHOW RESULTS --------
        st.subheader("📌 Annotated Detection Result")
        st.image(annotated_img_path, caption="Detections", use_column_width=True)

        st.subheader("📊 Detected Damages")
        st.dataframe(df[["class_name", "confidence", "severity"]])

        st.subheader("📈 Severity Distribution")
        st.image(chart_path, caption="Severity Chart", use_column_width=False)

        # -------- DOWNLOAD REPORT --------
        with open(pdf_path, "rb") as pdf_file:
            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_file,
                file_name="road_damage_report.pdf",
                mime="application/pdf"
            )
