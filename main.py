import cv2
import time
from detection import run_detection
from severity import add_severity_labels
from report import create_severity_chart, generate_pdf_report

# -------- CONFIG --------
MODEL_PATH = "best.pt"
IMG_PATH = "assets/road.jpg"
ANNOTATED_IMG_PATH = "assets/annotated_result.jpg"
CHART_PATH = "assets/severity_chart.png"
PDF_OUTPUT = "road_damage_report.pdf"

# -------- RUN DETECTION --------
start_time = time.time()
df, annotated_img = run_detection(MODEL_PATH, IMG_PATH)
processing_time = round(time.time() - start_time, 3)

# Save annotated image
cv2.imwrite(ANNOTATED_IMG_PATH, annotated_img)

# -------- ADD SEVERITY --------
df = add_severity_labels(df)

# -------- CREATE CHART --------
create_severity_chart(df, CHART_PATH)

# -------- GENERATE REPORT --------
generate_pdf_report(PDF_OUTPUT, processing_time, df, ANNOTATED_IMG_PATH, CHART_PATH)

print(f"✅ Report generated: {PDF_OUTPUT}")
