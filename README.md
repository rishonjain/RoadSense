# 🚧 RoadSense

RoadSense is a road damage detection system using YOLOv8 and Streamlit.  
It detects damages, classifies severity, and generates automatic PDF reports.

## 🚀 Run Locally

```bash
# Create venv
python -m venv venv
venv\Scripts\activate   # On Windows
source venv/bin/activate  # On Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py