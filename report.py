import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet

def create_severity_chart(df, chart_path):
    severity_counts = df["severity"].value_counts()
    plt.figure(figsize=(4,4))
    severity_counts.plot(kind="bar", color=["green", "orange", "red"])
    plt.title("Damage Severity Distribution")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(chart_path)
    plt.close()

def generate_pdf_report(output_path, processing_time, df, annotated_img_path, chart_path):
    styles = getSampleStyleSheet()
    report = SimpleDocTemplate(output_path, pagesize=A4)
    elements = []

    elements.append(Paragraph("<b>Road Damage Detection Report</b>", styles['Title']))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Processed in {processing_time} seconds. Total damages detected: {len(df)}.", styles['Normal']))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("<b>Annotated Detection Image:</b>", styles['Heading2']))
    elements.append(RLImage(annotated_img_path, width=400, height=300))
    elements.append(Spacer(1, 12))

    table_data = [["Class", "Confidence", "Severity"]]
    for _, row in df.iterrows():
        table_data.append([row["class_name"], f"{row['confidence']:.2f}", row["severity"]])

    table = Table(table_data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('BACKGROUND', (0,1), (-1,-1), colors.beige),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("<b>Severity Distribution Chart:</b>", styles['Heading2']))
    elements.append(RLImage(chart_path, width=300, height=300))

    report.build(elements)
