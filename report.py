from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

def generate_report(df, severity_results, filename):
    c = canvas.Canvas(filename, pagesize=A4)
    c.setFont("Helvetica", 12)
    c.drawString(50, 800, "🚧 Road Damage Detection Report")
    c.line(50, 795, 550, 795)

    c.drawString(50, 770, "Detected Damages:")
    y = 750
    for i, row in df.iterrows():
        c.drawString(50, y, f"Damage {i+1} - Class: {row['Class']} | Confidence: {row['Confidence']:.2f}")
        y -= 20

    c.drawString(50, y-20, "Severity Classification:")
    y -= 40
    for damage, severity in severity_results.items():
        c.drawString(50, y, f"{damage}: {severity}")
        y -= 20

    c.save()
