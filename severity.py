def classify_severity(df):
    severity_map = {}
    for i, row in df.iterrows():
        if row["Confidence"] > 0.8:
            severity = "High"
        elif row["Confidence"] > 0.5:
            severity = "Medium"
        else:
            severity = "Low"
        severity_map[f"Damage {i+1}"] = severity
    return severity_map
