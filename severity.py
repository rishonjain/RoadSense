def classify_severity(confidence, bbox_area):
    if bbox_area > 50000 or confidence > 0.85:
        return "Severe"
    elif bbox_area > 20000 or confidence > 0.65:
        return "Moderate"
    else:
        return "Minor"

def add_severity_labels(df):
    df["severity"] = df.apply(lambda row: classify_severity(row["confidence"], row["bbox_area"]), axis=1)
    return df
