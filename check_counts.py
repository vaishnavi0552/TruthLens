import pandas as pd
import re

fake = pd.read_csv("dataset/Fake.csv")
real = pd.read_csv("dataset/True.csv")

# Assign labels
fake["label"] = 0
real["label"] = 1

data = pd.concat(
    [fake, real],
    ignore_index=True
)

print("ORIGINAL RECORDS:", len(data))

# Check empty titles and text
data["title"] = data["title"].fillna("").astype(str).str.strip()
data["text"] = data["text"].fillna("").astype(str).str.strip()

print("Empty titles:", (data["title"] == "").sum())
print("Empty text:", (data["text"] == "").sum())

data = data[
    (data["title"] != "") &
    (data["text"] != "")
].copy()

# Create normalized duplicate key
def normalize(text):
    return re.sub(
        r"\s+", " ", str(text).lower()
    ).strip()

data["duplicate_key"] = (
    data["title"].apply(normalize)
    + " "
    + data["text"].apply(normalize)
)

# Check duplicate removal
print("\nRECORDS BEFORE DEDUPLICATION:", len(data))

data = data.drop_duplicates(
    subset=["duplicate_key"],
    keep="first"
)

print("RECORDS AFTER DEDUPLICATION:", len(data))

print("\nLABEL COUNTS:")
print(data["label"].value_counts().sort_index())

# Check existing cleaned file
cleaned = pd.read_csv("cleaned_news.csv")

print("\nEXISTING CLEANED FILE:", len(cleaned))

print("\nEXPECTED UNIQUE RECORDS:", len(data))

print("\nCOUNT CHECK COMPLETED!")