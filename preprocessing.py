# ==========================================
# FAKE NEWS DETECTION - DATA PREPROCESSING
# ==========================================

import pandas as pd
import re
import html

print("STARTING DATA PREPROCESSING")

# STEP 1: Load combined dataset

data = pd.read_csv("combined_news.csv")

print("\nOriginal dataset:", data.shape)

# STEP 2: Check and remove empty records

data["title"] = data["title"].fillna("").astype(str)
data["text"] = data["text"].fillna("").astype(str)

data["title"] = data["title"].str.strip()
data["text"] = data["text"].str.strip()

data = data[
    (data["title"] != "") &
    (data["text"] != "")
].copy()

print("After removing empty records:", data.shape)

# STEP 3: Normalize text for duplicate checking

def normalize_for_check(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

data["duplicate_key"] = (
    data["title"].apply(normalize_for_check)
    + " "
    + data["text"].apply(normalize_for_check)
)

# STEP 4: Find conflicting labels

label_counts = data.groupby(
    "duplicate_key"
)["label"].nunique()

conflicting_keys = label_counts[
    label_counts > 1
].index

print(
    "\nArticles with conflicting labels:",
    len(conflicting_keys)
)

# Remove conflicting articles rather than
# assigning an arbitrary label.

data = data[
    ~data["duplicate_key"].isin(conflicting_keys)
].copy()

# STEP 5: Remove duplicate articles

before = len(data)

data = data.drop_duplicates(
    subset=["duplicate_key"],
    keep="first"
).copy()

print("Duplicate articles removed:", before - len(data))

# STEP 6: Combine headline and article

data["content"] = (
    data["title"] + " " + data["text"]
)

# STEP 7: Text cleaning function

def clean_text(text):

    text = html.unescape(str(text))

    text = text.lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    # Remove HTML tags
    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )

    # Keep alphabetic characters and spaces
    text = re.sub(
        r"[^a-z\s]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# STEP 8: Apply cleaning

data["clean_text"] = data["content"].apply(
    clean_text
)

# STEP 9: Remove empty cleaned articles

data = data[
    data["clean_text"].str.len() > 0
].copy()

# STEP 10: Display examples

print("\nORIGINAL NEWS:")
print(data["content"].iloc[0][:300])

print("\nCLEANED NEWS:")
print(data["clean_text"].iloc[0][:300])

# STEP 11: Display final statistics

print("\nFINAL DATASET INFORMATION")

print("Total records:", len(data))

print("\nClass distribution:")
print(data["label"].value_counts())

# STEP 12: Save processed dataset

data.drop(
    columns=["duplicate_key"],
)

data.to_csv(
    "cleaned_news.csv",
    index=False
)

print("\nPREPROCESSING COMPLETED SUCCESSFULLY!")

print("Saved file: cleaned_news.csv")