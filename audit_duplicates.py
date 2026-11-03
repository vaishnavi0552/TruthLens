import pandas as pd
import re

# Load original datasets
fake = pd.read_csv("dataset/Fake.csv")
real = pd.read_csv("dataset/True.csv")

# Normalize article text
def normalize(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def create_keys(df):
    title = df["title"].fillna("").apply(normalize)
    body = df["text"].fillna("").apply(normalize)

    return title + " " + body

fake_keys = create_keys(fake)
real_keys = create_keys(real)

# Find matching articles across classes
fake_set = set(fake_keys)
real_set = set(real_keys)

common_keys = fake_set.intersection(real_set)

fake_conflicts = fake_keys.isin(common_keys).sum()
real_conflicts = real_keys.isin(common_keys).sum()

print("\nDUPLICATE AUDIT RESULTS")
print("-----------------------")

print("Original fake records:", len(fake))
print("Original real records:", len(real))

print("Unique fake articles:", fake_keys.nunique())
print("Unique real articles:", real_keys.nunique())

print("Common article keys:", len(common_keys))

print("Fake conflicting records:", fake_conflicts)
print("Real conflicting records:", real_conflicts)

print("\nSAMPLE MATCHING ARTICLES")

for key in list(common_keys)[:5]:
    print("\n", key[:200])

print("\nDUPLICATE AUDIT COMPLETED!")