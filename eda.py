# ==========================================
# FAKE NEWS DETECTION - EXPLORATORY DATA ANALYSIS
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("PROJECT: FAKE NEWS DETECTION USING NLP")

# STEP 1: Load datasets

fake = pd.read_csv("dataset/Fake.csv")
real = pd.read_csv("dataset/True.csv")

print("\n1. DATASET LOADED SUCCESSFULLY")

# STEP 2: Dataset dimensions

print("\n2. DATASET DIMENSIONS")

print("Fake News:", fake.shape)
print("Real News:", real.shape)

print("Total Articles:", len(fake) + len(real))

# STEP 3: Column information

print("\n3. COLUMN INFORMATION")

print("Columns:", list(fake.columns))

print("\nData Types:")
print(fake.dtypes)

# STEP 4: Missing values

print("\n4. MISSING VALUES")

print("Fake News:")
print(fake.isnull().sum())

print("\nReal News:")
print(real.isnull().sum())

# STEP 5: Duplicate records

print("\n5. DUPLICATE RECORDS")

print("Fake Duplicates:", fake.duplicated().sum())

print("Real Duplicates:", real.duplicated().sum())

# STEP 6: Assign labels

fake["label"] = 0
real["label"] = 1

# STEP 7: Combine datasets

data = pd.concat(
    [fake, real],
    ignore_index=True
)

print("\n6. COMBINED DATASET")

print(data.head())

print("\nTotal Records:", len(data))

# STEP 8: Class distribution

print("\n7. CLASS DISTRIBUTION")

print(data["label"].value_counts())

# STEP 9: Class distribution graph

plt.figure(figsize=(8, 5))

sns.countplot(
    x="label",
    data=data
)

plt.title("Fake News vs Real News Distribution")

plt.xlabel("News Category")
plt.ylabel("Number of Articles")

plt.xticks(
    [0, 1],
    ["Fake News", "Real News"]
)

plt.tight_layout()

plt.savefig("news_distribution.png")

plt.show()

# STEP 10: Save combined dataset

data.to_csv(
    "combined_news.csv",
    index=False
)

print("\nEDA COMPLETED SUCCESSFULLY!")

print("Graph saved as news_distribution.png")
print("Dataset saved as combined_news.csv")