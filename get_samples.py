import pandas as pd

# Load cleaned dataset
df = pd.read_csv("cleaned_news.csv")

# Extract 2 fake and 2 real articles
fake = df[df["label"] == 0].head(2)
real = df[df["label"] == 1].head(2)

# Combine samples
samples = pd.concat([fake, real])

# Keep only necessary columns
samples = samples[["title", "content", "label"]].copy()

# Add readable labels
samples["actual_category"] = samples["label"].map({
    0: "Fake News",
    1: "Real News"
})

# Save samples
samples.to_csv(
    "truthlens_test_samples.csv",
    index=False
)

print("\nSAMPLES EXTRACTED SUCCESSFULLY!")
print("Total samples:", len(samples))
print("\nCategory distribution:")
print(samples["actual_category"].value_counts())
print("\nSaved: truthlens_test_samples.csv")