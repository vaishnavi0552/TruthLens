import os
import re
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# =====================================
# 1. LOAD DATASET
# =====================================

print("\nFAKE NEWS DETECTION - MODEL TRAINING")

data = pd.read_csv("cleaned_news.csv")

print("Original dataset:", data.shape)

# Validate required columns
required = ["clean_text", "label"]

if not all(col in data.columns for col in required):
    raise ValueError("Required columns are missing!")

data = data[required].copy()

data["clean_text"] = (
    data["clean_text"].fillna("").astype(str).str.strip()
)

data["label"] = pd.to_numeric(
    data["label"], errors="coerce"
)

data = data.dropna(subset=["label"])
data = data[data["clean_text"] != ""].copy()

# Labels must be 0 and 1
data = data[data["label"].isin([0, 1])].copy()
data["label"] = data["label"].astype(int)

# =====================================
# 2. REMOVE DUPLICATES AND CONFLICTS
# =====================================

def normalize(text):
    return re.sub(r"\s+", " ", text.lower()).strip()

data["duplicate_key"] = data["clean_text"].apply(normalize)

# Identify identical text having conflicting labels
label_counts = data.groupby("duplicate_key")["label"].nunique()

conflicting_keys = label_counts[label_counts > 1].index

print("Conflicting duplicate groups:", len(conflicting_keys))

# Exclude ambiguous records
data = data[
    ~data["duplicate_key"].isin(conflicting_keys)
].copy()

# Remove repeated articles
data = data.drop_duplicates(
    subset=["duplicate_key"],
    keep="first"
)

data = data.drop(columns=["duplicate_key"])

print("Final dataset:", data.shape)
print("\nLabel distribution:")
print(data["label"].value_counts())

# =====================================
# 3. SPLIT DATASET
# =====================================

X = data["clean_text"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTRAINING RECORDS:", len(X_train))
print("TESTING RECORDS:", len(X_test))

# =====================================
# 4. DEFINE ML MODELS
# =====================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Naive Bayes": MultinomialNB(),

    "Linear SVM": LinearSVC(
        random_state=42
    )
}

os.makedirs("models", exist_ok=True)

results = []
best_f1 = -1
best_model = None
best_name = None
best_predictions = None

# =====================================
# 5. TRAIN AND EVALUATE
# =====================================

for name, classifier in models.items():

    print("\n" + "=" * 45)
    print("TRAINING:", name)

    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=50000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True
            )
        ),
        ("classifier", classifier)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["Fake News", "Real News"],
        zero_division=0
    ))

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    # Select model based on F1 score
    if f1 > best_f1:
        best_f1 = f1
        best_model = pipeline
        best_name = name
        best_predictions = predictions

# =====================================
# 6. MODEL COMPARISON
# =====================================

results_df = pd.DataFrame(results)

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))

results_df.to_csv(
    "models/model_comparison.csv",
    index=False
)

# =====================================
# 7. SAVE BEST MODEL
# =====================================

joblib.dump(
    best_model,
    "models/fake_news_model.joblib"
)

with open("models/model_info.json", "w") as f:
    json.dump({
        "best_model": best_name,
        "best_f1_score": best_f1,
        "label_mapping": {
            "0": "Fake News",
            "1": "Real News"
        }
    }, f, indent=4)

print("\nBEST MODEL:", best_name)
print("BEST F1 SCORE:", round(best_f1, 4))
print("MODEL SAVED SUCCESSFULLY!")

# =====================================
# 8. CONFUSION MATRIX
# =====================================

cm = confusion_matrix(
    y_test,
    best_predictions,
    labels=[0, 1]
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Fake News", "Real News"]
)

display.plot(cmap="Blues")
plt.title("Fake News Detection - Confusion Matrix")
plt.tight_layout()
plt.savefig("models/confusion_matrix.png", dpi=300)
plt.show()

print("\nMODEL TRAINING COMPLETED SUCCESSFULLY!")