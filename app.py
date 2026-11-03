import pandas as pd
from flask import Flask, request, jsonify, render_template
from pathlib import Path
import joblib
import re
import html
import logging

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "fake_news_model.joblib"

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

logging.basicConfig(level=logging.INFO)

# Load trained model
try:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)
    print("TRUTHLENS: MODEL LOADED SUCCESSFULLY!")

except Exception:
    app.logger.exception("Model loading failed")
    raise


# Text cleaning: keep this consistent with model training
def clean_text(text):
    text = html.unescape(str(text))
    text = text.lower()

    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


@app.route("/")
def home():
    return render_template("index.html")


# Simple backend health check
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "online",
        "model_loaded": model is not None,
        "application": "TruthLens"
    })


# News prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({
                "success": False,
                "error": "Invalid request. Please submit JSON data."
            }), 400

        news = data.get("news", "")

        if not isinstance(news, str) or not news.strip():
            return jsonify({
                "success": False,
                "error": "Please enter a news article."
            }), 400

        if len(news) > 20000:
            return jsonify({
                "success": False,
                "error": "Article exceeds the 20,000 character limit."
            }), 400

        cleaned_news = clean_text(news)

        if len(cleaned_news.split()) < 5:
            return jsonify({
                "success": False,
                "error": "Please enter at least five meaningful words."
            }), 400

        # Predict using the saved model
        raw_prediction = model.predict([cleaned_news])[0]
        prediction = int(raw_prediction)

        if prediction == 0:
            result = "Fake News"
            message = "The model classified this article as Fake News."
        elif prediction == 1:
            result = "Real News"
            message = "The model classified this article as Real News."
        else:
            return jsonify({
                "success": False,
                "error": "Unexpected prediction label from the model."
            }), 500

        app.logger.info("News prediction completed: %s", result)

        return jsonify({
            "success": True,
            "prediction": result,
            "label": prediction,
            "message": message,
            "model": "Linear Support Vector Machine",
            "disclaimer": (
                "This is an ML-based prediction, not independent "
                "verification of the news."
            )
        })

    except Exception:
        app.logger.exception("News prediction failed")

        return jsonify({
            "success": False,
            "error": "Prediction failed. Check the Flask terminal for details."
        }), 500


# ==========================================
# BULK CSV NEWS PREDICTION API
# ==========================================

@app.route("/predict_csv", methods=["POST"])
def predict_csv():

    try:
        if "file" not in request.files:
            return jsonify({"error": "Please upload a CSV file."}), 400

        file = request.files["file"]

        if not file.filename.lower().endswith(".csv"):
            return jsonify({"error": "Only CSV files are supported."}), 400

        df = pd.read_csv(file)

        if len(df) > 500:
            return jsonify({
                "error": "Maximum 500 articles allowed per upload."
            }), 400

        # Identify the article column
        if "content" in df.columns:
            articles = df["content"].fillna("").astype(str)

        elif "clean_text" in df.columns:
            articles = df["clean_text"].fillna("").astype(str)

        elif "text" in df.columns:
            articles = df["text"].fillna("").astype(str)

            if "title" in df.columns:
                articles = (
                    df["title"].fillna("").astype(str)
                    + " "
                    + articles
                )

        else:
            return jsonify({
                "error": "CSV must contain content, clean_text, or text column."
            }), 400

        # Clean articles using the existing preprocessing function
        cleaned = [clean_text(article) for article in articles]

        valid_indices = [
            i for i, article in enumerate(cleaned)
            if len(article.split()) >= 5
        ]

        if not valid_indices:
            return jsonify({
                "error": "No valid news articles found."
            }), 400

        valid_articles = [cleaned[i] for i in valid_indices]

        # Predict all articles together
        predictions = model.predict(valid_articles)

        results = []

        for index, prediction in zip(valid_indices, predictions):

            category = (
                "Fake News" if int(prediction) == 0
                else "Real News"
            )

            title = (
                str(df.iloc[index]["title"])
                if "title" in df.columns else
                f"Article {index + 1}"
            )

            results.append({
                "title": title,
                "prediction": category
            })

        fake_count = sum(
            r["prediction"] == "Fake News" for r in results
        )

        real_count = sum(
            r["prediction"] == "Real News" for r in results
        )

        return jsonify({
            "success": True,
            "total": len(results),
            "fake_count": fake_count,
            "real_count": real_count,
            "skipped": len(df) - len(results),
            "results": results
        })

    except Exception:
        app.logger.exception("CSV prediction failed")
        return jsonify({
            "error": "Unable to analyze the uploaded CSV."
        }), 500

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        use_reloader=False
    )
