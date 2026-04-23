
from pathlib import Path

import joblib
from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "fake_news_pipeline.joblib"

app = Flask(__name__)
_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run: python train_model.py"
            )
        _pipeline = joblib.load(MODEL_PATH)
    return _pipeline


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Please provide non-empty text."}), 400
    if len(text) > 20000:
        return jsonify({"error": "Text is too long (max 20000 characters)."}), 400

    try:
        pipe = get_pipeline()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503

    proba = pipe.predict_proba([text])[0]
    # classes_ order: index 0 = first class label sorted — sklearn uses sorted unique y
    classes = list(pipe.classes_)
    fake_idx = classes.index(0) if 0 in classes else 0
    real_idx = classes.index(1) if 1 in classes else 1
    p_fake = float(proba[fake_idx])
    p_real = float(proba[real_idx])
    pred = int(pipe.predict([text])[0])
    label = "Likely real news" if pred == 1 else "Likely fake news"

    return jsonify(
        {
            "prediction": pred,
            "label": label,
            "probability_fake": round(p_fake, 4),
            "probability_real": round(p_real, 4),
        }
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
