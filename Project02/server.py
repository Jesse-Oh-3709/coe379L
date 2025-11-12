from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io, json, os

app = Flask(__name__)

MODEL_PATH = os.path.join("artifacts", "best_model.keras")
SUMMARY_PATH = os.path.join("artifacts", "model_summary.json")

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
with open(SUMMARY_PATH, "r") as f:
    model_summary = json.load(f)

IMG_SIZE = (128, 128)
CLASS_NAMES = ["no_damage", "damage"]

@app.route("/summary", methods=["GET"])
def summary():
    """Return model metadata as JSON"""
    return jsonify(model_summary), 200

@app.route("/inference", methods=["POST"])
def inference():
    """Accept raw image bytes and return prediction JSON"""
    if not request.data:
        return jsonify({"error": "No image provided"}), 400

    try:
        image = Image.open(io.BytesIO(request.data)).convert("RGB")
        image = image.resize(IMG_SIZE)
        x = np.expand_dims(np.array(image) / 255.0, axis=0)
        preds = model.predict(x)
        label = CLASS_NAMES[int(preds[0][0] > 0.5)]
        return jsonify({"prediction": label}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)


