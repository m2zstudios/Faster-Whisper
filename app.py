from flask import Flask, request, jsonify
import tempfile
import os
from faster_whisper import WhisperModel

app = Flask(__name__)

# Load model once
model = WhisperModel("small", device="cpu")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    audio = request.files["file"]
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        audio.save(tmp.name)
        segments, info = model.transcribe(tmp.name)
        text = " ".join([seg.text for seg in segments])
        os.unlink(tmp.name)

    return jsonify({
        "language": info.language,
        "text": text
    })

@app.route("/", methods=["GET"])
def home():
    return "Whisper API is running 🧠"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
