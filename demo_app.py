#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import librosa
import numpy as np
from keras.models import load_model
from werkzeug.exceptions import BadRequest, HTTPException, NotFound
from werkzeug.serving import run_simple
from werkzeug.utils import secure_filename
from werkzeug.wrappers import Request, Response

ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "models" / "cnn.keras"
INDEX_PATH = ROOT_DIR / "demo" / "index.html"
RECORDINGS_DIR = ROOT_DIR / "recordings"
LABELS = [
    "angry",
    "calm",
    "disgust",
    "fearful",
    "happy",
    "neutral",
    "sad",
    "surprised",
]

RECORDINGS_DIR.mkdir(exist_ok=True)
(ROOT_DIR / ".mplconfig").mkdir(exist_ok=True)

MODEL = load_model(MODEL_PATH)


def extract_feature(data: np.ndarray, sr: int, mfcc: bool, chroma: bool, mel: bool) -> np.ndarray:
    if chroma:
        stft = np.abs(librosa.stft(data))

    result = np.array([], dtype=np.float32)

    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))

    if chroma:
        chroma_values = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
        result = np.hstack((result, chroma_values))

    if mel:
        mel_values = np.mean(librosa.feature.melspectrogram(y=data, sr=sr).T, axis=0)
        result = np.hstack((result, mel_values))

    return result.astype(np.float32)


def analyze_audio(audio_path: Path) -> dict[str, object]:
    data, sr = librosa.load(audio_path, sr=None, mono=True)
    if data.size == 0:
        raise BadRequest("Le fichier audio est vide.")

    features = extract_feature(data, sr, mfcc=True, chroma=True, mel=True)
    batch = np.expand_dims(np.expand_dims(features, axis=0), axis=2)
    prediction = MODEL.predict(batch, verbose=0)[0]
    best_index = int(np.argmax(prediction))
    scores = [
        {"label": label, "score": round(float(score), 4)}
        for label, score in sorted(zip(LABELS, prediction), key=lambda item: item[1], reverse=True)
    ]

    return {
        "label": LABELS[best_index],
        "confidence": round(float(prediction[best_index]), 4),
        "scores": scores,
        "sample_rate": int(sr),
        "duration_seconds": round(float(len(data) / sr), 2),
    }


def json_response(payload: dict[str, object], status: int = 200) -> Response:
    return Response(
        json.dumps(payload, ensure_ascii=False),
        status=status,
        content_type="application/json; charset=utf-8",
    )


@Request.application
def app(request: Request) -> Response:
    try:
        if request.method == "GET" and request.path == "/":
            return Response(INDEX_PATH.read_text(encoding="utf-8"), content_type="text/html; charset=utf-8")

        if request.method == "GET" and request.path == "/health":
            return json_response({"status": "ok", "model": str(MODEL_PATH.name)})

        if request.method == "POST" and request.path == "/api/analyze":
            audio = request.files.get("audio")
            if audio is None or not audio.filename:
                raise BadRequest("Aucun fichier audio n'a été envoyé.")

            safe_name = secure_filename(audio.filename) or "recording.wav"
            suffix = Path(safe_name).suffix.lower() or ".wav"
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            saved_path = RECORDINGS_DIR / f"{timestamp}_{Path(safe_name).stem}{suffix}"
            audio.save(saved_path)

            result = analyze_audio(saved_path)
            result["saved_file"] = str(saved_path.relative_to(ROOT_DIR))
            return json_response(result)

        raise NotFound()
    except BadRequest as exc:
        return json_response({"error": exc.description}, status=400)
    except HTTPException as exc:
        return json_response({"error": exc.description}, status=exc.code or 500)
    except Exception as exc:
        return json_response({"error": str(exc)}, status=500)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live speech emotion recognition demo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    run_simple(args.host, args.port, app, use_reloader=False, threaded=True)


if __name__ == "__main__":
    main()
