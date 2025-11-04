import whisper
import os
import uuid
import torch
from flask import Flask, request, render_template
from pydub import AudioSegment
import threading
import atexit

app = Flask(__name__, template_folder="templates")

MODEL_NAME = "medium"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FP16 = DEVICE == "cuda"

model = None
model_loading = False

def load_model():
    """Load the Whisper model in the background."""
    global model, model_loading
    model_loading = True
    try:
        print("Loading Whisper model...")
        model = whisper.load_model(MODEL_NAME, device=DEVICE)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
    finally:
        model_loading = False

def cleanup():
    """Clear GPU memory when the app exits."""
    global model
    if model is not None:
        model = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU memory cleared.")

threading.Thread(target=load_model, daemon=True).start()

atexit.register(cleanup)

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    if request.method == "POST":
        try:
            
            if model_loading:
                return render_template("index.html", result="Model is still loading, please try again shortly.")
            if model is None:
                return render_template("index.html", result="Model failed to load. Please contact support.")

            file = request.files["audio"]

            
            if file.content_length and file.content_length > 10 * 1024 * 1024:
                return render_template("index.html", result="Audio file too large (max 10MB allowed).")

            
            unique_name = f"{uuid.uuid4()}.wav"
            audio = AudioSegment.from_file(file)
            audio = audio.set_frame_rate(16000).set_channels(1)
            audio.export(unique_name, format="wav")

            
            result = model.transcribe(unique_name, fp16=FP16)["text"]


        except Exception as e:
            result = f"An error occurred during transcription: {str(e)}"

        finally:
            if os.path.exists(unique_name):
                os.remove(unique_name)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
