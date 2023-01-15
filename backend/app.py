import argparse
import time

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

import whisper

import time

app = Flask(__name__)
CORS(app)
whisper_model = None
global is_first_prompt
is_first_prompt = True

parser = argparse.ArgumentParser(description = "Duplex")
parser.add_argument("--port", type=int, default=8000, help = "backend port")
args = parser.parse_args()


def transcribe_whisper(filename: str):
  return whisper_model.transcribe(filename)["text"]

@app.route("/transcribe", methods=["POST"])
@cross_origin()
def receive_audio():
    print(f"Receiving new audio chunk")
    request.files['audio_file'].save('audio.wav')
    
    start = time.time()
    transcribed_text = transcribe_whisper("audio.wav")
    end = time.time()
    print('--> Whisper execution time {}'.format(end - start))

    return transcribed_text


@app.route("/", methods=["GET"])
@cross_origin()
def health_check():
    return jsonify(success=True)


with app.app_context():
  whisper_model = whisper.load_model("small.en")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=args.port, debug=False)
