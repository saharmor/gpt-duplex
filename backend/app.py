import argparse
import time

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

import whisper

app = Flask(__name__)
CORS(app)
whisper_model = None
chatgpt_client = None
global is_first_prompt
is_first_prompt = True

parser = argparse.ArgumentParser(description = "Duplex")
parser.add_argument("--port", type=int, default=8000, help = "backend port")
args = parser.parse_args()

chatgpt_initial_prompt = "You are a conversation bot that helps in booking reservations for restaurants via phone calls. Your answers are concise and to the point. Your answers should be less than 30 words. You also never share phone numbers unless someone asks you about your phone number.\nI want to book a place for two between 6pm-7.30pm today. My phone number is 4153775858 but you should never share it unless you have been explicitly asked for it by the restaurant.\nLet's start."
chatgpt_prompt_prefix = 'Restaurant:'


import time

def chat_gpt(prompt):
  response = chatgpt_client.send_message(prompt)
  return response

def transcribe_whisper(filename: str):
  return whisper_model.transcribe(filename)["text"]

def construct_prompt(audio_transcript: str):
  global is_first_prompt
  if is_first_prompt:
    is_first_prompt = False
    return "{} {} {}".format(chatgpt_initial_prompt, chatgpt_prompt_prefix, audio_transcript)
  
  return "{}{}".format(chatgpt_prompt_prefix, audio_transcript)


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


    prompt = construct_prompt(transcribed_text)
    print("Sending prompt [{}]".format(prompt))
    start = time.time()
    chat_output = chat_gpt(prompt)
    end = time.time()
    print('--> ChatGPT execution time {}'.format(end - start))

    print(chat_output['message'])
    return chat_output['message']


@app.route("/", methods=["GET"])
@cross_origin()
def health_check():
    return jsonify(success=True)


with app.app_context():
  whisper_model = whisper.load_model("small.en")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=args.port, debug=False)
