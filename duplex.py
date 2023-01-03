import io
import pyaudio
import math
import struct
import wave
import time
import os
from google.cloud import texttospeech
from pydub import AudioSegment
from pydub.playback import play
import requests

from gpt_client import GPTClient

Threshold = 100

SHORT_NORMALIZE = (1.0/32768.0)
chunk = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
swidth = 2

TIMEOUT_LENGTH = 2

f_name_directory = r'recordings'
duplex_backend_url = "{}/transcribe".format("https://stretch-mediawiki-thousands-clubs.trycloudflare.com")

gpt_client = GPTClient('restaurant')

class Recorder:

    @staticmethod
    def rms(frame):
        count = len(frame) / swidth
        format = "%dh" % (count)
        shorts = struct.unpack(format, frame)

        sum_squares = 0.0
        for sample in shorts:
            n = sample * SHORT_NORMALIZE
            sum_squares += n * n
        rms = math.pow(sum_squares / count, 0.5)

        return rms * 1000

    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=FORMAT,
                                  channels=CHANNELS,
                                  rate=RATE,
                                  input=True,
                                  output=True,
                                  frames_per_buffer=chunk)

    def record(self):
        print('Noise detected, recording beginning')
        rec = []
        current = time.time()
        end = time.time() + TIMEOUT_LENGTH

        while current <= end:
            data = self.stream.read(chunk, exception_on_overflow=False)
            if self.rms(data) >= Threshold:
                end = time.time() + TIMEOUT_LENGTH

            current = time.time()
            rec.append(data)

        print('Finished recording, sending to processing')

        n_files = len(os.listdir(f_name_directory))
        # output_filename = os.path.join(f_name_directory, '{}.wav'.format(34))
        output_filename = os.path.join(f_name_directory, '{}.wav'.format(n_files))
        self.write(b''.join(rec), output_filename)
        restaurant_reply = send_recording(output_filename).text
        print('--> Restaurant transcription [{}]'.format(restaurant_reply))
        bot_reply = gpt_client.get_bot_reply(restaurant_reply)
        print('--> GPT generated reply [{}]'.format(bot_reply))
        tts(bot_reply)

    def write(self, recording, output_filename):
        wf = wave.open(output_filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(recording)
        wf.close()

    def listen(self):
        print('Listening beginning')
        while True:
            input = self.stream.read(chunk, exception_on_overflow=False)
            rms_val = self.rms(input)
            if rms_val > Threshold:
                self.record()
                print('Returning to listening')


def send_recording(file_name: str):
    audio_files = {'audio_file': open(file_name, 'rb')}
    return requests.post(duplex_backend_url, files=audio_files)


def tts(text: str):
    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=text)

    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3, speaking_rate=1.2
    )
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    play_mp3(response.audio_content)


def play_mp3(audio_bytes):
    voiced_text = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
    play(voiced_text)



a = Recorder()

a.listen()