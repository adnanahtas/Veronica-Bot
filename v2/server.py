from flask import Flask, render_template, request, jsonify
from ctransformers import AutoModelForCausalLM, AutoConfig
import time
import whisper
import pyaudio
import wave
import os

app = Flask(__name__, template_folder="ui")

audio_file = 'audio.wav'
audio = pyaudio.PyAudio()
stream = None
recording = False

@app.route('/toggle-recording', methods=['POST'])
def toggle_recording():
    global stream, recording
    if not recording:
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100

        stream = audio.open(format=FORMAT, channels=CHANNELS,
                            rate=RATE, input=True,
                            frames_per_buffer=CHUNK)
        recording = True
    else:
        data = stream.read(1024)
        stream.stop_stream()
        stream.close()
        stream = None
        recording = False
        wf = wave.open(audio_file, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(44100)
        wf.writeframes(b''.join([data]))
        wf.close()
        model = whisper.load_model("base")
        transcription = whisper.transcribe(model, 'audio.wav')
        print(transcription["text"])

    return jsonify({'success': True})


def veronica_prompt(inp_role, inpt_prompt):
    t = time.time()
    role = inp_role
    question = inpt_prompt

    config = AutoConfig.from_pretrained("TheBloke/Mistral-7B-OpenOrca-GGUF")
    config.config.gpu_layers = 50
    config.config.mlock = True
    config.config.context_length = 4096
    config.config.batch_size = 16
    config.config.max_new_tokens = 8192
    llm = AutoModelForCausalLM.from_pretrained("TheBloke/Mistral-7B-Instruct-v0.1-GGUF", model_file="mistral-7b-instruct-v0.1.Q5_K_M.gguf", model_type="mistral", config=config)

    prompt = f"<s>[INST]You are a {role} named Veronica. You will respond in a way that is most natural and relevant for a {role}, while making sure to follow any specific instructions given to you[/INST]Sure, I will now respond to the conversation as if I am a {role} and my answers will be context relevant and related to my role of being a {role}</s>[INST]{question}[/INST]"
    response = llm(prompt)
    print(response)
    t = time.time() - t
    print("Time = ", t)
    return response


def transcribe():
    pass

@app.route('/home', methods=['GET', 'POST'])
def index():
    return render_template('verhome.html')
    
@app.route('/about', methods=['GET'])
def index_about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)


