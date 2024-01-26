from flask import Flask, render_template, request
from ctransformers import AutoModelForCausalLM, AutoConfig
import time
import whisper
import pyaudio
import wave
import os

app = Flask(__name__, template_folder="ui")

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

def record_audio():
    # Record audio
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    RECORD_SECONDS = 5
    WAVE_OUTPUT_FILENAME = "output.wav"

    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

    print("* recording")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* done recording")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save audio to file
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

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


