from flask import Flask, render_template, request, jsonify
from ctransformers import AutoModelForCausalLM, AutoConfig
import time
import whisper
import pyaudio
import wave
import threading
import pyttsx3

app = Flask(__name__, template_folder="ui")

audio_file = '11audio.wav'
audio = pyaudio.PyAudio()
stream = None
recording = False
frames = []
llm = ""
model = whisper.load_model("small.en")
speech = ""
micId = 1

# def veronica_prompt(inp_role, inpt_prompt):
#     t = time.time()
#     role = inp_role
#     question = inpt_prompt

#     config = AutoConfig.from_pretrained("TheBloke/Mistral-7B-Instruct-v0.2-GGUF")
#     config.config.gpu_layers = 50
#     config.config.context_length = 4096
#     config.config.batch_size = 16
#     config.config.max_new_tokens = 8192
#     llm = AutoModelForCausalLM.from_pretrained("TheBloke/Mistral-7B-Instruct-v0.2-GGUF", model_file="mistral-7b-instruct-v0.2.Q5_K_M.gguf", model_type="mistral", config=config)

#     prompt = f"<s>[INST]You are a {role} named Veronica. You will respond in a way that is most natural and relevant for a {role}, while making sure to follow any specific instructions given to you[/INST]Sure, I will now respond to the conversation as if I am a {role} and my answers will be context relevant and related to my role of being a {role}</s>[INST]{question}[/INST]"
#     response = llm(prompt)
#     print(response)
#     t = time.time() - t
#     print("Time = ", t)
#    return response

def init_model():
    config = AutoConfig.from_pretrained("TheBloke/Mistral-7B-Instruct-v0.1-GGUF")
    config.config.gpu_layers = 50
    # config.config.mlock = True
    config.config.context_length = 1024
    config.config.batch_size = 16
    config.config.max_new_tokens = 1024
    llm = AutoModelForCausalLM.from_pretrained("TheBloke/Mistral-7B-Instruct-v0.1-GGUF", model_file="mistral-7b-instruct-v0.1.Q5_K_M.gguf", model_type="mistral", config=config)
    print("model load successful!")
    return llm

def generate_response(model, role, question, uncensored = False):
    if(role == ""):
        role = "Personal Assistant"
    if(question == ""):
        question = "Introduce Yourself as a {role}"
    if(not uncensored):
        question = "Write some prose on: "+question
    prompt = f"<s>[INST]You are a {role} named Veronica. You will respond in a way that is most natural and relevant for a {role}, and limit your response length to be 500 words at most[/INST]Sure, I will now respond to the conversation as if I am a {role} and my answers will be context relevant and related to my role of being a {role}. I will also keep the answers limited to 500 words or less</s>[INST]{question}[/INST]"
    response = model(prompt)
    return response

def record_audio():
    global stream, frames, micId
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100

    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,input_device_index = int(micId),
                        frames_per_buffer=CHUNK)
    while recording:
        data = stream.read(CHUNK)
        frames.append(data)

def speak():
    global speech
    engine = pyttsx3.init()
    voice = engine.getProperty('voices')
    engine.setProperty('voice', voice[1].id)
    engine.say(speech)
    engine.runAndWait()
    del engine

voiceThread = threading.Thread(target = speak)

@app.route('/toggle-recording', methods=['POST'])
def toggle_recording():
    #global llm
    dt = time.time()
    global recording, model, speech, voiceThread
    transcription = ""
    response = ""
    if (voiceThread.is_alive()):
        del voiceThread
        print("Voisss aliev")
    voiceThread = threading.Thread(target = speak)
    tr = ""
    if not recording:
        recording = True
        audio_thread = threading.Thread(target=record_audio)
        audio_thread.start()
        print("Recording Started")
    else:
        #if (engine.isBusy):
         #   engine.stop()
        censorToggle = request.form.get("censor")
        role = request.form.get("role")
        print("Role = ", role, "Uncensored = ", censorToggle)
        global stream, frames
        recording = False
        stream.stop_stream()
        stream.close()
        stream = None
        wf = wave.open(audio_file, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(44100)
        wf.writeframes(b''.join(frames))
        wf.close()
        frames = []
        transcription = ""
        transcription = whisper.transcribe(model, '11audio.wav')
        print(transcription["text"])
        tr = transcription["text"]
        response = generate_response(llm, role, transcription["text"], censorToggle)
        dt = time.time() - dt
        print(response)
        print("Response Time = ",dt)
        speech = response
        voiceThread.start()
    return jsonify({'success': True, 'output':response,  'transcript':tr})
    
@app.route('/gen-from-text', methods=['POST'])
def generate_from_text():
    global voiceThread, speech
    if (voiceThread.is_alive()):
        del voiceThread
        print("Voisss aliev")
    voiceThread = threading.Thread(target = speak)
    dt = time.time()
    inptext = request.form.get("textprompt")
    role = request.form.get("role")
    print(inptext)
    censorToggle = request.form.get("censor")
    response = generate_response(llm, role, inptext, censorToggle)
    dt = time.time() - dt
    print(response)
    print("Response Time = ",dt)
    speech = response
    voiceThread.start()
    return jsonify({'success': True, 'output':response,  'input':inptext})

@app.route('/disp-mics', methods=['GET'])
def  get_mics():
    mic = ""
    mics = []
    info = audio.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    for i in range(0, numdevices):
        if (audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            mic = str(audio.get_device_info_by_host_api_device_index(0, i).get('name'))
            mics.append(mic)
    print(mics)
    return jsonify({'success': True, 'mics': mics})

@app.route('/select-mics', methods=['POST'])
def select_mic():
    global micId
    micId = int(request.form.get("selMic"))
    print("Selected Mic:  ", micId)
    return jsonify({'success': True})

@app.route('/home', methods=['GET', 'POST'])
def index():
    global llm 
    llm = init_model()
    # info = audio.get_host_api_info_by_index(0)
    # mics = []
    # numdevices = info.get('deviceCount')
    # for i in range(0, numdevices):
    #     if (audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
    #         print("Input Device id ", i, " - ", audio.get_device_info_by_host_api_device_index(0, i).get('name'))
    #         mics = ("Input Device id ", i, " - ", audio.get_device_info_by_host_api_device_index(0, i).get('name'))
    return render_template('verhome.html')

@app.route('/about', methods=['GET'])
def index_about():
    return render_template('about.html')

if __name__ == '__main__':
    info = audio.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    for i in range(0, numdevices):
        if (audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            print("Input Device id ", i, " - ", audio.get_device_info_by_host_api_device_index(0, i).get('name'))
    app.run(debug=True)