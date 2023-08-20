import pyaudio
import wave
import time
import requests
import json
import assemblyai as aai
import keyboard
import openai
import pyttsx3

# Constants for PyAudio
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
engine = pyttsx3.init()
voice = engine.getProperty('voices')
# engine.setProperty('voice', voice[0].id) #set the voice to index 0 for male voice
engine.setProperty('voice', voice[2].id)
recording = False
transcription = ""
filename = "temp.wav"
audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
frames = []
openai.api_key = "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"

# Function to handle button press event for starting recording
def start_recording():
    global recording
    global frames

    recording = True
    frames = []

    print("Recording started.")

    # Continue recording until the stop button is pressed
    while recording:
        data = stream.read(CHUNK)
        frames.append(data)

# Function to handle button press event for stopping recording
def stop_recording():
    global recording
    global frames
    global filename
    recording = False
    print("Recording stopped.")

    # Save the recorded audio as a WAV file

    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    process_transcription(filename)

# Function to send transcription to OpenAI API and receive output
def process_transcription(filename):
    global transcription
    global openai_api_key
    aai.settings.api_key = "90c53dc840e7456689bf3c847cc362bf"
    transcriber = aai.Transcriber()

    transcript = transcriber.transcribe(filename)
    transcription = transcript.text
    print(transcription)

    # Send prompt to OpenAI API
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
            {"role": "system", "content": "You are a intelligent Ai bot named Veronica"},
            {"role": "user", "content": transcription},
        ]
    )

    output = response['choices'][0]['message']['content']
    print(response)
    engine.say(output)
    engine.runAndWait()
# Cleanup
def cleanup():
    stream.stop_stream()
    stream.close()
    audio.terminate()

def main():
    while(True):
        print("Recording")
        data = stream.read(CHUNK)
        frames.append(data)
        if keyboard.is_pressed("0"):
            stop_recording()
            cleanup()
            break
    # while True:
    #     # Simulate button press events
    #     button_start_pressed = True  # Replace with actual button press detection
    #     button_stop_pressed = False  # Replace with actual button press detection
    #     if keyboard.is_pressed("0"):
    #         button_stop_pressed = True 
    #         button_start_pressed = False
    #         stop_recording()
    #         cleanup()
    #         break
    #     if button_start_pressed:
    #         start_recording()
    #     elif button_stop_pressed:
    #         stop_recording()
    #         cleanup()
    #         break
    #      # Adjust the sleep duration as per your requirement

if __name__ == "__main__":
    main()