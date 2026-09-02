#!/usr/bin/env python3

import os
import json
import pyaudio
import pyttsx3
from vosk import Model, KaldiRecognizer, SetLogLevel
import core

# Suppress internal C++ logs from Vosk
SetLogLevel(-1)

# Initialize TTS Engine
engine = pyttsx3.init()
voices = engine.getProperty('voices')
if voices:
    engine.setProperty('voice', voices[-2].id)

# Initialize Vosk Model
if not os.path.exists("model"):
    print("Model directory not found.")
    exit(1)

model = Model('model')
rec = KaldiRecognizer(model, 16000)

# Initialize PyAudio
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
stream.start_stream()

def speak(text):
    # Stop microphone stream while speaking to prevent self-listening loop
    stream.stop_stream()
    engine.say(text)
    engine.runAndWait()
    
    # Restart stream and clear any audio accumulated during speech
    stream.start_stream()
    stream.read(stream.get_read_available(), exception_on_overflow=False)

print("Lumi pronta e escutando...")

try:
    while True:
        # Avoid crash on buffer overflow
        data = stream.read(4000, exception_on_overflow=False)
        if len(data) == 0:
            break

        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            text = result.get('text', '')

            if text:
                print(f"Você disse: {text}")
                
                # Command: Shutdown
                if "desligar" in text.lower():
                    speak("Desligando...")
                    break

                # Command: Tell time (flexible matching)
                if "que horas são" in text.lower() or "horas" in text.lower():
                    speak(core.SystemInfo.get_time())

except KeyboardInterrupt:
    print("\nEncerrando...")

finally:
    # Cleanup audio resources
    stream.stop_stream()
    stream.close()
    p.terminate()