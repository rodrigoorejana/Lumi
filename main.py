#!/usr/bin/env python3

import os
import json
import pyaudio
import pyttsx3
from vosk import Model, KaldiRecognizer

import core
from nlu.classifier import classify

# --- Speech Synthesis (Text-to-Speech) ---
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[-2].id)

def speak(text):
    engine.say(text)
    engine.runAndWait()

# --- Speech Recognition (Vosk Model) ---
model = Model('model')
rec = KaldiRecognizer(model, 16000)

p = pyaudio.PyAudio()
stream = p.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=16000,
    input=True,
    frames_per_buffer=2048
)
stream.start_stream()

# --- Speech Recognition Loop ---
CONFIDENCE_THRESHOLD = 0.60  # Minimum required NLU confidence score

try:
    while True:
        data = stream.read(2048, exception_on_overflow=False)
        if len(data) == 0:
            break

        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            text = result.get('text', '').strip()

            # Process only if spoken text was recognized
            if text:
                # Unpack entity, action, and confidence score from the NLU classifier
                entity, action, confidence = classify(text)
                
                print(f"Text: '{text}' | Intent: {entity}/{action} | Confidence: {confidence:.2f}")

                # Validate confidence threshold before executing actions
                if confidence >= CONFIDENCE_THRESHOLD:
                    
                    # Handle time retrieval intent
                    if entity == 'time' and action == 'getTime':
                        current_time = core.SystemInfo.get_time()
                        speak(f"Agora são {current_time}")

                else:
                    print("Command not understood with sufficient confidence.")

except KeyboardInterrupt:
    print("\nShutting down Lumi...")
finally:
    # Ensure audio stream is properly closed on exit
    stream.stop_stream()
    stream.close()
    p.terminate()