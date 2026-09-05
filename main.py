#!/usr/bin/env python3

import os
import sys
import json
import pyaudio
import pyttsx3
from vosk import Model, KaldiRecognizer

# Add root directory to path for local module resolution
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import core
from nlu.classifier import classify

# --- Text-to-Speech Engine Initialization ---
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[-2].id)

def speak(text):
    """Synthesizes speech and blocks until completed."""
    engine.say(text)
    engine.runAndWait()

# --- Speech Recognition Setup (Vosk Model) ---
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

# --- Configuration & Main Event Loop ---
CONFIDENCE_THRESHOLD = 0.60  # Minimum confidence score required to trigger actions

try:
    while True:
        data = stream.read(2048, exception_on_overflow=False)
        if len(data) == 0:
            break

        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            text = result.get('text', '').strip()

            if text:
                # Extract entity, action, and confidence score from the NLU classifier
                entity, action, confidence = classify(text)
                
                print(f"Text: {text} | Intent: {entity}/{action} | Confidence: {confidence:.2f}")

                # Execute action only if the confidence score meets the threshold
                if confidence >= CONFIDENCE_THRESHOLD:
                    if entity == 'time' and action == 'getTime':
                        speak(core.SystemInfo.get_time())
                        rec.Reset()
                    elif entity == 'date' and action == 'getDate':
                        speak(core.SystemInfo.get_date())
                        rec.Reset()
                    elif entity == 'open' and action == 'getExplorer':
                        speak(core.SystemInfo.open_explorer())
                        rec.Reset()
                    elif entity == 'close' and action == 'closeExplorer':
                        speak(core.SystemInfo.close_explorer())
                        rec.Reset()
                    elif entity == 'open' and action == 'getNotepad':
                        speak(core.SystemInfo.open_notepad())
                        rec.Reset()
                    elif entity == 'close' and action == 'closeNotepad':
                        speak(core.SystemInfo.close_notepad())
                        rec.Reset()
                else:
                    print("Command not understood with sufficient confidence.")

except KeyboardInterrupt:
    print("\nEncerrando Lumi...")
finally:
    # Ensure audio stream resources are cleanly released
    stream.stop_stream()
    stream.close()
    p.terminate()