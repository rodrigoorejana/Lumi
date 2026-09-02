import os
import json
import pyaudio
from vosk import Model, KaldiRecognizer, SetLogLevel

# Suppress internal Vosk C++ logs
SetLogLevel(-1)

# Check if model exists
if not os.path.exists("model"):
    print("Please download the model from https://alphacephei.com/vosk/models and unpack as 'model' in the current folder.")
    exit(1)

print("Iniciando a Lumi...")

# Initialize Vosk model and recognizer
model = Model("model")
recognizer = KaldiRecognizer(model, 16000)

# Initialize PyAudio stream
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
stream.start_stream()

print("Pode falar (LUMI escutando offline)...")

try:
    while True:
        # Read audio chunk safely
        data = stream.read(4000, exception_on_overflow=False)
        if len(data) == 0:
            break

        # Process complete speech segment
        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            text = result.get("text", "")

            # Print recognized text only when non-empty
            if text:
                print(f"Você disse: {text}")

                if "desligar" in text.lower():
                    print("Desligando...")
                    break

except KeyboardInterrupt:
    print("\nLumi encerrada pelo usuário.")

finally:
    # Clean up audio stream resources
    stream.stop_stream()
    stream.close()
    p.terminate()