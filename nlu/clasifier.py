import os
import numpy as np
from tensorflow.keras.models import load_model

# --- Load Trained Model and Labels ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.keras')
LABELS_PATH = os.path.join(BASE_DIR, 'labels.txt')

model = load_model(MODEL_PATH)

# Retrieve dynamic input shape expected by the model (batch_size, max_seq, 256)
MAX_SEQ = model.input_shape[1]

# Load labels and strip empty trailing lines
with open(LABELS_PATH, 'r', encoding='utf-8') as f:
    labels = [line.strip() for line in f if line.strip()]

idx2label = {k: label for k, label in enumerate(labels)}

def classify(text: str):
    """
    Classifies an input text into an entity, action, and confidence score.
    """
    if not text:
        return None, None, 0.0

    # Encode input string into byte sequence
    text_bytes = bytes(text.lower().strip().encode('utf-8'))
    
    # Truncate text if it exceeds maximum sequence length trained
    if len(text_bytes) > MAX_SEQ:
        text_bytes = text_bytes[:MAX_SEQ]

    # Build input matrix matching model shape
    x = np.zeros((1, MAX_SEQ, 256), dtype='float32')
    for k, ch in enumerate(text_bytes):
        x[0, k, int(ch)] = 1.0

    # Run inference silently
    out = model.predict(x, verbose=0)
    idx = int(np.argmax(out[0]))
    confidence = float(out[0][idx])
    
    # Parse intent label (e.g., "time/getTime" -> entity="time", action="getTime")
    raw_label = idx2label.get(idx, "")
    if "/" in raw_label:
        entity, action = raw_label.split("/", 1)
    elif "\\" in raw_label:
        entity, action = raw_label.split("\\", 1)
    else:
        entity, action = raw_label, ""

    return entity, action, confidence


if __name__ == "__main__":
    while True:
        user_input = input("User: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        ent, act, conf = classify(user_input)
        print(f"Entity: {ent} | Action: {act} | Confidence: {conf:.2f}\n")