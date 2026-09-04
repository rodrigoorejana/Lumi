import os
import yaml
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Suppress TensorFlow info logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- 1. Load Dataset ---
dataset_path = os.path.join('nlu', 'train.yml')
data = yaml.safe_load(open(dataset_path, 'r', encoding='utf-8').read())

inputs, outputs = [], []

for command in data['commands']:
    inputs.append(command['input'].lower().strip())
    # Standardize separator to forward slash to match classifier logic
    outputs.append(f"{command['entity']}/{command['action']}")

# Calculate maximum sequence length in bytes
max_seq = max([len(bytes(x.encode('utf-8'))) for x in inputs])
print('Max sequence length (bytes):', max_seq)

# --- 2. Encode Inputs (One-Hot Byte Encoding) ---
input_data = np.zeros((len(inputs), max_seq, 256), dtype='float32')
for i, inp in enumerate(inputs):
    for k, ch in enumerate(bytes(inp.encode('utf-8'))):
        input_data[i, k, int(ch)] = 1.0

# --- 3. Encode Output Labels ---
labels = sorted(list(set(outputs)))
num_classes = len(labels)

# Save labels mapping file
labels_path = os.path.join('nlu', 'labels.txt')
with open(labels_path, 'w', encoding='utf-8') as f:
    for label in labels:
        f.write(label + '\n')

label2idx = {label: k for k, label in enumerate(labels)}
idx2label = {k: label for k, label in enumerate(labels)}

output_indices = [label2idx[out] for out in outputs]

# FIX: Pass num_classes instead of len(output_data)
output_data = to_categorical(output_indices, num_classes=num_classes)

print('Sample target vector:', output_data[0])

# --- 4. Build Neural Network ---
model = Sequential([
    # Explicit Input layer matching sequence length and byte vocabulary size
    Input(shape=(max_seq, 256)),
    LSTM(128),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['acc']
)

# --- 5. Training Callbacks & Execution ---
callbacks = [
    EarlyStopping(monitor='loss', patience=15, restore_best_weights=True),
    ModelCheckpoint(os.path.join('nlu', 'model.keras'), monitor='acc', save_best_only=True)
]

model.fit(
    input_data, 
    output_data, 
    epochs=128, 
    batch_size=4, 
    callbacks=callbacks
)

print("Model and labels saved successfully in 'nlu/' directory!")

# --- 6. Inference Method ---
def classify(text):
    text_bytes = bytes(text.lower().strip().encode('utf-8'))
    
    # Truncate text if it exceeds maximum trained sequence
    if len(text_bytes) > max_seq:
        text_bytes = text_bytes[:max_seq]

    # Initialize dynamic array matching trained max_seq
    x = np.zeros((1, max_seq, 256), dtype='float32')

    for k, ch in enumerate(text_bytes):
        x[0, k, int(ch)] = 1.0

    out = model.predict(x, verbose=0)
    idx = out.argmax()
    confidence = out[0][idx]
    
    return idx2label[idx], confidence

# Quick execution check
if __name__ == "__main__":
    intent, conf = classify("que horas são agora")
    print(f"Predicted Intent: {intent} | Confidence: {conf:.2f}")