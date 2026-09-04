import os
import re
import yaml
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Suppress TensorFlow informational messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- Helper to clean punctuation for Vosk compatibility ---
def clean_text(text: str) -> str:
    # Remove punctuation marks and force lower-case
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower().strip()

# --- 1. Load Training Dataset ---
dataset_path = os.path.join('nlu', 'train.yml')
data = yaml.safe_load(open(dataset_path, 'r', encoding='utf-8').read())

inputs, outputs = [], []

for item in data['commands']:
    # Handles both structure variants (nested 'command' key or direct keys)
    cmd = item['command'] if 'command' in item else item
    
    clean_input = clean_text(cmd['input'])
    inputs.append(clean_input)
    
    # Standardize path separator to forward slash for consistent intent parsing
    outputs.append(f"{cmd['entity']}/{cmd['action']}")

# Calculate maximum sequence length in UTF-8 bytes
max_seq = max([len(bytes(x.encode('utf-8'))) for x in inputs])
print('Max sequence length (bytes):', max_seq)

# --- 2. Encode Inputs (One-Hot Byte Encoding) ---
input_data = np.zeros((len(inputs), max_seq, 256), dtype='float32')
for i, inp in enumerate(inputs):
    for k, ch in enumerate(bytes(inp.encode('utf-8'))):
        input_data[i, k, int(ch)] = 1.0

# --- 3. Encode Target Labels ---
labels = sorted(list(set(outputs)))
num_classes = len(labels)

# Save target labels mapping file
labels_path = os.path.join('nlu', 'labels.txt')
with open(labels_path, 'w', encoding='utf-8') as f:
    for label in labels:
        f.write(label + '\n')

label2idx = {label: k for k, label in enumerate(labels)}
idx2label = {k: label for k, label in enumerate(labels)}

output_indices = [label2idx[out] for out in outputs]

# One-hot encode targets using the exact count of unique intent classes
output_data = to_categorical(output_indices, num_classes=num_classes)

# --- 4. Build Neural Network Architecture ---
model = Sequential([
    Input(shape=(max_seq, 256)),
    LSTM(128),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['acc']
)

# --- 5. Callbacks and Model Training ---
model_save_path = os.path.join('nlu', 'model.keras')

# Increased patience and epochs to ensure loss convergence on multi-class dataset
callbacks = [
    EarlyStopping(monitor='acc', patience=30, restore_best_weights=True),
    ModelCheckpoint(model_save_path, monitor='acc', save_best_only=True)
]

model.fit(
    input_data, 
    output_data, 
    epochs=200, 
    batch_size=4, 
    callbacks=callbacks
)

print(f"\nModel successfully trained and saved to '{model_save_path}'!")