import yaml
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical

# Load YAML training dataset
data = yaml.safe_load(open('nlu\\train.yml', 'r', encoding='utf-8').read())

inputs, outputs = [], []

for command in data['commands']:
    inputs.append(command['input'].lower())
    outputs.append(f"{command['entity']}/{command['action']}")


# Text processing strategy: UTF-8 Byte-level encoding (0-255)

max_seq = max([len(bytes(x.encode('utf-8'))) for x in inputs])

print('Max sequence length (bytes):', max_seq)

# Create input dataset using One-Hot Encoding: (samples, max_sequence, 256 bytes)
input_data = np.zeros((len(inputs), max_seq, 256), dtype='float32')
for i, inp in enumerate(inputs):
    for k, ch in enumerate(bytes(inp.encode('utf-8'))):
        input_data[i, k, int(ch)] = 1.0


# Target Output Processing (Categorical Intents)

labels = sorted(list(set(outputs)))

label2idx = {}
idx2label = {}

for k, label in enumerate(labels):
    label2idx[label] = k
    idx2label[k] = label

output_indices = []

for output in outputs:
    output_indices.append(label2idx[output])

# Convert target labels to one-hot vectors using unique class count
output_data = to_categorical(output_indices, num_classes=len(labels))


print("Sample target vector:", output_data[0])

# Define LSTM Model Architecture
model = Sequential()
model.add(LSTM(128, input_shape=(max_seq, 256)))
model.add(Dense(len(labels), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

# Train the model
model.fit(input_data, output_data, epochs=128, batch_size=4)


# Inference function: classify input string into an intent/action
def classify(text):
    text_bytes = bytes(text.lower().encode('utf-8'))
    
    # Truncate text if it exceeds maximum sequence length
    if len(text_bytes) > max_seq:
        text_bytes = text_bytes[:max_seq]

    # Initialize dynamic input tensor matching training shape
    x = np.zeros((1, max_seq, 256), dtype='float32')

    # Populate array with byte sequence
    for k, ch in enumerate(text_bytes):
        x[0, k, int(ch)] = 1.0

    # Perform intent prediction
    out = model.predict(x, verbose=0)
    idx = out.argmax()
    confidence = out[0][idx]
    
    print(f"Predicted intent: {idx2label[idx]} (Confidence: {confidence:.2f})")


# Interactive execution loop
while True:
    try:
        text = input('\nEnter command: ')
        if text:
            classify(text)
    except KeyboardInterrupt:
        print("\nExiting inference loop...")
        break