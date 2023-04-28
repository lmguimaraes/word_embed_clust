import numpy as np
import pandas as pd
from pathlib import Path
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense

MAX_SEQUENCE_LENGTH = 1000
VOCABULARY_SIZE = 10000
downloads_path = str(Path.home() / "Downloads")

data = pd.read_excel(downloads_path + '/data/first_wave.xlsx')

# Preprocess the data (tokenize, remove stop words, etc.)

# Convert the text data to sequences of word indices
tokenizer = Tokenizer(num_words=VOCABULARY_SIZE)
tokenizer.fit_on_texts(data['preprocessed_text'])
sequences = tokenizer.texts_to_sequences(data['preprocessed_text'])

# Pad the sequences to ensure that they are all the same length
padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

# Define the CNN model architecture
model = Sequential()
model.add(Embedding(VOCABULARY_SIZE, 128, input_length=MAX_SEQUENCE_LENGTH))
model.add(Conv1D(128, 5, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

topics = []

# Train the model
model.fit(padded_sequences, topics, epochs=10, validation_split=0.2)

# Get the predicted topics for each document
predicted_topics = np.argmax(model.predict(padded_sequences), axis=1)

# Print the topics for each document
for i, topic in enumerate(predicted_topics):
    print('Document {}: Topic {}'.format(i, topic))