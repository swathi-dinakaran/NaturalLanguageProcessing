import numpy as np
import pandas as pd
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, GRU, Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint

import re

def preprocess(text):
    # lower case text
    newString = text.lower()
    newString = re.sub(r"'s\b","",newString)
    # remove punctuations
    newString = re.sub("[^a-zA-Z]", " ", newString)
    long_words=[]
    # remove short word
    for i in newString.split():
        if len(i)>=3:
            long_words.append(i)
    return (" ".join(long_words)).strip()

import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

file_name = "G:\\NLP\\hw1\\training_pasted.xlsx"
data = pd.read_excel(r"G:\NLP\hw1\training_pasted.xlsx", encoding='unicode_escape', header=0, names=["text", "opinion"], error_bad_lines=False, lineterminator='\n')

text = data['text']
label = data['opinion']
train_percent = 0.8
train_cutoff = int(np.floor(train_percent * len(text)))


numpy.random.seed(7)
top_words = 5000
# truncate and pad input sequences
max_review_length = 500
# integer encode sequences of words
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text)
sequences = tokenizer.texts_to_sequences(text)
train_seq = sequence.pad_sequences(sequences[0: train_cutoff], maxlen=max_review_length)
test_seq = sequence.pad_sequences(sequences[train_cutoff + 1: len(text)], maxlen=max_review_length)
# create the model
y = to_categorical(label)

embedding_vector_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
model.add(LSTM(100))
model.add(Dense(3, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(train_seq, y[0: train_cutoff], epochs=3, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(test_seq, y[train_cutoff + 1: len(text)], verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))