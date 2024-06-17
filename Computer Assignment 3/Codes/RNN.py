import numpy as np
from matplotlib import pyplot as plt
from nltk.corpus import treebank
import seaborn as sns
from gensim.models import KeyedVectors
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import Dense, Input
from keras.layers import TimeDistributed
from keras.layers import LSTM, GRU, SimpleRNN, RNN
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import gensim.downloader as api
dataset = list(treebank.tagged_sents(tagset='universal'))
#%%
X = list()
Y = list()
for sen in dataset:
    X.append([w for w,t in sen])
    Y.append([t for w,t in sen])
del sen
#%%
words_count = len(set([word.lower() for sen in X for word in sen])) 
tags_count = len(set([tag.lower() for sen in Y for tag in sen]))
#%%
word_tokenizer = Tokenizer()
word_tokenizer.fit_on_texts(X)
X_encoded = word_tokenizer.texts_to_sequences(X)  
#%%
tag_tokenizer = Tokenizer()
tag_tokenizer.fit_on_texts(Y)
Y_encoded = tag_tokenizer.texts_to_sequences(Y)
#%%
MAX_SEQ_LENGTH = 100
X_padded = pad_sequences(X_encoded, maxlen=MAX_SEQ_LENGTH, padding="pre", truncating="post")
Y_padded = pad_sequences(Y_encoded, maxlen=MAX_SEQ_LENGTH, padding="pre", truncating="post")
#%%
X, Y = X_padded, Y_padded
del X_padded
del Y_padded
del X_encoded
del Y_encoded
#%%
VOCABULARY_SIZE = len(word_tokenizer.word_index) + 1
EMBEDDING_SIZE = 300
word2vec = api.load('word2vec-google-news-300')
#%%
embedding_weights = np.zeros((VOCABULARY_SIZE, EMBEDDING_SIZE))
word2id = word_tokenizer.word_index
for word, index in word2id.items():
    try:
        embedding_weights[index, :] = word2vec[word]
    except KeyError:
        pass
del word
#%%
Y = to_categorical(Y)
#%%
TEST_SIZE = 0.2
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TEST_SIZE, random_state=4)
VALID_SIZE = 0.25
X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=VALID_SIZE, random_state=4)
NUM_CLASSES = Y.shape[2]
#%%
rnn1_model = Sequential()
rnn1_model.add(Embedding(input_dim=VOCABULARY_SIZE, output_dim=EMBEDDING_SIZE, input_length=MAX_SEQ_LENGTH, trainable=True))
rnn1_model.add(SimpleRNN(32, return_sequences=True))
rnn1_model.add(TimeDistributed(Dense(NUM_CLASSES, activation='softmax')))
#%%
rnn1_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
#%%
rnn1_training = rnn1_model.fit(X_train, Y_train, batch_size=128, epochs=10, validation_data=(X_validation, Y_validation))
#%%
rnn1_loss, rnn1_accuracy = rnn1_model.evaluate(X_test, Y_test, verbose = 1)
#%%
rnn2_model = Sequential()
rnn2_model.add(Embedding(input_dim=VOCABULARY_SIZE, output_dim=EMBEDDING_SIZE, input_length=MAX_SEQ_LENGTH, trainable=True))
rnn2_model.add(SimpleRNN(64, return_sequences=True))
rnn2_model.add(TimeDistributed(Dense(NUM_CLASSES, activation='softmax')))
#%%
rnn2_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
#%%
rnn2_training = rnn2_model.fit(X_train, Y_train, batch_size=128, epochs=10, validation_data=(X_validation, Y_validation))
#%%
rnn2_loss, rnn2_accuracy = rnn2_model.evaluate(X_test, Y_test, verbose = 1)
#%%
rnn3_model = Sequential()
rnn3_model.add(Embedding(input_dim=VOCABULARY_SIZE, output_dim=EMBEDDING_SIZE, input_length=MAX_SEQ_LENGTH, trainable=True))
rnn3_model.add(SimpleRNN(128, return_sequences=True))
rnn3_model.add(TimeDistributed(Dense(NUM_CLASSES, activation='softmax')))
#%%
rnn3_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
#%%
rnn3_training = rnn3_model.fit(X_train, Y_train, batch_size=128, epochs=10, validation_data=(X_validation, Y_validation))
#%%
rnn3_loss, rnn3_accuracy = rnn3_model.evaluate(X_test, Y_test, verbose = 1)
#%%
lstm1_model = Sequential()
lstm1_model.add(Embedding(input_dim=VOCABULARY_SIZE, output_dim=EMBEDDING_SIZE, input_length=MAX_SEQ_LENGTH, weights=[embedding_weights], trainable=True))
lstm1_model.add(LSTM(32, return_sequences=True))
lstm1_model.add(TimeDistributed(Dense(NUM_CLASSES, activation='softmax')))
#%%
lstm1_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
#%%
lstm1_training = lstm1_model.fit(X_train, Y_train, batch_size=128, epochs=10, validation_data=(X_validation, Y_validation))
#%%
lstm1_loss, lstm1_accuracy = lstm1_model.evaluate(X_test, Y_test, verbose = 1)
#%%
lstm2_model = Sequential()
lstm2_model.add(Embedding(input_dim=VOCABULARY_SIZE, output_dim=EMBEDDING_SIZE, input_length=MAX_SEQ_LENGTH, weights=[embedding_weights], trainable=True))
lstm2_model.add(LSTM(64, return_sequences=True))
lstm2_model.add(TimeDistributed(Dense(NUM_CLASSES, activation='softmax')))
#%%
lstm2_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
#%%
lstm2_training = lstm2_model.fit(X_train, Y_train, batch_size=128, epochs=10, validation_data=(X_validation, Y_validation))
#%%
lstm2_loss, lstm2_accuracy = lstm2_model.evaluate(X_test, Y_test, verbose = 1)
#%%
lstm3_model = Sequential()
lstm3_model.add(Embedding(input_dim=VOCABULARY_SIZE, output_dim=EMBEDDING_SIZE, input_length=MAX_SEQ_LENGTH, weights=[embedding_weights], trainable=True))
lstm3_model.add(LSTM(128, return_sequences=True))
lstm3_model.add(TimeDistributed(Dense(NUM_CLASSES, activation='softmax')))
#%%
lstm3_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
#%%
lstm3_training = lstm3_model.fit(X_train, Y_train, batch_size=128, epochs=10, validation_data=(X_validation, Y_validation))
#%%
lstm3_loss, lstm3_accuracy = lstm3_model.evaluate(X_test, Y_test, verbose = 1)
#%%
gru1_model = Sequential()
gru1_model.add(Embedding(input_dim=VOCABULARY_SIZE, output_dim=EMBEDDING_SIZE, input_length= MAX_SEQ_LENGTH, weights=[embedding_weights], trainable=True))
gru1_model.add(GRU(32, return_sequences=True))
gru1_model.add(TimeDistributed(Dense(NUM_CLASSES, activation='softmax')))
#%%
gru1_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
#%%
gru1_training = gru1_model.fit(X_train, Y_train, batch_size=128, epochs=10, validation_data=(X_validation, Y_validation))
#%%
gru1_loss, gru1_accuracy = gru1_model.evaluate(X_test, Y_test, verbose = 1)
#%%
gru2_model = Sequential()
gru2_model.add(Embedding(input_dim=VOCABULARY_SIZE, output_dim=EMBEDDING_SIZE, input_length= MAX_SEQ_LENGTH, weights=[embedding_weights], trainable=True))
gru2_model.add(GRU(64, return_sequences=True))
gru2_model.add(TimeDistributed(Dense(NUM_CLASSES, activation='softmax')))
#%%
gru2_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
#%%
gru2_training = gru2_model.fit(X_train, Y_train, batch_size=128, epochs=10, validation_data=(X_validation, Y_validation))
#%%
gru2_loss, gru2_accuracy = gru2_model.evaluate(X_test, Y_test, verbose = 1)
#%%
gru3_model = Sequential()
gru3_model.add(Embedding(input_dim=VOCABULARY_SIZE, output_dim=EMBEDDING_SIZE, input_length= MAX_SEQ_LENGTH, weights=[embedding_weights], trainable=True))
gru3_model.add(GRU(128, return_sequences=True))
gru3_model.add(TimeDistributed(Dense(NUM_CLASSES, activation='softmax')))
#%%
gru3_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
#%%
gru3_training = gru3_model.fit(X_train, Y_train, batch_size=128, epochs=10, validation_data=(X_validation, Y_validation))
#%%
gru3_loss, gru3_accuracy = gru3_model.evaluate(X_test, Y_test, verbose = 1)