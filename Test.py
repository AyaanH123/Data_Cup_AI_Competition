# LSTM for sequence classification in the IMDB dataset
import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.preprocessing import sequence
import Json_Read as jr
from sklearn.feature_extraction.text import CountVectorizer
# fix random seed for reproducibility
np.random.seed(7)

df_two_fields = jr.create_dataframes_claim_label('../TrueNorthAI')
df_input = jr.create_dataframes_input('../TrueNorthAI')
# load the dataset but only keep the top n words, zero the rest
top_words = 5000
train_x = df_two_fields['claim']
train_y = df_two_fields['label']
valid_x = df_input['claim']
valid_y = df_input['label']
vectorizer = CountVectorizer()
train_x = vectorizer.fit_transform(train_x)
valid_x = vectorizer.fit_transform(valid_x)
# truncate and pad input sequences
max_review_length = 21536
train_X = sequence.pad_sequences(train_x.toarray(), maxlen=max_review_length)
valid_x = sequence.pad_sequences(valid_x.toarray(), maxlen=max_review_length)
# create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(train_x, train_y, epochs=7, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(valid_x, valid_y, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))