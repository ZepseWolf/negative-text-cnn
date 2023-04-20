import ctypes

import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
#from keras.preprocessing.text import Tokenizer
#from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split


print(tf.VERSION)
#print(keras.__version__)

def predictD(text=None):
    if text is None:
            return 'Failed'
    else:
            clean = pd.read_excel('C:/Users/Douglascat/Desktop/combinedData.xlsx')

            colums_text = clean['Text']

            tokenizer = keras.preprocessing.text.Tokenizer(num_words=10000, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~ '
                                                           , lower=True, split=' ', char_level=False, oov_token=None, document_count=0)
            tokenizer.fit_on_texts(colums_text)
            t = tokenizer.texts_to_sequences(colums_text)
            train_data = keras.preprocessing.sequence.pad_sequences(t, maxlen=100, padding='post', value=0)

            vocab_size = 10000

            model = keras.Sequential()
            model.add(keras.layers.Embedding(vocab_size, 16))
            model.add(keras.layers.GlobalAveragePooling1D())
            model.add(keras.layers.Dense(16, activation=tf.nn.relu))
            model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

            train_labels = clean['Class'].values

            X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.33, random_state=24)

            x_val = X_train[:2000]
            partial_x_train = X_train[2000:]

            y_val = y_train[:2000]
            partial_y_train = y_train[2000:]

            model.compile(optimizer='adam',
                          loss='binary_crossentropy',
                          metrics=['accuracy'])

            history = model.fit(partial_x_train,
                                partial_y_train,
                                epochs=40,
                                batch_size=500,
                                validation_data=(x_val, y_val),
                                verbose=1)

            results = model.evaluate(X_test, y_test)
            textToken = tokenizer.texts_to_sequences([text])
            textToken = keras.preprocessing.sequence.pad_sequences(textToken, maxlen=150, padding='post', value=0)
            result =  model.predict([textToken])
            
            return str(result[0][0])

#print(type(predictD("life saver of a 10 year old boy thanks to a romantic twist of fate that compelled him to become a bone marrow donor Four years ago young Rupert Cross was diagnosed with a rare blood disorder that required him to undergo chemotherapy in a specialist unit at Great Ormond Street Hospital for 80 days straight Though the prognosis was dire the boy is now completely healed after Higgins discovered he was a match for Rupert’s bone marrow")))
#print(predictD())
