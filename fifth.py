import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional, Activation
from tensorflow.keras.layers.experimental import preprocessing
import pandas as pd

tf.keras.backend.set_floatx('float64')

Name = "RECOMen{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir="logsri/".format(Name))

data = pd.read_csv("Book2.csv", encoding="cp850")
labels = pd.read_csv("labels.csv")

""""

model = Sequential()
model.add(LSTM(128, activation="tanh", input_shape=(data.shape[1:]), return_sequences=True))
model.add(Dropout(0.4))

model.add(LSTM(128, activation="tanh", return_sequences=True))
model.add(Dropout(0.4))

model.add(LSTM(128, activation="tanh", return_sequences=True))
model.add(Dropout(0.4))

model.add(LSTM(128, activation="tanh"))
model.add(Dropout(0.4))

model.add(Dense(128))
model.add(Activation("relu"))
model.add(Dropout(0.4))


model.add(Dense(128))
model.add(Activation("relu"))
model.add(Dropout(0.4))

model.add(Dense(1))
opt = tf.keras.optimizers.Adam(lr=0.0001, decay=1e-5)

model.compile(optimizer=opt,
              loss="mean_absolute_error",
              metrics=["accuracy"])
model.fit(data, labels, batch_size=100, validation_split=0.2, epochs=200, callbacks=[tensorboard])
"""