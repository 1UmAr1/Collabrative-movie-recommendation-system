import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, BatchNormalization, Flatten, LSTM, Bidirectional
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.callbacks import TensorBoard
import time
tf.keras.backend.set_floatx('float64')


Name = "summm{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir="logsrj/".format(Name))

data = pd.read_csv("Book2.csv", encoding="cp850")
train_dataset = data.sample(frac=0.7, random_state=0)
test_dataset = data.drop(train_dataset.index)


train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop("vote_average")
test_labels = test_dataset.pop("vote_average")

Normalizer = preprocessing.Normalization()
Normalizer.adapt(np.array(train_features))
train_features = np.asarray(train_features)
print(train_features.shape)
train_features = train_features.reshape(3362, 1, 46)

model = Sequential()
model.add(Bidirectional(LSTM(128, activation="tanh",  return_sequences=True)))


model.add(Bidirectional(LSTM(128, activation="tanh", return_sequences=True)))


model.add(Bidirectional(LSTM(128, activation="tanh", return_sequences=True)))
model.add(Dropout(0.2))
model.add(BatchNormalization())


model.add(Dense(128))
model.add(Activation("relu"))
model.add(Dropout(0.2))

model.add(Dense(1))

opt = tf.keras.optimizers.SGD(lr=0.0001, momentum=0.9)
model.compile(loss="mean_absolute_error",
              optimizer=opt,
              metrics=["accuracy"])


history = model.fit(train_features, train_labels, validation_split=0.2,
        verbose=0, epochs=100, callbacks=[tensorboard])
hist = pd.DataFrame(history.history)
hist["epoch"] = history.epoch
print(hist.head())
print(hist.tail())
model.summary()



