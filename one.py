import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import ast
from sklearn.model_selection import train_test_split
import sklearn.linear_model as linear_model
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.layers import Input, dot, concatenate
from keras.models import Model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Conv1D, MaxPooling1D, LSTM
Movies = pd.read_csv("Data/Precossed_Data.csv", encoding="cp850")
print(Movies.head())
Movies.rename(columns={"id": "movieId"}, inplace=True)
rating = pd.read_csv("Data/ratings_small.csv")
print(rating.head())

Data = pd.merge(Movies, rating, on="movieId")
print(Data.tail())


def variable_linreg_imputation(df, col_to_predict, ref_col):
    regr = linear_model.LinearRegression()
    test = df[[col_to_predict, ref_col]].dropna(how='any', axis=0)
    X = np.array(test[ref_col])
    Y = np.array(test[col_to_predict])
    X = X.reshape(len(X), 1)
    Y = Y.reshape(len(Y), 1)
    regr.fit(X, Y)

    test = df[df[col_to_predict].isnull() & df[ref_col].notnull()]
    for index, row in test.iterrows():
        value = float(regr.predict(row[ref_col]))
        df.set_value(index, col_to_predict, value)


variable_linreg_imputation(Data, 'popularity', "vote_count")

df = Data.copy(deep=True)
missing_df = df.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name', 'missing_count']
missing_df['filling_factor'] = (df.shape[0]
                                - missing_df['missing_count']) / df.shape[0] * 100
missing_df = missing_df.sort_values('filling_factor').reset_index(drop=True)
print(missing_df)


print(Data[Data['PDC1'].isnull()])
Data.dropna(subset=["PDC1"], inplace=True)


def get_values(data_str):
    if isinstance(data_str, float):
        pass
    else:
        values = []
        data_str = ast.literal_eval(data_str)
        if isinstance(data_str, list):
            for k_v in data_str:
                values.append(k_v['name'].replace(" ", ""))
            return str(values)[1:-1]
        else:
            return None


data = Data.sample(frac=1)
data_train_x = np.array(data[['userId', 'movieId']].values)
data_train_y = np.array(data['rating'].values)
x_train, x_test, y_train, y_test = train_test_split(data_train_x, data_train_y,
                                                    test_size=0.2, random_state=98)

n_factors = 50
n_users = data['userId'].max()
n_movies = data['movieId'].max()
user_input = Input(shape=(1,), name='User_Input')
user_embeddings = Embedding(input_dim=n_users+1, output_dim=n_factors, input_length=1,
                            name='User_Embedding')(user_input)
user_vector = Flatten(name='User_Vector')(user_embeddings)

movie_input = Input(shape=(1,), name='Movie_input')
movie_embeddings = Embedding(input_dim=n_movies+1, output_dim=n_factors, input_length=1,
                             name='Movie_Embedding')(movie_input)
movie_vector = Flatten(name='Movie_Vector')(movie_embeddings)

merged_vectors = concatenate([user_vector, movie_vector], name='Concatenation')
dense_layer_1 = Dense(100, activation='relu')(merged_vectors)
dense_layer_3 = Dropout(.5)(dense_layer_1)
dense_layer_2 = Dense(1)(dense_layer_3)
model = Model([user_input, movie_input], dense_layer_2)
opt = tf.keras.optimizers.Adam(learning_rate=0.0001, decay=0.00001)
model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
model.summary()
SVG(model_to_dot(model,  show_shapes=True, show_layer_names=True).create(prog='dot', format='svg'))
history = model.fit(x=[x_train[:, 0], x_train[:, 1]], y=y_train, batch_size=128, epochs=100,
                    validation_data=([x_test[:, 0], x_test[:, 1]], y_test))

loss, val_loss, accuracy, val_accuracy = history.history['loss'],\
                                            history.history['val_loss'],\
                                            history.history['accuracy'],\
                                            history.history['val_accuracy']

plt.figure(figsize=(12, 10))
plt.plot(loss, 'r--')
plt.plot(val_loss, 'b-')
plt.plot(accuracy, 'g--')
plt.plot(val_accuracy, '-')
plt.legend(['Training Loss', 'Validation Loss', 'Training Accuracy', 'Validation Accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
