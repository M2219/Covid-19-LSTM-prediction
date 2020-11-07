# Following script train stacked LSTM 
# for predicting Covid-19 death cases
# Mahmoud Tahmasebi

from __future__ import division
import numpy as np
from numpy import array
import pandas as pd
import matplotlib.pyplot as plt 
import pickle
import json
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten
import keras
from keras import backend as K
from sklearn.preprocessing import MinMaxScaler

np.set_printoptions(linewidth=np.inf)

def split_sequence(sequence, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequence)):

	    end_ix = i + n_steps_in
	    out_end_ix = end_ix + n_steps_out

	    if out_end_ix > len(sequence):
	    	break

	    seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
	    X.append(seq_x)
	    y.append(seq_y)
    return array(X), array(y)


################################
#        data preparation      #
################################
series = pd.read_csv("./covid-19-data/data/worldwide-aggregate.csv")
values = series.values
values_train_val = values[0:270, 3].reshape(-1, 1)

## Normalization

scaler = MinMaxScaler(feature_range = (0, 1))
scaler = scaler.fit(values_train_val)
print('Min: %f, Max: %f' % (scaler.data_min_, scaler.data_max_))
deaths_cases_normalized = scaler.transform(values_train_val)


# define input sequence
raw_seq = list(deaths_cases_normalized)

# choose a number of time steps
n_steps_in, n_steps_out = 30, 1

# split into samples
X, y = split_sequence(raw_seq, n_steps_in, n_steps_out)
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))

len_X = X.shape[0] - 1
print(len_X)
input_per = int(len_X * 0.8)

X_train = X[0:input_per, :]
y_train = y[0:input_per, :]
y_train = np.concatenate(y_train)

X_val = X[input_per:len_X - 30, :]
y_val = y[input_per:len_X - 30, :]
y_val = np.concatenate(y_val)

print("X_train", (X_train.shape))
print("y_train", (y_train.shape))

print("X_val", (X_val.shape))
print("y_val", (y_val.shape))


# define model
model = Sequential()
model.add(LSTM(80, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
#model.add(LSTM(100, activation='relu', return_sequences=True, dropout=0.1))
#model.add(LSTM(100, activation='relu', return_sequences=True, dropout=0.1))
model.add(LSTM(80, activation='relu', return_sequences=False, dropout=0.1))
model.add(Dense(1, activation='linear'))
opt= keras.optimizers.Adam(lr=0.001)

model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])
# fit model
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
history = model.fit(X_train, y_train, batch_size=8 ,epochs=500, verbose=1,\
validation_data = (X_val, y_val), callbacks=[es])

X_test = X
y_test = np.concatenate(y)

print("X_test.shape", X_test.shape)
print("y_test.shape", y_test.shape)

yhat = model.predict(X_test, verbose=0)
deaths_cases_inversed = scaler.inverse_transform(yhat)

np.save("yhat.npy", deaths_cases_inversed)
print("prediction", deaths_cases_inversed)
print(scaler.inverse_transform(np.concatenate(y_test).reshape(-1, 1)))

with open('./trainparam.pkl', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)


with open('trainparam.json', 'w') as file:
    json.dump(history.history, file)



