
"""

from pandas import read_csv
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt

def create_RNN(hidden_units, dense_units, input_shape, activation):
  model = Sequential()
  model.add(SimpleRNN(hidden_units, input_shape = input_shape, activation = activation[0]))
  model.add(Dense(units = dense_units, activation = activation[1]))
  model.compile(loss = 'mean_squared_error', optimizer = 'adam')
  return model

demo_model = create_RNN(2, 1, (3, 1), activation = ['linear', 'linear'])

x = np.array([3, 3, 3])
x_input = np.reshape(x, (1, 3, 1))
y_pred_model = demo_model.predict(x_input)

wx = demo_model.get_weights()[0]
print(wx)

wx = demo_model.get_weights()[0]
print('wx', wx)
wh = demo_model.get_weights()[1]
print('wh', wh)
bh = demo_model.get_weights()[2]
print('bh', bh)
wy = demo_model.get_weights()[3]
print('wy', wy)
by = demo_model.get_weights()[4]
print('by', by)

x = np.array([3, 3, 3])  #x[0], x[1], x[2]
print(x)
#reshape the input
x_input = np.reshape(x, (1, 3, 1))
print(x_input)
y_pred_model = demo_model.predict(x_input)   #prediction from network from keras library

m = 2   #hidden layers
h0 = np.ones(m)                           #initialization
h1 = np.dot(x[0], wx) + h0 + bh           #input layer
h2 = np.dot(x[1], wx) + np.dot(h1, wh) + bh   #1st hidden layer in RNN
h3 = np.dot(x[2], wx) + np.dot(h2, wh) + bh   #2nd hidden layer in RNN

o3 = np.dot(h3, wy) + by       #dense layer

print('h0', h0, 'h1', h1, 'h2', h2, 'h3', h3)
print('prediction from our network: ',o3)
print('prediction from Heras network: ', y_pred_model)
