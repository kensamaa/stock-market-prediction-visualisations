# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 20:10:40 2021

@author: kensama
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
# dataset https://finance.yahoo.com/quote/IAM.PA/history?p=IAM.PA
url = 'https://raw.githubusercontent.com/mwitiderrick/stockprice/master/NSE-TATAGLOBAL.csv'
#must change the path of the file

from sklearn.model_selection import train_test_split
df= pd.read_csv("E:/programing/python/platform/lab32Stockmarket/BAC.csv")
df = df.dropna(how='any',axis=0) #remove null rows


#dataset_train = df= pd.read_csv(url)
train, test = train_test_split(df, test_size=0.2,shuffle=False)
dataset_train = train
training_set = dataset_train.iloc[:, 1:2].values
print(dataset_train.head())

# MinMaxScaler from scikit-learn to scale our dataset into numbers between 0 and 1
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)
#we create data in 60 timesteps and convert it into an array using NumPy. 
#Then, we convert the data into a 3D array with X_train samples, 
#60 timestamps, and one feature at each step.
X_train = []
y_train = []
for i in range(60, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#           train the model

model = Sequential()

model.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50))
model.add(Dropout(0.2))

model.add(Dense(units=1))

model.compile(optimizer='adam',loss='mean_squared_error')

model.fit(X_train,y_train,epochs=20,batch_size=32)
model.save('bank.h5')

'''tf.keras.utils.plot_model(
    model, to_file='model.png',
    show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96
)'''
#                test

url = 'https://raw.githubusercontent.com/mwitiderrick/stockprice/master/tatatest.csv'
#must change the path of the file
#dataset_test = pd.read_csv(url)

dataset_test =test
real_stock_price = dataset_test.iloc[:, 1:2].values

dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []

for i in range(60, len(inputs)):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
#            plot

plt.plot(real_stock_price, color = 'black', label = 'bank america Stock Price')
plt.plot(predicted_stock_price, color = 'green', label = 'Predicted bank america Stock Price')
plt.title('bank america  Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('bank america   Stock Price')
plt.legend()
plt.show()