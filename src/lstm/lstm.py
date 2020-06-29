import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.optimizers import Adam

TICKER_DIR = "C:\\Users\\anton\\Documents\\antoniouaa\\msc_thesis\\data\\tickers\\ticker_data\\_Rolling"
ASSET_DIR = "C:\\Users\\anton\\Documents\\antoniouaa\\msc_thesis\\assets"

def fetch_tickers(path):
    return [tick for tick in os.listdir(path)]

def sliding_windows(data, seq_length):
    x, y = [], []
    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length]
        x.append(_x)
        y.append(_y)
    return np.array(x), np.array(y)    

def train_test_split(data, SC, seq_length=2, size=0.8):
    training_data = SC.fit_transform(training)
    x, y = sliding_windows(training_data, seq_length)
    train_size = int(len(y) * size)
    test_size = len(y) - train_size
    dataX = Variable(torch.Tensor(x))
    dataY = Variable(torch.Tensor(y))
    trainX = Variable(torch.Tensor(x[:train_size]))
    trainY = Variable(torch.Tensor(y[:train_size]))
    testX = Variable(torch.Tensor(x[train_size-1:]))
    testY = Variable(torch.Tensor(y[train_size-1:]))
    return dataX, dataY, trainX, trainY, testX, testY

def build_model(type_, loss="mean_squared_error", lr=0.01, batch=2, timesteps=2, features=2):
    m = Sequential()
    if type_.lower() == "lstm":
        m.add(LSTM(features, batch_input_shape=(batch, timesteps, features), stateful=True, return_sequences=True))
        m.add(LSTM(features, batch_input_shape=(batch, timesteps, features), stateful=True))
        m.add(Dense(features))
    if type_.lower() == "gru":
        m.add(GRU(features, batch_input_shape=(batch, timesteps, features), return_sequences=True))
        m.add(Dense(features))
    opt = Adam(learning_rate=lr)
    m.compile(loss="mean_squared_error", optimizer=opt)
    return m

def fit_model(model, trainX, trainY, num_epochs=2000, batch_size=2):
    for i in range(10):
        try:
            model.fit(trainX, trainY, epochs=num_epochs, batch_size=batch_size, verbose=2, shuffle=False)
            model.reset_states()
        except:
            pass
    return model

def model_predict(model, trainX, testX, trainY, testY, SC=MinMaxScaler, batch=2):
    train_predict = model.predict(trainX, batch_size=batch)
    model.reset_states()
    test_predict = model.predict(testX, batch_size=batch)
    train_predict = SC.inverse_transform(train_predict)
    trainY = SC.inverse_transform(trainY)
    test_predict = SC.inverse_transform(test_predict)
    testY = SC.inverse_transform(testY)
    return train_predict, trainY, test_predict, testY

def graph(training_data, train_predict, test_predict, ticker_name, SC, open_vals):
    offset = len(test_predict)
    train_predict_plot = np.empty_like(training_data[:, -1])
    train_predict_plot[:] = np.nan
    train_predict_plot[:len(train_predict)] = train_predict[:, -1]

    test_predict_plot = np.empty_like(training_data[:, -1])
    test_predict_plot[:] = np.nan
    test_predict_plot[len(train_predict)+1:len(training_data)-3] = test_predict[:, -1]

    fig = plt.figure()
    plt.plot(open_vals)
    plt.plot(train_predict_plot[:])
    plt.plot(test_predict_plot[:])
    fig.suptitle("Training and Testing datasets ontop of Actual Opening Prices")
    plt.ylabel("Opening Price ($)")
    plt.xlabel("Trading Days")
    fig.savefig(f"C:\\Users\\anton\\Documents\\antoniouaa\\msc_thesis\\assets\\{ticker_name}.jpg")
    plt.close()
    
tickers = fetch_tickers(TICKER_DIR)

for tick in tickers:
    ticker_name = tick[:-4]
    batch_size, timesteps, features = 2, 2, 2
    num_epochs, learning_rate = 1, 0.01
    train_size = 0.8

    training = pd.read_csv(f"{TICKER_DIR}\\{tick}", header=0)
    training = training.tail(160).iloc[:, 0:4:2].values
    open_vals = training[:, -1]

    scaler = MinMaxScaler()
    dataX, dataY, trainX, trainY, testX, testY = train_test_split(training, SC=scaler)
    model = build_model("gru", lr=learning_rate, batch=batch_size, timesteps=timesteps, features=features)
    model = fit_model(model, trainX, trainY, num_epochs=num_epochs, batch_size=batch_size)
    train_predict, trainY, test_predict, testY = model_predict(model, trainX, testX, trainY, testY, SC=scaler, batch=batch_size)
    graph(training, train_predict, test_predict, ticker_name, scaler, open_vals)
    keras.backend.clear_session()