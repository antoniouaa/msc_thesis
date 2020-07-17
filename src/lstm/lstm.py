import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
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

def build_model(loss="mean_squared_error", lr=0.01, batch=2, timesteps=2, features=2):
    m = Sequential()
    m.add(LSTM(features, batch_input_shape=(batch, timesteps, features), stateful=True, return_sequences=True))
    m.add(LSTM(features, batch_input_shape=(batch, timesteps, features), stateful=True))
    m.add(Dense(features))
    opt = Adam(learning_rate=lr)
    m.compile(loss="mean_squared_error", optimizer=opt, metrics=["accuracy"])
    return m

def fit_model(model, trainX, trainY, num_epochs=2000, batch_size=2):
    model.fit(trainX, trainY, epochs=num_epochs, batch_size=batch_size, verbose=2, shuffle=False)
    return model

def model_predict(model, trainX, testX, trainY, testY, SC=MinMaxScaler, batch=2):
    train_predict = model.predict(trainX, batch_size=batch)
    test_predict = model.predict(testX, batch_size=batch)
    train_predict = SC.inverse_transform(train_predict)
    test_predict = SC.inverse_transform(test_predict)
    trainY = SC.inverse_transform(trainY)
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
    # fig.savefig(f"C:\\Users\\anton\\Documents\\antoniouaa\\msc_thesis\\assets\\{ticker_name}.jpg")
    plt.close()


if __name__ == "__main__":
    tickers = fetch_tickers(TICKER_DIR)

    for tick in tickers:
        ticker_name = tick.split(".")[0]
        batch_size, timesteps, features = 1, 10, 10
        num_epochs, learning_rate = 1, 0.01
        train_size = 0.8

        dataset = pd.read_csv(f"{TICKER_DIR}\\{tick}", header=0)
        X = dataset.tail(160).values[:, 1:]
        y = dataset.tail(160).values[:, 0]

        scaler = MinMaxScaler()
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
        model = build_model("gru", lr=learning_rate, batch=batch_size, timesteps=timesteps, features=features)
        print(X_train.shape)
        print(y_train.shape)
        X_train = X_train.reshape(1, 10, X_train.shape[1])
        y_train = y_train.reshape(y_train.shape[0], 1)
        model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=2, shuffle=False)
        train_predict, y_train, test_predict, y_test = model_predict(model, X_train, X_test, y_train, y_test, SC=scaler, batch=batch_size)
        graph(training, train_predict, test_predict, ticker_name, scaler, open_vals)
        keras.backend.clear_session()