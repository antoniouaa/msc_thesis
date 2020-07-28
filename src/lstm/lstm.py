import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam

TICKER_DIR = "C:\\Users\\anton\\Documents\\antoniouaa\\msc_thesis\\data\\tickers\\ticker_data"
ASSET_DIR = "C:\\Users\\anton\\Documents\\antoniouaa\\msc_thesis\\assets"

def fetch_tickers(path):
    tickers = [tick for tick in os.listdir(os.path.abspath(TICKER_DIR)) if os.path.isfile(os.path.join(TICKER_DIR, tick))]
    return tickers

def split_sequences(sequences, n_steps):
	X, y = [], []
	for i, _ in enumerate(sequences):
		end_ix = i + n_steps
		if end_ix > len(sequences):
			break
		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix-1, :]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

def build_model(loss="mae", lr=0.01, n_steps=5, n_features=4):
    m = Sequential()
    m.add(LSTM(100, activation="relu", return_sequences=True, input_shape=(n_steps, n_features)))
    m.add(LSTM(n_features, activation="relu"))
    m.add(Dense(n_features))
    opt = Adam(learning_rate=lr)
    m.compile(loss=loss, optimizer=opt, metrics=["mae", "mse"])
    return m

def evaluate_model(model, X_test, y_test):
    scores = model.evaluate(X_test, y_test, verbose=0)
    print(*[f"History Item [{item}] reports value of {value}" for value, item in zip(scores, model.metrics_names)], sep="\n")
    return scores

if __name__ == "__main__":
    np.random.seed(1)
    tickers = fetch_tickers(TICKER_DIR)
    dfs = [pd.read_csv(os.path.join(TICKER_DIR, tick)) for tick in tickers]
    
    num_epochs, learning_rate = 100, 0.01
    n_steps = 3

    for tick, df in zip(tickers, dfs):
        keras.backend.clear_session()

        ticker_name = tick.split(".")[0]

        assert df.shape == (1006, 13)

        target = df["close"].tail(1000)
        dataset = df[["high", "low", "open"]].tail(1000)
        high = dataset["high"].values.reshape(len(dataset), 1)
        low = dataset["low"].values.reshape(len(dataset), 1)
        open_ = dataset["open"].values.reshape(len(dataset), 1)
        close = target.values.reshape(len(dataset), 1)
        d = np.hstack((open_, high, low, close))

        X, y = split_sequences(d, n_steps)
        assert X.shape == (998, 3, 4)
        assert y.shape == (998, 4)
        n_features = X.shape[2]

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=False)

        model = build_model(n_steps=n_steps, n_features=n_features)
        model.fit(X_train, y_train, epochs=num_epochs, verbose=0)

        print(f"Evaluating Ticker {ticker_name}")
        scores = evaluate_model(model, X_test, y_test)

        actual = df["close"].tail(X_test.shape[0]).values
        yhat = model.predict(X_test, verbose=0)
        predictions = yhat[:, -1]
        fig = plt.figure()
        plt.title("Actual v Predicted prices")
        plt.ylabel("Closing Price")
        plt.xlabel("Trading Day")
        plt.plot(actual, label="actual")
        plt.plot(predictions, label="predictions")
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(ASSET_DIR, f"{ticker_name}.png"))

        del model