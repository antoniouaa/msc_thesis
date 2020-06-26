import pandas as pd
import numpy as np
import torch

TICKER_DIR = "C:\\Users\\anton\\Documents\\antoniouaa\\msc_thesis\\data\\tickers\\ticker_data\\_Rolling"

df = pd.read_csv(f"{TICKER_DIR}\\MA_AIZP.csv", header=0)

df_filled = df.dropna()

x = df_filled.to_numpy()
# print(x)

np.random.seed(2)

T = 20
L = 1000
N = 100

x = np.empty((N, L), 'int64')
x[:] = np.array(range(L)) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)
data = np.sin(x / 1.0 / T).astype('float64')
print(data[:, :])