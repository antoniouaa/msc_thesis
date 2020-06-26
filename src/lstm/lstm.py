import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.lstm1 = nn.LSTMCell(1, 12)
        self.lstm2 = nn.LSTMCell(12, 12)
        self.linear = nn.Linear(12, 1)

    def forward(self, input, future=0):
        outputs = []
        h_t = torch.zeros(input.size(0), 12, dtype=torch.double)     
        c_t = torch.zeros(input.size(0), 12, dtype=torch.double)     
        h_t2 = torch.zeros(input.size(0), 12, dtype=torch.double)     
        c_t2 = torch.zeros(input.size(0), 12, dtype=torch.double)     

        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        for i in range(future):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs

def fetch_data(url):
    df = pd.read_csv(f"{TICKER_DIR}\\MA_AIZP.csv", header=0)
    df_filled = df.dropna()
    return df_filled.to_numpy()

def draw(yi, color):
    plt.plot(np.arange(input.size(1)), yi[:input.size(1)], color, linewidth = 2.0)
    plt.plot(np.arange(input.size(1), input.size(1) + future), yi[input.size(1):], color + ':', linewidth = 2.0)

TICKER_DIR = "C:\\Users\\anton\\Documents\\antoniouaa\\msc_thesis\\data\\tickers\\ticker_data\\_Rolling"

if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    data = fetch_data(TICKER_DIR)[14:, :]
    length = int(data.shape[0] * 0.8)
    input = torch.from_numpy(data[3:length, :-1])
    target = torch.from_numpy(data[3:length, 1:])
    test_input = torch.from_numpy(data[length:, :-1])
    test_target = torch.from_numpy(data[length:, 1:])

    seq = Sequence()
    seq.double()
    criterion = nn.MSELoss()

    optimizer = optim.LBFGS(seq.parameters(), lr=0.04)

    for i in range(15):
        print(f"STEP: {i}")
        def closure():
            optimizer.zero_grad()
            out = seq(input)
            loss = criterion(out, target)
            print(f"Loss: {loss.item()}")
            loss.backward()
            return loss
        optimizer.step(closure)
        
        with torch.no_grad():
            future = 10
            pred = seq(test_input, future=future)
            loss = criterion(pred[:, :-future], test_target)
            print(F"Test Loss: {loss.item()}")
            y = pred.detach().numpy()

        plt.figure(figsize=(30, 10))
        plt.title("Predicting future values for time sequence\n(Dash is predicted)", fontsize=30)
        plt.xlabel("time", fontsize=20)
        plt.ylabel("Values", fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        draw(y[0], "r")
        draw(y[1], "g")
        draw(y[2], "b")
        #plt.savefig("prediction.pdf")
        #plt.show()
        plt.close()