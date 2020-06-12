import os
import requests
import csv

BASE_PATH = os.path.dirname(__file__)

def fetch_credentials():
    FILENAME = "API_KEY.txt"
    FILEPATH = os.path.abspath(os.path.join(BASE_PATH, "..", FILENAME))

    credentials = {}
    with open(FILEPATH) as f:
        for line in f:
            words = line.split("=")
            credentials[words[0]] = words[1]
        return credentials

def read_tickers(filename="supported_tickers.csv"):
    FILEPATH = os.path.abspath(os.path.join(BASE_PATH, "../data/tickers", filename))
    with open(FILEPATH) as f:
        tickers = [row.split(",")[0] for row in f]
        return tickers[1:]

def make_request(ticker, start_date=None, end_date=None):
    URL = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices?startDate={start_date}&endDate={end_date}&token={credentials['TIINGO']}"
    print(URL)
    req = requests.get(URL)
    req_json = req.json()
    return req_json

def save_to_csv(data, filename="ticker_data.csv", headers=[]):
    FILEPATH = os.path.abspath(os.path.join(BASE_PATH, "../data/tickers/ticker_data", filename))
    with open(FILEPATH, "w", newline="") as datacsv:
        csvwriter = csv.writer(datacsv, delimiter=",")
        csvwriter.writerow(headers)
        for line in data:
            csvwriter.writerow(line.values())

if __name__ == "__main__":
    credentials = fetch_credentials()
    tickers = read_tickers()
    for ticker in tickers[0:len(tickers):200]:
        try:
            data = make_request(ticker, start_date="2018-01-01", end_date="2018-12-01")
            if data:
                headers = data[0].keys()
                save_to_csv(data, filename=f"{ticker}.csv", headers=headers)
        except:
            pass
