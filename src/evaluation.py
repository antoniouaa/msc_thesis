import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


class Evaluator:

    def __init__(self, ideal, preds):
        self.ideal = ideal
        self.preds = preds

    def rmse(self):
        return mean_squared_error(self.ideal, self.preds, squared=False)

    def mae(self):
        return mean_absolute_error(self.ideal, self.preds)

    def __repr__(self):
        return f"RMSE = [{self.rmse()}]\nMAE = [{self.mae()}]"


if __name__ == "__main__":
    print("~~~~~TESTING EVALUATION METRICS~~~~~")
    ideal = np.array([0.00, 0.166, 0.333])
    preds = np.array([0.00, 0.254, 0.998])

    evaluator = Evaluator(ideal, preds)

    print("Ideal values\n", *ideal)
    print("Predictions\n", *preds)
    print(evaluator)
    