import pandas as pd

class MovingAverageBaseline:
    """
    A simple baseline model that predicts the next price as the average of the last n prices
    """

    def __init__(self, window_size: int = 5):
        self.window_size = window_size

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits the model to the data.
        """
        pass

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predicts the next price as the average of the last n prices.
        """
        return X['close'].rolling(window=self.window_size).mean()
