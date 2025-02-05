
import pandas as pd

class CurrentPriceBaseline:

    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits the model to the data.
        """
        pass


    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predicts the next price as the current price.
        """
        return X['close']

