from abc import ABC, abstractmethod
from typing import List

from .trade import Trade

class TradeSource(ABC):

    @abstractmethod
    def get_trades(self) -> List[Trade]:
        """
        Retrieve the trades from whatever source you connect to.
        """
        pass

    @abstractmethod
    def is_done(self) -> bool:
        """
        Returns True if there are no more trades to retrieve, False otherwise.
        """
        pass