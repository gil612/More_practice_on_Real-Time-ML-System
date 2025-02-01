from typing import List, Tuple
import requests
from loguru import logger
import json


class KrakenRestAPI:

    URL = "https://api.kraken.com/0/public/Trades?pair={product_id}&since={since_sec}"

    def __init__(
        self,
        product_ids_list: List[str],
        last_n_days: int,
    ) -> None:
        """Basic initialization of the Kraken Rest API

        Args:
            product_ids_list: List[str]: The product IDs to fetch trades for.
            last_n_days: int: The number of days to fetch trades for.
        """
        self.product_ids_list = product_ids_list
        self.from_ms = None
        self.to_ms = None
        self.last_n_days = last_n_days

        self.to_ms, self.from_ms = self._init_from_to_ms(last_n_days)

        logger.info(f"Initialized KrakenRestAPI with from_ms: {self.from_ms} and to_ms: {self.to_ms}")

        self.last_trade_ms = self.from_ms

        

        # we use it to check if we are done fetching the historical data
        # if the latest batch of trades we get exceeds this timestamp, we are done fetching historical data
        self._is_done = False

      
        since_sec = self.last_trade_ms // 1000

    # To make the code more explicit, I'm using a static method to initialize the from and to timestamps in milliseconds
    @staticmethod
    def _init_from_to_ms(last_n_days: int) -> Tuple[int, int]:
        """Initialize the from and to timestamps in milliseconds"""
        from datetime import datetime, timezone

        today_date = datetime.now(timezone.utc).replace(
            hour=0, 
            minute=0, 
            second=0, 
            microsecond=0
        )

        to_ms = int(today_date.timestamp() * 1000)
        from_ms = to_ms - last_n_days * 24 * 60 * 60 * 1000
        return to_ms, from_ms

    def get_trades(self) -> List[dict]:
        """
        Fetches a batch of trades from the Kraken REST API and returns them as a list of dictionaries.

        Args:
            None

        Returns:
            List[Trade]: A list of dictionaries representing the trades.
        """
        payload = {}
        headers = {'Accept': 'application/json'}

        # replacing the placeholders in the URL with actual values for the first product id and since_ms
        since_sec = self.last_trade_ms // 1000
        url = self.URL.format(product_id=self.product_ids_list[0], since_sec=since_sec)
        response = requests.request("GET", url, headers=headers, data=payload)

        data = json.loads(response.text)
        

        trades = [
            {
                'price': float(trade[0]),
                'volume': float(trade[1]),
                'time': int(trade[2]),
                'product_id': self.product_ids_list[0],
            }
            for trade in data['result'][self.product_ids_list[0]]
        ]

        # filter out trades that are after the end timestamp
        trades = [trade for trade in trades if trade['time'] <= self.to_ms // 1000]

        last_ts_in_ns = int(data['result']['last'])

        self.last_trade_ms = last_ts_in_ns // 1_000_000
        self._is_done = self.last_trade_ms >= self.to_ms


        logger.debug(f'Fetched {len(trades)} trades')
        # log the last trade timestamp
        logger.debug(f'Last trade timestamp: {self.last_trade_ms}')

    

        return trades


    def is_done(self) -> bool:
        return self._is_done
