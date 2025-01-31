from typing import List
import requests
from loguru import logger
import json


class KrakenRestAPI:
    URL = "https://api.kraken.com/0/public/Trades"

    URL = "https://api.kraken.com/0/public/Trades?pair={product_id}&since={since_ms}"

    def __init__(
        self,
        product_ids_list: List[str],
        from_ms: int,
        to_ms: int,
    ) -> None:
        """Basic initialization of the Kraken Rest API

        Args:
            product_ids_list: List[str]: The product IDs to fetch trades for.
            from_ms: int: The timestamp of the earliest trade to fetch.
            to_ms: int: The timestamp of the latest trade to fetch.
        """
        self.product_ids_list = product_ids_list
        self.from_ms = from_ms
        self.to_ms = to_ms

        # we use it to check if we are done fetching the historical data
        # if the latest batch of trades we get exceeds this timestamp, we are done fetching historical data
        self._is_done = False

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
        url = self.URL.format(product_id=self.product_ids_list[0], since_ms=self.from_ms)
        response = requests.request("GET", url, headers=headers, data=payload)
        print(response.text)

        data = json.loads(response.text)
        if data['error'] != []:
            logger.error(f"Error fetching trades for {self.product_ids_list[0]}: {data['error']}")
            raise Exception(data['error'])

        trades = [
            {
                'price': float(trade[0]),
                'volume': float(trade[1]),
                'time': int(trade[2]),
                'product_id': self.product_ids_list[0],
            }
            for trade in data['result'][self.product_ids_list[0]]
        ]
        last_ts_in_ns = int(data['result']['last'])

        last_ts = last_ts_in_ns // 1_000_000
       
        if last_ts > self.to_ms:
            self._is_done = True

        logger.debug(f"Fetched {len(trades)} trades for {self.product_ids_list[0]}")
        logger.debug(f'Last trade timestamp: {trades[-1]["time"]}')

    

        return trades


    def is_done(self) -> bool:
        return self._is_done
