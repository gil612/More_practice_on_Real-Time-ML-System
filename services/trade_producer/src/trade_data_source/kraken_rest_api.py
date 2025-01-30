from typing import List
import requests


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

        # parse string into dict
        import json
        data = json.loads(response.text)
        breakpoint()


    def is_done(self) -> bool:
        return False
