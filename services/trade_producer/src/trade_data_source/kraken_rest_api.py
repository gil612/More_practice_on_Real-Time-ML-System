from typing import List, Tuple, Optional, Dict
import requests
from loguru import logger
import json
from trade_data_source.trade import Trade
from pathlib import Path

class KrakenRestAPI:

    URL = "https://api.kraken.com/0/public/Trades?pair={product_id}&since={since_sec}"

    def __init__(
        self,
        product_ids_list: List[str],
        last_n_days: int,
        cache_dir: Optional[str] = None,
        n_threads: Optional[int] = 1,
    ) -> None:
        """Basic initialization of the Kraken Rest API

        Args:
            product_ids_list: List[str]: The product IDs to fetch trades for.
            last_n_days: int: The number of days to fetch trades for.
            cache_dir: Optional[str]: Directory to cache trade data
            n_threads: Optional[int]: Number of threads to use (not used in this class but needed for interface compatibility)
        """
        self.product_ids_list = product_ids_list
        self.from_ms = None
        self.to_ms = None
        self.last_n_days = last_n_days
        self._is_done = False  # Initialize _is_done attribute

        self.to_ms, self.from_ms = self._init_from_to_ms(last_n_days)

        logger.info(f"Initialized KrakenRestAPI with from_ms: {self.from_ms} and to_ms: {self.to_ms}")

        self.last_trade_ms = self.from_ms

        

        # cache_dir is the directory where we will store the historical data to speed up
        # service restarts
        self.use_cache = False
        if cache_dir is not None:
            self.cache = CachedTradeData(cache_dir)
            self.use_cache = True


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

    def get_trades(self) -> List[Trade]:
        """
        Fetches a batch of trades from the Kraken REST API and returns them as a list of Trade objects.

        Returns:
            List[Trade]: A list of Trade objects representing the trades.
        """
        payload = {}
        headers = {'Accept': 'application/json'}

        since_sec = self.last_trade_ms // 1000
        url = self.URL.format(product_id=self.product_ids_list[0], since_sec=since_sec)
        
        logger.debug(f"Requesting trades from URL: {url}")
        response = requests.request("GET", url, headers=headers, data=payload)
        
        # Log response details
        logger.debug(f"Response status code: {response.status_code}")
        logger.debug(f"Response headers: {response.headers}")
        logger.debug(f"Response text: {response.text[:500]}...")  # First 500 chars
        
        # Check response status
        if response.status_code != 200:
            logger.error(f"Error from Kraken API: {response.text}")
            return []

        try:
            data = json.loads(response.text)
            
            # Check for Kraken API errors
            if 'error' in data and data['error']:
                logger.error(f"Kraken API error: {data['error']}")
                return []

            # It can happen that we get an error response from the Kraken REST API
            trades = []
            for trade in data['result'][self.product_ids_list[0]]:
                trade_obj = Trade(
                    product_id=self.product_ids_list[0],
                    price=float(trade[0]),
                    quantity=float(trade[1]),
                    timestamp_ms=int(float(trade[2]) * 1000)
                )
                trades.append(trade_obj)

            # filter out trades that are after the end timestamp
            filtered_trades = []
            for trade in trades:
                if trade.timestamp_ms <= self.to_ms:
                    filtered_trades.append(trade)
            trades = filtered_trades

            last_ts_in_ns = int(data['result']['last'])
            self.last_trade_ms = last_ts_in_ns // 1_000_000
            self._is_done = self.last_trade_ms >= self.to_ms

            logger.debug(f'Fetched {len(trades)} trades')
            logger.debug(f'Last trade timestamp: {self.last_trade_ms}')

            return trades
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            return []
        except KeyError as e:
            logger.error(f"Missing key in response: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return []


    def is_done(self) -> bool:
        return self._is_done
    

    
class CachedTradeData:
    """
    A class to handle the caching of trade data fetched from the Kraken REST API.
    """

    def __init__(self, cache_dir: str) -> None:
        self.cache_dir = Path(cache_dir)

        if not self.cache_dir.exists():
            # create the cache directory if it does not exist
            self.cache_dir.mkdir(parents=True)

    def read(self, url: str) -> List[Trade]:
        """
        Reads from the cache the trade data for the given url
        """
        file_path = self._get_file_path(url)

        if file_path.exists():
            # read the data from the parquet file
            import pandas as pd

            data = pd.read_parquet(file_path)
            # transform the data to a list of Trade objects
            return [Trade(**trade) for trade in data.to_dict(orient='records')]

        return []

    def write(self, url: str, trades: List[Trade]) -> None:
        """
        Saves the given trades to a parquet file in the cache directory.
        """
        if not trades:
            return

        # transform the trades to a pandas DataFrame
        import pandas as pd

        data = pd.DataFrame([trade.model_dump() for trade in trades])

        # write the DataFrame to a parquet file
        file_path = self._get_file_path(url)
        data.to_parquet(file_path)

    def has(self, url: str) -> bool:
        """
        Returns True if the cache has the trade data for the given url, False otherwise.
        """
        file_path = self._get_file_path(url)
        return file_path.exists()

    def _get_file_path(self, url: str) -> str:
        """
        Returns the file path where the trade data for the given url is (or will be) stored.
        """
        # use the given url to generate a unique file name in a deterministic way
        import hashlib

        url_hash = hashlib.md5(url.encode()).hexdigest()
        return self.cache_dir / f'{url_hash}.parquet'
        # return self.cache_dir / f'{product_id.replace ("/","-")}_{from_ms}.parquet'

class KrakenRestAPIMultipleProducts:

    def __init__(
            self,
            product_ids_list: List[str],
            last_n_days: int,
            n_threads: Optional[int] = 1,
            cache_dir: Optional[str] = None,

    ) -> None:
        self.product_ids_list = product_ids_list

        self.kraken_apis = [
            KrakenRestAPI(product_ids_list=[product_id], last_n_days=last_n_days, n_threads=n_threads, cache_dir=cache_dir)
            for product_id in product_ids_list
            ]
        
        self.n_threads = n_threads
        
    def get_trades_for_one_product(self, kraken_api: KrakenRestAPI) -> List[Trade]:
        """
        Gets trades for a single product using the given KrakenRestAPI instance.
        Used for parallel processing.

        Args:
            kraken_api (KrakenRestAPI): The API instance to use for fetching trades

        Returns:
            List[Trade]: List of trades for the product
        """
        if kraken_api.is_done():
            return []
        return kraken_api.get_trades()

    def get_trades(self) -> List[Trade]:
        """
        Gets trade data from each kraken_api in self.kraken_apis and returns a list
        with all trades from all kraken_apis.

        Returns:
            List[Trade]: A list of trades from all product_ids
        """
        if self.n_threads == 1:
            # Sequential version
            trades: List[Trade] = []
            for kraken_api in self.kraken_apis:
                if not kraken_api.is_done():
                    trades.extend(kraken_api.get_trades())
            return trades
        else:
            # Parallel version
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
                # Get trades for each product in parallel
                all_trades = list(executor.map(self.get_trades_for_one_product, self.kraken_apis))
                # Flatten the list of lists into a single list
                return [trade for sublist in all_trades for trade in sublist]
    
    def is_done(self) -> bool:
        """
        Checks if all kraken_apis are done fetching historical data.

        Args:
            None

        Returns:
            bool: True if all kraken_apis are done fetching historical data, False otherwise.
        """
        for kraken_api in self.kraken_apis: 
            if not kraken_api.is_done():
                return False
        return True
