import sys
import os

# Add the src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List

from loguru import logger
from quixstreams import Application

# Use absolute imports
from trade_data_source.kraken_websocket_api import (
    KrakenWebsocketAPI,
    Trade,
)


def produce_trades(
    kafka_broker_address: str,
    kafka_topic: str,
    product_ids_list: List[str],
    live_or_historical: str,
    last_n_days: int,
    n_threads: int = 10,
    cache_dir_historical_data: str = '/tmp/historical_trade_data',
):
    """
    Reads trades from the Kraken Websocket API and saves them in the given `kafka_topic`

    Args:
        kafka_broker_address (str): The address of the Kafka broker
        kafka_topic (str): The Kafka topic to save the trades
        product_id (str): The product id to get the trades from
        live_or_historical (str): The type of data source to use. 'live' or 'historical'
        last_n_days (int): The number of days to fetch historical data from.
        n_threads (int): Number of threads to use for historical data fetching
        cache_dir_historical_data (str): Directory to cache historical trade data

    Returns:
        None
    """    

    assert live_or_historical in ['live', 'historical'], f"Invalid value for live_or_historical: {live_or_historical}"
    # Create an Application instance with Kafka config using the provided address
    app = Application(broker_address=kafka_broker_address)

    # Define a topic "my_topic" with JSON serialization
    topic = app.topic(name=kafka_topic, value_serializer='json')

    # create a kraken api object,
    if live_or_historical == 'live':
        kraken_api = KrakenWebsocketAPI(product_ids_list=product_ids_list)
    else:
        # Use absolute import here too
        from trade_data_source.kraken_rest_api import KrakenRestAPIMultipleProducts
        kraken_api = KrakenRestAPIMultipleProducts(
            product_ids_list=product_ids_list,
            last_n_days=last_n_days,
            n_threads=n_threads,
            cache_dir=cache_dir_historical_data,
        )


    # Create a Producer instance
    with app.get_producer() as producer:

        while True:

            # Check if we are done fetching historical data
            if kraken_api.is_done():
                logger.info("Done fetching historical data")
                break

            trades: List[Trade] = kraken_api.get_trades()

          
            
            for trade in trades:
                # Convert Trade object to dictionary before serializing
                message = topic.serialize(key=trade.product_id, value=trade.to_dict())
                producer.produce(topic=topic.name, value=message.value, key=message.key)
        


                logger.debug(f"Pushed trade to Kafka: {trade}")




if __name__ == "__main__":
    from config import config  # Update this import too

    from trade_data_source.kraken_websocket_api import KrakenWebsocketAPI
    kraken_api = KrakenWebsocketAPI(product_ids_list=config.product_ids_list)

    
    produce_trades(
        kafka_broker_address=config.kafka_broker_address,
        kafka_topic=config.kafka_topic,
        product_ids_list=config.product_ids_list,

        # extra parameters i need when running the trade_producer against historical data from Kraken REST API
        live_or_historical=config.live_or_historical,
        last_n_days=config.last_n_days,
        n_threads=config.n_threads,
        cache_dir_historical_data=config.cache_dir_historical_data,
    )