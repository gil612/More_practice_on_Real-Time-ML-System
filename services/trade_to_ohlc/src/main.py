from quixstreams import Application
from datetime import timedelta
from loguru import logger

def init_ohlcv_candle(trade:dict):
    """
    Retruns the initial state of the OHLCV candle when the first 'trade' in that window is received.
    """
    return {
        "open": trade['price'], 
        "high": trade['price'],
        "low": trade['price'],
        "close": trade['price'],
        "volume": trade['quantity'],
        "timestamp": None
    }

def update_ohlcv_candle(candle:dict, trade:dict):
    """
    Updates the OHLCV candle with the latest trade.
    """
    candle['high'] = max(candle['high'], trade['price'])
    candle['low'] = min(candle['low'], trade['price'])
    candle['close'] = trade['price']
    candle['volume'] += trade['quantity']
    # candle['timestamp'] = trade['timestamp']
    return candle

def transform_trade_to_ohlcv(
        kafka_broker_address: str,
        kafka_input_topic: str,
        kafka_output_topic: str,
        kafka_consumer_group: str,
        ohlcv_window_seconds: int
) -> None:
    """
    Args:
        kafka_broker_address: The address of the Kafka broker
        kafka_input_topic: The topic to consume trades from
        kafka_output_topic: The topic to produce OHLCV to
        kafka_consumer_group: The consumer group to use

    Returns:
        None
    """
    app = Application(
        broker_address=kafka_broker_address,
        consumer_group=kafka_consumer_group,
    )

    input_topic = app.topic(name=kafka_input_topic, value_deserializer="json")
    output_topic = app.topic(name=kafka_output_topic, value_serializer="json")

    # create a Quix Streams Dataframe
    sdf = app.dataframe(input_topic)

    sdf = (
        sdf.tumbling_window(duration_ms=timedelta(seconds=ohlcv_window_seconds))
        .reduce(reducer=update_ohlcv_candle, initializer=init_ohlcv_candle)
        .final()
    )
    

    # unpack the dictionary into separate columns
    sdf['open'] = sdf['value']['open']
    sdf['high'] = sdf['value']['high']
    sdf['low'] = sdf['value']['low']
    sdf['close'] = sdf['value']['close']
    sdf['volume'] = sdf['value']['volume']
    sdf['timestamp_ms'] = sdf['end']


    sdf = sdf[['timestamp_ms', 'open', 'high', 'low', 'close', 'volume']]


    # print the output to the console
    sdf.update(logger.debug)

    sdf = sdf.to_topic(output_topic)
    
    app.run()

if __name__ == "__main__":
    from config import config
    transform_trade_to_ohlcv(
        kafka_broker_address=config.kafka_broker_address,
        kafka_input_topic=config.kafka_input_topic,
        kafka_output_topic=config.kafka_output_topic,
        kafka_consumer_group=config.kafka_consumer_group,
        ohlcv_window_seconds=config.ohlcv_window_seconds
    )


