from quixstreams import Application
from datetime import timedelta
from loguru import logger
from config import config

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
    broker_address: str,
    input_topic: str,
    output_topic: str,
    consumer_group: str,
    ohlcv_window_seconds: int,
):
    """
    Transforms trade data into OHLCV candles.
    """
    logger.info(f"Starting with input_topic={input_topic}, output_topic={output_topic}, consumer_group={consumer_group}")
    
    # Create a Quix Streams application
    app = Application(
        broker_address=broker_address,
        consumer_group=consumer_group,
        auto_offset_reset="earliest",  # Important for historical data processing
    )

    # Create Topic objects with explicit names
    input_topic = app.topic(input_topic, value_deserializer="json")
    output_topic = app.topic(output_topic, value_serializer="json")

    # Log the actual topic names being used
    logger.info(f"Using topics: input={input_topic.name}, output={output_topic.name}")

    # Create a Quix Streams DataFrame
    sdf = app.dataframe(input_topic)

    # Add debug logging for incoming trades
    sdf.update(lambda x: logger.debug(f"Received trade: {x}"))

    sdf = (
        sdf.tumbling_window(duration_ms=timedelta(seconds=ohlcv_window_seconds))
        .reduce(reducer=update_ohlcv_candle, initializer=init_ohlcv_candle)
        .final()
    )

    # Transform the window output into the format we want
    def transform_window_output(row):
        candle = row['value']  # Get the candle dictionary from the window
        return {
            'timestamp_ms': row['end'],  # Use window end time as timestamp
            'open': candle['open'],
            'high': candle['high'],
            'low': candle['low'],
            'close': candle['close'],
            'volume': candle['volume']
        }

    sdf = sdf.apply(transform_window_output)

    # Add debug logging for transformed output
    sdf.update(lambda x: logger.debug(f"Transformed output: {x}"))

    # push these message to the output topic
    sdf = sdf.to_topic(output_topic)
    
    app.run()  # Remove sdf argument

if __name__ == "__main__":
    logger.info("Starting trade_to_ohlc service...")
    logger.info(f"Using configuration: {config.model_dump()}")
    
    transform_trade_to_ohlcv(
        broker_address=config.kafka_broker_address,
        input_topic=config.kafka_input_topic,
        output_topic=config.kafka_output_topic,
        consumer_group=config.kafka_consumer_group,
        ohlcv_window_seconds=config.ohlcv_window_seconds,
    )

