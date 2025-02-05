Some notes on the trade to OHLC service:

- It consumes trades from the `trades` topic
- It produces OHLCV to the `ohlcv` topic
- It uses a tumbling window of 60 seconds to aggregate the trades into OHLCV

The concept of tumbling windows is that it will group the trades into buckets of time, and then apply the function to each bucket.


