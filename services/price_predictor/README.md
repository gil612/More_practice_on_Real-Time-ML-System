some notes on the price predictor:

- It consumes OHLCV from the `ohlcv` topic
- It produces predictions to the `predictions` topic

Currently with 505 rows in the `ohlcv` topic, test size is 30% of data, target price is would using the shift method, current 5 minutes.

Mean absolute error is 286.5