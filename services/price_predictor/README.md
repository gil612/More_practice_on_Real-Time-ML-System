some notes on the price predictor:

- It consumes OHLCV from the `ohlcv` topic
- It produces predictions to the `predictions` topic

Currently with 505 rows in the `ohlcv` topic, test size is 30% of data, target price is would using the shift method, current 5 minutes.

Mean absolute error is 286.5

Update:

Increased the dataset to 2580 rows, test size is 30% of data, target price is would using the shift method, current 5 minutes.

Mean absolute error is 156.87

Update 2:

Increased the dataset to 9640 rows, test size is 30% of data, target price is would using the shift method, current 5 minutes.

Mean absolute error is 90.3


Model training:
sklearn website: https://scikit-learn.org/
When we deal with tabular data?, the model that we can use is:
- XGBoost - hyperparameter tuning cab get terribly slow
- LightGBM - not as powerful as XGBoost but its better computationally
- CatBoost - Categorical values. good for non-linear relationships.

I need to understand the results of the model.

# All suported Indicators and Functions
https://ta-lib.github.io/ta-lib-python/funcs.html




- 