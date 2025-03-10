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


we can use an alternative library called ta which is a pure Python implementation of technical analysis indicators and doesn't require C dependencies.

https://technical-analysis-library-in-python.readthedocs.io/en/latest/ta.html

My Comet_ml project experiments:
https://www.comet.com/gil612/price-predictor/view/new/panels


Build inference pipeline:
- Create Predictor Service:
    - recieves info from OHLC Feature Group in Feature Store
    - recieves info from ML model in Model metadata
    - generates the prediction to Predcitions FG in FS

- Create CDC (change data capture) Pattern
    Objective is to use functions to deal with a database like (Insert, Update, Delete)
    Example: If there are changes in the row, send a notification with what are changes, aend them to a kafka topic and I will pick that in process wherever need.
    Target system can be Database, Cache, Search Index, DAta Warehouse, Data Lake
    Problem: Hopsworks does not support CDC pattern
- Create REST API:
    - recieves info from Predictions Feature group in Feature Store



- 