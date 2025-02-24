from typing import Optional
import hashlib
from comet_ml import Experiment
from loguru import logger
from sklearn.metrics import mean_absolute_error

from config import CometConfig, HopsworksConfig
from feature_engineering import add_technical_indicators

def train_model(
    comet_config: CometConfig,
    hopsworks_config: HopsworksConfig,
    feature_view_name: str,
    feature_view_version: int,
    feature_group_name: str,
    feature_group_version: int,
    ohlc_window_sec: int,
    product_id: str,
    last_n_days: int,
    forecast_steps: int,
    perc_test_data: Optional[float] = 0.3,
    n_search_trials: Optional[int] = 0,
    n_splits: Optional[int] = 3,
):
    """
    Reads features from the Feature Store
    Trains a predictive model,
    Saves the model to the model registry

    Args:
        comet_config: CometConfig
            Configuration object for Comet.
        hopsworks_config: HopsworksConfig
            Configuration object for Hopsworks.
        feature_view_name: str
            Name of the feature view to read data from.
        feature_view_version: int
            Version of the feature view to read data from.
        feature_group_name: str
            Name of the feature group to read data from.
        feature_group_version: int
            Version of the feature group to read data from.
        ohlc_window_sec: int
            Time window in seconds for OHLC (Open, High, Low, Close) data.
        product_id: str
            Identifier for the product to predict prices for.
        last_n_days: int
            Number of past days to consider for training the model.
        forecast_steps: int
            Number of steps to forecast into the future.
        perc_test_data: float
            Percentage of the data to use for testing.
        n_search_trials: int
            Number of search trials for the XGBoost model.
        n_splits: int
            Number of splits for the XGBoost model.

    Returns:
        None
    """
    # create a comet experiment
    experiment = Experiment(
        api_key=comet_config.comet_api_key,
        project_name=comet_config.comet_project_name,
    )
    experiment.log_parameter("last_n_days", last_n_days)
    experiment.log_parameter("forecast_steps", forecast_steps)
    experiment.log_parameter("n_search_trials", n_search_trials)
    experiment.log_parameter("n_splits", n_splits)
    
    # Load feature data from the Feature Store
    from ohlc_data_reader import OhlcDataReader

    ohlc_data_reader = OhlcDataReader(
        ohlc_window_sec=ohlc_window_sec,
        hopsworks_config=hopsworks_config,
        feature_view_name=feature_view_name,
        feature_view_version=feature_view_version,
        feature_group_name=feature_group_name,
        feature_group_version=feature_group_version,
    )

    # read the sorted data from the offline store
    # data is sorted by timestamp_ms
    ohlc_data = ohlc_data_reader.read_from_offline_store(
        product_id=product_id,
        last_n_days=last_n_days
    )
    logger.debug(f"Read {len(ohlc_data)} rows from the offline store")
    experiment.log_parameter("n_raw_feature_rows", len(ohlc_data))

    # log a hash of the dataset to comet
    
    dataset_hash = hashlib.sha256(ohlc_data.to_numpy().tobytes()).hexdigest()
    experiment.log_parameter("ohlc_data_hash", dataset_hash)

    # split the data into training and testing
    logger.debug(f"Splitting the data into training and testing")
    test_size = int(len(ohlc_data) * perc_test_data)
    train_df = ohlc_data[:-test_size]
    test_df = ohlc_data[-test_size:]
    logger.debug(f"Training data: {len(train_df)} rows")
    logger.debug(f"Testing data: {len(test_df)} rows")
    experiment.log_parameter("n_train_rows_before_dropna", len(train_df))
    experiment.log_parameter("n_test_rows_before_dropna", len(test_df))

    # add a column with the target price we want our model to predict
    # for both training data and testing data
    train_df['target_price'] = train_df['close'].shift(-forecast_steps)
    test_df['target_price'] = test_df['close'].shift(-forecast_steps)
    logger.debug(f"Added target price column to training and testing data")

    # remove rows with NaN values
    train_df = train_df.dropna()
    test_df = test_df.dropna()
    logger.debug(f"Removed rows with NaN values")
    logger.debug(f"Training data after removing NaN values: {len(train_df)} rows")
    logger.debug(f"Testing data after removing NaN values: {len(test_df)} rows")
    experiment.log_parameter("n_train_rows_after_dropna", len(train_df))
    experiment.log_parameter("n_test_rows_after_dropna", len(test_df))

    # split the data into features and target
    X_train = train_df.drop(columns=['target_price'])
    y_train = train_df['target_price']
    X_test = test_df.drop(columns=['target_price'])
    y_test = test_df['target_price']
    logger.debug(f"Split the data into features and target")
    
    # keep only the features that are needed for the model
    # TODO: think if I want these hard-coded here or if I want to provide them as parameters
    # to my training function
    X_train = X_train[['open', 'high', 'low', 'close', 'volume']]
    X_test = X_test[['open', 'high', 'low', 'close', 'volume']]
    
    # add technical indicators to the features
    X_train = add_technical_indicators(X_train)
    X_test = add_technical_indicators(X_test)
    logger.debug(f"Added technical indicators to the features")
    logger.debug(f"X_train: {X_train.columns}")
    logger.debug(f"X_test: {X_test.columns}")
    experiment.log_parameter('features', X_train.columns.tolist())

    # Dropping rows with NaN values
    # extract row indices from X_train where any of the technical indicators is not NaN
    nan_rows_train = X_train.isna().any(axis=1)
    # count number of NaN rows
    logger.debug(f"Number of NaN rows in X_train: {nan_rows_train.sum()}")
    # keep only the rows where the technical indicators are not NaN
    X_train = X_train.loc[~nan_rows_train]
    y_train = y_train.loc[~nan_rows_train]

    # extract row indices from X_test where any of the technical indicators is not NaN
    nan_rows_test = X_test.isna().any(axis=1)
    # count number of NaN rows
    logger.debug(f"Number of NaN rows in X_test: {nan_rows_test.sum()}")
    # keep only the rows where the technical indicators are not NaN
    X_test = X_test.loc[~nan_rows_test]
    y_test = y_test.loc[~nan_rows_test]

    # log the number of NaN rows and the percentage of dropped rows
    experiment.log_parameter("n_nan_rows_train", nan_rows_train.sum())
    experiment.log_parameter("n_nan_rows_test", nan_rows_test.sum())
    experiment.log_parameter("perc_dropped_rows_train", nan_rows_train.sum() / len(X_train) * 100)
    experiment.log_parameter("perc_dropped_rows_test", nan_rows_test.sum() / len(X_test) * 100)

    # log dimensions of the features and target
    logger.debug(f"X_train: {X_train.shape}")
    logger.debug(f"y_train: {y_train.shape}")
    logger.debug(f"X_test: {X_test.shape}")
    logger.debug(f"y_test: {y_test.shape}")
    
    # log the shapes to Comet ML
    experiment.log_parameter("X_train_shape", X_train.shape)
    experiment.log_parameter("y_train_shape", y_train.shape)
    experiment.log_parameter("X_test_shape", X_test.shape)
    experiment.log_parameter("y_test_shape", y_test.shape)

    # breakpoint()

    # build a model
    from models.current_price_baseline import CurrentPriceBaseline
    model = CurrentPriceBaseline()
    model.fit(X_train, y_train)
    logger.debug(f"Model built")

    # evaluate the model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    logger.debug(f"Mean absolute error of CurrentPriceBaseline: {mae}")
    experiment.log_metric("mae_CurrentPriceBaseline", mae)

 # train an XGBoost model
    from xgboost import XGBRegressor
    from models.xgboost_model import XGBoostModel

    
    xgb_model = XGBoostModel()
    xgb_model.fit(X_train, y_train, n_search_trials=n_search_trials, n_splits=n_splits)
    y_pred = xgb_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    logger.debug(f"Mean absolute error: {mae}")
    experiment.log_metric("mae", mae)



    # compute mae on the training data for debugging purposes
    y_train_pred = model.predict(X_train)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    logger.debug(f"Mean absolute error on the training data of CurrentPriceBaseline: {mae_train}")
    experiment.log_metric("mae_training_CurrentPriceBaseline", mae_train)

   
    
    # compute mae on the training data for debugging purposes
    y_train_pred = xgb_model.predict(X_train)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    logger.debug(f"Mean absolute error on the training data: {mae_train}")
    experiment.log_metric("mae_training", mae_train)

    # push model to the model registry
    # TODO: implement this

    experiment.end()

if __name__ == '__main__':

    from config import config, hopsworks_config, comet_config

    train_model(
        comet_config=comet_config,
        hopsworks_config=hopsworks_config,
        feature_view_name=config.feature_view_name,
        feature_view_version=config.feature_view_version,
        feature_group_name=config.feature_group_name,
        feature_group_version=config.feature_group_version,
        ohlc_window_sec=config.ohlc_window_sec,
        product_id=config.product_id,
        last_n_days=config.last_n_days,
        forecast_steps=config.forecast_steps,
        n_search_trials=config.n_search_trials,
        n_splits=config.n_splits,
    )