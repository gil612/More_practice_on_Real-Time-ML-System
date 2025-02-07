from config import HopsworksConfig, config, hopsworks_config, comet_config
from ohlc_data_reader import OhlcDataReader
from loguru import logger
from typing import Optional
from sklearn.metrics import mean_absolute_error
from comet_ml import Experiment

def train_model(
        comet_config: comet_config,
        hopsworks_config: HopsworksConfig,
        feature_view_name: str,
        feature_view_version: int,
        feature_group_name: str,
        feature_group_version: int,
        ohlc_window_sec: int,
        product_id: str,
        last_n_days: int,
        forecast_steps: int,
        perc_test_data: Optional[float] = 0.3
):
    """
    Reads features from the Feature store
    Trains a predictive model
    Saves the model to the model registry
    """

    experiment = Experiment(
        api_key=comet_config.comet_api_key,
        project_name=comet_config.comet_project_name,
    )


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
        last_n_days=last_n_days,
    )
    logger.debug(f"Read {len(ohlc_data)} rows from the offline store")
    experiment.log_parameters({
        "n_raw_feature_rows": len(ohlc_data)
    })
    # split the data into train and test
    logger.debug(f"Splitting the data into train and test")
    test_size = int(len(ohlc_data) * perc_test_data)
    train_df = ohlc_data[:-test_size]
    test_df = ohlc_data[-test_size:]
    logger.debug(f"Training data: {len(train_df)} rows")
    logger.debug(f"Testing data: {len(test_df)} rows")
    experiment.log_parameters({
        "n_training_rows_before_dropna": len(train_df),
        "n_testing_rows_before_dropna": len(test_df)
    })



    # add a column with the target price we want our model to predict
    train_df['target_price'] = train_df['close'].shift(-forecast_steps)
    test_df['target_price'] = test_df['close'].shift(-forecast_steps)
    logger.debug("Added target price column to the training and testing data")

    # remove rows with NaN values
    train_df = train_df.dropna()
    test_df = test_df.dropna()
    logger.debug("Removed rows with NaN values")
    logger.debug(f"Training data after removing NaN values: {len(train_df)} rows")
    logger.debug(f"Testing data after removing NaN values: {len(test_df)} rows")
    experiment.log_parameters({
        "n_training_rows_after_dropna": len(train_df),
        "n_testing_rows_after_dropna": len(test_df)
    })



    # split the data into X and y
    X_train = train_df.drop(columns=['target_price'])
    y_train = train_df['target_price']
    X_test = test_df.drop(columns=['target_price'])
    y_test = test_df['target_price']
    logger.debug("Split the data into X and y")
    # log dimensions of the features and target
    logger.debug(f"X_train: {X_train.shape}")
    logger.debug(f"y_train: {y_train.shape}")
    logger.debug(f"X_test: {X_test.shape}")
    logger.debug(f"y_test: {y_test.shape}")
    # log the shape of the features and target
    experiment.log_parameters({
        "X_train_shape": X_train.shape,
        "y_train_shape": y_train.shape,
        "X_test_shape": X_test.shape,
        "y_test_shape": y_test.shape
    })


    # build a model
    from src.models.current_price_baseline import CurrentPriceBaseline
    model = CurrentPriceBaseline()
    model.fit(X_train, y_train)
    logger.debug("Model built")
    
    # evaluate the model
    y_pred = model.predict(X_test)

    # push model to the model registry
    from src.models.current_price_baseline import CurrentPriceBaseline
    model = CurrentPriceBaseline()
    model.fit(X_train, y_train)
    logger.debug("Model built")
    
    # evaluate the model
    y_pred = model.predict(X_test)

    
    mae = mean_absolute_error(y_test, y_pred)
    logger.debug(f"Mean Absolute Error: {mae}")
    experiment.log_metric("mae", mae)



    # add a column with the target price we want to predict
    ohlc_data['target_price'] = ohlc_data['close'].shift(-forecast_steps)

    # Build a model

    # push model to the model registry

if __name__ == "__main__":
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
    )