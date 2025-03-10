from pydantic import BaseModel
import json
import joblib

from src.hopsworks_api import push_value_to_feature_group
from src.config import comet_config, CometConfig
from src.model_registry import get_model_name
from src.price_predictor import PricePredictor


def predict(
    product_id: str,
    ohlc_window_sec: int,
    forecast_steps: int,
    model_status: str,
):
    """
    Loads the model from the model registry, and then enters a loop where it
    - connect to the feature store,
    - fetches the most recent OHLCV data,
    - predicts the price for the next 5 minutes, and
    - stores the prediction in an online feature group.

    """
    # we create a predictor object, that loads the model from the registry
    predictor = PricePredictor.from_model_registry(
        product_id=product_id,
        ohlc_window_sec=ohlc_window_sec,
        forecast_steps=forecast_steps,
        status=model_status,
    )

    # breakpoint()

    while True:

        prediction = predictor.predict()
        
        # # store the prediction in the online feature group
        # push_value_to_feature_group(
        #     value=prediction,
        #     feature_group_name=config.online_feature_group_name,
        #     feature_group_version=config.online_feature_group_version,
        #     feature_group_primary_keys=config.online_feature_group_primary_keys,
        #     feature_group_event_time=config.online_feature_group_event_time,
        #     start_offline_materialization=config.start_offline_materialization
        # )


if __name__ == "__main__":

    from src.config import config
    predict(
        product_id=config.product_id,
        ohlc_window_sec=config.ohlc_window_sec,
        forecast_steps=config.forecast_steps,
        model_status=config.model_status,
    )