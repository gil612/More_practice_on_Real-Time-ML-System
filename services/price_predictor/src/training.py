from src.config import HopsworksConfig

def train_model(
        feature_view_name: str,
        feature_view_version: int,
        feature_group_name: str,
        feature_group_version: int,
        ohlc_window_sec: int,
        product_id: str,
        last_n_days: int,
        hopsworks_config: HopsworksConfig,

):
    """
    Reads features from the Feature store
    Trains a predictive model
    Saves the model to the model registry

    """
    # Load feature data from the Feature Store
    from src.ohlc_data_reader import OhlcDataReader

    ohlc_data_reader = OhlcDataReader(
        ohlc_window_sec=ohlc_window_sec,
        hopsworks_config=hopsworks_config,
        feature_view_name=feature_view_name,
        feature_view_version=feature_view_version,
        feature_group_name=feature_group_name,
        feature_group_version=feature_group_version,

    )

    ohlc_data_reader.read_from_offline_store(
        product_id=product_id,
        last_n_days=last_n_days,

    )

    breakpoint()

    # Build a model


    # push model to the model registry


if __name__ == "__main__":
    from config import config, hopsworks_config
    train_model(
        feature_view_name=config.feature_view_name,
        feature_view_version=config.feature_view_version,
        feature_group_name=config.feature_group_name,
        feature_group_version=config.feature_group_version,
        ohlc_window_sec=config.ohlc_window_sec,
        product_id=config.product_id,
        last_n_days=config.last_n_days,
        hopsworks_config=hopsworks_config,



    )
