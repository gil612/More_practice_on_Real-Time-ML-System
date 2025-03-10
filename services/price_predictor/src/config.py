from pydantic_settings import BaseSettings, SettingsConfigDict

class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")
    feature_view_name: str = "ohlcv_feature_view_2"
    feature_view_version: int = 12
    feature_group_name: str 
    feature_group_version: int
    ohlc_window_sec: int
    product_id: str
    last_n_days: int
    forecast_steps: int
    n_search_trials: int
    n_splits: int
    last_n_minutes: int
    model_status: str


class HopsworksConfig(BaseSettings):
    hopsworks_project_name: str
    hopsworks_api_key: str
    class Config:
        env_file = "hopsworks_credentials.env"

class CometConfig(BaseSettings):
    comet_api_key: str
    comet_project_name: str
    comet_workspace: str
    class Config:
        env_file = "comet_credentials.env"
    



config = AppConfig()
hopsworks_config = HopsworksConfig()
comet_config = CometConfig()



