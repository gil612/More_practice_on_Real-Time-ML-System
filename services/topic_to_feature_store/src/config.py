from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='allow'  # Allow extra fields
    )

    kafka_broker_address: str
    kafka_input_topic: str
    kafka_consumer_group: str
    feature_group_name: str
    feature_group_version: int
    feature_group_primary_keys: List[str]
    feature_group_event_time: str	
    start_offline_materialization: bool
    batch_size: Optional[int] = 1

class HopsworksConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file="credentials.env",
        env_file_encoding='utf-8',
        extra='allow'
    )

    hopsworks_project_name: str
    hopsworks_api_key: str


config = AppConfig()
hopsworks_config = HopsworksConfig()