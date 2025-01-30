from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict
import os

class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra='ignore'
    )
    kafka_broker_address: str
    kafka_input_topic: str
    kafka_consumer_group: str
    feature_group_name: str
    feature_group_version: int
    feature_group_primary_keys: List[str]
    feature_group_event_time: str
    start_offline_materialization: bool

class HopsworksConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file="hopsworks_credentials.env")
    hopsworks_project_name: str
    hopsworks_api_key: str

config = Config()
hopsworks_config = HopsworksConfig()

# Remove these as they're not needed anymore since we're using the HopsworksConfig class
# HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME")
# HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")