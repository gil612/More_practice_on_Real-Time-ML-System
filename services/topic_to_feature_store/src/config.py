from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict

class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")
    kafka_broker_address: str
    kafka_input_topic: str
    kafka_consumer_group: str


    feature_group_name: str
    feature_group_version: int
    feature_group_primary_key: List[str]
    feature_group_event_time: str

class HopsworksConfig(BaseSettings):
        model_config = SettingsConfigDict(env_file="hopsworks_credentials.env")
        hopsworks_project_name: str
        hopsworks_api_key: str

config = Config()
hopsworks_config = HopsworksConfig()