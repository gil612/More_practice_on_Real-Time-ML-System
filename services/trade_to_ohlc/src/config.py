from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict



class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")
    kafka_broker_address: str
    kafka_input_topic: str
    kafka_output_topic: str
    kafka_consumer_group: str
    ohlcv_window_seconds: int

config = Config()