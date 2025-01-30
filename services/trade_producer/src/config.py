from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict



class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")
    kafka_broker_address: str
    kafka_topic: str
    product_ids: str

    @property
    def product_ids_list(self) -> List[str]:
        """Convert comma-separated string to list of strings"""
        return [x.strip() for x in self.product_ids.split(',')]
    
    live_or_historical: str = 'live'
    last_n_days: int = 7

config = Config()