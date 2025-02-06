from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict
from loguru import logger
import os



class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding='utf-8', 
        case_sensitive=False,
        extra='allow',
        env_prefix=""  # No prefix for env vars
    )
    
    kafka_broker_address: str
    kafka_input_topic: str
    kafka_output_topic: str
    kafka_consumer_group: str
    ohlcv_window_seconds: int

    @classmethod
    def load(cls):
        logger.info("Starting config load...")
        
        # First load from environment variables
        env_vars = {
            "KAFKA_BROKER_ADDRESS": os.environ.get("KAFKA_BROKER_ADDRESS"),
            "KAFKA_INPUT_TOPIC": os.environ.get("KAFKA_INPUT_TOPIC"),
            "KAFKA_OUTPUT_TOPIC": os.environ.get("KAFKA_OUTPUT_TOPIC"),
            "KAFKA_CONSUMER_GROUP": os.environ.get("KAFKA_CONSUMER_GROUP"),
            "OHLCV_WINDOW_SECONDS": os.environ.get("OHLCV_WINDOW_SECONDS")
        }
        
        # Filter out None values and convert types
        env_vars = {k: v for k, v in env_vars.items() if v is not None}
        
        # Convert OHLCV_WINDOW_SECONDS to int if present
        if "OHLCV_WINDOW_SECONDS" in env_vars:
            env_vars["OHLCV_WINDOW_SECONDS"] = int(env_vars["OHLCV_WINDOW_SECONDS"])
        
        # Create instance with .env file as fallback
        instance = cls()
        
        # Override with environment variables
        for key, value in env_vars.items():
            setattr(instance, key.lower(), value)
        
        logger.info("Final config values:")
        for key, value in instance.model_dump().items():
            logger.info(f"{key}={value} (type: {type(value).__name__})")
            
        return instance

# Initialize config
logger.info("Initializing configuration...")
config = Config.load()