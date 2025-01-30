from typing import List
import hopsworks
from loguru import logger
from config import hopsworks_config
import pandas as pd

def push_value_to_feature_group(
    value: dict,
    feature_group_name: str,
    feature_group_version: int,
    feature_group_primary_key: list,
    feature_group_event_time: str,
):
    """
    Push a value to a feature group
    """
    logger.debug(f"Received value: {value}")
    
    # Login to Hopsworks
    project = hopsworks.login(
        project=hopsworks_config.hopsworks_project_name,
        api_key_value=hopsworks_config.hopsworks_api_key
    )

    # Get the feature store
    feature_store = project.get_feature_store()

    try:
        # Try to get existing feature group
        feature_group = feature_store.get_feature_group(
            name=feature_group_name,
            version=feature_group_version,
        )
    except:
        logger.info(f"Feature group {feature_group_name} not found, creating it...")
        
        # Convert sample dict to DataFrame to get schema
        sample_df = pd.DataFrame([value])
        
        # Add timestamp column if it doesn't exist
        if 'timestamp' not in sample_df.columns:
            sample_df['timestamp'] = pd.to_datetime(sample_df['timestamp_ms'], unit='ms')
        
        # Add id column as primary key
        sample_df['id'] = 1  # Will be auto-incremented
        
        logger.debug(f"Sample DataFrame columns: {sample_df.columns}")
        
        # Create the feature group
        feature_group = feature_store.create_feature_group(
            name=feature_group_name,
            version=feature_group_version,
            primary_key=['id'],  # Use id as primary key
            event_time='timestamp',
            online_enabled=True,
            description="OHLCV data from crypto trades"
        )
        
        # Initialize the feature group with schema
        feature_group.save(sample_df)

    # Insert the data
    df = pd.DataFrame([value])
    # Add the same transformations for insert data
    if 'timestamp' not in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms')
    
    # Add auto-incrementing id
    last_id = 1  # TODO: Get last id from feature store
    df['id'] = last_id + 1
        
    feature_group.insert(df)
    logger.info(f"Inserted data into feature group: {value}")
