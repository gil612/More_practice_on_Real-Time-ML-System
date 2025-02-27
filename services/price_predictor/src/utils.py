import pandas as pd
import hashlib
import numpy as np
import logging

logger = logging.getLogger(__name__)

def hash_dataframe(df: pd.DataFrame) -> str:
    """
    Compute a deterministic hash of a pandas DataFrame.
    
    Args:
        df: pandas DataFrame to hash
        
    Returns:
        str: A deterministic hex string hash
    """
    logger.debug(f"Initial DataFrame shape: {df.shape}")
    logger.debug(f"Initial columns: {list(df.columns)}")
    
    # Make a copy and sort columns
    df = df.copy()
    df = df.reindex(sorted(df.columns), axis=1)
    logger.debug(f"Sorted columns: {list(df.columns)}")
    
    # Get only numeric columns and sort them
    numeric_cols = sorted(df.select_dtypes(include=[np.number]).columns)
    logger.debug(f"Numeric columns: {numeric_cols}")
    
    # Create a string from the rounded numeric values only
    values_str = ""
    for col in numeric_cols:
        # Log first few values for debugging
        logger.debug(f"First 5 values of {col}: {df[col].values[:5]}")
        values = [f"{x:.4f}" for x in df[col].values]
        values_str += "".join(values)
    
    # Log part of the string for debugging
    logger.debug(f"First 100 chars of values_str: {values_str[:100]}")
    
    # Create hash of the string representation
    hash_value = hashlib.sha256(values_str.encode('utf-8')).hexdigest()
    logger.debug(f"Generated hash: {hash_value}")
    
    return hash_value
