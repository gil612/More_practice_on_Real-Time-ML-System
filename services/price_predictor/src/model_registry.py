

def get_model_name(
    product_id: str,
    ohlc_window_sec: int,
    forecast_steps: int
) -> str:
    """
    Returns the name of the model in the model registry given the
    - product_id
    - ohlc_window_sec
    - forecast_steps

    The model name is used to identify the model in the model registry.
    """
    return f"price_predictor_{product_id.replace('/', '_')}_{ohlc_window_sec}s_{forecast_steps}steps"