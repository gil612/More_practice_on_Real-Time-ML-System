from pydantic import BaseModel

class Trade(BaseModel):
    """
    A class that represents a trade.
    """
    product_id: str
    quantity: float
    price: float
    timestamp_ms: int