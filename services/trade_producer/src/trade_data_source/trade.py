from pydantic import BaseModel

class Trade(BaseModel):
    product_id: str
    quantity: float
    price: float
    timestamp_ms: int