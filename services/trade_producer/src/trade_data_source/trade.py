from dataclasses import dataclass, asdict

@dataclass
class Trade:
    product_id: str
    quantity: float
    price: float
    timestamp_ms: int

    def to_dict(self):
        return asdict(self)