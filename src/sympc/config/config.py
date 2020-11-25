from dataclasses import dataclass, field

@dataclass
class Config:
    ring_size: int = field()
    min_value: int = field()
    max_value: int = field()

    encoder_precision: int = field()
    encoder_base: int = field()

    def __init__(
            self,
            ring_size: int = 2**62,
            encoder_precision: int = 4,
            encoder_base: int = 10
    ):
        self.ring_size = ring_size
        self.min_value = -(ring_size // 2)
        self.max_value = (ring_size - 1) // 2

        self.encoder_precision = encoder_precision
        self.encoder_base = encoder_base
