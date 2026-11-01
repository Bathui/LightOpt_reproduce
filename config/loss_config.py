import math
from dataclasses import dataclass

@dataclass(frozen=True)
class LossConfig:
    overlap_weight: float = 0.1
    attraction_weight: float = 0.2

    n_max: int = 3

    beta_w: float = 1.0
    beta_p: float = 1.0
    beta_m: float = 1.0

    under_eps: float = 0.1
    dark_eps: float = 1e-8
    volume: float = 15.0
    eps: float = 1e-8

    spot_sign_correction: bool = True

    def __post_init__(self):
        positive = (
            self.beta_w,
            self.beta_p,
            self.beta_m,
            self.eps,
        )

        nonnegative = (
            self.overlap_weight,
            self.attraction_weight,
            self.dark_eps,
            self.volume,
        )

        if not all(math.isfinite(v) and v > 0 for v in positive):
            raise ValueError(
                "Beta values and eps must be finite and positive"
            )

        if not all(math.isfinite(v) and v >= 0 for v in nonnegative):
            raise ValueError(
                "Weights, dark_eps and volume must be finite and nonnegative"
            )

        if not isinstance(self.n_max, int) or self.n_max < 0:
            raise ValueError("n_max must be a nonnegative integer")

        if not 0 <= self.under_eps <= 1:
            raise ValueError("under_eps must be in [0,1]")