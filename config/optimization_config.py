from dataclasses import dataclass
import math

@dataclass(frozen=True)
class OptimizationConfig:
    steps: int = 300
    intensity_learning_rate: float = 0.05
    position_learning_rate: float = 0.001
    range_learning_rate: float = 0.001
    color_learning_rate: float = 0.001
    direction_learning_rate: float = 0.001
    initial_intensity_scale: float = 0.5
    log_every: int = 50

    def __post_init__(self):
        if not isinstance(self.steps, int) or self.steps < 1:
            raise ValueError("steps must be a positive integer")

        if not isinstance(self.log_every, int) or self.log_every < 1:
            raise ValueError("log_every must be a positive integer")

        if (
            not math.isfinite(self.intensity_learning_rate)
            or self.intensity_learning_rate <= 0
        ):
            raise ValueError(
                "intensity_learning_rate must be finite and positive"
            )

        if (
            not math.isfinite(self.initial_intensity_scale)
            or self.initial_intensity_scale < 0
        ):
            raise ValueError(
                "initial_intensity_scale must be finite and nonnegative"
            )

        for name in (
            "position_learning_rate",
            "range_learning_rate",
            "color_learning_rate",
            "direction_learning_rate",
        ):
            value = getattr(self, name)

            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    f"{name} must be finite and positive"
                )