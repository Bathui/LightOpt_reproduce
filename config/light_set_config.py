import math
from dataclasses import dataclass


@dataclass(frozen=True)
class LightSetConfig:
    importance_threshold: float
    dbscan_eps: float = 0.5
    dbscan_min_samples: int = 8
    height_min: float = 0.1
    height_max: float = 5.0
    height_steps: int = 24
    range_margin: float = 1.1
    angle_margin_degrees: float = 2.0
    point_angle_degrees: float = 130.0
    inner_angle_ratio: float = 0.8
    attenuation_smoothness: float = 1.0
    max_additions: int = 8
    require_reconstruction_improvement: bool = True

    def __post_init__(self):
        positive = (
            self.dbscan_eps,
            self.height_min,
            self.height_max,
            self.attenuation_smoothness,
        )
        if not all(math.isfinite(v) and v > 0 for v in positive):
            raise ValueError("Distances and attenuation smoothness must be positive")

        if (
            not math.isfinite(self.importance_threshold)
            or self.importance_threshold < 0
        ):
            raise ValueError("importance_threshold must be finite and nonnegative")

        counts = (
            self.dbscan_min_samples,
            self.height_steps,
            self.max_additions,
        )
        if not all(
            isinstance(v, int) and not isinstance(v, bool) and v >= 1
            for v in counts
        ):
            raise ValueError("Counts must be positive integers")

        if self.height_max < self.height_min:
            raise ValueError("height_max must be >= height_min")
        if not math.isfinite(self.range_margin) or self.range_margin <= 1:
            raise ValueError("range_margin must be > 1")
        if not 0 < self.angle_margin_degrees < 90:
            raise ValueError("angle_margin_degrees must be in (0,90)")
        if not 0 < self.point_angle_degrees < 180:
            raise ValueError("point_angle_degrees must be in (0,180)")
        if not 0 < self.inner_angle_ratio < 1:
            raise ValueError("inner_angle_ratio must be in (0,1)")