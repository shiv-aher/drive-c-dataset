from __future__ import annotations
from typing import Dict

from simulation.base import BaseDegradation

from simulation.physics_aware.fog import FogDegradation
from simulation.physics_aware.defocus import DefocusDegradation
from simulation.physics_aware.occlusion import LensOcclusionDegradation
from simulation.physics_aware.glare import GlareDegradation

from simulation.weather.rain import RainDegradation
from simulation.weather.snow import SnowDegradation

from simulation.digital.motion_blur import MotionBlurDegradation
from simulation.digital.compression import CompressionArtifactsDegradation

from simulation.sensor.noise import SensorNoiseDegradation
from simulation.sensor.exposure import ExposureShiftDegradation
from simulation.sensor.vignetting import VignettingDegradation
from simulation.sensor.low_light import LowLightDegradation


def build_degradations() -> Dict[str, BaseDegradation]:
    """
    Keys MUST match camera_issues.yaml issue names.
    """
    return {
        "haze_fog": FogDegradation(),
        "defocus_blur": DefocusDegradation(),
        "lens_occlusion": LensOcclusionDegradation(),
        "glare_flare": GlareDegradation(),
        "rain": RainDegradation(),
        "snow": SnowDegradation(),
        "motion_blur": MotionBlurDegradation(),
        "sensor_noise": SensorNoiseDegradation(),
        "vignetting": VignettingDegradation(),
        "compression": CompressionArtifactsDegradation(),
        "exposure_shift": ExposureShiftDegradation(),
        "low_light": LowLightDegradation(),
    }
