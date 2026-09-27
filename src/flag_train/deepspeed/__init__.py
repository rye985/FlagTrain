"""DeepSpeed-related implementations."""

from flag_train.runtime import device as runtime_device
from flag_train.runtime.backend import SpecOpRegistrar

from .blocked_flash import blocked_flash
from .lamb import lamb
from .multi_tensor_adam import multi_tensor_adam

__all__ = [
    "blocked_flash",
    "lamb",
    "multi_tensor_adam",
]


SpecOpRegistrar(registry=globals(), vendor=runtime_device.vendor_name).apply()
