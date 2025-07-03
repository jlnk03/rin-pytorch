import os

os.environ["KERAS_BACKEND"] = "torch"

try:
    from .Rin import Rin  # type: ignore
    from .RinDiffusionModel import RinDiffusionModel  # type: ignore
    from .Trainer import Trainer  # type: ignore

    __all__ = ["Rin", "RinDiffusionModel", "Trainer"]
except Exception:  # pragma: no cover
    # In minimal environments (e.g., CI or unit-test), optional heavy
    # dependencies such as wandb / pillow may be missing.  We still want the
    # lightweight sub-packages (e.g. ``rin_pytorch.modules``) to be importable
    # so we silently skip exposing the higher-level training API if its
    # requirements are unavailable.
    __all__ = []
