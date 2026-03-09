"""
Scripts — modular framework for time series classification.
"""

__version__ = "0.1.0"

from . import models
from . import interpretability
from . import dataloaders
from . import utils
from . import train

__all__ = ["models", "interpretability", "dataloaders", "utils", "train"]
