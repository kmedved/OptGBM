"""OptGBM package."""

import logging

try:
    from importlib.metadata import version
except ImportError:  # pragma: no cover - Python < 3.8
    from importlib_metadata import version  # type: ignore

try:
    __version__ = version(__name__)
except Exception:  # pragma: no cover
    pass

from lightgbm import *  # noqa

from . import basic  # noqa
from . import sklearn  # noqa
from . import typing  # noqa
from . import utils  # noqa
from .sklearn import *  # noqa

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()

logger.addHandler(handler)

logger.setLevel(logging.INFO)
