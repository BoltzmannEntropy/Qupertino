"""mlxQ-prefixed alias for tensor module."""
from .tensor import *  # noqa: F401,F403

__all__ = [name for name in globals().keys() if not name.startswith('_')]

