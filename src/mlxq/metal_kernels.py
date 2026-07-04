"""Back-compat shim: the Metal shaders moved to the mlxq.shaders package.

Import from mlxq.shaders directly in new code; this module re-exports the
public API so existing imports keep working.
"""
from .shaders import *  # noqa: F401,F403
from .shaders import __all__  # noqa: F401
