"""chainorder: chain decomposition and order parameters for ReO3-type anion ordering."""
from importlib.metadata import version

from chainorder.decompose import SublatticeOccupation
from chainorder import order_params

__version__ = version("chainorder")
__all__ = ["SublatticeOccupation", "order_params"]
