#---------------
from typing import Any
from typing import Self

#---------------
import importlib

class _KnownIntervals (
):

    #---------------
    _instanceofki = None

    #---------------
    rr: Any = None
    empty: Any = None
    singleton: Any = None
    above: Any = None
    notbelow: Any = None
    notabove: Any = None
    below: Any = None
    open: Any = None
    openclosed: Any = None
    closed: Any = None
    closedopen: Any = None

    #---------------
    def __new__ (
        cls
    ) -> Self:
        if cls._instanceofki is None:
            cls._instanceofki = super().__new__(cls)
            cls.rr = importlib.import_module("splinekit.interval.rr").RR()
            cls.empty = importlib.import_module(
                "splinekit.interval.empty"
            ).Empty()
            cls.singleton = importlib.import_module(
                "splinekit.interval.singleton"
            )
            cls.above = importlib.import_module("splinekit.interval.above")
            cls.notbelow = importlib.import_module(
                "splinekit.interval.not_below"
            )
            cls.notabove = importlib.import_module(
                "splinekit.interval.not_above"
            )
            cls.below = importlib.import_module("splinekit.interval.below")
            cls.open = importlib.import_module("splinekit.interval.open")
            cls.openclosed = importlib.import_module(
                "splinekit.interval.open_closed"
            )
            cls.closed = importlib.import_module("splinekit.interval.closed")
            cls.closedopen = importlib.import_module(
                "splinekit.interval.closed_open"
            )
        return cls._instanceofki
