"""
This concrete class honors ``Interval`` and ``Universal``; it handles
the interval of all finite real numbers.
"""

#---------------
from typing import Self

#---------------
import math

#---------------
from splinekit.interval.interval import Interval
from splinekit.interval.universal import Universal

#---------------
class RR (
    Interval,
    Universal
):

    r"""
    .. _RR:

    All finite real numbers
    :math:`{\mathbb{R}}=(-\infty,\infty).`
    """

    #---------------
    @property
    def infimum (
        self
    ) -> float:
        """
        ``float("-inf")``
        """
        return float("-inf")

    #---------------
    @property
    def supremum (
        self
    ) -> float:
        """
        ``float("inf")``
        """
        return float("inf")

    #---------------
    @property
    def isleftopen (
        self
    ) -> bool:
        """
        ``True``
        """
        return True

    #---------------
    @property
    def isrightopen (
        self
    ) -> bool:
        """
        ``True``
        """
        return True

    #---------------
    @property
    def isopen (
        self
    ) -> bool:
        """
        ``True``
        """
        return True

    #---------------
    @property
    def ishalfopen (
        self
    ) -> bool:
        """
        ``False``
        """
        return False

    #---------------
    @property
    def isclosed (
        self
    ) -> bool:
        """
        ``True``
        """
        return True

    #---------------
    @property
    def isleftbounded (
        self
    ) -> bool:
        """
        ``False``
        """
        return False

    #---------------
    @property
    def isrightbounded (
        self
    ) -> bool:
        """
        ``False``
        """
        return False

    #---------------
    @property
    def ishalfbounded (
        self
    ) -> bool:
        """
        ``False``
        """
        return False

    #---------------
    @property
    def isbounded (
        self
    ) -> bool:
        """
        ``False``
        """
        return False

    #---------------
    @property
    def isdegenerate (
        self
    ) -> bool:
        """
        ``False``
        """
        return False

    #---------------
    @property
    def isproper (
        self
    ) -> bool:
        """
        ``True``
        """
        return True

    #---------------
    @property
    def interior (
        self
    ) -> Self:
        """
        ``RR()``
        """
        return self

    #---------------
    @property
    def closure (
        self
    ) -> Self:
        """
        ``RR()``
        """
        return self

    #---------------
    @property
    def diameter (
        self
    ) -> float:
        """
        ``float("inf")``
        """
        return float("inf")

    #---------------
    @property
    def midpoint (
        self
    ) -> float:
        """
        ``float("nan")``
        """
        return float("nan")

    #---------------
    def sortorder (
        self
    ) -> float:
        """
        ``float("inf")``
        """
        return float("inf")

    #---------------
    def __contains__ (
        self,
        x: float
    ) -> bool:
        return math.isfinite(x)

    #---------------
    def __hash__ (
        self
    ) -> int:
        return hash(self.__class__)

    #---------------
    def __str__ (
        self
    ) -> str:
        return "RR"

    #---------------
    def __repr__ (
        self
    ) -> str:
        return "RR()"
