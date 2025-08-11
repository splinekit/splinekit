"""
This concrete class honors ``Interval`` and ``Universal``; it handles
the empty interval of real numbers.
"""

#---------------
from typing import Self

#---------------
from splinekit.interval.interval import Interval
from splinekit.interval.universal import Universal

#---------------
class Empty (
    Interval,
    Universal
):

    r"""
    .. _Empty:

    The empty interval
    :math:`\{x\in{\mathbb{R}}|x\neq x\}=\emptyset` that contains no number.
    """

    #---------------
    @property
    def infimum (
        self
    ) -> float:
        """
        ``float("nan")``
        """
        return float("nan")

    #---------------
    @property
    def supremum (
        self
    ) -> float:
        """
        ``float("nan")``
        """
        return float("nan")

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
        ``True``
        """
        return True

    #---------------
    @property
    def isrightbounded (
        self
    ) -> bool:
        """
        ``True``
        """
        return True

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
        ``True``
        """
        return True

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
        ``False``
        """
        return False

    #---------------
    @property
    def interior (
        self
    ) -> Self:
        """
        ``Empty()``
        """
        return self

    #---------------
    @property
    def closure (
        self
    ) -> Self:
        """
        ``Empty()``
        """
        return self

    #---------------
    @property
    def diameter (
        self
    ) -> float:
        """
        ``float("nan")``
        """
        return float("nan")

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
        ``float("-inf")``
        """
        return float("-inf")

    #---------------
    def __contains__ (
        self,
        x: float
    ) -> bool:
        """
        ``False``
        """
        del x
        return False

    #---------------
    def __hash__ (
        self
    ) -> int:
        return hash(self.__class__)

    #---------------
    def __str__ (
        self
    ) -> str:
        return "{}"

    #---------------
    def __repr__ (
        self
    ) -> str:
        return "Empty()"
