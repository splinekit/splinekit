"""
This concrete class honors ``Interval`` and ``HalfBounded``; it handles
left-bounded intervals not below  a finite real number.
"""

#---------------
from typing import cast
from typing import Any
from typing import Self

#---------------
import importlib
import math

#---------------
from splinekit.interval.half_bounded import HalfBounded
from splinekit.interval.interval import Interval
from splinekit.interval.rr import RR
from splinekit.interval.empty import Empty

#---------------
class NotBelow (
    Interval,
    HalfBounded
):

    r"""
    .. _NotBelow:

    The right-unbounded interval
    :math:`\{x\in{\mathbb{R}}|x\geq x_{0}\}=[x_{0},\infty).`
    """

    #---------------
    _above: Any = importlib.import_module("splinekit.interval.above")

    #---------------
    @property
    def infimum (
        self
    ) -> float:
        """
        ``self.threshold``
        """
        return self._threshold

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
        ``False``
        """
        return False

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
        ``False``
        """
        return False

    #---------------
    @property
    def ishalfopen (
        self
    ) -> bool:
        """
        ``True``
        """
        return True

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
        ``False``
        """
        return False

    #---------------
    @property
    def ishalfbounded (
        self
    ) -> bool:
        """
        ``True``
        """
        return True

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
        ``Above(self.threshold)``
        """
        return cast(Self, self._above.Above(self._threshold))

    #---------------
    @property
    def closure (
        self
    ) -> Self:
        """
        ``NotBelow(self.threshold)``
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
        ``self.threshold``
        """
        return self._threshold

    #---------------
    def __contains__ (
        self,
        x: float
    ) -> bool:
        """
        """
        return math.isfinite(x) and (self._threshold <= x)

    #---------------
    def __hash__ (
        self
    ) -> int:
        return hash(self.__class__)

    #---------------
    def __str__ (
        self
    ) -> str:
        return "{x in RR | %s <= x}" % self._threshold

    #---------------
    def __repr__ (
        self
    ) -> str:
        return f"NotBelow({self._threshold})"

    #---------------
    def __new__ (
        cls,
        threshold: float
    ) -> Self:

        r"""
        Creates the right-unbounded interval
        :math:`\{x\in{\mathbb{R}}|x\geq x_{0}\}=[x_{0},\infty)` that contains
        all numbers that are not smaller than the threshold :math:`x_{0}.`

        Parameters
        ----------
        threshold
            The inclusive threshold above which every number belongs to this
            interval.

        Returns
        -------
        Interval
            A ``NotBelow`` interval if ``threshold`` is a finite number;
            ``RR()`` if ``threshold == float("-inf")``; and the ``Empty``
            interval otherwise.
        """

        if math.isfinite(threshold):
            return super().__new__(cls, threshold)
        if (math.isnan(threshold)
            or (math.isinf(threshold) and 0.0 < threshold)):
            return cast(Self, Empty())
        return cast(Self, RR())
