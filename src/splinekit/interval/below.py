"""
This concrete class honors ``Interval`` and ``HalfBounded``; it handles
right-bounded intervals below a finite real number.

====

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
class Below (
    Interval,
    HalfBounded
):

    r"""
    .. _Below:

    The left-unbounded interval
    :math:`\{x\in{\mathbb{R}}|x<x_{0}\}=(—\infty,x_{0}).`

    ====

    """

    #---------------
    _not_above: Any = importlib.import_module("splinekit.interval.not_above")

    #---------------
    @property
    def infimum (
        self
    ) -> float:
        """
        ``float("-inf")``

        ----

        """
        return float("-inf")

    #---------------
    @property
    def supremum (
        self
    ) -> float:
        """
        ``self.threshold``

        ----

        """
        return self._threshold

    #---------------
    @property
    def isleftopen (
        self
    ) -> bool:
        """
        ``True``

        ----

        """
        return True

    #---------------
    @property
    def isrightopen (
        self
    ) -> bool:
        """
        ``True``

        ----

        """
        return True

    #---------------
    @property
    def isopen (
        self
    ) -> bool:
        """
        ``True``

        ----

        """
        return True

    #---------------
    @property
    def ishalfopen (
        self
    ) -> bool:
        """
        ``False``

        ----

        """
        return False

    #---------------
    @property
    def isclosed (
        self
    ) -> bool:
        """
        ``False``

        ----

        """
        return False

    #---------------
    @property
    def isleftbounded (
        self
    ) -> bool:
        """
        ``False``

        ----

        """
        return False

    #---------------
    @property
    def isrightbounded (
        self
    ) -> bool:
        """
        ``True``

        ----

        """
        return True

    #---------------
    @property
    def ishalfbounded (
        self
    ) -> bool:
        """
        ``True``

        ----

        """
        return True

    #---------------
    @property
    def isbounded (
        self
    ) -> bool:
        """
        ``False``

        ----

        """
        return False

    #---------------
    @property
    def isdegenerate (
        self
    ) -> bool:
        """
        ``False``

        ----

        """
        return False

    #---------------
    @property
    def isproper (
        self
    ) -> bool:
        """
        ``True``

        ----

        """
        return True

    #---------------
    @property
    def interior (
        self
    ) -> Self:
        """
        ``Below(self.threshold)``

        ----

        """
        return self

    #---------------
    @property
    def closure (
        self
    ) -> Self:
        """
        ``NotAbove(self.threshold)``

        ----

        """
        return cast(Self, self._not_above.NotAbove(self._threshold))

    #---------------
    @property
    def diameter (
        self
    ) -> float:
        """
        ``float("inf")``

        ----

        """
        return float("inf")

    #---------------
    @property
    def midpoint (
        self
    ) -> float:
        """
        ``float("nan")``

        ----

        """
        return float("nan")

    #---------------
    def sortorder (
        self
    ) -> float:
        """
        ``self.threshold``

        ----

        """
        return self._threshold

    #---------------
    def __contains__ (
        self,
        x: float
    ) -> bool:
        """
        ``math.isfinite(x) and (x < self.threshold)``

        ----

        """
        return math.isfinite(x) and (x < self._threshold)

    #---------------
    def __hash__ (
        self
    ) -> int:
        return hash(self.__class__)

    #---------------
    def __str__ (
        self
    ) -> str:
        return "{x in RR | x < %s}" % self._threshold

    #---------------
    def __repr__ (
        self
    ) -> str:
        return f"Below({self._threshold})"

    #---------------
    def __new__ (
        cls,
        threshold: float
    ) -> Self:

        r"""
        Creates the left-unbounded interval
        :math:`\{x\in{\mathbb{R}}|x<x_{0}\}=(—\infty,x_{0})` that contains
        all numbers that are smaller than the threshold :math:`x_{0}.`

        Parameters
        ----------
        threshold
            The threshold below which every number belongs to this interval.

        Returns
        -------
        Interval
            A ``Below`` interval if ``threshold`` is a finite number; ``RR()``
            if ``threshold == float("inf")``; and the ``Empty`` interval
            otherwise.
        """

        if math.isfinite(threshold):
            return super().__new__(cls, threshold)
        if (math.isnan(threshold)
            or (math.isinf(threshold) and threshold < 0.0)):
            return cast(Self, Empty())
        return cast(Self, RR())
