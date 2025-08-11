"""
This concrete class honors ``Interval`` and ``Bounded``; it handles
open-closed intervals between a pair of finite real numbers.
"""

#---------------
from typing import cast
from typing import Any
from typing import Self
from typing import Tuple

#---------------
import importlib
import math

#---------------
from splinekit.interval.bounded import Bounded
from splinekit.interval.interval import Interval
from splinekit.interval.rr import RR
from splinekit.interval.empty import Empty
from splinekit.interval.above import Above
from splinekit.interval.not_above import NotAbove
from splinekit.interval.open import Open

#---------------
class OpenClosed (
    Interval,
    Bounded
):

    r"""
    .. _OpenClosed:

    The bounded interval
    :math:`\{x\in{\mathbb{R}}|x_{0}<x\leq x_{1}\}=(x_{0},x_{1}].`
    """

    #---------------
    _closed: Any = importlib.import_module("splinekit.interval.closed")

    #---------------
    @property
    def infimum (
        self
    ) -> float:
        """
        Left endpoint.
        """
        return self._leftbound

    #---------------
    @property
    def supremum (
        self
    ) -> float:
        """
        Right endpoint.
        """
        return self._rightbound

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
        ``False``
        """
        return False

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
        ``False``
        """
        return False

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
        ``True``
        """
        return True

    #---------------
    @property
    def interior (
        self
    ) -> Self:
        """
        ``Open((self.infimum, self.supremum))``
        """
        return cast(Self, Open((self._leftbound, self._rightbound)))

    #---------------
    @property
    def closure (
        self
    ) -> Self:
        """
        ``Closed((self.infimum, self.supremum))``
        """
        return cast(Self, self._closed.Closed((
            self._leftbound,
            self._rightbound
        )))

    #---------------
    @property
    def diameter (
        self
    ) -> float:
        """
        ``self.supremum - self.infimum``
        """
        return self._rightbound - self._leftbound

    #---------------
    @property
    def midpoint (
        self
    ) -> float:
        """
        ``0.5 * (self.supremum + self.infimum)``
        """
        return 0.5 * (self._leftbound + self._rightbound)

    #---------------
    def sortorder (
        self
    ) -> float:
        """
        ``self.midpoint``
        """
        return self.midpoint

    #---------------
    def __contains__ (
        self,
        x: float
    ) -> bool:
        """
        """
        return self._leftbound < x <= self._rightbound

    #---------------
    def __hash__ (
        self
    ) -> int:
        return hash(self.__class__)

    #---------------
    def __str__ (
        self
    ) -> str:
        return "{x in RR | %s < x <= %s}" % (self._leftbound, self._rightbound)

    #---------------
    def __repr__ (
        self
    ) -> str:
        return f"OpenClosed(({self._leftbound}, {self._rightbound}))"

    #---------------
    def __new__ (
        cls,
        bounds: Tuple[float, float]
    ) -> Self:

        r"""
        Creates the bounded open-closed interval
        :math:`\{x\in{\mathbb{R}}|x_{0}<x\leq x_{1}\}=(x_{0},x_{1}]` that
        contains all numbers that are larger than :math:`x_{0}` and not
        larger than :math:`x_{1}.`

        Parameters
        ----------
        bounds
            The tuple (infimum, supremum).

        Returns
        -------
        Interval
            An ``OpenClosed`` interval if ``bounds`` contains two finite
            numbers, with the first number being smaller than the second;
            otherwise, an interval from {``RR``, ``Empty``, ``Above``,
            ``NotAbove``}.
        """

        if (math.isfinite(bounds[0]) and math.isfinite(bounds[1]) and
            (bounds[0] != bounds[1])):
            instance = super().__new__(cls, bounds)
            return instance
        if ((math.isinf(bounds[0]) and bounds[0] < 0.0) and
            math.isfinite(bounds[1])):
            return cast(Self, NotAbove(bounds[1]))
        if (math.isfinite(bounds[0]) and
            (math.isinf(bounds[1]) and 0.0 < bounds[1])):
            return cast(Self, Above(bounds[0]))
        if ((math.isinf(bounds[0]) and bounds[0] < 0.0) and
            (math.isinf(bounds[1]) and 0.0 < bounds[1])):
            return cast(Self, RR())
        return cast(Self, Empty())
