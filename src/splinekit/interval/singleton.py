"""
This concrete class honors ``Interval`` and ``Degenerate``; it handles
intervals made of just one finite real number.

====

"""

#---------------
from typing import cast
from typing import Self

#---------------
import math

#---------------
from splinekit.interval.interval import Interval
from splinekit.interval.degenerate import Degenerate
from splinekit.interval.empty import Empty

#---------------
class Singleton (
    Interval,
    Degenerate
):

    r"""
    .. _Singleton:

    The degenerate interval
    :math:`\{x\in{\mathbb{R}}|x=x_{0}\}=[x_{0},x_{0}].`

    ====

    """

    #---------------
    @property
    def infimum (
        self
    ) -> float:
        """
        ``self.value``

        ----

        """
        return self._value

    #---------------
    @property
    def supremum (
        self
    ) -> float:
        """
        ``self.value``

        ----

        """
        return self._value

    #---------------
    @property
    def isleftopen (
        self
    ) -> bool:
        """
        ``False``

        ----

        """
        return False

    #---------------
    @property
    def isrightopen (
        self
    ) -> bool:
        """
        ``False``

        ----

        """
        return False

    #---------------
    @property
    def isopen (
        self
    ) -> bool:
        """
        ``False``

        ----

        """
        return False

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
        ``True``

        ----

        """
        return True

    #---------------
    @property
    def isleftbounded (
        self
    ) -> bool:
        """
        ``True``

        ----

        """
        return True

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
        ``False``

        ----

        """
        return False

    #---------------
    @property
    def isbounded (
        self
    ) -> bool:
        """
        ``True``

        ----

        """
        return True

    #---------------
    @property
    def isdegenerate (
        self
    ) -> bool:
        """
        ``True``

        ----

        """
        return True

    #---------------
    @property
    def isproper (
        self
    ) -> bool:
        """
        ``False``

        ----

        """
        return False

    #---------------
    @property
    def interior (
        self
    ) -> Self:
        """
        ``Empty()``

        ----

        """
        return cast(Self, Empty())

    #---------------
    @property
    def closure (
        self
    ) -> Self:
        """
        ``Singleton(self.value)``

        ----

        """
        return self

    #---------------
    @property
    def diameter (
        self
    ) -> float:
        """
        ``0.0``

        ----

        """
        return 0.0

    #---------------
    @property
    def midpoint (
        self
    ) -> float:
        """
        ``self.value``

        ----

        """
        return self._value

    #---------------
    def sortorder (
        self
    ) -> float:
        """
        ``self.value``

        ----

        """
        return self._value

    #---------------
    def __contains__ (
        self,
        x: float
    ) -> bool:
        return self._value == x

    #---------------
    def __hash__ (
        self
    ) -> int:
        return hash(self.__class__)

    #---------------
    def __str__ (
        self
    ) -> str:
        return "{x in RR | %s = x}" % self._value

    #---------------
    def __repr__ (
        self
    ) -> str:
        return f"Singleton({self._value})"

    def __new__ (
        cls,
        value: float
    ) -> Self:

        r"""
        Creates the degenerate interval
        :math:`\{x\in{\mathbb{R}}|x=x_{0}\}=[x_{0},x_{0}]` that contains
        exactly one number.

        Parameters
        ----------
        value
            The number contained by this degenerate interval.

        Returns
        -------
        Interval
            A ``Singleton`` interval if ``value`` is a finite number; an
            ``Empty`` interval otherwise.
        """

        if math.isfinite(value):
            instance = super().__new__(cls, value)
            return instance
        return cast(Self, Empty())
