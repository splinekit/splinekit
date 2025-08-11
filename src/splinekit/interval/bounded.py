"""
This abstract class handles bounded intervals of real numbers.
"""

#---------------
from typing import Self
from typing import Tuple

#---------------
from abc import ABC
from abc import abstractmethod

#---------------
class Bounded (
    ABC
):

    """
    A bounded interval that depends on a pair of endpoints.
    """

    #---------------
    @property
    def bounds (
        self
    ) -> Tuple[float, float]:

        """
        The two endpoints of this interval.

        Returns
        -------
        tuple of float
            The left- and right-endpoint of this interval, in that order.
        """

        return (self._leftbound, self._rightbound)

    #---------------
    @property
    def leftbound (
        self
    ) -> float:

        """
        The infimum of this bounded interval.

        Returns
        -------
        float
            The left endpoint of this interval.
        """

        return self._leftbound

    #---------------
    @leftbound.setter
    def leftbound (
        self,
        value: float
    ):
        self._leftbound = value

    #---------------
    @property
    def rightbound (
        self
    ) -> float:

        """
        The supremum of this bounded interval.

        Returns
        -------
        float
            The right endpoint of this interval.
        """

        return self._rightbound

    #---------------
    @rightbound.setter
    def rightbound (
        self,
        value: float
    ):
        self._rightbound = value

    #---------------
    @abstractmethod
    def __new__ (
        cls,
        bounds: Tuple[float, float]
    ) -> Self:

        r"""
        Creates a bounded interval that contains all numbers related
        to a lower bound and an upper bound.

        Parameters
        ----------
        bounds
            The tuple (infimum, supremum).

        Returns
        -------
        Interval
            A bounded interval.
        """

        instance = super().__new__(cls)
        setattr(instance, "leftbound", bounds[0])
        setattr(instance, "rightbound", bounds[1])
        return instance
