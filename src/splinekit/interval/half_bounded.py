"""
This abstract class handles half-bounded intervals of real numbers.
"""

#---------------
from typing import Self

#---------------
from abc import ABC
from abc import abstractmethod

#---------------
class HalfBounded (
    ABC
):

    """
    A half-bounded interval that depends on a threshold.
    """

    #---------------
    @property
    def threshold (
        self
    ) -> float:

        """
        The threshold of this half-bounded interval.

        Returns
        -------
        float
            The threshold.
        """

        return self._threshold

    #---------------
    @threshold.setter
    def threshold (
        self,
        value: float
    ):
        self._threshold = value

    #---------------
    @abstractmethod
    def __new__ (
        cls,
        threshold: float
    ) -> Self:

        """
        Creates a half-bounded interval that contains all numbers related
        to a threshold.

        Parameters
        ----------
        threshold
            The threshold.

        Returns
        -------
        Interval
            A half-bounded interval.
        """

        instance = super().__new__(cls)
        setattr(instance, "threshold", threshold)
        return instance
