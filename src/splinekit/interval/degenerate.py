"""
This abstract class handles degenerate intervals of real numbers.

====

"""

#---------------
from typing import Self

#---------------
from abc import ABC

#---------------
class Degenerate (
    ABC
):

    """
    Notes
    -----
    Subclasses that implement this class have the property ``value``.

    ====

    """

    #---------------
    @property
    def value (
        self
    ) -> float:

        r"""
        The value contained by this degenerate interval.

        Returns
        -------
        float
            The value.


        ----

        """

        return self._value

    #---------------
    @value.setter
    def value (
        self,
        value: float
    ):
        self._value = value

    #---------------
    def __new__ (
        cls,
        value: float
    ) -> Self:

        """
        Creates a degenerate interval that contains exactly one number.

        Parameters
        ----------
        value
            The number contained by this degenerate interval.

        Returns
        -------
        Interval
            A degenerate interval.
        """

        instance = super().__new__(cls)
        setattr(instance, "value", value)
        return instance
