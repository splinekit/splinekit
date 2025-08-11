"""
This abstract class handles non-parametric intervals of real numbers.
"""

#---------------
from typing import Self

#---------------
from abc import ABC

#---------------
class Universal (
    ABC
):

    """
    Notes
    -----
    Subclasses that implement this class have a single instance.
    """

    #---------------
    _instance = None
    def __new__ (
        cls
    ) -> Self:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
