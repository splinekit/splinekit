r"""
This abstract class handles intervals of real numbers.

Abstract Properties
===================
- :ref:`infimum <infimum>`
- :ref:`supremum <supremum>`
- :ref:`isleftopen <isleftopen>`
- :ref:`isrightopen <isrightopen>`
- :ref:`isopen <isopen>`
- :ref:`ishalfopen <ishalfopen>`
- :ref:`isclosed <isclosed>`
- :ref:`isleftbounded <isleftbounded>`
- :ref:`isrightbounded <isrightbounded>`
- :ref:`ishalfbounded <ishalfbounded>`
- :ref:`isbounded <isbounded>`
- :ref:`isdegenerate <isdegenerate>`
- :ref:`isproper <isproper>`
- :ref:`interior <interior>`
- :ref:`closure <closure>`
- :ref:`diameter <diameter>`
- :ref:`midpoint <midpoint>`

Abstract Methods
================
- :ref:`sortorder <sortorder>`

Class Methods
=============
- :ref:`ismember <ismember>`
- :ref:`enclosure <enclosure>`
- :ref:`complement <complement>`
- :ref:`intersection <intersection>`
- :ref:`union <union>`
- :ref:`gravitycenter <gravitycenter>`

Instance Methods
================
- :ref:`iskissing <iskissing>`
- :ref:`isoverlapping <isoverlapping>`
- :ref:`partition <partition>`
- :ref:`copy <copy>`

Abstract Operators
==================
- :ref:`__contains__ <contains>` ``in`` Interval membership: :math:`\in`

Operators
=========
- :ref:`__eq__ <eq>` ``==`` Equality of intervals: :math:`=`
- :ref:`__ne__ <ne>` ``!=`` Inequality of intervals: :math:`\neq`
- :ref:`__lt__ <lt>` ``<`` Proper subset: :math:`\subset`
- :ref:`__le__ <le>` ``<=`` Subset: :math:`\subseteq`
- :ref:`__neg__ <neg>` ``-`` Unary complement in ``RR()``:
  :math:`(\cdot)^{{\mathrm{c}}}`
- :ref:`__sub__ <sub>` ``-`` Pairwise difference: :math:`\setminus`
- :ref:`__and__ <and>` ``&`` Intersection: :math:`\cap`
- :ref:`__or__ <or>` ``|`` Union: :math:`\cup`
- :ref:`__xor__ <xor>` ``^`` Symmetric difference: :math:`\Delta`

Concrete Classes
================
- :ref:`RR <RR>`
- :ref:`Empty <Empty>`
- :ref:`Singleton <Singleton>`
- :ref:`Above <Above>`
- :ref:`NotBelow <NotBelow>`
- :ref:`NotAbove <NotAbove>`
- :ref:`Below <Below>`
- :ref:`Open <Open>`
- :ref:`OpenClosed <OpenClosed>`
- :ref:`Closed <Closed>`
- :ref:`ClosedOpen <ClosedOpen>`

"""

#---------------
from typing import cast
from typing import List
from typing import Self
from typing import Set
from typing import Tuple

#---------------
import math

#---------------
from abc import ABC
from abc import abstractmethod

#---------------
from splinekit.interval._known_intervals import _KnownIntervals

#---------------
from splinekit.interval.universal import Universal
from splinekit.interval.degenerate import Degenerate
from splinekit.interval.half_bounded import HalfBounded
from splinekit.interval.bounded import Bounded

#---------------
class Interval (
    ABC
):

    r"""
        .. _Interval:

    The abstract class that concrete intervals of real numbers must implement.

    The known concrete classes that implement this abstract class are
        - ``RR`` All finite real numbers :math:`{\mathbb{R}}=(-\infty,\infty)`
        - ``Empty`` The empty interval
          :math:`\{x\in{\mathbb{R}}|x\neq x\}=\emptyset`
        - ``Singleton`` The degenerate interval
          :math:`\{x\in{\mathbb{R}}|x=x_{0}\}=[x_{0},x_{0}]`
        - ``Above`` The right-unbounded interval
          :math:`\{x\in{\mathbb{R}}|x>x_{0}\}=(x_{0},\infty)`
        - ``NotBelow`` The right-unbounded interval
          :math:`\{x\in{\mathbb{R}}|x\geq x_{0}\}=[x_{0},\infty)`
        - ``NotAbove`` The left-unbounded interval
          :math:`\{x\in{\mathbb{R}}|x\leq x_{0}\}=(—\infty,x_{0}]`
        - ``Below`` The left-unbounded interval
          :math:`\{x\in{\mathbb{R}}|x<x_{0}\}=(—\infty,x_{0})`
        - ``Open`` The bounded interval
          :math:`\{x\in{\mathbb{R}}|x_{0}<x<x_{1}\}=(x_{0},x_{1})`
        - ``OpenClosed`` The bounded interval
          :math:`\{x\in{\mathbb{R}}|x_{0}<x\leq x_{1}\}=(x_{0},x_{1}]`
        - ``Closed`` The bounded interval
          :math:`\{x\in{\mathbb{R}}|x_{0}\leq x\leq x_{1}\}=[x_{0},x_{1}]`
        - ``ClosedOpen`` The bounded interval
          :math:`\{x\in{\mathbb{R}}|x_{0}\leq x<x_{1}\}=[x_{0},x_{1})`
    """

    #---------------
    @property
    @abstractmethod
    def infimum (
        self
    ) -> float:

        r"""
        .. _infimum:

        The value of the left endpoint of this interval.

        Returns
        -------
        float
            Left endpoint.

        Notes
        -----
        `infimum = float("nan")` for the following class.
            - ``Empty`` The empty interval
              :math:`\{x\in{\mathbb{R}}|x\neq x\}=\emptyset`
        `infimum = float("-inf")` for the following classes.
            - ``RR`` All finite real numbers
              :math:`{\mathbb{R}}=(-\infty,\infty)`
            - ``NotAbove`` The left-unbounded interval
              :math:`\{x\in{\mathbb{R}}|x\leq x_{0}\}=(—\infty,x_{0}]`
            - ``Below`` The left-unbounded interval
              :math:`\{x\in{\mathbb{R}}|x<x_{0}\}=(—\infty,x_{0})`
        `math.isfinite(infimum) = True` for the following classes.
            - ``Singleton`` The degenerate interval
              :math:`\{x\in{\mathbb{R}}|x=x_{0}\}=[x_{0},x_{0}]`
            - ``Above`` The right-unbounded interval
              :math:`\{x\in{\mathbb{R}}|x>x_{0}\}=(x_{0},\infty)`
            - ``NotBelow`` The right-unbounded interval
              :math:`\{x\in{\mathbb{R}}|x\geq x_{0}\}=[x_{0},\infty)`
            - ``Open`` The bounded interval
              :math:`\{x\in{\mathbb{R}}|x_{0}<x<x_{1}\}=(x_{0},x_{1})`
            - ``OpenClosed`` The bounded interval
              :math:`\{x\in{\mathbb{R}}|x_{0}<x\leq x_{1}\}=(x_{0},x_{1}]`
            - ``Closed`` The bounded interval
              :math:`\{x\in{\mathbb{R}}|x_{0}\leq x\leq x_{1}\}=[x_{0},x_{1}]`
            - ``ClosedOpen`` The bounded interval
              :math:`\{x\in{\mathbb{R}}|x_{0}\leq x<x_{1}\}=[x_{0},x_{1})`
        """

    #---------------
    @property
    @abstractmethod
    def supremum (
        self
    ) -> float:

        r"""
        .. _supremum:

        The value of the right endpoint of this interval.

        Returns
        -------
        float
            Right endpoint.

        Notes
        -----
        `supremum = float("nan")` for the following class.
            - ``Empty`` The empty interval
              :math:`\{x\in{\mathbb{R}}|x\neq x\}=\emptyset`
        `supremum = float("inf")` for the following classes.
            - ``RR`` All finite real numbers
              :math:`{\mathbb{R}}=(-\infty,\infty)`
            - ``Above`` The right-unbounded interval
              :math:`\{x\in{\mathbb{R}}|x>x_{0}\}=(x_{0},\infty)`
            - ``NotBelow`` The right-unbounded interval
              :math:`\{x\in{\mathbb{R}}|x\geq x_{0}\}=[x_{0},\infty)`
        `math.isfinite(supremum) = True` for the following classes.
            - ``Singleton`` The degenerate interval
              :math:`\{x\in{\mathbb{R}}|x=x_{0}\}=[x_{0},x_{0}]`
            - ``NotAbove`` The left-unbounded interval
              :math:`\{x\in{\mathbb{R}}|x\leq x_{0}\}=(—\infty,x_{0}]`
            - ``Below`` The left-unbounded interval
              :math:`\{x\in{\mathbb{R}}|x<x_{0}\}=(—\infty,x_{0})`
            - ``Open`` The bounded interval
              :math:`\{x\in{\mathbb{R}}|x_{0}<x<x_{1}\}=(x_{0},x_{1})`
            - ``OpenClosed`` The bounded interval
              :math:`\{x\in{\mathbb{R}}|x_{0}<x\leq x_{1}\}=(x_{0},x_{1}]`
            - ``Closed`` The bounded interval
              :math:`\{x\in{\mathbb{R}}|x_{0}\leq x\leq x_{1}\}=[x_{0},x_{1}]`
            - ``ClosedOpen`` The bounded interval
              :math:`\{x\in{\mathbb{R}}|x_{0}\leq x<x_{1}\}=[x_{0},x_{1})`
        """

    #---------------
    @property
    @abstractmethod
    def isleftopen (
        self
    ) -> bool:

        r"""
        .. _isleftopen:

        If the infimum of this interval is a finite number, tests whether
        it fails to be a member of the interval; else, the interval is
        vacuously said to be left-open.

        Returns
        -------
        bool
            - `False` if this interval is not open to the left.
            - `True` if this interval is open to the left.

        Examples
        --------
        Left-openness of some bounded interval.
            >>> import splinekit.interval as ivl
            >>> ivl.OpenClosed((1.0, 5.0)).isleftopen
            True

        Notes
        -----
        `isleftopen = True` for the following classes.
            - ``RR`` All finite real numbers
              :math:`{\mathbb{R}}=(-\infty,\infty)`
            - ``Empty`` The empty interval
              :math:`\{x\in{\mathbb{R}}|x\neq x\}=\emptyset`
            - ``Above`` The right-unbounded interval
              :math:`\{x\in{\mathbb{R}}|x>x_{0}\}=(x_{0},\infty)`
            - ``NotAbove`` The left-unbounded interval
              :math:`\{x\in{\mathbb{R}}|x\leq x_{0}\}=(—\infty,x_{0}]`
            - ``Below`` The left-unbounded interval
              :math:`\{x\in{\mathbb{R}}|x<x_{0}\}=(—\infty,x_{0})`
            - ``Open`` The bounded interval
              :math:`\{x\in{\mathbb{R}}|x_{0}<x<x_{1}\}=(x_{0},x_{1})`
            - ``OpenClosed`` The bounded interval
              :math:`\{x\in{\mathbb{R}}|x_{0}<x\leq x_{1}\}=(x_{0},x_{1}]`
        `isleftopen = False` for the following classes.
            - ``Singleton`` The degenerate interval
              :math:`\{x\in{\mathbb{R}}|x=x_{0}\}=[x_{0},x_{0}]`
            - ``NotBelow`` The right-unbounded interval
              :math:`\{x\in{\mathbb{R}}|x\geq x_{0}\}=[x_{0},\infty)`
            - ``Closed`` The bounded interval
              :math:`\{x\in{\mathbb{R}}|x_{0}\leq x\leq x_{1}\}=[x_{0},x_{1}]`
            - ``ClosedOpen`` The bounded interval
              :math:`\{x\in{\mathbb{R}}|x_{0}\leq x<x_{1}\}=[x_{0},x_{1})`
        """

    #---------------
    @property
    @abstractmethod
    def isrightopen (
        self
    ) -> bool:

        r"""
        .. _isrightopen:

        If the supremum of this interval is a finite number, tests whether
        it fails to be a member of the interval; else, the interval is
        vacuously said to be right-open.

        Returns
        -------
        bool
            - `False` if this interval is not open to the right.
            - `True` if this interval is open to the right.

        Examples
        --------
        Right-openness of some bounded interval.
            >>> import splinekit.interval as ivl
            >>> ivl.OpenClosed((1.0, 5.0)).isrightopen
            False

        Notes
        -----
        `isrightopen = True` for the following classes.
            - ``RR`` All finite real numbers
              :math:`{\mathbb{R}}=(-\infty,\infty)`
            - ``Empty`` The empty interval
              :math:`\{x\in{\mathbb{R}}|x\neq x\}=\emptyset`
            - ``Above`` The right-unbounded interval
              :math:`\{x\in{\mathbb{R}}|x>x_{0}\}=(x_{0},\infty)`
            - ``NotBelow`` The right-unbounded interval
              :math:`\{x\in{\mathbb{R}}|x\geq x_{0}\}=[x_{0},\infty)`
            - ``Below`` The left-unbounded interval
              :math:`\{x\in{\mathbb{R}}|x<x_{0}\}=(—\infty,x_{0})`
            - ``Open`` The bounded interval
              :math:`\{x\in{\mathbb{R}}|x_{0}<x<x_{1}\}=(x_{0},x_{1})`
            - ``ClosedOpen`` The bounded interval
              :math:`\{x\in{\mathbb{R}}|x_{0}\leq x<x_{1}\}=[x_{0},x_{1})`
        `isrightopen = False` for the following classes.
            - ``Singleton`` The degenerate interval
              :math:`\{x\in{\mathbb{R}}|x=x_{0}\}=[x_{0},x_{0}]`
            - ``NotAbove`` The left-unbounded interval
              :math:`\{x\in{\mathbb{R}}|x\leq x_{0}\}=(—\infty,x_{0}]`
            - ``OpenClosed`` The bounded interval
              :math:`\{x\in{\mathbb{R}}|x_{0}<x\leq x_{1}\}=(x_{0},x_{1}]`
            - ``Closed`` The bounded interval
              :math:`\{x\in{\mathbb{R}}|x_{0}\leq x\leq x_{1}\}=[x_{0},x_{1}]`
        """

    #---------------
    @property
    @abstractmethod
    def isopen (
        self
    ) -> bool:

        r"""
        .. _isopen:

        Tests if this interval is jointly left-open and right-open.

        Returns
        -------
        bool
            - `False` if this interval is not open.
            - `True` if this interval is open.

        Examples
        --------
        Openness of some bounded interval.
            >>> import splinekit.interval as ivl
            >>> ivl.OpenClosed((1.0, 5.0)).isopen
            False

        Notes
        -----
        `isopen = True` for the following classes.
            - ``RR`` All finite real numbers
              :math:`{\mathbb{R}}=(-\infty,\infty)`
            - ``Empty`` The empty interval
              :math:`\{x\in{\mathbb{R}}|x\neq x\}=\emptyset`
            - ``Above`` The right-unbounded interval
              :math:`\{x\in{\mathbb{R}}|x>x_{0}\}=(x_{0},\infty)`
            - ``Below`` The left-unbounded interval
              :math:`\{x\in{\mathbb{R}}|x<x_{0}\}=(—\infty,x_{0})`
            - ``Open`` The bounded interval
              :math:`\{x\in{\mathbb{R}}|x_{0}<x<x_{1}\}=(x_{0},x_{1})`
        `isopen = False` for the following classes.
            - ``Singleton`` The degenerate interval
              :math:`\{x\in{\mathbb{R}}|x=x_{0}\}=[x_{0},x_{0}]`
            - ``NotBelow`` The right-unbounded interval
              :math:`\{x\in{\mathbb{R}}|x\geq x_{0}\}=[x_{0},\infty)`
            - ``NotAbove`` The left-unbounded interval
              :math:`\{x\in{\mathbb{R}}|x\leq x_{0}\}=(—\infty,x_{0}]`
            - ``OpenClosed`` The bounded interval
              :math:`\{x\in{\mathbb{R}}|x_{0}<x\leq x_{1}\}=(x_{0},x_{1}]`
            - ``Closed`` The bounded interval
              :math:`\{x\in{\mathbb{R}}|x_{0}\leq x\leq x_{1}\}=[x_{0},x_{1}]`
            - ``ClosedOpen`` The bounded interval
              :math:`\{x\in{\mathbb{R}}|x_{0}\leq x<x_{1}\}=[x_{0},x_{1})`
        """

        return self.isleftopen and self.isrightopen

    #---------------
    @property
    @abstractmethod
    def ishalfopen (
        self
    ) -> bool:

        r"""
        .. _ishalfopen:

        Tests if this interval is open at one end only.

        Returns
        -------
        bool
            - `False` if this interval is not half-open.
            - `True` if this interval is half-open.

        Examples
        --------
        Half-openness of some bounded interval.
            >>> import splinekit.interval as ivl
            >>> ivl.OpenClosed((1.0, 5.0)).ishalfopen
            True

        Notes
        -----
        `ishalfopen = True` for the following classes.
            - ``NotBelow`` The right-unbounded interval
              :math:`\{x\in{\mathbb{R}}|x\geq x_{0}\}=[x_{0},\infty)`
            - ``NotAbove`` The left-unbounded interval
              :math:`\{x\in{\mathbb{R}}|x\leq x_{0}\}=(—\infty,x_{0}]`
            - ``OpenClosed`` The bounded interval
              :math:`\{x\in{\mathbb{R}}|x_{0}<x\leq x_{1}\}=(x_{0},x_{1}]`
            - ``ClosedOpen`` The bounded interval
              :math:`\{x\in{\mathbb{R}}|x_{0}\leq x<x_{1}\}=[x_{0},x_{1})`
        `ishalfopen = False` for the following classes.
            - ``RR`` All finite real numbers
              :math:`{\mathbb{R}}=(-\infty,\infty)`
            - ``Empty`` The empty interval
              :math:`\{x\in{\mathbb{R}}|x\neq x\}=\emptyset`
            - ``Singleton`` The degenerate interval
              :math:`\{x\in{\mathbb{R}}|x=x_{0}\}=[x_{0},x_{0}]`
            - ``Above`` The right-unbounded interval
              :math:`\{x\in{\mathbb{R}}|x>x_{0}\}=(x_{0},\infty)`
            - ``Below`` The left-unbounded interval
              :math:`\{x\in{\mathbb{R}}|x<x_{0}\}=(—\infty,x_{0})`
            - ``Open`` The bounded interval
              :math:`\{x\in{\mathbb{R}}|x_{0}<x<x_{1}\}=(x_{0},x_{1})`
            - ``Closed`` The bounded interval
              :math:`\{x\in{\mathbb{R}}|x_{0}\leq x\leq x_{1}\}=[x_{0},x_{1}]`
        """

        return ((self.isleftopen and not self.isrightopen) or
            (self.isrightopen and not self.isleftopen))

    #---------------
    @property
    @abstractmethod
    def isclosed (
        self
    ) -> bool:

        r"""
        .. _isclosed:

        If the infimum and the supremum of this interval are two finite
        numbers, tests if they are jointly members of the interval;
        else, the interval is vacuously said to be closed.

        Returns
        -------
        bool
            - `False` if this interval is not closed.
            - `True` if this interval is closed.

        Examples
        --------
        Closedness of some bounded interval.
            >>> import splinekit.interval as ivl
            >>> ivl.OpenClosed((1.0, 5.0)).isclosed
            False

        Notes
        -----
        `isclosed = True` for the following classes.
            - ``RR`` All finite real numbers
              :math:`{\mathbb{R}}=(-\infty,\infty)`
            - ``Empty`` The empty interval
              :math:`\{x\in{\mathbb{R}}|x\neq x\}=\emptyset`
            - ``Singleton`` The degenerate interval
              :math:`\{x\in{\mathbb{R}}|x=x_{0}\}=[x_{0},x_{0}]`
            - ``NotBelow`` The right-unbounded interval
              :math:`\{x\in{\mathbb{R}}|x\geq x_{0}\}=[x_{0},\infty)`
            - ``NotAbove`` The left-unbounded interval
              :math:`\{x\in{\mathbb{R}}|x\leq x_{0}\}=(—\infty,x_{0}]`
            - ``Closed`` The bounded interval
              :math:`\{x\in{\mathbb{R}}|x_{0}\leq x\leq x_{1}\}=[x_{0},x_{1}]`
        `isclosed = False` for the following classes.
            - ``Above`` The right-unbounded interval
              :math:`\{x\in{\mathbb{R}}|x>x_{0}\}=(x_{0},\infty)`
            - ``Below`` The left-unbounded interval
              :math:`\{x\in{\mathbb{R}}|x<x_{0}\}=(—\infty,x_{0})`
            - ``Open`` The bounded interval
              :math:`\{x\in{\mathbb{R}}|x_{0}<x<x_{1}\}=(x_{0},x_{1})`
            - ``OpenClosed`` The bounded interval
              :math:`\{x\in{\mathbb{R}}|x_{0}<x\leq x_{1}\}=(x_{0},x_{1}]`
            - ``ClosedOpen`` The bounded interval
              :math:`\{x\in{\mathbb{R}}|x_{0}\leq x<x_{1}\}=[x_{0},x_{1})`
        """

        return self.closure == self

    #---------------
    @property
    @abstractmethod
    def isleftbounded (
        self
    ) -> bool:

        r"""
        .. _isleftbounded:

        Tests whether the infimum of this interval fails to be an infinite
        number.

        Returns
        -------
        bool
            - `False` if this interval is not bounded to the left.
            - `True` if this interval is bounded to the left.

        Examples
        --------
        Left-boundedness of some half-bounded interval.
            >>> import splinekit.interval as ivl
            >>> ivl.Above(3.0).isleftbounded
            True

        Notes
        -----
        `isleftbounded = True` for the following classes.
            - ``Empty`` The empty interval
              :math:`\{x\in{\mathbb{R}}|x\neq x\}=\emptyset`
            - ``Singleton`` The degenerate interval
              :math:`\{x\in{\mathbb{R}}|x=x_{0}\}=[x_{0},x_{0}]`
            - ``Above`` The right-unbounded interval
              :math:`\{x\in{\mathbb{R}}|x>x_{0}\}=(x_{0},\infty)`
            - ``NotBelow`` The right-unbounded interval
              :math:`\{x\in{\mathbb{R}}|x\geq x_{0}\}=[x_{0},\infty)`
            - ``Open`` The bounded interval
              :math:`\{x\in{\mathbb{R}}|x_{0}<x<x_{1}\}=(x_{0},x_{1})`
            - ``OpenClosed`` The bounded interval
              :math:`\{x\in{\mathbb{R}}|x_{0}<x\leq x_{1}\}=(x_{0},x_{1}]`
            - ``Closed`` The bounded interval
              :math:`\{x\in{\mathbb{R}}|x_{0}\leq x\leq x_{1}\}=[x_{0},x_{1}]`
            - ``ClosedOpen`` The bounded interval
              :math:`\{x\in{\mathbb{R}}|x_{0}\leq x<x_{1}\}=[x_{0},x_{1})`
        `isleftbounded = False` for the following classes.
            - ``RR`` All finite real numbers
              :math:`{\mathbb{R}}=(-\infty,\infty)`
            - ``NotAbove`` The left-unbounded interval
              :math:`\{x\in{\mathbb{R}}|x\leq x_{0}\}=(—\infty,x_{0}]`
            - ``Below`` The left-unbounded interval
              :math:`\{x\in{\mathbb{R}}|x<x_{0}\}=(—\infty,x_{0})`
        """

        return not math.isinf(self.infimum)

    #---------------
    @property
    @abstractmethod
    def isrightbounded (
        self
    ) -> bool:

        r"""
        .. _isrightbounded:

        Tests whether the supremum of this interval fails to be an infinite
        number.

        Returns
        -------
        bool
            - `False` if this interval is not bounded to the right.
            - `True` if this interval is bounded to the right.

        Examples
        --------
        Right-boundedness of some half-bounded interval.
            >>> import splinekit.interval as ivl
            >>> ivl.Above(3.0).isrightbounded
            False

        Notes
        -----
        `isrightbounded = True` for the following classes.
            - ``Empty`` The empty interval
              :math:`\{x\in{\mathbb{R}}|x\neq x\}=\emptyset`
            - ``Singleton`` The degenerate interval
              :math:`\{x\in{\mathbb{R}}|x=x_{0}\}=[x_{0},x_{0}]`
            - ``NotAbove`` The left-unbounded interval
              :math:`\{x\in{\mathbb{R}}|x\leq x_{0}\}=(—\infty,x_{0}]`
            - ``Below`` The left-unbounded interval
              :math:`\{x\in{\mathbb{R}}|x<x_{0}\}=(—\infty,x_{0})`
            - ``Open`` The bounded interval
              :math:`\{x\in{\mathbb{R}}|x_{0}<x<x_{1}\}=(x_{0},x_{1})`
            - ``OpenClosed`` The bounded interval
              :math:`\{x\in{\mathbb{R}}|x_{0}<x\leq x_{1}\}=(x_{0},x_{1}]`
            - ``Closed`` The bounded interval
              :math:`\{x\in{\mathbb{R}}|x_{0}\leq x\leq x_{1}\}=[x_{0},x_{1}]`
            - ``ClosedOpen`` The bounded interval
              :math:`\{x\in{\mathbb{R}}|x_{0}\leq x<x_{1}\}=[x_{0},x_{1})`
        `isrightbounded = False` for the following classes.
            - ``RR`` All finite real numbers
              :math:`{\mathbb{R}}=(-\infty,\infty)`
            - ``Above`` The right-unbounded interval
              :math:`\{x\in{\mathbb{R}}|x>x_{0}\}=(x_{0},\infty)`
            - ``NotBelow`` The right-unbounded interval
              :math:`\{x\in{\mathbb{R}}|x\geq x_{0}\}=[x_{0},\infty)`
        """

        return not math.isinf(self.supremum)

    #---------------
    @property
    @abstractmethod
    def ishalfbounded (
        self
    ) -> bool:

        r"""
        .. _ishalfbounded:

        Tests if this interval is bounded at one end only.

        Returns
        -------
        bool
            - `False` if this interval is not half-bounded.
            - `True` if this interval is half-bounded.

        Examples
        --------
        Half-boundedness of some half-bounded interval.
            >>> import splinekit.interval as ivl
            >>> ivl.Above(3.0).ishalfbounded
            True

        Notes
        -----
        `ishalfbounded = True` for the following classes.
            - ``Above`` The right-unbounded interval
              :math:`\{x\in{\mathbb{R}}|x>x_{0}\}=(x_{0},\infty)`
            - ``NotBelow`` The right-unbounded interval
              :math:`\{x\in{\mathbb{R}}|x\geq x_{0}\}=[x_{0},\infty)`
            - ``NotAbove`` The left-unbounded interval
              :math:`\{x\in{\mathbb{R}}|x\leq x_{0}\}=(—\infty,x_{0}]`
            - ``Below`` The left-unbounded interval
              :math:`\{x\in{\mathbb{R}}|x<x_{0}\}=(—\infty,x_{0})`
        `ishalfbounded = False` for the following classes.
            - ``RR`` All finite real numbers
              :math:`{\mathbb{R}}=(-\infty,\infty)`
            - ``Empty`` The empty interval
              :math:`\{x\in{\mathbb{R}}|x\neq x\}=\emptyset`
            - ``Singleton`` The degenerate interval
              :math:`\{x\in{\mathbb{R}}|x=x_{0}\}=[x_{0},x_{0}]`
            - ``Open`` The bounded interval
              :math:`\{x\in{\mathbb{R}}|x_{0}<x<x_{1}\}=(x_{0},x_{1})`
            - ``OpenClosed`` The bounded interval
              :math:`\{x\in{\mathbb{R}}|x_{0}<x\leq x_{1}\}=(x_{0},x_{1}]`
            - ``Closed`` The bounded interval
              :math:`\{x\in{\mathbb{R}}|x_{0}\leq x\leq x_{1}\}=[x_{0},x_{1}]`
            - ``ClosedOpen`` The bounded interval
              :math:`\{x\in{\mathbb{R}}|x_{0}\leq x<x_{1}\}=[x_{0},x_{1})`
        """

        return ((self.isleftbounded and not self.isrightbounded) or
            (self.isrightbounded and not self.isleftbounded))

    #---------------
    @property
    @abstractmethod
    def isbounded (
        self
    ) -> bool:

        r"""
        .. _isbounded:

        Tests if this interval is bounded at both ends jointly.

        Returns
        -------
        bool
            - `False` if this interval is not bounded.
            - `True` if this interval is bounded.

        Examples
        --------
        Boundedness of some half-bounded interval.
            >>> import splinekit.interval as ivl
            >>> ivl.Above(3.0).isbounded
            False

        Notes
        -----
        `isbounded = True` for the following classes.
            - ``Empty`` The empty interval
              :math:`\{x\in{\mathbb{R}}|x\neq x\}=\emptyset`
            - ``Singleton`` The degenerate interval
              :math:`\{x\in{\mathbb{R}}|x=x_{0}\}=[x_{0},x_{0}]`
            - ``Open`` The bounded interval
              :math:`\{x\in{\mathbb{R}}|x_{0}<x<x_{1}\}=(x_{0},x_{1})`
            - ``OpenClosed`` The bounded interval
              :math:`\{x\in{\mathbb{R}}|x_{0}<x\leq x_{1}\}=(x_{0},x_{1}]`
            - ``Closed`` The bounded interval
              :math:`\{x\in{\mathbb{R}}|x_{0}\leq x\leq x_{1}\}=[x_{0},x_{1}]`
            - ``ClosedOpen`` The bounded interval
              :math:`\{x\in{\mathbb{R}}|x_{0}\leq x<x_{1}\}=[x_{0},x_{1})`
        `isbounded = False` for the following classes.
            - ``RR`` All finite real numbers
              :math:`{\mathbb{R}}=(-\infty,\infty)`
            - ``Above`` The right-unbounded interval
              :math:`\{x\in{\mathbb{R}}|x>x_{0}\}=(x_{0},\infty)`
            - ``NotBelow`` The right-unbounded interval
              :math:`\{x\in{\mathbb{R}}|x\geq x_{0}\}=[x_{0},\infty)`
            - ``NotAbove`` The left-unbounded interval
              :math:`\{x\in{\mathbb{R}}|x\leq x_{0}\}=(—\infty,x_{0}]`
            - ``Below`` The left-unbounded interval
              :math:`\{x\in{\mathbb{R}}|x<x_{0}\}=(—\infty,x_{0})`
        """

        return self.isleftbounded and self.isrightbounded

    #---------------
    @property
    @abstractmethod
    def isdegenerate (
        self
    ) -> bool:

        r"""
        .. _isdegenerate:

        Tests if this interval contains a single number.

        Returns
        -------
        bool
            - `False` if this interval is not degenerate.
            - `True` if this interval is degenerate.

        Examples
        --------
        Degeneracy of the Empty interval.
            >>> import splinekit.interval as ivl
            >>> ivl.Empty().isdegenerate
            False

        Notes
        -----
        `isdegenerate = True` for the following class.
            - ``Singleton`` The degenerate interval
              :math:`\{x\in{\mathbb{R}}|x=x_{0}\}=[x_{0},x_{0}]`
        `isdegenerate = False` for the following classes.
            - ``RR`` All finite real numbers
              :math:`{\mathbb{R}}=(-\infty,\infty)`
            - ``Empty`` The empty interval
              :math:`\{x\in{\mathbb{R}}|x\neq x\}=\emptyset`
            - ``Above`` The right-unbounded interval
              :math:`\{x\in{\mathbb{R}}|x>x_{0}\}=(x_{0},\infty)`
            - ``NotBelow`` The right-unbounded interval
              :math:`\{x\in{\mathbb{R}}|x\geq x_{0}\}=[x_{0},\infty)`
            - ``NotAbove`` The left-unbounded interval
              :math:`\{x\in{\mathbb{R}}|x\leq x_{0}\}=(—\infty,x_{0}]`
            - ``Below`` The left-unbounded interval
              :math:`\{x\in{\mathbb{R}}|x<x_{0}\}=(—\infty,x_{0})`
            - ``Open`` The bounded interval
              :math:`\{x\in{\mathbb{R}}|x_{0}<x<x_{1}\}=(x_{0},x_{1})`
            - ``OpenClosed`` The bounded interval
              :math:`\{x\in{\mathbb{R}}|x_{0}<x\leq x_{1}\}=(x_{0},x_{1}]`
            - ``Closed`` The bounded interval
              :math:`\{x\in{\mathbb{R}}|x_{0}\leq x\leq x_{1}\}=[x_{0},x_{1}]`
            - ``ClosedOpen`` The bounded interval
              :math:`\{x\in{\mathbb{R}}|x_{0}\leq x<x_{1}\}=[x_{0},x_{1})`
        """

        if math.isnan(self.infimum) and math.isnan(self.supremum):
            return False
        return self.infimum == self.supremum

    #---------------
    @property
    @abstractmethod
    def isproper (
        self
    ) -> bool:

        r"""
        .. _isproper:

        Tests if this interval contains a range of numbers.

        Returns
        -------
        bool
            - `False` if this interval is not proper.
            - `True` if this interval is proper.

        Examples
        --------
        Properness of the Singleton interval.
            >>> import splinekit.interval as ivl
            >>> ivl.Singleton(6.0).isproper
            False

        Notes
        -----
        `isproper = True` for the following class.
            - ``RR`` All finite real numbers
              :math:`{\mathbb{R}}=(-\infty,\infty)`
            - ``Above`` The right-unbounded interval
              :math:`\{x\in{\mathbb{R}}|x>x_{0}\}=(x_{0},\infty)`
            - ``NotBelow`` The right-unbounded interval
              :math:`\{x\in{\mathbb{R}}|x\geq x_{0}\}=[x_{0},\infty)`
            - ``NotAbove`` The left-unbounded interval
              :math:`\{x\in{\mathbb{R}}|x\leq x_{0}\}=(—\infty,x_{0}]`
            - ``Below`` The left-unbounded interval
              :math:`\{x\in{\mathbb{R}}|x<x_{0}\}=(—\infty,x_{0})`
            - ``Open`` The bounded interval
              :math:`\{x\in{\mathbb{R}}|x_{0}<x<x_{1}\}=(x_{0},x_{1})`
            - ``OpenClosed`` The bounded interval
              :math:`\{x\in{\mathbb{R}}|x_{0}<x\leq x_{1}\}=(x_{0},x_{1}]`
            - ``Closed`` The bounded interval
              :math:`\{x\in{\mathbb{R}}|x_{0}\leq x\leq x_{1}\}=[x_{0},x_{1}]`
            - ``ClosedOpen`` The bounded interval
              :math:`\{x\in{\mathbb{R}}|x_{0}\leq x<x_{1}\}=[x_{0},x_{1})`
        `isproper = False` for the following classes.
            - ``Empty`` The empty interval
              :math:`\{x\in{\mathbb{R}}|x\neq x\}=\emptyset`
            - ``Singleton`` The degenerate interval
              :math:`\{x\in{\mathbb{R}}|x=x_{0}\}=[x_{0},x_{0}]`
        """

        if math.isnan(self.infimum) and math.isnan(self.supremum):
            return False
        return self.infimum < self.supremum

    #---------------
    @property
    @abstractmethod
    def interior (
        self
    ) -> Self:

        r"""
        .. _interior:

        The largest open interval that is contained in this interval.

        Returns
        -------
        Interval
            The interior of this interval.

        Examples
        --------
        The interior of an OpenClosed interval is an Open interval.
            >>> import splinekit.interval as ivl
            >>> ivl.OpenClosed((1.0, 5.0)).interior
            Open((1.0, 5.0))

        Notes
        -----
        The following class has `RR()` as interior.
            - ``RR`` All finite real numbers
              :math:`{\mathbb{R}}=(-\infty,\infty)`
        The following classes have `Empty()` as interior.
            - ``Empty`` The empty interval
              :math:`\{x\in{\mathbb{R}}|x\neq x\}=\emptyset`
            - ``Singleton`` The degenerate interval
              :math:`\{x\in{\mathbb{R}}|x=x_{0}\}=[x_{0},x_{0}]`
        The following classes have an object of class `Above` as interior.
            - ``Above`` The right-unbounded interval
              :math:`\{x\in{\mathbb{R}}|x>x_{0}\}=(x_{0},\infty)`
            - ``NotBelow`` The right-unbounded interval
              :math:`\{x\in{\mathbb{R}}|x\geq x_{0}\}=[x_{0},\infty)`
        The following classes have an object of class `Below` as interior.
            - ``NotAbove`` The left-unbounded interval
              :math:`\{x\in{\mathbb{R}}|x\leq x_{0}\}=(—\infty,x_{0}]`
            - ``Below`` The left-unbounded interval
              :math:`\{x\in{\mathbb{R}}|x<x_{0}\}=(—\infty,x_{0})`
        The following classes have an object of class `Open` as interior.
            - ``Open`` The bounded interval
              :math:`\{x\in{\mathbb{R}}|x_{0}<x<x_{1}\}=(x_{0},x_{1})`
            - ``OpenClosed`` The bounded interval
              :math:`\{x\in{\mathbb{R}}|x_{0}<x\leq x_{1}\}=(x_{0},x_{1}]`
            - ``Closed`` The bounded interval
              :math:`\{x\in{\mathbb{R}}|x_{0}\leq x\leq x_{1}\}=[x_{0},x_{1}]`
            - ``ClosedOpen`` The bounded interval
              :math:`\{x\in{\mathbb{R}}|x_{0}\leq x<x_{1}\}=[x_{0},x_{1})`
        """

    #---------------
    @property
    @abstractmethod
    def closure (
        self
    ) -> Self:

        r"""
        .. _closure:

        The smallest closed interval that contains this interval.

        Returns
        -------
        Interval
            The closure of this interval.

        Examples
        --------
        The closure of an OpenClosed interval is a Closed interval.
            >>> import splinekit.interval as ivl
            >>> ivl.OpenClosed((1.0, 5.0)).closure
            Closed((1.0, 5.0))

        Notes
        -----
        The following class has `RR()` as closure.
            - ``RR`` All finite real numbers
              :math:`{\mathbb{R}}=(-\infty,\infty)`
        The following class has `Empty()` as closure.
            - ``Empty`` The empty interval
              :math:`\{x\in{\mathbb{R}}|x\neq x\}=\emptyset`
        The following class has `Singleton()` as closure.
            - ``Singleton`` The degenerate interval
              :math:`\{x\in{\mathbb{R}}|x=x_{0}\}=[x_{0},x_{0}]`
        The following classes have an object of class `NotBelow` as closure.
            - ``Above`` The right-unbounded interval
              :math:`\{x\in{\mathbb{R}}|x>x_{0}\}=(x_{0},\infty)`
            - ``NotBelow`` The right-unbounded interval
              :math:`\{x\in{\mathbb{R}}|x\geq x_{0}\}=[x_{0},\infty)`
        The following classes have an object of class `NotAbove` as closure.
            - ``NotAbove`` The left-unbounded interval
              :math:`\{x\in{\mathbb{R}}|x\leq x_{0}\}=(—\infty,x_{0}]`
            - ``Below`` The left-unbounded interval
              :math:`\{x\in{\mathbb{R}}|x<x_{0}\}=(—\infty,x_{0})`
        The following classes have an object of class `Closed` as closure.
            - ``Open`` The bounded interval
              :math:`\{x\in{\mathbb{R}}|x_{0}<x<x_{1}\}=(x_{0},x_{1})`
            - ``OpenClosed`` The bounded interval
              :math:`\{x\in{\mathbb{R}}|x_{0}<x\leq x_{1}\}=(x_{0},x_{1}]`
            - ``Closed`` The bounded interval
              :math:`\{x\in{\mathbb{R}}|x_{0}\leq x\leq x_{1}\}=[x_{0},x_{1}]`
            - ``ClosedOpen`` The bounded interval
              :math:`\{x\in{\mathbb{R}}|x_{0}\leq x<x_{1}\}=[x_{0},x_{1})`
        """

    #---------------
    @property
    @abstractmethod
    def diameter (
        self
    ) -> float:

        r"""
        .. _diameter:

        The distance between the two endpoints of this interval.

        Returns
        -------
        float
            The diameter of this interval.

        Examples
        --------
        Here is the diameter of an OpenClosed interval.
            >>> import splinekit.interval as ivl
            >>> ivl.OpenClosed((1.0, 5.0)).diameter
            4.0

        Notes
        -----
        `diameter = float("nan")` for the following class.
            - ``Empty`` The empty interval
              :math:`\{x\in{\mathbb{R}}|x\neq x\}=\emptyset`
        `diameter = float("inf")` for the following classes.
            - ``RR`` All finite real numbers
              :math:`{\mathbb{R}}=(-\infty,\infty)`
            - ``Above`` The right-unbounded interval
              :math:`\{x\in{\mathbb{R}}|x>x_{0}\}=(x_{0},\infty)`
            - ``NotBelow`` The right-unbounded interval
              :math:`\{x\in{\mathbb{R}}|x\geq x_{0}\}=[x_{0},\infty)`
            - ``NotAbove`` The left-unbounded interval
              :math:`\{x\in{\mathbb{R}}|x\leq x_{0}\}=(—\infty,x_{0}]`
            - ``Below`` The left-unbounded interval
              :math:`\{x\in{\mathbb{R}}|x<x_{0}\}=(—\infty,x_{0})`
        `diameter = 0.0` for the following class.
            - ``Singleton`` The degenerate interval
              :math:`\{x\in{\mathbb{R}}|x=x_{0}\}=[x_{0},x_{0}]`
        `math.isfinite(diameter) = True` for the following classes.
            - ``Open`` The bounded interval
              :math:`\{x\in{\mathbb{R}}|x_{0}<x<x_{1}\}=(x_{0},x_{1})`
            - ``OpenClosed`` The bounded interval
              :math:`\{x\in{\mathbb{R}}|x_{0}<x\leq x_{1}\}=(x_{0},x_{1}]`
            - ``Closed`` The bounded interval
              :math:`\{x\in{\mathbb{R}}|x_{0}\leq x\leq x_{1}\}=[x_{0},x_{1}]`
            - ``ClosedOpen`` The bounded interval
              :math:`\{x\in{\mathbb{R}}|x_{0}\leq x<x_{1}\}=[x_{0},x_{1})`
        """

        if math.isinf(self.infimum) or math.isinf(self.supremum):
            return float("inf")
        return self.supremum - self.infimum

    #---------------
    @property
    @abstractmethod
    def midpoint (
        self
    ) -> float:

        r"""
        .. _midpoint:

        The midway point between the two endpoints of this interval.

        Returns
        -------
        float
            The midpoint of this interval.

        Examples
        --------
        Here is the midpoint of an OpenClosed interval.
            >>> import splinekit.interval as ivl
            >>> ivl.OpenClosed((1.0, 5.0)).midpoint
            3.0

        Notes
        -----
        `midpoint = float("nan")` for the following classes.
            - ``RR`` All finite real numbers
              :math:`{\mathbb{R}}=(-\infty,\infty)`
            - ``Empty`` The empty interval
              :math:`\{x\in{\mathbb{R}}|x\neq x\}=\emptyset`
            - ``Above`` The right-unbounded interval
              :math:`\{x\in{\mathbb{R}}|x>x_{0}\}=(x_{0},\infty)`
            - ``NotBelow`` The right-unbounded interval
              :math:`\{x\in{\mathbb{R}}|x\geq x_{0}\}=[x_{0},\infty)`
            - ``NotAbove`` The left-unbounded interval
              :math:`\{x\in{\mathbb{R}}|x\leq x_{0}\}=(—\infty,x_{0}]`
            - ``Below`` The left-unbounded interval
              :math:`\{x\in{\mathbb{R}}|x<x_{0}\}=(—\infty,x_{0})`
        `math.isfinite(midpoint) = True` for the following classes.
            - ``Singleton`` The degenerate interval
              :math:`\{x\in{\mathbb{R}}|x=x_{0}\}=[x_{0},x_{0}]`
            - ``Open`` The bounded interval
              :math:`\{x\in{\mathbb{R}}|x_{0}<x<x_{1}\}=(x_{0},x_{1})`
            - ``OpenClosed`` The bounded interval
              :math:`\{x\in{\mathbb{R}}|x_{0}<x\leq x_{1}\}=(x_{0},x_{1}]`
            - ``Closed`` The bounded interval
              :math:`\{x\in{\mathbb{R}}|x_{0}\leq x\leq x_{1}\}=[x_{0},x_{1}]`
            - ``ClosedOpen`` The bounded interval
              :math:`\{x\in{\mathbb{R}}|x_{0}\leq x<x_{1}\}=[x_{0},x_{1})`
        """

        if math.isfinite(self.infimum) and math.isfinite(self.supremum):
            if self.infimum == self.supremum:
                return self.infimum
            return 0.5 * (self.infimum + self.supremum)
        return float("nan")

    #---------------
    @abstractmethod
    def sortorder (
        self
    ) -> float:

        r"""
        .. _sortorder:

        Convenience method to sort the intervals. This method is not
        consistent with equal.

        Returns
        -------
        float
            A scalar proxy of this interval.

        Notes
        -----
        `sortorder = float("-inf")` for the following class.
            - ``Empty`` The empty interval
              :math:`\{x\in{\mathbb{R}}|x\neq x\}=\emptyset`
        `sortorder = float("inf")` for the following class.
            - ``RR`` All finite real numbers
              :math:`{\mathbb{R}}=(-\infty,\infty)`
        `sortorder` is :math:`x_{0}` for the following classes.
            - ``Singleton`` The degenerate interval
              :math:`\{x\in{\mathbb{R}}|x=x_{0}\}=[x_{0},x_{0}]`
            - ``Above`` The right-unbounded interval
              :math:`\{x\in{\mathbb{R}}|x>x_{0}\}=(x_{0},\infty)`
            - ``NotBelow`` The right-unbounded interval
              :math:`\{x\in{\mathbb{R}}|x\geq x_{0}\}=[x_{0},\infty)`
            - ``NotAbove`` The left-unbounded interval
              :math:`\{x\in{\mathbb{R}}|x\leq x_{0}\}=(—\infty,x_{0}]`
            - ``Below`` The left-unbounded interval
              :math:`\{x\in{\mathbb{R}}|x<x_{0}\}=(—\infty,x_{0})`
        `sortorder` is :math:`\left(x_{0}+x_{1}\right)/2` for the following classes.
            - ``Open`` The bounded interval
              :math:`\{x\in{\mathbb{R}}|x_{0}<x<x_{1}\}=(x_{0},x_{1})`
            - ``OpenClosed`` The bounded interval
              :math:`\{x\in{\mathbb{R}}|x_{0}<x\leq x_{1}\}=(x_{0},x_{1}]`
            - ``Closed`` The bounded interval
              :math:`\{x\in{\mathbb{R}}|x_{0}\leq x\leq x_{1}\}=[x_{0},x_{1}]`
            - ``ClosedOpen`` The bounded interval
              :math:`\{x\in{\mathbb{R}}|x_{0}\leq x<x_{1}\}=[x_{0},x_{1})`
        """

    #---------------
    @abstractmethod
    def __contains__ (
        self,
        x: float
    ) -> bool:

        r"""
        .. _contains:

        Tests if the number :math:`x` is contained in this interval :math:`U,`
        as in :math:`\left(x\in U\right).`

        Parameters
        ----------
        x : float
            Real number whose membership is queried.

        Returns
        -------
        bool
            - `False` if ``x`` is not a member of this interval.
            - `True` if ``x`` is a member of this interval.

        Examples
        --------
        Load the library.
            >>> import splinekit.interval as ivl
        ``0.0`` is in ``Singleton(0.0)``.
            >>> 0.0 in ivl.Singleton(0.0)
            True
        ``float("inf")`` is not in ``RR()``.
            >>> float("inf") in ivl.RR()
            False
        """

    #---------------
    @abstractmethod
    def __hash__ (
        self
    ) -> int:
        return hash(self.__class__)

    #---------------
    @classmethod
    def ismember (
        cls,
        x: float,
        s: Set[Self]
    ) -> bool:

        """
        .. _ismember:

        Tests the membership of a real number to a set of intervals.

        Parameters
        ----------
        x : float
            Real number whose membership is queried.
        s : set of Interval
            Set of intervals that are interrogated for the membership of ``x``.

        Returns
        -------
        bool
            - `False` if ``x`` does not belong to any interval in ``s``.
            - `True` if ``x`` belongs to at least one interval in ``s``.

        Examples
        --------
        Load the library.
            >>> import splinekit.interval as ivl
        Every finite number is a member of ``RR()``.
            >>> ivl.Interval.ismember(1.0, {ivl.RR()})
            True
        Infinite numbers do not belong to ``RR()``.
            >>> ivl.Interval.ismember(float("-inf"), {ivl.RR()})
            False
        """

        for s0 in s:
            if x in s0:
                return True
        return False

    #---------------
    @classmethod
    def enclosure (
        cls,
        s: Set[Self]
    ) -> Self:

        """
        .. _enclosure:

        Determines the enclosure (or span) of a set of intervals.

        Parameters
        ----------
        s : set of Interval
            Set of intervals whose enclosure is sought for.

        Returns
        -------
        Interval
            The smallest interval that contains the numbers found in ``s``.

        Examples
        --------
        Load the library.
            >>> import splinekit.interval as ivl
        Here is the enclosure of a pair of bounded intervals.
            >>> ivl.Interval.enclosure({ivl.Open((2.0, 3.0)), ivl.Closed((5.0, 8.0))})
            OpenClosed([2.0, 8.0])
        The enclosure of ``{RR()}`` is ``RR()``.
            >>> ivl.Interval.enclosure({ivl.RR()})
            RR()
        The enclosure of ``{Empty()}`` is ``Empty()``.
            >>> ivl.Interval.enclosure({ivl.Empty()})
            Empty()
        The enclosure of an empty set of intervals is ``Empty()``.
            >>> ivl.Interval.enclosure({})
            Empty()
        """

        _ki = _KnownIntervals()
        le = float("nan")
        re = float("nan")
        lc = False
        rc = False
        for s0 in s:
            if _ki.empty == s:
                continue
            if math.isnan(le):
                le = s0.infimum
                lc = not s0.isleftopen
            elif s0.infimum < le:
                le = s0.infimum
                lc = not s0.isleftopen
            if s0.infimum == le:
                lc |= not s0.isleftopen
            if math.isnan(re):
                re = s0.supremum
                rc = not s0.isrightopen
            elif re < s0.supremum:
                re = s0.supremum
                rc = not s0.isrightopen
            if s0.supremum == re:
                rc |= not s0.isrightopen
        if math.isinf(le) and math.isinf(re) and (not lc) and (not rc):
            return cast(Self, _ki.rr)
        if math.isnan(le) and math.isnan(re) and (not lc) and (not rc):
            return cast(Self, _ki.empty)
        if le == re and lc and rc:
            return cast(Self, _ki.singleton.Singleton(le))
        if math.isfinite(le) and math.isinf(re) and (not lc) and (not rc):
            return cast(Self, _ki.above.Above(le))
        if math.isfinite(le) and math.isinf(re) and lc and (not rc):
            return cast(Self, _ki.notbelow.NotBelow(le))
        if math.isinf(le) and math.isfinite(re) and (not lc) and rc:
            return cast(Self, _ki.notabove.NotAbove(re))
        if math.isinf(le) and math.isfinite(re) and (not lc) and (not rc):
            return cast(Self, _ki.below.Below(re))
        if math.isfinite(le) and math.isfinite(re) and (not lc) and (not rc):
            return cast(Self, _ki.open.Open((le, re)))
        if math.isfinite(le) and math.isfinite(re) and (not lc) and rc:
            return cast(Self, _ki.openclosed.OpenClosed((le, re)))
        if math.isfinite(le) and math.isfinite(re) and lc and rc:
            return cast(Self, _ki.closed.Closed((le, re)))
        if math.isfinite(le) and math.isfinite(re) and lc and (not rc):
            return cast(Self, _ki.closedopen.ClosedOpen((le, re)))
        raise ValueError("Internal error (unexpected enclosure)")

    #---------------
    @classmethod
    def complement (
        cls,
        s: Set[Self]
    ) -> Set[Self]:

        """
        .. _complement:

        Determines the complement (with respect to the real numbers) of a
        set of intervals.

        Parameters
        ----------
        s : set of Interval
            Set of intervals whose complement is sought for.

        Returns
        -------
        set of Interval
            A set of nonoverlapping intervals that contain every number
            not found in ``s``.

        Examples
        --------
        Load the library.
            >>> import splinekit.interval as ivl
        Here is the complement of a pair of bounded intervals.
            >>> ivl.Interval.complement({ivl.Open((2.0, 3.0)), ivl.Closed((5.0, 8.0))})
            {Above(8.0), NotAbove(2.0), ClosedOpen([3.0, 5.0])}
        The complement of the set ``{Empty()}`` is ``{RR()}``.
            >>> ivl.Interval.complement({ivl.Empty()})
            {RR()}
        The complement of an empty set of intervals is ``{RR()}``.
            >>> ivl.Interval.complement({})
            {RR()}
        """

        _ki = _KnownIntervals()
        su = Interval.union(cast(Set[Interval], s))
        if _ki.rr in su:
            return {cast(Self, _ki.empty)}
        if _ki.empty in su:
            return {cast(Self, _ki.rr)}
        epset = {float("-inf"), float("inf")}
        for ivl in s:
            if isinstance(ivl, Degenerate):
                epset.add(ivl.value)
            elif isinstance(ivl, HalfBounded):
                epset.add(ivl.threshold)
            elif isinstance(ivl, Bounded):
                epset.add(ivl.infimum)
                epset.add(ivl.supremum)
        ep = list(epset)
        ep.sort()
        c: set[Self] = set()
        lb = float("-inf")
        for ub in ep:
            ivl = cast(Self, _ki.open.Open((lb, ub)))
            lb = ub
            if _ki.empty == ivl:
                continue
            sgt = _ki.singleton.Singleton(ivl.supremum)
            if isinstance(ivl, _ki.below.Below):
                if not Interval.ismember(
                    ivl.threshold - math.ulp(ivl.threshold),
                    cast(Set[Interval], s)
                ):
                    if _ki.empty != sgt:
                        if Interval.ismember(
                            sgt.value,
                            cast(Set[Interval], s)
                        ):
                            c.add(cast(Self, ivl))
                        else:
                            c.add(cast(
                                Self,
                                _ki.notabove.NotAbove(sgt.value)
                            ))
                    continue
                if _ki.empty != sgt:
                    if not Interval.ismember(
                        sgt.value,
                        cast(Set[Interval], s)
                    ):
                        c.add(cast(Self, sgt))
                continue
            if isinstance(ivl, _ki.open.Open):
                if not Interval.ismember(ivl.midpoint, cast(Set[Interval], s)):
                    if _ki.empty != sgt:
                        if Interval.ismember(
                            sgt.value,
                            cast(Set[Interval], s)
                        ):
                            c.add(cast(Self, ivl))
                        else:
                            c.add(cast(Self, _ki.openclosed.OpenClosed((
                                ivl.infimum,
                                ivl.supremum
                            ))))
                    continue
                if _ki.empty != sgt:
                    if not Interval.ismember(
                        sgt.value,
                        cast(Set[Interval], s)
                    ):
                        c.add(cast(Self, sgt))
                continue
            if isinstance(ivl, _ki.above.Above):
                if not Interval.ismember(
                    ivl.threshold + math.ulp(ivl.threshold),
                    cast(Set[Interval], s)
                ):
                    c.add(cast(Self, ivl))
        return cast(Set[Self], Interval.union(cast(Set[Interval], c)))

    #---------------
    @classmethod
    def intersection (
        cls,
        s: Set[Self]
    ) -> Self:

        """
        .. _intersection:

        Determines the intersection of a set of intervals.

        Parameters
        ----------
        s : set of Interval
            Set of intervals whose intersection is sought for.

        Returns
        -------
        Interval
            The smallest interval that contains all those numbers found
            jointly in every interval of ``s``.

        Examples
        --------
        Load the library.
            >>> import splinekit.interval as ivl
        Here is the intersection of a pair of overlapping intervals.
            >>> ivl.Interval.intersection({ivl.Open((2.0, 8.0)), ivl.NotAbove((5.0))})
            OpenClosed([2.0, 5.0])
        The intersection of an empty set of intervals is ``{Empty()}``.
            >>> ivl.Interval.intersection({})
            Empty()
        """

        _ki = _KnownIntervals()
        if 0 == len(s):
            return cast(Self, _ki.empty)
        r = cast(Self, _ki.rr)
        for ivl in s:
            r &= ivl
        return r

    #---------------
    @classmethod
    def union (
        cls,
        s: Set[Self]
    ) -> Set[Self]:

        """
        .. _union:

        Determines the union of a set of intervals.

        Parameters
        ----------
        s : set of Interval
            Set of intervals whose union is sought for.

        Returns
        -------
        set of Interval
            A set of nonoverlapping intervals that contain every number
            found in ``s`` and that contain no other number.

        Examples
        --------
        Load the library.
            >>> import splinekit.interval as ivl
        Here is the union of a pair of overlapping intervals.
            >>> ivl.Interval.union({ivl.Open((2.0, 8.0)), ivl.NotAbove((5.0))})
            {Below(8.0)}
        The union of the set ``{Empty()}`` is ``{Empty()}``.
            >>> ivl.Interval.union({ivl.Empty()})
            {Empty()}
        The union of an empty set of intervals is ``{Empty()}``.
            >>> ivl.Interval.union({})
            {Empty()}
        """

        _ki = _KnownIntervals()
        i0 = cast(Self, _ki.empty)
        if 0 == len(s):
            return {i0}
        u = list(s)
        def neigh (
            p: int
        ) -> List[int]:
            if i0 == u[p]:
                return []
            return [p] + [
                k for k in range(p + 1, len(u))
                if (i0 != u[k]) and (u[p].iskissing(u[k]) or
                    u[p].isoverlapping(u[k]))
            ]
        e = [neigh(p) for p in range(len(u))]
        def cc (
            p: int
        ) -> List[Self]:
            if not p in e[p]:
                return []
            c = [u[p]]
            q = e[p][-1]
            del e[p][-1]
            c += cc(q)
            while 0 < len(e[p]):
                q = e[p][-1]
                del e[p][-1]
                c += cc(q)
            for q in range(p):
                try:
                    k = e[q].index(p)
                    del e[q][k]
                    c += cc(q)
                except ValueError:
                    continue
            return c
        s0: List[Self] = []
        for p in range(len(u)):
            c: List[Self] = cc(p)
            if 0 == len(c):
                continue
            v = c[0]
            for v0 in c:
                v = list(v | v0)[0]
            s0 += [v]
        return set(s0) if 0 < len(s0) else {i0}

    #---------------
    @classmethod
    def gravitycenter (
        cls,
        ivls: List[Tuple[Self, float]]
    ) -> float:

        r"""
        .. _gravitycenter:

        Determines the center of gravity of a set of weighted intervals.

        Parameters
        ----------
        ivls : list of tuple (Interval, float)
            List of weighted intervals whose center of gravity is sought for.
            For nondegenerate intervals, the second member of the tuple gives
            the density of the interval, in mass per unit length;
            for Singleton intervals, the second memebr of the tuple gives the
            pointwise mass.

        Returns
        -------
        float
            The center of gravity.

        Examples
        --------
        Load the library.
            >>> import splinekit.interval as ivl
        Here is the center of gravity of a pair of bounded intervals.
            >>> ivl.Interval.gravitycenter([(ivl.Open((2.0, 8.0)), 1.0), (ivl.Singleton(3.0), 6.0)])
            4.0
        Here is the center of gravity of a balanced trio of unbounded intervals.
            >>> ivl.Interval.gravitycenter([(ivl.Below(2.0), 1.0), (ivl.Below(9.0), 3.0), (ivl.Above(8.0), 4.0)])
            1.5

        Notes
        -----
            The center of gravity :math:`g` is such that
            :math:`0=S(g)+U(g)+V(g)+B(g),`
            where the contribution of the :math:`S_{0}` singletons of value
            :math:`x_{k}` and weight :math:`w_{k}` is
            :math:`S(g)=\sum_{k=1}^{S_{0}}\,w_{k}\,(x_{k}-g),`
            the contribution of the :math:`U_{0}` intervals that are not
            left-bounded, have threshold :math:`u_{k},` and have linear density
            :math:`\mu_{k}` is :math:`U(g)=\sum_{k=1}^{U_{0}}\,
            \int_{-\infty}^{u_{k}}\,\mu_{k}\,(u-g)\,{\mathrm{d}}u,` the
            contribution of the :math:`V_{0}` intervals that are not
            right-bounded, have threshold :math:`v_{k},` and have linear
            density :math:`\nu_{k}` is :math:`V(g)=\sum_{k=1}^{V_{0}}\,
            \int_{v_{k}}^{\infty}\,\nu_{k}\,(v-g)\,{\mathrm{d}}v,` and the
            contribution of the :math:`B_{0}` bounded intervals with
            infimum :math:`a_{k},` supremum :math:`b_{k},` and linear density
            :math:`\rho_{k}` is :math:`B(g)=\sum_{k=1}^{B_{0}}\,
            \int_{a_{k}}^{b_{k}}\,\rho_{k}\,(x-g)\,{\mathrm{d}}x.`

            For intervals such that
            :math:`\sum_{k=1}^{U_{0}}\,\mu_{k}=\sum_{k=1}^{V_{0}}\,\nu_{k},`
            the system is balanced and a center of gravity generally exists
            as a real number.

            For intervals such that
            :math:`\sum_{k=1}^{U_{0}}\,\mu_{k}>\sum_{k=1}^{V_{0}}\,\nu_{k},`
            the system is unbalanced and leans to the left. The center of
            gravity is returned as ``float("-inf")``.

            For intervals such that
            :math:`\sum_{k=1}^{U_{0}}\,\mu_{k}<\sum_{k=1}^{V_{0}}\,\nu_{k},`
            the system is unbalanced and leans to the right. The center of
            gravity is returned as ``float("inf")``.

            In some cases, the center of gravity will fail to exist at all.
            If such is the case, ``float("nan")`` is returned.
        """

        _ki = _KnownIntervals()
        balance = 0.0
        min_thr = float("inf")
        max_thr = float("-inf")
        n_thr = 0
        s = 0.0
        s2 = 0.0
        for (ivl, w) in ivls:
            if isinstance(ivl, Universal):
                continue
            if isinstance(ivl, Degenerate):
                s += w
                s2 += 2.0 * w * ivl.value
                continue
            if isinstance(ivl, (_ki.notabove.NotAbove, _ki.below.Below)):
                s += w * ivl.threshold
                s2 += w * (ivl.threshold ** 2)
                balance -= w
                min_thr = min(min_thr, ivl.threshold)
                max_thr = max(max_thr, ivl.threshold)
                n_thr += 1
                continue
            if isinstance(ivl, (_ki.above.Above, _ki.notbelow.NotBelow)):
                s -= w * ivl.threshold
                s2 -= w * (ivl.threshold ** 2)
                balance += w
                min_thr = min(min_thr, ivl.threshold)
                max_thr = max(max_thr, ivl.threshold)
                n_thr += 1
                continue
            if isinstance(ivl, Bounded):
                s += w * (ivl.supremum - ivl.infimum)
                s2 += w * (ivl.supremum ** 2 - ivl.infimum ** 2)
                continue
            raise ValueError("Internal error (unexpected interval class)")
        if not math.isclose(
                balance,
                0.0,
                rel_tol = 0.0,
                abs_tol = (math.ulp(n_thr * (1.0 + max_thr - min_thr))
                    if (math.isfinite(min_thr) and math.isfinite(max_thr))
                    else 0.0)
            ):
            return balance * float("inf")
        if 0.0 != s:
            return 0.5 * s2 / s
        return float("nan")

    #---------------
    def iskissing (
        self,
        other: Self
    ) -> bool:

        """
        .. _iskissing:

        Tests if this interval ``self`` is neighboring the interval ``other``,
        as in
        ``(self & other == Empty()) and (1 == len(union({self, other})))``.

        Parameters
        ----------
        other : Interval
            Interval whose vicinity is queried.

        Returns
        -------
        bool
            - `False` if this interval is not kissing ``other``.
            - `True` if this interval is kissing ``other``.

        Examples
        --------
        Load the library.
            >>> import splinekit.interval as ivl
        Here are two kissing intervals.
            >>> ivl.Closed((1.0, 3.0)).iskissing(ivl.Above(3.0))
            True
        Here are two non-kissing intervals.
            >>> ivl.Below(0.0).iskissing(ivl.Above(0.0))
            False
        """

        if self.infimum == other.supremum:
            return self.isleftopen ^ other.isrightopen
        if self.supremum == other.infimum:
            return self.isrightopen ^ other.isleftopen
        return False

    #---------------
    def isoverlapping (
        self,
        other: Self
    ) -> bool:

        """
        .. _isoverlapping:

        Tests if this interval ``self`` and the interval ``other`` share
        at least one number.

        Parameters
        ----------
        other : Interval
            Interval whose intersection is queried.

        Returns
        -------
        bool
            - `False` if this interval is not overlapping ``other``.
            - `True` if this interval is overlapping ``other``.

        Examples
        --------
        Load the library.
            >>> import splinekit.interval as ivl
        Here are two overlapping intervals.
            >>> ivl.Closed((1.0, 3.0)).isoverlapping(ivl.NotBelow(3.0))
            True
        Here are two non-overlapping intervals.
            >>> ivl.Below(0.0).isoverlapping(ivl.Above(0.0))
            False
        """

        _ki = _KnownIntervals()
        return _ki.empty != (self & other)

    #---------------
    def partition (
        self,
        cuts: Set[float]
    ) -> List[Self]:

        """
        .. _partition:

        Creates a list of nonoverlapping intervals whose union is equal to
        this interval. The created list contains an alternance of open and
        degenerate intervals, where ``cuts`` imposes the values associated
        to the degenerate ``Singleton`` intervals.

        Parameters
        ----------
        cuts : set of float
            Endpoints of the proper intervals of the partition.

        Returns
        -------
        list of Interval
            Kissing intervals that partition this interval.

        Examples
        --------
        Here is some partition of a left-unbounded interval. Observe that it
            is not required that every element of ``cuts`` belong to this interval.
            >>> import splinekit.interval as ivl
            >>> ivl.NotAbove(5.0).partition({2.0, 4.0, 6.0})
            [Below(2.0), Singleton(2.0), Open((2.0, 4.0)), Singleton(4.0), Open((4.0, 5.0)), Singleton(5.0)]
        """

        _ki = _KnownIntervals()
        if 0 == len(cuts):
            return [self]
        p: List[Self] = []
        c = cuts.copy()
        c0 = float("-inf")
        c.add(c0)
        c.add(float("inf"))
        s = list(c)
        s.sort()
        for c1 in s:
            ivl = cast(Self, _ki.openclosed.OpenClosed((c0, c1))) & self
            c0 = c1
            if _ki.empty == ivl:
                continue
            if ivl.isrightopen:
                p.append(cast(Self, ivl))
            else:
                s1 = cast(Self, _ki.singleton.Singleton(ivl.supremum))
                p.append(list(ivl - s1)[0])
                if _ki.empty != s1:
                    p.append(s1)
        return p if 0 < len(p) else [self]

    #---------------
    def copy (
        self
    ) -> Self:

        """
        .. _copy:

        Creates a copy of this interval.

        Returns
        -------
        Interval
            A copy of this interval.

        Notes
        -----
        The classes ``Empty`` and ``RR`` have a single instance.
            >>> import splinekit.interval as ivl
            >>> ivl.Empty() is ivl.Empty().copy()
            True
            >>> ivl.RR() is ivl.RR().copy()
            True
        """

        _ki = _KnownIntervals()
        if isinstance(self, _ki.rr.__class__):
            return cast(Self, _ki.rr)
        if isinstance(self, _ki.empty.__class__):
            return cast(Self, _ki.empty)
        if isinstance(self, Degenerate):
            return cast(Self, _ki.singleton.Singleton(self._value))
        if isinstance(self, _ki.above.Above):
            return cast(Self, _ki.above.Above(self._threshold))
        if isinstance(self, _ki.notbelow.NotBelow):
            return cast(Self, _ki.notbelow.NotBelow(self._threshold))
        if isinstance(self, _ki.notabove.NotAbove):
            return cast(Self, _ki.notabove.NotAbove(self._threshold))
        if isinstance(self, _ki.below.Below):
            return cast(Self, _ki.below.Below(self._threshold))
        if isinstance(self, _ki.open.Open):
            return cast(Self, _ki.open.Open((
                self._leftbound,
                self._rightbound
            )))
        if isinstance(self, _ki.openclosed.OpenClosed):
            return cast(Self, _ki.openclosed.OpenClosed((
                self._leftbound,
                self._rightbound
            )))
        if isinstance(self, _ki.closed.Closed):
            return cast(Self, _ki.closed.Closed((
                self._leftbound,
                self._rightbound
            )))
        if isinstance(self, _ki.closedopen.ClosedOpen):
            return cast(Self, _ki.closedopen.ClosedOpen((
                self._leftbound,
                self._rightbound
            )))
        raise ValueError("Internal error (unexpected interval class)")

    #---------------
    def __eq__ (
        self,
        other: object
    ) -> bool:

        r"""
        .. _eq:

        Tests if this interval :math:`U` is equal to the interval :math:`V,`
        as in :math:`\left(U=V\right)\Leftrightarrow
        \left(U\subseteq V\wedge V\subseteq U\right).`

        Parameters
        ----------
        other : Interval
            Interval whose equality is queried.

        Returns
        -------
        bool
            - `True` if this interval and ``other`` have all the same members.
            - `False` otherwise.

        Examples
        --------
        Load the library.
            >>> import splinekit.interval as ivl
        Here are two equal intervals.
            >>> ivl.Closed((3.0, 3.0)) == ivl.Singleton(3.0)
            True
        Here are two different intervals.
            >>> ivl.OpenClosed((1.0, 3.0)) == ivl.ClosedOpen((1.0, 3.0))
            False
        """

        if not isinstance(self, other.__class__):
            return False
        if isinstance(self, Universal):
            return True
        if isinstance(self, Degenerate):
            return self._value == cast(Degenerate, other)._value
        if isinstance(self, HalfBounded):
            return self._threshold == cast(HalfBounded, other)._threshold
        if isinstance(self, Bounded):
            return ((self._leftbound == cast(Bounded, other)._leftbound) and
                (self._rightbound == cast(Bounded, other)._rightbound))
        raise ValueError("Objects cannot be compared")

    #---------------
    def __ne__ (
        self,
        other: object
    ) -> bool:

        r"""
        .. _ne:

        Tests if this interval :math:`U` differs from the interval :math:`V,`
        as in :math:`\left(U\neq V\right)\Leftrightarrow\neg\left(U=V\right).`

        Parameters
        ----------
        other : Interval
            Interval whose inequality is queried.

        Returns
        -------
        bool
            - `True` if this interval and ``other`` have all the same members.
            - `False` otherwise.

        Examples
        --------
        Load the library.
            >>> import splinekit.interval as ivl
        Here are two equal intervals.
            >>> ivl.Closed((3.0, 3.0)) != ivl.Singleton(3.0)
            False
        Here are two different intervals.
            >>> ivl.OpenClosed((1.0, 3.0)) != ivl.ClosedOpen((1.0, 3.0))
            True
        """

        if not isinstance(self, other.__class__):
            return True
        if isinstance(self, Universal):
            return False
        if isinstance(self, Degenerate):
            return self._value != cast(Degenerate, other)._value
        if isinstance(self, HalfBounded):
            return self._threshold != cast(HalfBounded, other)._threshold
        if isinstance(self, Bounded):
            return ((self._leftbound != cast(Bounded, other)._leftbound) or
                (self._rightbound != cast(Bounded, other)._rightbound))
        raise ValueError("Objects cannot be compared")

    #---------------
    def __lt__ (
        self,
        other: Self
    ) -> bool:

        r"""
        .. _lt:

        ``<`` Proper subset: :math:`\subset`

        Tests if this interval :math:`U` is a proper subset of the interval
        :math:`V,` as in :math:`\left(U\subset V\right)\Leftrightarrow
        \left(U\subseteq V\wedge U\neq V\right).`

        Parameters
        ----------
        other : Interval
            Interval that may or may not be a proper superset of this interval.

        Returns
        -------
        bool
            - `False` if this interval is not a proper subset of ``other``.
            - `True` if this interval is a proper subset of ``other``.

        Examples
        --------
        Load the library.
            >>> import splinekit.interval as ivl
        The positive numbers form a proper subset of the nonnegative numbers.
            >>> ivl.Above(0.0) < ivl.NotBelow(0.0)
            True
        The empty interval is a proper subset of every interval except itself.
            >>> ivl.Empty() < ivl.Empty()
            False
        """

        if self == other:
            return False
        return self <= other

    #---------------
    def __le__ (
        self,
        other: Self
    ) -> bool:

        r"""
        .. _le:

        ``<=`` Subset: :math:`\subseteq`

        Tests if this interval :math:`U` is a subset of the interval :math:`V,`
        as in :math:`\left(U\subseteq V\right)\Leftrightarrow
        \left(\forall x\in U:x\in V\right).`

        Parameters
        ----------
        other : Interval
            Interval that may or may not be a superset of this interval.

        Returns
        -------
        bool
            - `False` if this interval is not a subset of ``other``.
            - `True` if this interval is a subset of ``other``.

        Examples
        --------
        The empty interval is a subset of every interval, including itself.
            >>> import splinekit.interval as ivl
            >>> ivl.Empty() <= ivl.Empty()
            True
        """

        _ki = _KnownIntervals()
        if isinstance(self, _ki.empty.__class__):
            return True
        return (((self.infimum in other) or ((self.infimum == other.infimum)
            and (not self.infimum in self)
            and (not other.infimum in other))) and
            ((self.supremum in other) or ((self.supremum == other.supremum)
            and (not self.supremum in self)
            and (not other.supremum in other))))

    #---------------
    def __neg__ (
        self
    ) -> Set[Self]:

        r"""
        .. _neg:

        ``-`` Unary complement in ``RR()``: :math:`(\cdot)^{{\mathrm{c}}}`

        Creates a set of mutually nonoverlapping intervals that contain
        every real number except those found in this interval :math:`U,` as in
        :math:`U^{{\mathrm{c}}}:=\{x\in{\mathbb{R}}|x\not\in U\}.`

        Returns
        -------
        set of Interval
            The complement in ``RR()`` to this interval.

        Examples
        --------
        The complement of the negative numbers is the nonnegative numbers.
            >>> import splinekit.interval as ivl
            >>> -ivl.Below(0.0)
            {NotBelow(0.0)}
        """

        _ki = _KnownIntervals()
        if isinstance(self, _ki.rr.__class__):
            return {cast(Self, _ki.empty)}
        if isinstance(self, _ki.empty.__class__):
            return {cast(Self, _ki.rr)}
        if isinstance(self, Degenerate):
            return {
                cast(Self, _ki.below.Below(self.value)),
                cast(Self, _ki.above.Above(self.value))
            }
        if isinstance(self, _ki.above.Above):
            return {cast(Self, _ki.notabove.NotAbove(self.threshold))}
        if isinstance(self, _ki.notbelow.NotBelow):
            return {cast(Self, _ki.below.Below(self.threshold))}
        if isinstance(self, _ki.notabove.NotAbove):
            return {cast(Self, _ki.above.Above(self.threshold))}
        if isinstance(self, _ki.below.Below):
            return {cast(Self, _ki.notbelow.NotBelow(self.threshold))}
        if isinstance(self, _ki.open.Open):
            return {
                cast(Self, _ki.notabove.NotAbove(self.infimum)),
                cast(Self, _ki.notbelow.NotBelow(self.supremum))
            }
        if isinstance(self, _ki.openclosed.OpenClosed):
            return {
                cast(Self, _ki.notabove.NotAbove(self.infimum)),
                cast(Self, _ki.above.Above(self.supremum))
            }
        if isinstance(self, _ki.closed.Closed):
            return {
                cast(Self, _ki.below.Below(self.infimum)),
                cast(Self, _ki.above.Above(self.supremum))
            }
        if isinstance(self, _ki.closedopen.ClosedOpen):
            return {
                cast(Self, _ki.below.Below(self.infimum)),
                cast(Self, _ki.notbelow.NotBelow(self.supremum))
            }
        raise ValueError("Internal error (unexpected interval class)")

    #---------------
    def __sub__ (
        self,
        other: Self
    ) -> Set[Self]:

        r"""
        .. _sub:

        ``-`` Pairwise difference: :math:`\setminus`

        Creates a set of nonoverlapping intervals that contains all elements
        of this interval :math:`U` except those of the interval :math:`V,` as
        in :math:`U\setminus V:=\{x\in U|x\not\in V\}.`

        Parameters
        ----------
        other : Interval
            The interval been removed from this interval.

        Returns
        -------
        set of Interval
            The pairwise difference between this interval and ``other``.

        Examples
        --------
        The pairwise difference between the nonnegative numbers and the positive numbers is a single number.
            >>> import splinekit.interval as ivl
            >>> ivl.NotBelow(0.0) - ivl.Above(0.0)
            {Singleton(0.0)}
        """

        return cast(Set[Self], Interval.union({self & o for o in -other}))

    #---------------
    def __and__ (
        self,
        other: Self
    ) -> Self:

        r"""
        .. _and:

        ``&`` Intersection: :math:`\cap`

        Creates an interval that contains every element that is a member
        of both this interval :math:`U` and the interval :math:`V,` as in
        :math:`U\cap V:=\{x\in U|x\in V\}.`

        Parameters
        ----------
        other : Interval
            The interval whose intersection with this interval is sought for.

        Returns
        -------
        Interval
            The intersection between this interval and ``other``.

        Examples
        --------
        The intersection between the negative numbers and the positive numbers is empty.
            >>> import splinekit.interval as ivl
            >>> ivl.Below(0.0) & ivl.Above(0.0)
            Empty()
        """

        _ki = _KnownIntervals()
        if isinstance(other, _ki.rr.__class__):
            return self
        if isinstance(self, _ki.rr.__class__):
            return other
        if (isinstance(self, _ki.empty.__class__) or
            isinstance(other, _ki.empty.__class__)):
            return cast(Self, _ki.empty)
        infimum = max(self.infimum, other.infimum)
        supremum = min(self.supremum, other.supremum)
        if supremum < infimum:
            return cast(Self, _ki.empty)
        if supremum == infimum:
            return(
                cast(Self, _ki.singleton.Singleton(supremum))
                if (supremum in self) and (supremum in other)
                else cast(Self, _ki.empty)
            )
        if float("-inf") == infimum:
            return(
                cast(Self, _ki.notabove.NotAbove(supremum))
                if (supremum in self) and (supremum in other)
                else cast(Self, _ki.below.Below(supremum))
            )
        if float("inf") == supremum:
            return(
                cast(Self, _ki.notbelow.NotBelow(infimum))
                if (infimum in self) and (infimum in other)
                else cast(Self, _ki.above.Above(infimum))
            )
        if (supremum in self) and (supremum in other):
            if (infimum in self) and (infimum in other):
                return cast(Self, _ki.closed.Closed((infimum, supremum)))
            return cast(Self, _ki.openclosed.OpenClosed((infimum, supremum)))
        if (infimum in self) and (infimum in other):
            return cast(Self, _ki.closedopen.ClosedOpen((infimum, supremum)))
        return cast(Self, _ki.open.Open((infimum, supremum)))

    #---------------
    def __or__ (
        self,
        other: Self
    ) -> Set[Self]:

        r"""
        .. _or:

        ``|`` Union: :math:`\cup`

        Creates a set of nonoverlapping intervals that contains all elements
        of this interval :math:`U` and all elements of the the interval
        :math:`V,` as in
        :math:`U\cup V:=\{x\in{\mathbb{R}}|x\in U\vee x\in V\}.`

        Parameters
        ----------
        other : Interval
            The interval whose union with this interval is sought for.

        Returns
        -------
        set of Interval
            The union between this interval and ``other``.

        Examples
        --------
        The nonnegative numbers result from the union between the number ``0.0`` and the positive numbers.
            >>> import splinekit.interval as ivl
            >>> ivl.Singleton(0.0) | ivl.Above(0.0)
            {NotBelow(0.0)}
        """

        _ki = _KnownIntervals()
        if self <= other:
            return {other}
        if other <= self:
            return {self}
        u = self
        v = other
        if v.infimum < u.infimum:
            u = other
            v = self
        if u.supremum < v.infimum:
            return {u, v}
        if (u.infimum == v.infimum) and (u.isleftopen and not v.isleftopen):
            (u, v) = (v, u)
        if (v.infimum in u) and (v.supremum in u):
            return {u}
        if ((v.infimum in u) and not v.supremum in u) or (u.supremum in v):
            if u.isleftbounded:
                if u.isleftopen:
                    if v.isrightbounded:
                        if v.isrightopen:
                            return {cast(
                                Self,
                                _ki.open.Open((u.infimum, v.supremum))
                            )}
                        return {cast(
                            Self,
                            _ki.openclosed.OpenClosed((u.infimum, v.supremum))
                        )}
                    return {cast(Self, _ki.above.Above(u.infimum))}
                if v.isrightbounded:
                    if v.isrightopen:
                        return {cast(
                            Self,
                            _ki.closedopen.ClosedOpen((u.infimum, v.supremum))
                        )}
                    return {cast(Self, _ki.closed.Closed((
                        u.infimum,
                        v.supremum
                    )))}
                return {cast(Self, _ki.notbelow.NotBelow(u.infimum))}
            if v.isrightbounded:
                if v.isrightopen:
                    return {cast(Self, _ki.below.Below(v.supremum))}
                return {cast(Self, _ki.notabove.NotAbove(v.supremum))}
            return {cast(Self, _ki.rr)}
        return {u, v}

    #---------------
    def __xor__ (
        self,
        other: Self
    ) -> Set[Self]:

        r"""
        .. _xor:

        ``^`` Symmetric difference: :math:`\Delta`

        Creates a set of nonoverlapping intervals that contains the elements
        that belong either to this interval :math:`U` or to the interval
        :math:`V,` as in :math:`U\Delta V:=\left(U\setminus V\right)\cup
        \left(V\setminus U\right).`

        Parameters
        ----------
        other : Interval
            The interval whose symmetric difference with this interval is
            sought for.

        Returns
        -------
        set of Interval
            The symmetric difference between this interval and ``other``.

        Examples
        --------
        Here is the symmetric difference between two bounded intervals.
            >>> import splinekit.interval as ivl
            >>> ivl.Closed((-3.0, 1.0)) ^ ivl.Closed((-1.0, 3.0))
            {OpenClosed((1.0, 3.0)), ClosedOpen((-3.0, -1.0))}
        """

        return (self - other).union(other - self)
