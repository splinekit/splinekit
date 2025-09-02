#---------------
from __future__ import annotations

#---------------
from typing import Any
from typing import List
from typing import NamedTuple
from typing import cast

#---------------
from dataclasses import dataclass

#---------------
from splinekit.interval import Closed
from splinekit.interval import ClosedOpen
from splinekit.interval import Empty
from splinekit.interval import Interval
from splinekit.interval import Open
from splinekit.interval import OpenClosed
from splinekit.interval import Singleton

#---------------
from splinekit.spline_utilities import _divmod

#---------------
class PeriodicNonuniformPiecewise:

    r"""
    .. _PeriodicNonuniformPiecewise:

    The class that maintains a valid periodic partition of a closed-open
    proper interval of integer diameter with zero as left bound.

    Let a :ref:`Piece object<Piece>` define some arbitrary attribute over
    a domain made of an interval. Consider now a list of such pieces and
    assume that their combined intervals make for a partition of the
    :ref:`closed-open<ClosedOpen>` interval :math:`[0, P),` where the period
    :math:`P` is a :ref:`positive<def-positive>` integer. An instance of the
    class ``PeriodicNonuniformPiecewise`` maintains a database of such pieces
    and provides a convenience method to access the appropriate piece at any
    :math:`x\in{\mathbb{R}}`.

    ====

    """

    #---------------
    @property
    def pieces (
        self
    ) -> List[Piece]:

        """
        The list of all pieces associated to this
        ``PeriodicNonuniformPiecewise`` instance, sorted by increasing domain.

        Returns
        -------
        list of Piece
            All pieces as a list.


        ----

        """

        return self._pieces

    #---------------
    @property
    def period (
        self
    ) -> int:

        """
        The period associated to this ``PeriodicNonuniformPiecewise`` instance.

        Returns
        -------
        int
            The period.


        ----

        """

        return self._period

    #---------------
    class Piece (
        NamedTuple
    ):

        """
        .. _Piece:

        Each piece of a ``PeriodicNonuniformPiecewise`` object is a named
        tuple made of two fields. The first field is named ``domain`` and is
        an :ref:`Interval<Interval>` object that describes an arbitrary domain
        over which the piece extends. The second field is named ``item`` and
        carries the attributes of the piece—often, a function.

        ====

        """

        domain: Interval

        """
        The domain of the piece.

        ----

        """

        item: Any

        """
        The item of the piece.

        ----

        """

    #---------------
    def __init__ (
        self,
        pieces: List[Piece],
        *,
        period: int
    ) -> None:

        """
        .. _Piece-init:

        The constructor for this class.

        Together, the pieces must partition a closed-open interval with ``0``
        as left bound and ``period`` as right bound. This requirement is
        checked at instanciation time.

        Parameters
        ----------
        pieces : List[Piece]
            The list of pieces, in no particular order.
        period : int
            The positive period.

        Raises
        ------
        ValueError
            Raised when the pieces do not provide a valid partition of the
            :ref:`closed-open<ClosedOpen>` interval ``[0, period)``.


        ----

        """

        if 0 == len(pieces):
            raise ValueError("Pieces must contain at least one element")
        if period < 1:
            raise ValueError("Period must be positive")
        ps = pieces.copy()
        ps.sort(key = lambda p: p.domain.sortorder())
        previous_sup = 0.0
        previous_sup_contained = True
        p = 0
        while isinstance(ps[p].domain, Empty):
            p += 1
            if len(ps) == p:
                raise ValueError("Pieces must partition [0, period)")
            continue
        if isinstance(ps[p].domain, Singleton):
            if 0.0 != cast(Singleton, ps[p].domain).value:
                raise ValueError("Pieces must partition [0, period)")
        elif isinstance(ps[p].domain, Closed):
            if 0.0 != cast(Closed, ps[p].domain).infimum:
                raise ValueError("Pieces must partition [0, period)")
            previous_sup = ps[p].domain.supremum
        elif isinstance(ps[p].domain, ClosedOpen):
            if 0.0 != cast(ClosedOpen, ps[p].domain).infimum:
                raise ValueError("Pieces must partition [0, period)")
            previous_sup = ps[p].domain.supremum
            previous_sup_contained = False
        else:
            raise ValueError("Pieces must partition [0, period)")
        p += 1
        while p < len(ps) - 1:
            if previous_sup_contained:
                if isinstance(ps[p].domain, Open):
                    if cast(Open, ps[p].domain).infimum != previous_sup:
                        raise ValueError("Pieces must partition [0, period)")
                    previous_sup = cast(Open, ps[p].domain).supremum
                    previous_sup_contained = False
                elif isinstance(ps[p].domain, OpenClosed):
                    if cast(OpenClosed, ps[p].domain).infimum != previous_sup:
                        raise ValueError("Pieces must partition [0, period)")
                    previous_sup = cast(OpenClosed, ps[p].domain).supremum
                    previous_sup_contained = True
                else:
                    raise ValueError("Pieces must partition [0, period)")
            else:
                if isinstance(ps[p].domain, Singleton):
                    if cast(Singleton, ps[p].domain).value != previous_sup:
                        raise ValueError("Pieces must partition [0, period)")
                    previous_sup_contained = True
                elif isinstance(ps[p].domain, ClosedOpen):
                    if cast(ClosedOpen, ps[p].domain).infimum != previous_sup:
                        raise ValueError("Pieces must partition [0, period)")
                    previous_sup = cast(ClosedOpen, ps[p].domain).supremum
                    previous_sup_contained = False
                elif isinstance(ps[p].domain, Closed):
                    if cast(Closed, ps[p].domain).infimum != previous_sup:
                        raise ValueError("Pieces must partition [0, period)")
                    previous_sup = cast(Closed, ps[p].domain).supremum
                    previous_sup_contained = True
                else:
                    raise ValueError("Pieces must partition [0, period)")
            p += 1
        if previous_sup_contained:
            if isinstance(ps[-1].domain, Open):
                if float(period) != cast(Open, ps[-1].domain).supremum:
                    raise ValueError("Pieces must partition [0, period)")
            else:
                raise ValueError("Pieces must partition [0, period)")
        else:
            if isinstance(ps[-1].domain, ClosedOpen):
                if float(period) != cast(ClosedOpen, ps[-1].domain).supremum:
                    raise ValueError("Pieces must partition [0, period)")
            else:
                raise ValueError("Pieces must partition [0, period)")
        self._pieces = ps
        self._period = period

    #---------------
    def __str__ (
        self
    ) -> str:
        return ("{len(pieces) = " + format(len(self._pieces)) +
            ", period = " + format(self._period) +
            "}"
        )

    #---------------
    def __repr__ (
        self
    ) -> str:
        return (f"periodic_nonuniform_piecewise({self.pieces}, "
            + f"period = {self.period})"
        )

    #---------------
    def item_at (
        self,
        x: float
    ) -> Any:

        """
        .. _Piece-item_at:

        Convenience method to access individual pieces.

        The pieces are periodized with period ``period``. Accordingly, the
        index ``x`` is free to take any real value.

        Parameters
        ----------
        x : float
            Floating-point index of the piece.

        Returns
        -------
        Any
            The item associated to the piece whose interval contains ``x``.
        """

        (_, x_wrapped) = _divmod(x, self._period)
        for piece in self._pieces:
            if x_wrapped in piece.domain:
                return piece.item
        raise ValueError("Pieces must partition [0, period)")
