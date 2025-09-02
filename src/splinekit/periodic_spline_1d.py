"""
This class handles periodic splines.

Properties
==========
- :ref:`period <periodic_spline_1d-attributes>`
- :ref:`degree <periodic_spline_1d-attributes>`
- :ref:`delay <periodic_spline_1d-attributes>`
- :ref:`spline_coeff <periodic_spline_1d-attributes>`

Constructors
============
-   Zero-Valued Spline

    - :ref:`init <periodic_spline_1d-init>`
-   Periodized Spline Bases

    - :ref:`periodized_b_spline <periodic_spline_1d-periodized_b_spline>`
    - :ref:`periodized_cardinal_b_spline <periodic_spline_1d-periodized_cardinal_b_spline>`
    - :ref:`periodized_dual_b_spline <periodic_spline_1d-periodized_dual_b_spline>`
    - :ref:`periodized_orthonormal_b_spline <periodic_spline_1d-periodized_orthonormal_b_spline>`
-   From Data

    - :ref:`from_samples <periodic_spline_1d-from_samples>`
    - :ref:`from_smoothed_samples <periodic_spline_1d-from_smoothed_samples>`
    - :ref:`from_spline_coeff <periodic_spline_1d-from_spline_coeff>`

Class Methods
=============
-   Combine Two Splines

    - :ref:`add <periodic_spline_1d-add>`
    - :ref:`add_projected <periodic_spline_1d-add_projected>`
    - :ref:`convolve <periodic_spline_1d-convolve>`
    - :ref:`normed_cross_correlate <periodic_spline_1d-normed_cross_correlate>`
-   Evaluate Nonstandard Spline Representations

    - :ref:`eval_piecewise_polynomials_at <periodic_spline_1d-eval_piecewise_polynomials_at>`
    - :ref:`eval_piecewise_signs_at <periodic_spline_1d-eval_piecewise_signs_at>`

Instance Methods
================
-   Utility Methods

    - :ref:`copy <periodic_spline_1d-copy>`
    - :ref:`plot <periodic_spline_1d-plot>`
-   Evaluate a Spline

    - :ref:`at <periodic_spline_1d-at>`
    - :ref:`get_samples <periodic_spline_1d-get_samples>`
-   Empirical Statistics

    - :ref:`mean <periodic_spline_1d-mean>`
    - :ref:`variance <periodic_spline_1d-variance>`
    - :ref:`lower_bound <periodic_spline_1d-lower_bound>`
    - :ref:`upper_bound <periodic_spline_1d-upper_bound>`
-   Locations of Interest

    - :ref:`get_knots <periodic_spline_1d-get_knots>`
    - :ref:`zeros <periodic_spline_1d-zeros>`
    - :ref:`zero_crossings <periodic_spline_1d-zero_crossings>`
    - :ref:`Extremum <periodic_spline_1d-Extremum>`
    - :ref:`extrema <periodic_spline_1d-extrema>`
-   Arithmetic and Geometric Operations

    - :ref:`plus <periodic_spline_1d-plus>`
    - :ref:`times <periodic_spline_1d-times>`
    - :ref:`negated <periodic_spline_1d-negated>`
    - :ref:`mirrored <periodic_spline_1d-mirrored>`
    - :ref:`fractionalized_delay <periodic_spline_1d-fractionalized_delay>`
    - :ref:`delayed_by <periodic_spline_1d-delayed_by>`
-   Calculus

    - :ref:`differentiated <periodic_spline_1d-differentiated>`
    - :ref:`gradient <periodic_spline_1d-gradient>`
    - :ref:`anti_grad <periodic_spline_1d-anti_grad>`
    - :ref:`integrate <periodic_spline_1d-integrate>`

-   Nonstandard Representations

    - :ref:`fourier_coeff <periodic_spline_1d-fourier_coeff>`
    - :ref:`piecewise_polynomials <periodic_spline_1d-piecewise_polynomials>`
    - :ref:`piecewise_signs <periodic_spline_1d-piecewise_signs>`

-   Resampling and Projections

    - :ref:`projected <periodic_spline_1d-projected>`
    - :ref:`upscaled <periodic_spline_1d-upscaled>`
    - :ref:`upscaled_projected <periodic_spline_1d-upscaled_projected>`
    - :ref:`downscaled_projected <periodic_spline_1d-downscaled_projected>`
    - :ref:`rescaled_projected <periodic_spline_1d-rescaled_projected>`

====

"""

#---------------
from __future__ import annotations

#---------------
from typing import cast
from typing import List
from typing import NamedTuple
from typing import Tuple

#---------------
from math import ceil
from math import floor
from math import fsum
from math import gcd
from math import isclose
from math import sqrt
from math import ulp

#---------------
import cmath
import scipy

#---------------
import matplotlib as plt
import numpy as np

#---------------
from splinekit import PeriodicNonuniformPiecewise

#---------------
from splinekit import b_spline
from splinekit import diff_b_spline
from splinekit import integrated_b_spline
from splinekit import mscale_filter

#---------------
from splinekit.bases import Bases

#---------------
from splinekit.interval import Above
from splinekit.interval import Below
from splinekit.interval import Closed
from splinekit.interval import ClosedOpen
from splinekit.interval import Empty
from splinekit.interval import Interval
from splinekit.interval import NotAbove
from splinekit.interval import NotBelow
from splinekit.interval import Open
from splinekit.interval import OpenClosed
from splinekit.interval import RR
from splinekit.interval import Singleton

#---------------
from splinekit.spline_padding import change_basis_p
from splinekit.spline_padding import samples_to_coeff_p

#---------------
from splinekit.spline_utilities import _b
from splinekit.spline_utilities import _divmod
from splinekit.spline_utilities import _inv_w
from splinekit.spline_utilities import _sgn
from splinekit.spline_utilities import _w

#---------------
class PeriodicSpline1D:

    r"""
    .. _periodic_spline_1d:

    The class that maintains and operates on a polynomial one-dimensional
    uniform spline with periodic padding.

    A :ref:`uniform spline<def-uniform_spline>` is a
    :ref:`piecewise-polynomial function<def-piecewise_polynomial>`. The class
    ``PeriodicSpline1D`` is handling a version that is periodic, with the
    period :math:`K\in{\mathbb{N}}+1` being a :ref:`positive
    integer<def-positive>`. The spline is

    ..  math::

        f(x)=\sum_{k\in{\mathbb{Z}}}\,c[{k\bmod K}]\,\beta^{n}(x-\delta x-k),

    where :math:`x\in{\mathbb{R}}` is the argument at which the spline
    :math:`f` is evaluated, :math:`n` is the :ref:`nonnegative<def-negative>`
    degree of the polynomial spline, :math:`\beta^{n}` is a
    :ref:`B-spline<def-b_spline>` basis, and :math:`\delta x` is a delay that
    controls the placement of the uniform knots. The array of coefficients
    :math:`c` is giving its shape to the spline.

    .. _periodic_spline_1d-attributes:

    Attributes
    ----------
    period : int
        (R/O) The :ref:`positive<def-positive>` period
        :math:`K\in{\mathbb{N}}+1.`
    degree : int
        (R/W) The :ref:`nonnegative<def-negative>` degree
        :math:`n\in{\mathbb{N}}` of the polynomial spline.
    delay : float
        (R/W) The delay :math:`\delta x\in{\mathbb{R}}.`
    spline_coeff : np.ndarray[tuple[int], np.dtype[np.float64]]
        (R/W) The spline coefficients :math:`c\in{\mathbb{R}}^{K}.` The setter
        creates a local copy and updates the period.


    Raises
    ------
    ValueError
        Raised when ``degree`` is :ref:`negative<def-negative>`.
    ValueError
        Raised when ``spline_coeff`` fails to be a ``numpy`` one-dimensional
        ``float`` array that contains at least one element.


    ====

    """

    #---------------
    @property
    def period (
        self
    ) -> int:
        return self._period

    #---------------
    @property
    def degree (
        self
    ) -> int:
        return self._degree

    #---------------
    @degree.setter
    def degree (
        self,
        degree: int
    ) -> None:
        if 0 > degree:
            raise ValueError("Degree must be nonnegative")
        self._degree = degree

    #---------------
    @property
    def delay (
        self
    ) -> float:
        return self._delay

    #---------------
    @delay.setter
    def delay (
        self,
        delay: float
    ) -> None:
        self._delay = delay

    #---------------
    @property
    def spline_coeff (
        self
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        return cast(
            np.ndarray[tuple[int], np.dtype[np.float64]],
            self._spline_coeff
        )

    #---------------
    @spline_coeff.setter
    def spline_coeff (
        self,
        data: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> None:
        if np.ndarray != type(data):
            raise ValueError("Data must be a numpy array")
        if 1 != len(data.shape):
            raise ValueError("Data must be a one-dimensional numpy array")
        if float != data.dtype:
            raise ValueError(
                "Data must be of type np.ndarray[tuple[int], np.dtype[np.float64]]"
            )
        if 0 == len(data):
            raise ValueError("Data must contain at least one element")
        self._spline_coeff = np.copy(data)
        self._period = len(data)

    #---------------
    def __init__ (
        self
    ) -> None:

        """
        .. _periodic_spline_1d-init:

        The default constructor for this class.

        Creates a piecewise-constant periodic spline of unit period,
        vanishing delay, and value zero.

        Parameters
        ----------
        None

        Examples
        --------
        Load the library.
            >>> import splinekit as sk
        Create a default spline.
            >>> sk.PeriodicSpline1D()
            PeriodicSpline1D([0.], degree = 0, delay = 0.0)

        ----

        """

        self.spline_coeff = cast(
            np.ndarray[tuple[int], np.dtype[np.float64]],
            np.array([0.0], dtype = float)
        )
        self.degree = 0
        self.delay = 0.0

    #---------------
    def __str__ (
        self
    ) -> str:
        return ("{degree = " + format(self._degree) +
            ", delay = " + format(self._delay) +
            ", period = " + format(self._period) + "}"
        )

    #---------------
    def __repr__ (
        self
    ) -> str:
        return (f"PeriodicSpline1D({self._spline_coeff}, " +
            f"degree = {self._degree}, delay = {self._delay})"
        )

    #---------------
    @classmethod
    def periodized_b_spline (
        cls,
        *,
        period: int,
        degree: int
    ) -> PeriodicSpline1D:

        r"""
        .. _periodic_spline_1d-periodized_b_spline:

        The constructor of a periodized B-spline.

        This constructor populates the attributes of a ``PeriodicSpline1D``
        object that represents the one-dimensional spline

        ..  math::

            f(x)=\sum_{k\in{\mathbb{Z}}}\,\beta^{n}(k\,K+x),

        as defined :ref:`here<periodic_spline_1d>`. This spline has period
        :math:`K` and is the periodized version of a centered
        :ref:`B-spline<def-b_spline>` of degree :math:`n.`

        Parameters
        ----------
        period : int
            The :ref:`positive<def-positive>` period of the
            periodized B-spline.
        degree : int
            The :ref:`nonnegative<def-negative>` degree of the periodized
            B-spline.

        Returns
        -------
        PeriodicSpline1D
            The periodized B-spline centered at the origin.

        Examples
        --------
        Load the library.
            >>> import splinekit as sk
        Create a nonic B-spline of period ``6``.
            >>> sk.PeriodicSpline1D.periodized_b_spline(period = 6, degree = 9)
            PeriodicSpline1D([1. 0. 0. 0. 0. 0.], degree = 9, delay = 0.0)

        Raises
        ------
        ValueError
            Raised when ``period`` is :ref:`nonpositive<def-positive>`.
        ValueError
            Raised when ``degree`` is :ref:`negative<def-negative>`.


        ----

        """

        if 1 > period:
            raise ValueError("Period must be at least one")
        if 0 > degree:
            raise ValueError("Degree must be nonnegative")
        coeffs = np.zeros(period, dtype = float)
        coeffs[0] = 1.0
        return cls.from_spline_coeff(
            coeffs,
            degree = degree
        )

    #---------------
    @classmethod
    def periodized_cardinal_b_spline (
        cls,
        *,
        period: int,
        degree: int
    ) -> PeriodicSpline1D:

        r"""
        .. _periodic_spline_1d-periodized_cardinal_b_spline:

        The constructor of a periodized cardinal B-spline.

        This constructor populates the attributes of a ``PeriodicSpline1D``
        object that represents the one-dimensional spline

        ..  math::

            f(x)=\sum_{k\in{\mathbb{Z}}}\,\eta^{n}(k\,K+x),

        as defined :ref:`here<periodic_spline_1d>`. This spline has period
        :math:`K` and is the periodized version of a centered
        :ref:`cardinal B-spline<def-cardinal_b_spline>` of degree :math:`n.`

        Parameters
        ----------
        period : int
            The :ref:`positive<def-positive>` period of the periodized
            cardinal B-spline.
        degree : int
            The :ref:`nonnegative<def-negative>` degree of the periodized
            cardinal B-spline.

        Returns
        -------
        PeriodicSpline1D
            The periodized cardinal B-spline centered at the origin.

        Examples
        --------
        Load the library.
            >>> import splinekit as sk
        Create a nonic cardinal B-spline of period ``6``.
            >>> sk.PeriodicSpline1D.periodized_cardinal_b_spline(period = 6, degree = 9)
            PeriodicSpline1D([10.54181435 -8.30274513  6.41054444 -5.75741296  6.41054444 -8.30274513], degree = 9, delay = 0.0)

        Raises
        ------
        ValueError
            Raised when ``period`` is :ref:`nonpositive<def-positive>`.
        ValueError
            Raised when ``degree`` is :ref:`negative<def-negative>`.


        ----

        """

        if 1 > period:
            raise ValueError("Period must be at least one")
        if 0 > degree:
            raise ValueError("Degree must be nonnegative")
        coeffs = np.zeros(period, dtype = float)
        coeffs[0] = 1.0
        change_basis_p(
            coeffs,
            degree = degree,
            source_basis = cast(Bases, Bases.CARDINAL),
            target_basis = cast(Bases, Bases.BASIC),
        )
        return cls.from_spline_coeff(
            coeffs,
            degree = degree
        )

    #---------------
    @classmethod
    def periodized_dual_b_spline (
        cls,
        *,
        period: int,
        dual_degree: int,
        primal_degree: int
    ) -> PeriodicSpline1D:

        r"""
        .. _periodic_spline_1d-periodized_dual_b_spline:

        The constructor of a periodized dual B-spline.

        This constructor populates the attributes of a ``PeriodicSpline1D``
        object that represents the one-dimensional spline

        ..  math::

            f(x)=\sum_{k\in{\mathbb{Z}}}\,\mathring{\beta}^{m,n}(k\,K+x),

        as defined :ref:`here<periodic_spline_1d>`. This spline has period
        :math:`K` and is the periodized version of a centered
        :ref:`dual B-spline<def-dual_b_spline>` of dual degree :math:`m`
        and primal degree :math:`n.`

        Parameters
        ----------
        period : int
            The :ref:`positive<def-positive>` period of the periodized dual
            B-spline.
        dual_degree : int
            The :ref:`nonnegative<def-negative>` dual degree of the periodized
            dual B-spline.
        primal_degree : int
            The :ref:`nonnegative<def-negative>` primal degree of the
            periodized dual B-spline.

        Returns
        -------
        PeriodicSpline1D
            The periodized dual B-spline centered at the origin.

        Examples
        --------
        Load the library.
            >>> import splinekit as sk
        Create a nonic dual B-spline of quadratic primal degree and period ``6``.
            >>> sk.PeriodicSpline1D.periodized_dual_b_spline(period = 6, dual_degree = 9, primal_degree = 2)
            PeriodicSpline1D([ 34.24972088 -31.03664923  27.43176629 -26.039955    27.43176629 -31.03664923], degree = 9, delay = 0.0)

        Raises
        ------
        ValueError
            Raised when ``period`` is :ref:`nonpositive<def-positive>`.
        ValueError
            Raised when ``dual_degree`` is :ref:`negative<def-negative>`.
        ValueError
            Raised when ``primal_degree`` is :ref:`negative<def-negative>`.


        ----

        """

        if 1 > period:
            raise ValueError("Period must be at least one")
        if 0 > dual_degree:
            raise ValueError("Dual degree must be nonnegative")
        if 0 > primal_degree:
            raise ValueError("Primal degree must be nonnegative")
        coeffs = np.zeros(period, dtype = float)
        coeffs[0] = 1.0
        change_basis_p(
            coeffs,
            degree = primal_degree + dual_degree + 1,
            source_basis = cast(Bases, Bases.CARDINAL),
            target_basis = cast(Bases, Bases.BASIC),
        )
        return cls.from_spline_coeff(
            coeffs,
            degree = dual_degree
        )

    #---------------
    @classmethod
    def periodized_orthonormal_b_spline (
        cls,
        *,
        period: int,
        degree: int
    ) -> PeriodicSpline1D:

        r"""
        .. _periodic_spline_1d-periodized_orthonormal_b_spline:

        The constructor of a periodized orthonormal B-spline.

        This constructor populates the attributes of a ``PeriodicSpline1D``
        object that represents the one-dimensional spline

        ..  math::

            f(x)=\sum_{k\in{\mathbb{Z}}}\,\phi^{n}(k\,K+x),

        as defined :ref:`here<periodic_spline_1d>`. This spline has period
        :math:`K` and is the periodized version of a centered
        :ref:`orthonormal B-spline<def-orthonormal_b_spline>` of degree
        :math:`n.`

        Parameters
        ----------
        period : int
            The :ref:`positive<def-positive>` period of the periodized
            orthonormal B-spline.
        degree : int
            The :ref:`nonnegative<def-negative>` degree of the periodized
            orthonormal B-spline.

        Returns
        -------
        PeriodicSpline1D
            The periodized orthonormal B-spline centered at the origin.

        Examples
        --------
        Load the library.
            >>> import splinekit as sk
        Create a nonic orthonormal B-spline of period ``6``.
            >>> sk.PeriodicSpline1D.periodized_orthonormal_b_spline(period = 6, degree = 9)
            PeriodicSpline1D([ 13.70088094 -11.4607243    9.56634894  -8.91213021   9.56634894  -11.4607243 ], degree = 9, delay = 0.0)

        Raises
        ------
        ValueError
            Raised when ``period`` is :ref:`nonpositive<def-positive>`.
        ValueError
            Raised when ``degree`` is :ref:`negative<def-negative>`.


        ----

        """

        if 1 > period:
            raise ValueError("Period must be at least one")
        if 0 > degree:
            raise ValueError("Degree must be nonnegative")
        coeffs = np.zeros(period, dtype = float)
        coeffs[0] = 1.0
        change_basis_p(
            coeffs,
            degree = degree,
            source_basis = cast(Bases, Bases.ORTHONORMAL),
            target_basis = cast(Bases, Bases.BASIC),
        )
        return cls.from_spline_coeff(
            coeffs,
            degree = degree
        )

    #---------------
    @classmethod
    def from_samples (
        cls,
        samples: np.ndarray[tuple[int], np.dtype[np.float64]],
        *,
        degree: int,
        delay: float = 0.0,
        regularized: bool = False
    ) -> PeriodicSpline1D:

        r"""
        .. _periodic_spline_1d-from_samples:

        A constructor from data samples.

        This constructor populates the attributes of a ``PeriodicSpline1D``
        object, which represents the one-dimensional spline

        ..  math::

            f(x)=\sum_{k\in{\mathbb{Z}}}\,c[{k\bmod K}]\,
            \beta^{n}(x-\delta x-k),

        as defined :ref:`here<periodic_spline_1d>`. The provided data samples
        :math:`\left(s[k]\right)_{k=0}^{K-1}` are assumed to have been taken
        at the integers.

        *   When ``regularized == False``, this constructor determines the
            spline coefficients :math:`\left(c[k]\right)_{k=0}^{K-1}` such
            that the periodic spline :math:`f` interpolates the samples
            :math:`s,` meaning that :math:`\left(f(k)\right)_{k=0}^{K-1}=
            \left(s[k]\right)_{k=0}^{K-1}.` In general, this condition can be
            satisfied for any delay :math:`\delta x\in{\mathbb{R}},` provided
            the period :math:`K\in2\,{\mathbb{N}}+1` is :ref:`odd<def-odd>`.
            Alternatively, the interpolation condition can be satisfied for
            any non-half-integer delay :math:`\delta x\in{\mathbb{R}}\setminus
            \left({\mathbb{Z}}+\frac{1}{2}\right),` this time without
            restriction on the parity of the positive period
            :math:`K\in{\mathbb{N}}+1.`
        *   When ``regularized == True``, the interpolation condition is still
            honored for all odd periods. For :ref:`even<def-even>` periods
            :math:`K\in2\,{\mathbb{N}}+2,` however, this constructor
            determines the spline coefficients
            :math:`\left(c[k]\right)_{k=0}^{K-1}` that minimize
            the criterion
            :math:`J=\sum_{k=0}^{K-1}\,\left(f(k)-s[k]\right)^{2}` such that
            :math:`\sum_{k=0}^{K/2-1}\,f(2\,k)=\sum_{k=0}^{K/2-1}\,f(2\,k+1),`
            which happens to lift the restriction of non-half-integer delays.

        Parameters
        ----------
        samples : np.ndarray[tuple[int], np.dtype[np.float64]]
            A one-dimensional ``numpy`` array of real data samples. The
            period of the spline will be set to the length of this array.
        degree : int
            The :ref:`nonnegative<def-negative>` degree of the polynomial
            spline.
        delay : float
            The delay of this spline.
        regularized : bool
            The regularization directive.

        Returns
        -------
        PeriodicSpline1D
            The spline.
    
        Examples
        --------
        Load the libraries.
            >>> import numpy as np
            >>> import splinekit as sk
        Create an undelayed cubic spline from a few arbitrary samples.
            >>> samples = np.array([1, 5, -3], dtype = float)
            >>> sk.PeriodicSpline1D.from_samples(samples, degree = 3)
            PeriodicSpline1D([ 1.  9. -7.], degree = 3, delay = 0.0)

        Raises
        ------
        ValueError
            Raised when ``samples`` fails to be a ``numpy`` one-dimensional
            ``float`` array that contains at least one element.
        ValueError
            Raised when ``degree`` is :ref:`negative<def-negative>`.


        ----

        """

        if np.ndarray != type(samples):
            raise ValueError("Samples must be a numpy array")
        if 1 != len(samples.shape):
            raise ValueError("Samples must be a one-dimensional numpy array")
        if float != samples.dtype:
            raise ValueError(
                "Samples must be of type np.ndarray[tuple[int], np.dtype[np.float64]]"
            )
        if 0 == len(samples):
            raise ValueError("Samples must contain at least one element")
        if 0 > degree:
            raise ValueError("Degree must be nonnegative")
        p0 = len(samples)
        if 1 == p0:
            return cls.from_spline_coeff(
                samples,
                degree = degree,
                delay = delay
            )
        dx = delay - round(delay)
        s = np.copy(samples)
        if regularized and 0 == p0 % 2:
            even_half_mean = fsum(samples[k] for k in range(0, p0, 2)) / p0
            odd_half_mean = fsum(samples[k] for k in range(1, p0, 2)) / p0
            de = odd_half_mean - even_half_mean
            do = even_half_mean - odd_half_mean
            s = cast(
                np.ndarray[tuple[int], np.dtype[np.float64]],
                np.fromiter(
                    (
                        sk + (de if 0 == k % 2 else do)
                        for (k, sk) in enumerate(s)
                    ),
                    dtype = float,
                    count = p0
                )
            )
        if 0.0 == dx:
            s = np.concatenate((
                s[round(delay) % p0 : ],
                s[ : round(delay) % p0]
            ))
            samples_to_coeff_p(cast(
                np.ndarray[tuple[int], np.dtype[np.float64]], s),
                degree = degree
            )
            return cls.from_spline_coeff(
                cast(
                    np.ndarray[tuple[int], np.dtype[np.float64]],
                    s
                ),
                degree = degree,
                delay = delay
            )
        mat = scipy.linalg.toeplitz(
            np.fromiter(
                (
                    fsum(
                        b_spline(c - delay - p * p0, degree)
                        for p in range(
                            int((c - delay - 0.5 * (degree - 1.0)) // p0),
                            int((c - delay + 0.5 * (degree + 1.0)) // p0) + 1
                        )
                    )
                    for c in range(p0)
                ),
                dtype = float,
                count = p0
            ),
            np.fromiter(
                (
                    fsum(
                        b_spline(-r - delay - p * p0, degree)
                        for p in range(
                            int((-r - delay - 0.5 * (degree - 1.0)) // p0),
                            int((-r - delay + 0.5 * (degree + 1.0)) // p0) + 1
                        )
                    )
                    for r in range(p0)
                ),
                dtype = float,
                count = p0
            )
        )
        if isclose(
            0.5,
            abs(dx),
            rel_tol = sqrt(ulp(1.0)),
            abs_tol = sqrt(ulp(1.0))
        ):
            return cls.from_spline_coeff(
                np.linalg.lstsq(mat, s)[0],
                degree = degree,
                delay = delay
            )
        return cls.from_spline_coeff(
            np.linalg.solve(mat, s),
            degree = degree,
            delay = delay
        )

    #---------------
    @classmethod
    def from_smoothed_samples (
        cls,
        samples: np.ndarray[tuple[int], np.dtype[np.float64]],
        *,
        degree: int,
        delay: float = 0.0,
        smoothing: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> PeriodicSpline1D:

        r"""
        .. _periodic_spline_1d-from_smoothed_samples:

        A constructor from data samples.

        This constructor populates the attributes of a ``PeriodicSpline1D``
        object, which represents the one-dimensional spline

        ..  math::

            f(x)=\sum_{k\in{\mathbb{Z}}}\,c[{k\bmod K}]\,
            \beta^{n}(x-\delta x-k),

        as defined :ref:`here<periodic_spline_1d>`. The provided data samples
        :math:`\left(s[k]\right)_{k=0}^{K-1}` are assumed to have been taken
        at the integers.

        The constructed spline minimizes the criterion

        ..  math::

            J=\sum_{k=0}^{K-1}\,\left(f(k)-s[k]\right)^{2}+\sum_{m=0}^{n}\,
            \lambda[m]\,\int_{0}^{K}\,
            \left(\frac{{\mathrm{d}}^{m}f(x)}{{\mathrm{d}}x^{m}}\right)^{2}\,
            {\mathrm{d}}x.

        The first term encourages the spline to interpolate the samples, while
        the second term discourages the spline derivatives to be large, as
        controlled by the vector :math:`\lambda` of smoothing parameters.

        Parameters
        ----------
        samples : np.ndarray[tuple[int], np.dtype[np.float64]]
            A one-dimensional ``numpy`` array of real data samples. The
            period of the spline will be set to the length of this array.
        degree : int
            The :ref:`nonnegative<def-negative>` degree of the polynomial
            spline.
        delay : float
            The delay of this spline.
        smoothing : np.ndarray[tuple[int], np.dtype[np.float64]]
            A one-dimensional ``numpy`` array of ``n + 1`` regularization
            weights.

        Returns
        -------
        PeriodicSpline1D
            The spline.
    
        Examples
        --------
        Load the libraries.
            >>> import numpy as np
            >>> import splinekit as sk
        Create an undelayed cubic spline with highly attenuated second-order derivatives.
            >>> samples = np.array([1, 5, -3], dtype = float)
            >>> regularization = np.array([0, 0, 5, 0], dtype = float)
            >>> sk.PeriodicSpline1D.from_smoothed_samples(samples, degree = 3, smoothing = regularization)
            PeriodicSpline1D([1.         1.08791209 0.91208791], degree = 3, delay = 0.0)

        Raises
        ------
        ValueError
            Raised when ``samples`` fails to be a ``numpy`` one-dimensional
            ``float`` array that contains at least one element.
        ValueError
            Raised when ``degree`` is :ref:`negative<def-negative>`.
        ValueError
            Raised when ``smoothing`` fails to be a ``numpy`` one-dimensional
            ``float`` array that contains ``degree + 1`` elements.


        ----

        """

        if np.ndarray != type(samples):
            raise ValueError("Samples must be a numpy array")
        if 1 != len(samples.shape):
            raise ValueError("Samples must be a one-dimensional numpy array")
        if float != samples.dtype:
            raise ValueError(
                "Samples must be of type np.ndarray[tuple[int], np.dtype[np.float64]]"
            )
        if 0 == len(samples):
            raise ValueError("Samples must contain at least one element")
        if 0 > degree:
            raise ValueError("Degree must be nonnegative")
        if np.ndarray != type(smoothing):
            raise ValueError("Smoothing must be a numpy array")
        if 1 != len(smoothing.shape):
            raise ValueError("Smoothing must be a one-dimensional numpy array")
        if float != smoothing.dtype:
            raise ValueError(
                "Smoothing must be of type np.ndarray[tuple[int], np.dtype[np.float64]]"
            )
        if degree + 1 != len(smoothing):
            raise ValueError("smoothing must be of length (degree + 1)")
        p0 = len(samples)
        if 1 == p0:
            return cls.from_spline_coeff(
                cast(
                    np.ndarray[tuple[int], np.dtype[np.float64]],
                    np.array(
                        [
                            _sgn(samples[0]) * float("inf")
                                if isclose(
                                    1.0 + smoothing[0],
                                    0.0,
                                    rel_tol = sqrt(ulp(1.0)),
                                    abs_tol = sqrt(ulp(1.0))
                                )
                                else samples[0] / (1.0 + smoothing[0])
                        ],
                        dtype = float
                    )
                ),
                degree = degree,
                delay = delay
            )
        rdftbm = np.fft.rfft(np.fromiter(
            (
                fsum(
                    b_spline(k - delay - p * p0, degree)
                    for p in range(
                        int((k - delay - 0.5 * (degree - 1.0)) // p0),
                        int((k - delay + 0.5 * (degree + 1.0)) // p0) + 1
                    )
                )
                for k in range(p0)
            ),
            dtype = float,
            count = p0
        ))
        rdftbp = np.fft.rfft(np.fromiter(
            (
                fsum(
                    b_spline(k + delay - p * p0, degree)
                    for p in range(
                        int((k + delay - 0.5 * (degree - 1.0)) // p0),
                        int((k + delay + 0.5 * (degree + 1.0)) // p0) + 1
                    )
                )
                for k in range(p0)
            ),
            dtype = float,
            count = p0
        ))
        rdftdb = [
            np.fft.rfft(np.fromiter(
                (
                    fsum(
                        diff_b_spline(
                            k - p * p0,
                            degree = 2 * degree + 1,
                            differentiation_order = 2 * m
                        )
                        for p in range(
                            int((k - degree) // p0),
                            int((k + degree + 1) // p0) + 1
                        )
                    )
                    for k in range(p0)
                ),
                dtype = float,
                count = p0
            ))
            for m in range(degree + 1)
        ]
        rdftdenum = np.fromiter(
            (
                rdftbm[nu] * rdftbp[nu] + sum(
                    ((-1) ** m) * lmbd * rdftdb[m][nu]
                    for (m, lmbd) in enumerate(smoothing)
                )
                for nu in range(p0 // 2 + 1)
            ),
            dtype = complex,
            count = p0 // 2 + 1
        )
        rdftsamples = np.fft.rfft(samples)
        return cls.from_spline_coeff(
            cast(
                np.ndarray[tuple[int], np.dtype[np.float64]],
                np.fft.irfft(
                    np.fromiter(
                        (
                            0.0
                                if isclose(
                                    abs(rdftdenum[nu]),
                                    0.0,
                                    rel_tol = sqrt(ulp(1.0)),
                                    abs_tol = sqrt(ulp(1.0))
                                )
                                else (rdftsamples[nu] * rdftbp[nu] /
                                    rdftdenum[nu])
                            for nu in range(p0 // 2 + 1)
                        ),
                        dtype = complex,
                        count = p0 // 2 + 1
                    ),
                    n = p0,
                )
            ),
            degree = degree,
            delay = delay
        )

    #---------------
    @classmethod
    def from_spline_coeff (
        cls,
        spline_coeff: np.ndarray[tuple[int], np.dtype[np.float64]],
        *,
        degree: int,
        delay: float = 0.0
    ) -> PeriodicSpline1D:

        """
        .. _periodic_spline_1d-from_spline_coeff:

        A constructor from spline coefficients.

        This constructor populates the attributes of a ``PeriodicSpline1D``
        object with the provided parameters. The created object retains a copy
        of the array ``spline_coeff``.

        Parameters
        ----------
        spline_coeff : np.ndarray[tuple[int], np.dtype[np.float64]]
            A one-dimensional ``numpy`` array of real spline coefficients. The
            period of the spline will be set to the length of this array.
        degree : int
            The :ref:`nonnegative<def-negative>` degree of the polynomial
            spline.
        delay : float
            The delay of this spline.

        Returns
        -------
        PeriodicSpline1D
            The spline.

        Examples
        --------
        Load the libraries.
            >>> import numpy as np
            >>> import splinekit as sk
        Create an undelayed cubic spline from a few arbitrary coefficients.
            >>> c = np.array([1, 9, -7], dtype = float)
            >>> sk.PeriodicSpline1D.from_spline_coeff(c, degree = 3)
            PeriodicSpline1D([ 1.  9. -7.], degree = 3, delay = 0.0)

        Raises
        ------
        ValueError
            Raised when ``spline_coeff`` fails to be a ``numpy``
            one-dimensional ``float`` array that contains at least one element.
        ValueError
            Raised when ``degree`` is :ref:`negative<def-negative>`.


        ----

        """

        if np.ndarray != type(spline_coeff):
            raise ValueError("Spline_coeff must be a numpy array")
        if 1 != len(spline_coeff.shape):
            raise ValueError(
                "Spline_coeff must be a one-dimensional numpy array"
            )
        if float != spline_coeff.dtype:
            raise ValueError(
                "Spline_coeff must be of type np.ndarray[tuple[int], np.dtype[np.float64]]"
            )
        if 0 > degree:
            raise ValueError("Degree must be nonnegative")
        s = cls()
        s.spline_coeff = spline_coeff
        s.degree = degree
        s.delay = delay
        return s

    #---------------
    @classmethod
    def add (
        cls,
        augend: PeriodicSpline1D,
        addend: PeriodicSpline1D
    ) -> PeriodicSpline1D:

        r"""
        .. _periodic_spline_1d-add:

        Create a spline from the sum of two splines.

        Given two ``PeriodicSpline1D`` objects of identical period :math:`K,`
        degree :math:`n,` and delay :math:`\delta x,` this constructor
        returns a new ``PeriodicSpline1D`` spline that represents their sum.
        The created object has period :math:`K,` degree :math:`n,` and delay
        :math:`\delta x,` too.

        Parameters
        ----------
        augend : PeriodicSpline1D
            A ``PeriodicSpline1D`` spline.
        addend : PeriodicSpline1D
            A ``PeriodicSpline1D`` spline.

        Returns
        -------
        PeriodicSpline1D
            The spline equal to the sum of ``augend`` and ``addend``.

        Examples
        --------
        Load the libraries.
            >>> import numpy as np
            >>> import splinekit as sk
        Create two undelayed cubic splines and sum them.
            >>> c1 = np.array([1, 9, -7], dtype = float)
            >>> s1 = sk.PeriodicSpline1D.from_spline_coeff(c1, degree = 3)
            >>> c2 = np.array([6, -3, -2], dtype = float)
            >>> s2 = sk.PeriodicSpline1D.from_spline_coeff(c2, degree = 3)
            >>> sk.PeriodicSpline1D.add(s1, s2)
            PeriodicSpline1D([ 7.  6. -9.], degree = 3, delay = 0.0)

        Raises
        ------
        ValueError
            Raised when ``augend.period != addend.period``.
        ValueError
            Raised when ``augend.degree != addend.degree``.
        ValueError
            Raised when ``augend.delay != addend.delay``.


        ----

        """

        if augend.period != addend.period:
            raise ValueError("The periods must match")
        if augend.degree != addend.degree:
            raise ValueError("The degrees must match")
        if augend.delay != addend.delay:
            raise ValueError("The delays must match")
        return PeriodicSpline1D.from_spline_coeff(
            cast(
                np.ndarray[tuple[int], np.dtype[np.float64]],
                np.add(augend.spline_coeff, addend.spline_coeff)
            ),
            degree = augend.degree,
            delay = augend.delay
        )

    #---------------
    @classmethod
    def add_projected (
        cls,
        augend: PeriodicSpline1D,
        addend: PeriodicSpline1D,
        *,
        degree: int,
        delay: float = 0.0
    ) -> PeriodicSpline1D:

        r"""
        .. _periodic_spline_1d-add_projected:

        Create a spline of arbitrary degree and delay from the sum of two
        splines.

        Given two ``PeriodicSpline1D`` objects of identical period :math:`K,`
        independent degree :math:`(n_{1}, n_{2}),` and independent delay
        :math:`(\delta x_{1}, \delta x_{2}),` this constructor
        returns a new ``PeriodicSpline1D`` spline that represents the
        continuous least-squares approximation of their sum by a
        ``PeriodicSpline1D`` spline of period :math:`K,`
        :ref:`nonnegative<def-negative>` degree :math:`n,` and delay
        :math:`\delta x\in{\mathbb{R}}.`

        With

        ..  math::

            \begin{eqnarray*}
            s_{1}(x)&=&\sum_{k\in{\mathbb{Z}}}\,c_{1}[{k\bmod K}]\,
            \beta^{n_{1}}(x-\delta x_{1}-k)\\
            s_{2}(x)&=&\sum_{k\in{\mathbb{Z}}}\,c_{2}[{k\bmod K}]\,
            \beta^{n_{2}}(x-\delta x_{2}-k),
            \end{eqnarray*}

        the created sum

        ..  math::

            f(x)=\sum_{k\in{\mathbb{Z}}}\,c[{k\bmod K}]\,
            \beta^{n}(x-\delta x-k)

        minimizes the criterion :math:`J=\int_{0}^{K}\,\left(f(x)-s_{1}(x)-
        s_{2}(x)\right)^{2}\,{\mathrm{d}}x` in terms of the coefficients
        :math:`c.`

        Parameters
        ----------
        augend : PeriodicSpline1D
            A ``PeriodicSpline1D`` spline.
        addend : PeriodicSpline1D
            A ``PeriodicSpline1D`` spline.
        degree : int
            The :ref:`nonnegative<def-negative>` degree of the created
            polynomial spline.
        delay : float
            The delay of the created spline.

        Returns
        -------
        PeriodicSpline1D
            The spline of degree ``degree`` and delay ``delay`` that best
            approximates the sum of ``augend`` and ``addend``.

        Examples
        --------
        Load the libraries.
            >>> import numpy as np
            >>> import splinekit as sk
        Create arbitrary splines and sum them.
            >>> c1 = np.array([1, 9, -7], dtype = float)
            >>> s1 = sk.PeriodicSpline1D.from_spline_coeff(c1, degree = 3, delay = 8.4)
            >>> c2 = np.array([6, -3, -2], dtype = float)
            >>> s2 = sk.PeriodicSpline1D.from_spline_coeff(c2, degree = 1, delay = 0.1)
            >>> sk.PeriodicSpline1D.add_projected(s1, s2, degree = 0, delay = -4.25)
            PeriodicSpline1D([-4.72426875  6.28725     2.43701875], degree = 0, delay = -4.25)

        Raises
        ------
        ValueError
            Raised when ``augend.period != addend.period``.
        ValueError
            Raised when ``degree`` is :ref:`negative<def-negative>`.


        ----

        """

        if augend.period != addend.period:
            raise ValueError("The periods must match")
        if 0 > degree:
            raise ValueError("Degree must be nonnegative")
        if (
            (augend.degree == addend.degree) and
                (augend.delay == addend.delay)
        ):
            return PeriodicSpline1D.add(augend, addend)
        return PeriodicSpline1D.add(
            augend.projected(degree = degree, delay = delay),
            addend.projected(degree = degree, delay = delay)
        )

    #---------------
    @classmethod
    def convolve (
        cls,
        s1: PeriodicSpline1D,
        s2: PeriodicSpline1D
    ) -> PeriodicSpline1D:

        r"""
        .. _periodic_spline_1d-convolve:

        Create a spline from the convolution of two splines.

        Given two ``PeriodicSpline1D`` objects of identical period :math:`K,`
        independent degree :math:`(n_{1}, n_{2}),` and independent delay
        :math:`(\delta x_{1}, \delta x_{2}),` this constructor returns a new
        ``PeriodicSpline1D`` spline that represents their convolution. The
        created object has period :math:`K,` too.

        With

        ..  math::

            \begin{eqnarray*}
            s_{1}(x)&=&\sum_{k\in{\mathbb{Z}}}\,c_{1}[{k\bmod K}]\,
            \beta^{n_{1}}(x-\delta x_{1}-k)\\
            s_{2}(x)&=&\sum_{k\in{\mathbb{Z}}}\,c_{2}[{k\bmod K}]\,
            \beta^{n_{2}}(x-\delta x_{2}-k),
            \end{eqnarray*}

        the created convolution result is

        ..  math::

            f(x)=\int_{0}^{K}\,s_{1}(y)\,s_{2}(x-y)\,{\mathrm{d}}y.

        Parameters
        ----------
        s1 : PeriodicSpline1D
            A ``PeriodicSpline1D`` spline.
        s2 : PeriodicSpline1D
            A ``PeriodicSpline1D`` spline.

        Returns
        -------
        PeriodicSpline1D
            The spline equal to the convolution of ``s1`` and ``s2``.

        Examples
        --------
        Load the libraries.
            >>> import numpy as np
            >>> import splinekit as sk
        Create arbitrary splines and sum them.
            >>> c1 = np.array([1, 9, -7], dtype = float)
            >>> s1 = sk.PeriodicSpline1D.from_spline_coeff(c1, degree = 3, delay = 8.4)
            >>> c2 = np.array([6, -3, -2], dtype = float)
            >>> s2 = sk.PeriodicSpline1D.from_spline_coeff(c2, degree = 1, delay = 0.1)
            >>> sk.PeriodicSpline1D.convolve(s1, s2)
            PeriodicSpline1D([  9.  65. -71.], degree = 5, delay = 8.5)

        Raises
        ------
        ValueError
            Raised when ``s1.period != s2.period``.


        ----

        """

        if s1.period != s2.period:
            raise ValueError("The periods must match")
        return PeriodicSpline1D.from_spline_coeff(
            cast(
                np.ndarray[tuple[int], np.dtype[np.float64]],
                np.fromiter(
                    (
                        fsum(
                            s1q * s2.spline_coeff[(k - q) % s2.period]
                            for (q, s1q) in enumerate(s1.spline_coeff)
                        )
                        for k in range(s1.period)
                    ),
                    dtype = float,
                    count = s1.period
                )
            ),
            degree = s2.degree + s1.degree + 1,
            delay = s2.delay + s1.delay
        )

    #---------------
    @classmethod
    def normed_cross_correlate (
        cls,
        s1: PeriodicSpline1D,
        s2: PeriodicSpline1D
    ) -> PeriodicSpline1D:

        r"""
        .. _periodic_spline_1d-normed_cross_correlate:

        Create a spline from the normalized cross-correlation of two splines.

        Given two ``PeriodicSpline1D`` objects of identical period :math:`K,`
        independent degree :math:`(n_{1}, n_{2}),` and independent delay
        :math:`(\delta x_{1}, \delta x_{2}),` this constructor returns a new
        ``PeriodicSpline1D`` spline that represents their zero-normalized
        cross-correlation. The created object has period :math:`K,` too.

        With

        ..  math::

            \begin{eqnarray*}
            s_{1}(x)&=&\sum_{k\in{\mathbb{Z}}}\,c_{1}[{k\bmod K}]\,
            \beta^{n_{1}}(x-\delta x_{1}-k)\\
            s_{2}(x)&=&\sum_{k\in{\mathbb{Z}}}\,c_{2}[{k\bmod K}]\,
            \beta^{n_{2}}(x-\delta x_{2}-k),
            \end{eqnarray*}

        the created zero-normalized cross-correlation result
        :math:`f(x)\in[-1,1]` is

        ..  math::

            f(x)=\frac{1}{K}\,\int_{0}^{K}\,\frac{s_{1}(y)-
            {\mathrm{E}}\{s_{1}\}}{\sqrt{{\mathrm{Var}}\{s_{1}\}}}\,
            \frac{s_{2}(y+x)}{\sqrt{{\mathrm{Var}}\{s_{2}\}}}\,{\mathrm{d}}y,

        where :math:`{\mathrm{E}}` is the
        :ref:`mean value<periodic_spline_1d-mean>` of a spline and
        :math:`{\mathrm{Var}}` is its
        :ref:`variance<periodic_spline_1d-variance>`.

        Parameters
        ----------
        s1 : PeriodicSpline1D
            A ``PeriodicSpline1D`` spline.
        s2 : PeriodicSpline1D
            A ``PeriodicSpline1D`` spline.

        Returns
        -------
        PeriodicSpline1D
            The spline equal to the zero-normalized cross-correlation of ``s1``
            and ``s2``.

        Examples
        --------
        Load the libraries.
            >>> import numpy as np
            >>> import splinekit as sk
        Create arbitrary splines and sum them.
            >>> c1 = np.array([1, 9, -7], dtype = float)
            >>> s1 = sk.PeriodicSpline1D.from_spline_coeff(c1, degree = 3, delay = 8.4)
            >>> c2 = np.array([6, -3, -2], dtype = float)
            >>> s2 = sk.PeriodicSpline1D.from_spline_coeff(c2, degree = 1, delay = 0.1)
            >>> sk.PeriodicSpline1D.normed_cross_correlate(s1, s2)
            PeriodicSpline1D([-0.30586209 -2.44689675  2.75275884], degree = 5, delay = -8.3)

        Raises
        ------
        ValueError
            Raised when ``s1.period != s2.period``.
        ValueError
            Raised when ``0.0 == s1.variance()``.
        ValueError
            Raised when ``0.0 == s2.variance()``.


        ----

        """

        if s1.period != s2.period:
            raise ValueError("The periods must match")
        v1 = s1.variance()
        v2 = s2.variance()
        if (0.0 == v1 ** 2) or (0.0 == v2 ** 2):
            raise ValueError("The variance of either spline must not vanish")
        m1 = s1.mean()
        s0 = 1.0 / (s1.period * sqrt(v1 * v2))
        return PeriodicSpline1D.from_spline_coeff(
            cast(
                np.ndarray[tuple[int], np.dtype[np.float64]],
                np.fromiter(
                    (
                        s0 * fsum(
                            (s1q - m1) * s2.spline_coeff[(k + q) % s2.period]
                            for (q, s1q) in enumerate(s1.spline_coeff)
                        )
                        for k in range(s1.period)
                    ),
                    dtype = float,
                    count = s1.period
                )
            ),
            degree = s2.degree + s1.degree + 1,
            delay = s2.delay - s1.delay
        )

    #---------------
    @classmethod
    def eval_piecewise_polynomials_at (
        cls,
        x: float,
        polynomials: PeriodicNonuniformPiecewise
    ) -> float:

        """
        .. _periodic_spline_1d-eval_piecewise_polynomials_at:

        The ordinate of a piecewise polynomial at an arbitrary abscissa.

        Evaluated at the abscissa ``x``, this function returns the value of a
        given :ref:`PeriodicNonuniformPiecewise<PeriodicNonuniformPiecewise>`
        object that represents a periodic
        :ref:`piecewise polynomial<def-piecewise_polynomial>`, as made
        available by the function
        :ref:`piecewise_polynomials<periodic_spline_1d-piecewise_polynomials>`.

        The evaluation proceeds in two steps. The first step is to identify
        to which piece of the piecewise polynomial the abscissa belongs. The
        second step is to evaluate the ordinate by direct computation of the
        polynomial.

        Parameters
        ----------
        x : float
            The abscissa.
        polynomials : PeriodicNonuniformPiecewise
            A piecewise-polynomial function.

        Returns
        -------
        float
            The value of the piecewise polynomial evaluated at ``x``.

        Examples
        --------
        Load the libraries.
            >>> import numpy as np
            >>> import splinekit as sk
        Create an arbitrary spline and evaluate it from its pievewise-polynomial representation.
            >>> c = np.array([1, 9, -7], dtype = float)
            >>> s = sk.PeriodicSpline1D.from_spline_coeff(c, degree = 3)
            >>> pp = s.piecewise_polynomials()
            >>> x = -18.5
            >>> print(sk.PeriodicSpline1D.eval_piecewise_polynomials_at(x, pp))
            >>> print(s.at(x))
            >>> print(s.at(x) - sk.PeriodicSpline1D.eval_piecewise_polynomials_at(x, pp))
            -2.499999999999999
            -2.5
            -8.881784197001252e-16

        See Also
        --------
        at : Evaluation of this spline at ``x``.


        ----

        """

        (_, x_wrapped) = _divmod(x, polynomials.period)
        for piece in polynomials.pieces:
            if x_wrapped in piece.domain:
                return float(piece.item(x_wrapped))
        raise ValueError(
            "Internal error (polynomials must partition [0.0, period))"
        )

    #---------------
    @classmethod
    def eval_piecewise_signs_at (
        cls,
        x: float,
        signs: PeriodicNonuniformPiecewise
    ) -> float:

        # TODO: docstring
        # TODO: tests
        r"""
        .. _periodic_spline_1d-eval_piecewise_signs_at:


        ----

        """

        (_, x_wrapped) = _divmod(x, signs.period)
        for piece in signs.pieces:
            if x_wrapped in piece.domain:
                return float(piece.item)
        raise ValueError(
            "Internal error (signs must partition [0.0, period))"
        )

    #---------------
    def copy (
        self
    ) -> PeriodicSpline1D:

        """
        .. _periodic_spline_1d-copy:

        A copy of this object.

        This function returns a deep copy of this object.

        Parameters
        ----------
        None

        Returns
        -------
        PeriodicSpline1D
            The copied spline.
    
        Examples
        --------
        Load the libraries.
            >>> import numpy as np
            >>> import splinekit as sk
        Create an arbitrary spline and duplicate it.
            >>> c = np.array([1, 9, -7], dtype = float)
            >>> s1 = sk.PeriodicSpline1D.from_spline_coeff(c, degree = 3, delay = 8.4)
            >>> s2 = s1.copy()
            >>> s1
            PeriodicSpline1D([ 1.  9. -7.], degree = 3, delay = 8.4)
            >>> s2
            PeriodicSpline1D([ 1.  9. -7.], degree = 3, delay = 8.4)

        ----

        """

        return PeriodicSpline1D.from_spline_coeff(
            cast(
                np.ndarray[tuple[int], np.dtype[np.float64]],
                self._spline_coeff
            ),
            degree = self._degree,
            delay = self._delay
        )

    #---------------
    def plot (
        self,
        subplot: Tuple[plt.Figure, plt.Axes],
        *,
        plotdomain: Interval = RR(),
        plotrange: Interval = RR(),
        plotpoints: int,
        line_fmt: str = "-C0",
        marker_fmt: str = "oC0",
        stem_fmt: str = "-C0",
        knot_marker: str = "o",
        knot_color: str = "k",
        periodbound_marker_fmt: str = "or",
        periodbound_stem_fmt: str = "-r"
    ):

        # TODO: docstring
        # TODO: tests
        r"""
        .. _periodic_spline_1d-plot:


        ----

        """

        start = -1.0
        stop = self._period + 1.0
        start_included = False
        stop_included = False
        if isinstance(plotdomain, Empty):
            raise ValueError("Invalid plotdomain")
        if isinstance(plotdomain, Above):
            if 2 > plotpoints:
                raise ValueError("At least two plotpoints are required")
            start = plotdomain.threshold
            stop = start + self._period
            stop_included = True
        elif isinstance(plotdomain, NotBelow):
            if 2 > plotpoints:
                raise ValueError("At least two plotpoints are required")
            start = plotdomain.threshold
            start_included = True
            stop = start + self._period
        elif isinstance(plotdomain, NotAbove):
            if 2 > plotpoints:
                raise ValueError("At least two plotpoints are required")
            start = plotdomain.threshold - self._period
            stop = plotdomain.threshold
            stop_included = True
        elif isinstance(plotdomain, Below):
            if 2 > plotpoints:
                raise ValueError("At least two plotpoints are required")
            start = plotdomain.threshold - self._period
            start_included = True
            stop = plotdomain.threshold
        elif isinstance(plotdomain, RR):
            if 2 > plotpoints:
                raise ValueError("At least two plotpoints are required")
        elif isinstance(plotdomain, Singleton):
            if 1 != plotpoints:
                raise ValueError("A Singleton requires that plotpoints = 1")
            start = plotdomain.value
            start_included = True
            stop = plotdomain.value
            stop_included = True
        elif isinstance(plotdomain, Open):
            if 2 > plotpoints:
                raise ValueError("At least two plotpoints are required")
            start = plotdomain.infimum
            stop = plotdomain.supremum
        elif isinstance(plotdomain, OpenClosed):
            if 2 > plotpoints:
                raise ValueError("At least two plotpoints are required")
            start = plotdomain.infimum
            stop = plotdomain.supremum
            stop_included = True
        elif isinstance(plotdomain, Closed):
            if 2 > plotpoints:
                raise ValueError("At least two plotpoints are required")
            start = plotdomain.infimum
            start_included = True
            stop = plotdomain.supremum
            stop_included = True
        elif isinstance(plotdomain, ClosedOpen):
            if 2 > plotpoints:
                raise ValueError("At least two plotpoints are required")
            start = plotdomain.infimum
            start_included = True
            stop = plotdomain.supremum
        else:
            raise ValueError("Internal error (unexpected interval class)")
        minrange = 0.0
        minrange_provided = False
        maxrange = 0.0
        maxrange_provided = False
        if isinstance(plotrange, (Empty, Singleton)):
            raise ValueError("Invalid plotrange")
        if isinstance(plotrange, RR):
            pass
        elif isinstance(plotrange, (Above, NotBelow)):
            minrange = plotrange.threshold
            minrange_provided = True
        elif isinstance(plotrange, (NotAbove, Below)):
            maxrange = plotrange.threshold
            maxrange_provided = True
        elif isinstance(plotrange, (Open, OpenClosed, Closed, ClosedOpen)):
            minrange = plotrange.infimum
            minrange_provided = True
            maxrange = plotrange.supremum
            maxrange_provided = True
        else:
            raise ValueError("Internal error (unexpected interval class)")
        if 1 == plotpoints:
            fig = subplot[0]
            stems = subplot[1].twinx()
            stems.sharey(subplot[1])
            stems.spines.right.set_visible(False)
            stems.spines.top.set_visible(False)
            x = start if start == stop else 0.5 * (start + stop)
            if not x in plotdomain:
                return
            fx = self.at(x)
            if fx not in plotrange:
                return
            if not minrange_provided:
                minrange = fx - 0.25
            if not maxrange_provided:
                maxrange = fx + 0.25
            plt.pyplot.ylim(minrange, maxrange)
            if floor(x) != x:
                (markerline, stemlines, baseline) = stems.stem(
                    [x],
                    [fx],
                    linefmt = "None",
                    markerfmt = marker_fmt,
                    bottom = 0.0
                )
            elif 0 != int(x) % self._period:
                (markerline, stemlines, baseline) = stems.stem(
                    [x],
                    [fx],
                    linefmt = stem_fmt,
                    markerfmt = marker_fmt,
                    bottom = 0.0
                )
                c = plt.colors.to_hex(markerline.get_markerfacecolor())
                markerline.set_markerfacecolor(c + "00")
                markerline.set_markeredgecolor(c + "ff")
                stemlines.set_linewidth(0.25)
            else:
                (markerline, stemlines, baseline) = stems.stem(
                    [x],
                    [fx],
                    linefmt = periodbound_stem_fmt,
                    markerfmt = periodbound_marker_fmt,
                    bottom = 0.0
                )
                c = plt.colors.to_hex(markerline.get_markerfacecolor())
                markerline.set_markerfacecolor(c + "00")
                markerline.set_markeredgecolor(c + "ff")
                stemlines.set_linewidth(0.25)
            data = stems.twinx()
            data.sharey(subplot[1])
            data.axis("off")
            for (k, k0) in enumerate(self.get_knots(plotdomain)):
                data.plot(
                    k0,
                    self.at(k0),
                    scalex = False,
                    scaley = False,
                    marker = knot_marker,
                    markerfacecolor = knot_color,
                    markersize = 3.5
                )
            plt.pyplot.ylim(minrange, maxrange)
        elif 0 == self._degree:
            fig = subplot[0]
            data = subplot[1].twinx()
            data.sharey(subplot[1])
            data.spines.right.set_visible(False)
            data.spines.top.set_visible(False)
            domain = cast(Interval, Empty())
            if start_included:
                if stop_included:
                    domain = Closed((start, stop))
                else:
                    domain = ClosedOpen((start, stop))
            else:
                if stop_included:
                    domain = OpenClosed((start, stop))
                else:
                    domain = Open((start, stop))
            knots = self.get_knots(domain)
            if 0 == len(knots):
                fx0 = self.at(0.5 * (start + stop))
                xx = np.array([start, stop], dtype = float)
                fxx = np.array([fx0, fx0], dtype = float)
                data.plot(xx, fxx, line_fmt)
                if not minrange_provided:
                    (minrange, _) = plt.pyplot.ylim()
                if not maxrange_provided:
                    (_, maxrange) = plt.pyplot.ylim()
                plt.pyplot.ylim(minrange, maxrange)
                stems = data.twinx()
                stems.sharey(subplot[1])
                stems.axis("off")
                intdomain = (
                    ceil(start)
                        if ceil(start) in domain
                        else ceil(start) + 1,
                    floor(stop)
                        if floor(stop) in domain
                        else floor(stop) - 1
                )
                if intdomain[0] == intdomain[1]:
                    if 0 != intdomain[0] % self._period:
                        (markerline, stemlines, baseline) = stems.stem(
                            [intdomain[0]],
                            [fx0],
                            linefmt = stem_fmt,
                            markerfmt = marker_fmt,
                            bottom = 0.0
                        )
                        c = plt.colors.to_hex(markerline.get_markerfacecolor())
                        markerline.set_markerfacecolor(c + "00")
                        markerline.set_markeredgecolor(c + "ff")
                        stemlines.set_linewidth(0.25)
                    else:
                        (markerline, stemlines, baseline) = stems.stem(
                            [intdomain[0]],
                            [fx0],
                            linefmt = periodbound_stem_fmt,
                            markerfmt = periodbound_marker_fmt,
                            bottom = 0.0
                        )
                        c = plt.colors.to_hex(markerline.get_markerfacecolor())
                        markerline.set_markerfacecolor(c + "00")
                        markerline.set_markeredgecolor(c + "ff")
                        stemlines.set_linewidth(0.25)
                    plt.pyplot.ylim(minrange, maxrange)
            else:
                f_left = self.at(0.5 * (start + knots[0]))
                xx = np.array([start, knots[0]], dtype = float)
                fxx = np.array([f_left, f_left], dtype = float)
                data.plot(xx, fxx, line_fmt)
                for (k, k0) in enumerate(knots):
                    f_right = self.at(k0 + 0.5)
                    data.plot(
                        k0,
                        0.5 * (f_left + f_right),
                        scalex = False,
                        scaley = False,
                        marker = knot_marker,
                        markerfacecolor = knot_color,
                        markeredgecolor = "#00000000",
                        markersize = 3.5
                    )
                    if 0 == k:
                        f_left = f_right
                        continue
                    xx = np.array([k0 - 1.0, k0], dtype = float)
                    fxx = np.array([f_left, f_left], dtype = float)
                    data.plot(xx, fxx, line_fmt)
                    f_left = f_right
                f_right = self.at(0.5 * (knots[-1] + stop))
                xx = np.array([knots[-1], stop], dtype = float)
                fxx = np.array([f_right, f_right], dtype = float)
                data.plot(xx, fxx, line_fmt)
                if not minrange_provided:
                    (minrange, _) = plt.pyplot.ylim()
                if not maxrange_provided:
                    (_, maxrange) = plt.pyplot.ylim()
                plt.pyplot.ylim(minrange, maxrange)
                stems = data.twinx()
                stems.sharey(subplot[1])
                stems.axis("off")
                intdomain = (
                    ceil(start)
                        if ceil(start) in domain
                        else ceil(start) + 1,
                    floor(stop)
                        if floor(stop) in domain
                        else floor(stop) - 1
                )
                if intdomain[0] <= intdomain[1]:
                    kr = np.arange(intdomain[0], intdomain[1] + 1)
                    kp = kr[0 == kr % self._period]
                    knp = kr[0 != kr % self._period]
                    fp = np.fromiter(
                        (self.at(float(k0)) for k0 in kp),
                        dtype = float,
                        count = len(kp)
                    )
                    fnp = np.fromiter(
                        (self.at(float(k0)) for k0 in knp),
                        dtype = float,
                        count = len(knp)
                    )
                    if 0 < len(kp):
                        (markerline, stemlines, baseline) = stems.stem(
                            kp,
                            fp,
                            linefmt = periodbound_stem_fmt,
                            markerfmt = periodbound_marker_fmt,
                            bottom = 0.0
                        )
                        c = plt.colors.to_hex(markerline.get_markerfacecolor())
                        markerline.set_markerfacecolor(c + "00")
                        markerline.set_markeredgecolor(c + "ff")
                        stemlines.set_linewidth(0.25)
                        baseline.set_alpha(0.0)
                    if 0 < len(knp):
                        (markerline, stemlines, baseline) = stems.stem(
                            knp,
                            fnp,
                            linefmt = stem_fmt,
                            markerfmt = marker_fmt,
                            bottom = 0.0
                        )
                        c = plt.colors.to_hex(markerline.get_markerfacecolor())
                        markerline.set_markerfacecolor(c + "00")
                        markerline.set_markeredgecolor(c + "ff")
                        stemlines.set_linewidth(0.25)
                        baseline.set_alpha(0.0)
                    plt.pyplot.ylim(minrange, maxrange)
        else:
            fig = subplot[0]
            data = subplot[1].twinx()
            data.sharey(subplot[1])
            data.spines.right.set_visible(False)
            data.spines.top.set_visible(False)
            domain = cast(Interval, Empty())
            if start_included:
                if stop_included:
                    domain = Closed((start, stop))
                else:
                    domain = ClosedOpen((start, stop))
            else:
                if stop_included:
                    domain = OpenClosed((start, stop))
                else:
                    domain = Open((start, stop))
            xx = np.linspace(start, stop, num = plotpoints)
            fxx = np.fromiter(
                (self.at(x0) for x0 in xx),
                dtype = float,
                count = plotpoints
            )
            data.plot(xx, fxx, line_fmt)
            kk = self.get_knots(domain)
            for k0 in kk:
                data.plot(
                    k0,
                    self.at(k0),
                    scalex = False,
                    scaley = False,
                    marker = knot_marker,
                    markerfacecolor = knot_color,
                    markeredgecolor = "#00000000",
                    markersize = 3.5
                )
            if not minrange_provided:
                (minrange, _) = plt.pyplot.ylim()
            if not maxrange_provided:
                (_, maxrange) = plt.pyplot.ylim()
            plt.pyplot.ylim(minrange, maxrange)
            stems = data.twinx()
            stems.sharey(subplot[1])
            stems.axis("off")
            intdomain = (
                ceil(start)
                    if ceil(start) in domain
                    else ceil(start) + 1,
                floor(stop)
                    if floor(stop) in domain
                    else floor(stop) - 1
            )
            if intdomain[0] <= intdomain[1]:
                kr = np.arange(intdomain[0], intdomain[1] + 1)
                kp = kr[0 == kr % self._period]
                knp = kr[0 != kr % self._period]
                fp = np.fromiter(
                    (self.at(float(k0)) for k0 in kp),
                    dtype = float,
                    count = len(kp)
                )
                fnp = np.fromiter(
                    (self.at(float(k0)) for k0 in knp),
                    dtype = float,
                    count = len(knp)
                )
                if 0 < len(kp):
                    (markerline, stemlines, baseline) = stems.stem(
                        kp,
                        fp,
                        linefmt = periodbound_stem_fmt,
                        markerfmt = periodbound_marker_fmt,
                        bottom = 0.0
                    )
                    c = plt.colors.to_hex(markerline.get_markerfacecolor())
                    markerline.set_markerfacecolor(c + "00")
                    markerline.set_markeredgecolor(c + "ff")
                    stemlines.set_linewidth(0.25)
                    baseline.set_alpha(0.0)
                if 0 < len(knp):
                    (markerline, stemlines, baseline) = stems.stem(
                        knp,
                        fnp,
                        linefmt = stem_fmt,
                        markerfmt = marker_fmt,
                        bottom = 0.0
                    )
                    c = plt.colors.to_hex(markerline.get_markerfacecolor())
                    markerline.set_markerfacecolor(c + "00")
                    markerline.set_markeredgecolor(c + "ff")
                    stemlines.set_linewidth(0.25)
                    baseline.set_alpha(0.0)
                plt.pyplot.ylim(minrange, maxrange)
        fig.tight_layout()

    #---------------
    def at (
        self,
        x: float
    ) -> float:

        r"""
        .. _periodic_spline_1d-at:

        The ordinate of this spline at an arbitrary abscissa.

        This function returns the value of this spline :math:`f` of period
        :math:`K,` degree :math:`n`, and delay :math:`\delta x,` as defined
        for :math:`x\in{\mathbb{R}}` by

        ..  math::

            \begin{eqnarray*}
            f(x)&=&\sum_{k\in{\mathbb{Z}}}\,c[{k\bmod K}]\,
            \beta^{n}(x-\delta x-k)\\
            &=&{\mathbf{c}}^{{\mathsf{T}}}\,{\mathbf{W}}^{n}\,
            {\mathbf{v}}^{n}(\chi).
            \end{eqnarray*}

        There, :math:`c` are the :math:`K` spline coefficients,
        :math:`{\mathbf{c}}=\left(c[{\left(k-\Xi\right)\bmod
        K}]\right)_{k=0}^{n}\in{\mathbb{R}}^{n+1}` is a vector of
        :math:`\left(n+1\right)` spline coefficients, :math:`{\mathbf{W}}^{n}`
        is the :ref:`B-spline evaluation matrix<w_frac>`, and
        :math:`{\mathbf{v}}^{n}(\chi)` is the
        :ref:`Vandermonde vector<def-vandermonde_vector>` of argument
        :math:`\chi,` with :math:`\xi=\left(\frac{n-1}{2}-x+\delta x\right)
        \in{\mathbb{R}},` :math:`\Xi=\left\lceil\xi\right\rceil\in
        {\mathbb{Z}},` and :math:`\chi=\left(\Xi-\xi\right)\in[0,1).`

        As computed above, the fact that the Vandermonde vector has the domain
        :math:`[0,1)` greatly favors numerical stability since the range of
        each of its components is :math:`[0,1].`

        Parameters
        ----------
        x : float
            Argument.

        Returns
        -------
        float
            The value of this spline evaluated at ``x``.

        Examples
        --------
        Load the libraries.
            >>> import numpy as np
            >>> import splinekit as sk
        Create an undelayed spline and evaluate it at the arbitrary abscissa ``-18.5``.
            >>> c = np.array([1, 9, -7], dtype = float)
            >>> s = sk.PeriodicSpline1D.from_spline_coeff(c, degree = 3)
            >>> s.at(-18.5)
            -2.5

        ----

        """

        xi = 0.5 * (self._degree - 1.0) - x + self._delay
        if 0 == self._degree:
            if 0.0 == xi - round(xi):
                y = floor(x - self._delay)
                return 0.5 * (self._spline_coeff[y % self._period] +
                    self._spline_coeff[(y + 1) % self._period]
                )
            return self._spline_coeff[round(x - self._delay) % self._period]
        xi0 = floor(-xi)
        xi1 = xi0 + self._degree + 1
        if 0 <= xi0 and xi1 <= self._period:
            return float(
                self._spline_coeff[xi0 : xi1] @ (_w(self._degree) @
                    np.vander(
                        [ceil(xi) - xi],
                        self._degree + 1,
                        increasing = True
                    )[0]
                )
            )
        return float(
            np.fromiter(
                (
                    self._spline_coeff[k % self._period]
                    for k in range(xi0, xi1)
                ),
                dtype = float,
                count = self._degree + 1
            ) @ (_w(self._degree) @ np.vander(
                    [ceil(xi) - xi],
                    self._degree + 1,
                    increasing = True
                )[0]
            )
        )

    #---------------
    def get_samples (
        self,
        starting_at: float,
        *,
        support_length: int,
        oversampling: int = 1
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:

        r"""
        .. _periodic_spline_1d-get_samples:

        Evaluation of this spline at an array of arguments spaced by a
        regular step.

        This function returns a ``numpy`` array of :math:`\left(L\,M\right)`
        samples of this spline :math:`f` of period :math:`K,` degree :math:`n`,
        and delay :math:`\delta x,` spaced by the regular step
        :math:`\frac{1}{M}`. The first sample of the array corresponds to
        :math:`f(x_{0}).` The returned array is

        ..  math::

            \begin{eqnarray*}
            \left(f(x_{0}+\frac{q}{M})\right)_{q=0}^{L\,M-1}&=&
            \left(\sum_{k\in{\mathbb{Z}}}\,c[{k\bmod K}]\,
            \beta^{n}(x_{0}+\frac{q}{M}-\delta x-k)\right)_{q=0}^{L\,M-1}\\
            &=&\left({\mathbf{c}}_{q}^{{\mathsf{T}}}\,{\mathbf{W}}^{n}\,
            {\mathbf{v}}^{n}(\chi_{x_{0}})\right)_{q=0}^{L\,M-1}.
            \end{eqnarray*}

        There, :math:`L` is ``support_length``, :math:`M` is ``oversampling``,
        :math:`x_{0}` is ``starting_at``, and :math:`c` are the :math:`K`
        spline coefficients.

        The terms :math:`{\mathbf{c}},` :math:`{\mathbf{W}}^{n},`
        :math:`{\mathbf{v}}^{n},` and :math:`\chi` are taken from the function
        :ref:`splinekit.PeriodicSpline1D.at<periodic_spline_1d-at>`. Because
        :math:`\left({\mathbf{W}}^{n}\,{\mathbf{v}}^{n}(\chi_{x_{0}})\right)`
        does not depend on :math:`q,` the computation of ``get_samples`` is
        faster than repeated calls to the function
        ``splinekit.PeriodicSpline1D.at``.

        Parameters
        ----------
        starting_at : float
            First argument.
        support_length : int
            The :ref:`nonnegative<def-negative>` length of the support of the
            spline over which the array of samples is taken.
        oversampling : int
            The :ref:`positive<def-positive>` number of samples taken per unit
            length of the spline.

        Returns
        -------
        np.ndarray[tuple[int], np.dtype[np.float64]]
            The array of values of this spline.

        Examples
        --------
        Load the libraries.
            >>> import numpy as np
            >>> import splinekit as sk
        Create an undelayed cubic spline.
            >>> c = np.array([1, 9, -7], dtype = float)
            >>> s = sk.PeriodicSpline1D.from_spline_coeff(c, degree = 3)
        Evaluate ``s(x)`` for ``x in [-18.5, -18.25, -18.0, -17.75]``.
            >>> s.get_samples(-18.5, support_length = 1, oversampling = 4)
            array([-2.5   , -0.9375,  1.    ,  2.9375])

        Raises
        ------
        ValueError
            Raised when ``support_length`` is :ref:`negative<def-negative>`.
        ValueError
            Raised when ``oversampling`` is :ref:`nonpositive<def-positive>`.


        ----

        """

        if 0 > support_length:
            raise ValueError("The support length must be nonnegative")
        if 0 >= oversampling:
            raise ValueError("Oversampling must be positive")
        xi = 0.5 * (self._degree - 1.0) - starting_at + self._delay
        if 1 == oversampling:
            if 0 == self._degree:
                if 0.0 == xi - round(xi):
                    y = floor(starting_at - self._delay)
                    return cast(
                        np.ndarray[tuple[int], np.dtype[np.float64]],
                        np.fromiter(
                            (
                                0.5 * (self._spline_coeff[k % self._period] +
                                    self._spline_coeff[(k + 1) %
                                    self._period])
                                for k in range(y, y + support_length)
                            ),
                            dtype = float,
                            count = support_length
                        )
                    )
                y = round(starting_at - self._delay)
                return cast(
                    np.ndarray[tuple[int], np.dtype[np.float64]],
                    np.fromiter(
                        (
                            self._spline_coeff[k % self._period]
                            for k in range(y, y + support_length)
                        ),
                        dtype = float,
                        count = support_length
                    )
                )
            xi0 = floor(-xi)
            w = _w(self._degree) @ np.vander(
                [ceil(xi) - xi],
                self._degree + 1,
                increasing = True
            )[0]
            return cast(
                np.ndarray[tuple[int], np.dtype[np.float64]],
                np.fromiter(
                    (
                        (
                            self._spline_coeff[k : k + self._degree + 1]
                            if (0 <= k and k + self._degree < self._period)
                            else np.fromiter(
                                (
                                    self._spline_coeff[q % self._period]
                                    for q in range(k, k + self._degree + 1)
                                ),
                                dtype = float,
                                count = self._degree + 1
                            )
                        ) @ w
                        for k in range(xi0, xi0 + support_length)
                    ),
                    dtype = float,
                    count = support_length
                )
            )
        if 0 == self._degree:
            def at0 (
                k
            ):
                sigma = xi - k / oversampling
                y = starting_at + k / oversampling - self._delay
                if 0 == sigma - round(sigma):
                    y = floor(y)
                    return 0.5 * (self._spline_coeff[y % self._period] +
                        self._spline_coeff[(y + 1) % self._period]
                    )
                return self._spline_coeff[round(y) % self._period]
            return cast(
                np.ndarray[tuple[int], np.dtype[np.float64]],
                np.fromiter(
                    (at0(k) for k in range(support_length * oversampling)),
                    dtype = float,
                    count = support_length * oversampling
                )
            )
        def w0 (
            x
        ):
            if 1.0 <= x:
                x -= 1.0
                return np.concatenate(
                    (
                        [0.0],
                        _w(self._degree) @ np.vander(
                            [x],
                            self._degree + 1,
                            increasing = True
                        )[0]
                    ),
                    axis = None
                )
            return np.concatenate(
                (
                    _w(self._degree) @
                        np.vander([x], self._degree + 1, increasing = True)[0],
                    [0.0]
                ),
                axis = None
            )
        x0 = ceil(xi) - xi
        w = np.array(
            [w0(x0 + k / oversampling) for k in range(oversampling)],
            dtype = float
        )
        xi = int(floor(-xi))
        return np.ravel(np.fromiter(
            (
                w @ (
                    self._spline_coeff[k0 : k0 + self._degree + 2]
                    if (0 <= k0 and k0 + self._degree + 1 < self._period)
                    else np.fromiter(
                        (
                            self._spline_coeff[k % self._period]
                            for k in range(k0, k0 + self._degree + 2)
                        ),
                        dtype = float,
                        count = self._degree + 2
                    )
                )
                for k0 in range(xi, xi + support_length)
            ),
            dtype = np.dtype((float, oversampling)),
            count = support_length
        ))

    #---------------
    def mean (
        self
    ) -> float:

        r"""
        .. _periodic_spline_1d-mean:

        The mean of this ``PeriodicSpline1D`` object.

        This function returns the mean value of this spline :math:`f,` as
        defined by

        ..  math::

            E\{f\}=\lim_{(x_{0},x_{1})\rightarrow(-\infty,\infty)}
            \frac{1}{x_{1}-x_{0}}\,\int_{x_{0}}^{x_{1}}\,f(x)\,{\mathrm{d}}x.

        Parameters
        ----------
        None

        Returns
        -------
        float
            The mean of this spline.
    
        Examples
        --------
        Load the libraries.
            >>> import numpy as np
            >>> import splinekit as sk
        Create a spline and compute its mean.
            >>> c = np.array([1, 9, -7], dtype = float)
            >>> s = sk.PeriodicSpline1D.from_spline_coeff(c, degree = 3)
            >>> s.mean()
            1.0

        ----

        """

        return fsum(self._spline_coeff) / self._period

    #---------------
    def variance (
        self
    ) -> float:

        r"""
        .. _periodic_spline_1d-variance:

        The variance of this ``PeriodicSpline1D`` object.

        This function returns the variance over one period of this spline
        :math:`f` of period :math:`K,` as defined by

        ..  math::

            {\mathrm{Var}}\{f\}=\frac{1}{K}\,\int_{0}^{K}\,
            \left(f(x)-E\{f\}\right)^{2}\,{\mathrm{d}}x,

        where :math:`E\{f\}` is the :ref:`mean<periodic_spline_1d-mean>` of
        :math:`f.`

        Parameters
        ----------
        None

        Returns
        -------
        float
            The variance of this spline.
    
        Examples
        --------
        Load the libraries.
            >>> import numpy as np
            >>> import splinekit as sk
        Create a spline and compute its variance.
            >>> c = np.array([1, 9, -7], dtype = float)
            >>> s = sk.PeriodicSpline1D.from_spline_coeff(c, degree = 3)
            >>> s.variance()
            9.371428571428572

        ----

        """

        h = _b(2 * self._degree + 1)
        v = np.fromiter(
            (
                fsum(
                    hq * self._spline_coeff[(k + self._degree - q) %
                        self._period]
                    for (q, hq) in enumerate(h)
                )
                for k in range(self._period)
            ),
            dtype = float,
            count = self._period
        )
        return (fsum(vk * self._spline_coeff[k] for (k, vk) in enumerate(v)) /
            self._period - self.mean() ** 2)

    #---------------
    def lower_bound (
        self,
        *,
        degree: int
    ) -> PeriodicSpline1D:

        r"""
        .. _periodic_spline_1d-lower_bound:

        A spline that is never larger than this ``PeriodicSpline1D`` object.

        Assume that this spline :math:`s` is of period :math:`K` and degree
        :math:`m.` Then, this function returns another spline :math:`f` of
        same period :math:`K` and arbitrary :ref:`nonnegative<def-negative>`
        degree :math:`n` such that :math:`f(x)\leq g(x)` for all
        :math:`x\in{\mathbb{R}}.` The delay of :math:`f` is adjusted so that
        the knots of :math:`f` coincide with those of :math:`s.`

        Parameters
        ----------
        degree : int
            The :ref:`nonnegative<def-negative>` degree of the spline that
            lower-bounds this spline.

        Returns
        -------
        PeriodicSpline1D
            The lower-bounding spline.

        Examples
        --------
        Load the libraries.
            >>> import numpy as np
            >>> import splinekit as sk
        Create a spline and get a lower-bounding spline of lower degree.
            >>> c = np.array([1, 9, -7], dtype = float)
            >>> s = sk.PeriodicSpline1D.from_spline_coeff(c, degree = 3)
            >>> s.lower_bound(degree = 1)
            PeriodicSpline1D([  5. -11. -11.], degree = 1, delay = 1.0)
        Here is a lower-bounding spline of higher degree.
            >>> s.lower_bound(degree = 4)
            PeriodicSpline1D([-29.   7.  -5.], degree = 4, delay = -0.5)

        Notes
        -----
        The bound relies on two properties. One is that
        :math:`x^{n}\geq x^{n+1}` for :math:`x\in(0,1)` and
        :math:`n\in{\mathbb{N}}.` The other is that
        :ref:`B-splines<def-b_spline>` are nonnegative. The bound is
        reasonably sharp and fast to compute.

        Raises
        ------
        ValueError
            Raised when ``degree`` is :ref:`negative<def-negative>`.


        ----

        """

        if 0 > degree:
            raise ValueError("Degree must be nonnegative")
        if degree == self._degree:
            return self.copy()
        def low (
            p: int
        ) -> float:
            a0 = (
                self._spline_coeff[p : p + self._degree + 1]
                if (0 <= p and p + self._degree < self._period)
                else np.fromiter(
                    (
                        self._spline_coeff[q % self._period]
                        for q in range(p, p + self._degree + 1)
                    ),
                    dtype = float,
                    count = self._degree + 1
                )
            ) @ _w(self._degree)
            for q in range(1, self._degree + 1):
                a0[0] += min(0.0, a0[q])
            return a0[0]
        if 0 == degree:
            return PeriodicSpline1D.from_spline_coeff(
                cast(
                    np.ndarray[tuple[int], np.dtype[np.float64]],
                    np.fromiter(
                        (low(p) for p in range(self._period)),
                        dtype = float,
                        count = self._period
                    )
                ),
                degree = 0,
                delay = self._delay + 0.5 *self._degree
            )
        g = cast(
            np.ndarray[tuple[int], np.dtype[np.float64]],
            np.full((self._period), np.inf)
        )
        if self._degree <= degree:
            for p in range(self._period):
                a0 = np.append(
                    (
                        self._spline_coeff[p : p + self._degree + 1]
                        if (0 <= p and p + self._degree < self._period)
                        else np.fromiter(
                            (
                                self._spline_coeff[q % self._period]
                                for q in range(p, p + self._degree + 1)
                            ),
                            dtype = float,
                            count = self._degree + 1
                        )
                    ) @ _w(self._degree),
                    np.zeros(degree - self._degree, dtype = float)
                ) @ _inv_w(degree)
                for (q, a0q) in enumerate(a0):
                    k = (p + q) % self._period
                    g[k] = min(g[k], a0q)
        else:
            for p in range(self._period):
                a0 = (
                    self._spline_coeff[p : p + self._degree + 1]
                    if (0 <= p and p + self._degree < self._period)
                    else np.fromiter(
                        (
                            self._spline_coeff[q % self._period]
                            for q in range(p, p + self._degree + 1)
                        ),
                        dtype = float,
                        count = self._degree + 1
                    )
                ) @ _w(self._degree)
                for q in range(degree + 1, self._degree + 1):
                    a0[degree] += min(0.0, a0[q])
                a0 = a0[0 : degree + 1] @ _inv_w(degree)
                for (q, a0q) in enumerate(a0):
                    k = (p + q) % self._period
                    g[k] = min(g[k], a0q)
        return PeriodicSpline1D.from_spline_coeff(
            g,
            degree = degree,
            delay = self._delay + 0.5 * (self._degree - degree)
        )

    #---------------
    def upper_bound (
        self,
        *,
        degree: int
    ) -> PeriodicSpline1D:

        r"""
        .. _periodic_spline_1d-upper_bound:

        A spline that is never smaller than this ``PeriodicSpline1D`` object.

        Assume that this spline :math:`s` is of period :math:`K` and degree
        :math:`m.` Then, this function returns another spline :math:`f` of
        same period :math:`K` and arbitrary :ref:`nonnegative<def-negative>`
        degree :math:`n` such that :math:`f(x)\geq g(x)` for all
        :math:`x\in{\mathbb{R}}.` The delay of :math:`f` is adjusted so that
        the knots of :math:`f` coincide with those of :math:`s.`

        Parameters
        ----------
        degree : int
            The :ref:`nonnegative<def-negative>` degree of the spline that
            upper-bounds this spline.

        Returns
        -------
        PeriodicSpline1D
            The upper-bounding spline.

        Examples
        --------
        Load the libraries.
            >>> import numpy as np
            >>> import splinekit as sk
        Create a spline and get an upper-bounding spline of lower degree.
            >>> c = np.array([1, 9, -7], dtype = float)
            >>> s = sk.PeriodicSpline1D.from_spline_coeff(c, degree = 3)
            >>> s.upper_bound(degree = 1)
            PeriodicSpline1D([9. 9. 5.], degree = 1, delay = 1.0)
        Here is an upper-bounding spline of higher degree.
            >>> s.upper_bound(degree = 4)
            PeriodicSpline1D([-5. 31.  7.], degree = 4, delay = -0.5)

        Notes
        -----
        The bound relies on two properties. One is that
        :math:`x^{n}\geq x^{n+1}` for :math:`x\in(0,1)` and
        :math:`n\in{\mathbb{N}}.` The other is that
        :ref:`B-splines<def-b_spline>` are nonnegative. The bound is
        reasonably sharp and fast to compute.

        Raises
        ------
        ValueError
            Raised when ``degree`` is :ref:`negative<def-negative>`.


        ----

        """

        if 0 > degree:
            raise ValueError("Degree must be nonnegative")
        if degree == self._degree:
            self.copy()
        def up (
            p: int
        ) -> float:
            a0 = (
                self._spline_coeff[p : p + self._degree + 1]
                if (0 <= p and p + self._degree < self._period)
                else np.fromiter(
                    (
                        self._spline_coeff[q % self._period]
                        for q in range(p, p + self._degree + 1)
                    ),
                    dtype = float,
                    count = self._degree + 1
                )
            ) @ _w(self._degree)
            for q in range(1, self._degree + 1):
                a0[0] += max(0.0, a0[q])
            return a0[0]
        if 0 == degree:
            return PeriodicSpline1D.from_spline_coeff(
                cast(
                    np.ndarray[tuple[int], np.dtype[np.float64]],
                    np.fromiter(
                        (up(p) for p in range(self._period)),
                        dtype = float,
                        count = self._period
                    )
                ),
                degree = 0,
                delay = self._delay + 0.5 *self._degree
            )
        g = cast(
            np.ndarray[tuple[int], np.dtype[np.float64]],
            np.full((self._period), -np.inf)
        )
        if self._degree <= degree:
            for p in range(self._period):
                a0 = np.append(
                        (
                            self._spline_coeff[p : p + self._degree + 1]
                            if (0 <= p and p + self._degree < self._period)
                            else np.fromiter(
                                (
                                    self._spline_coeff[q % self._period]
                                    for q in range(p, p + self._degree + 1)
                                ),
                                dtype = float,
                                count = self._degree + 1
                            )
                        ) @ _w(self._degree),
                        np.zeros(degree - self._degree, dtype = float)
                    ) @ _inv_w(degree)
                for (q, a0q) in enumerate(a0):
                    k = (p + q) % self._period
                    g[k] = max(g[k], a0q)
        else:
            for p in range(self._period):
                a0 = (
                    self._spline_coeff[p : p + self._degree + 1]
                    if (0 <= p and p + self._degree < self._period)
                    else np.fromiter(
                        (
                            self._spline_coeff[q % self._period]
                            for q in range(p, p + self._degree + 1)
                        ),
                        dtype = float,
                        count = self._degree + 1
                    )
                ) @ _w(self._degree)
                for q in range(degree + 1, self._degree + 1):
                    a0[degree] += max(0.0, a0[q])
                a0 = a0[0 : degree + 1] @ _inv_w(degree)
                for (q, a0q) in enumerate(a0):
                    k = (p + q) % self._period
                    g[k] = max(g[k], a0q)
        return PeriodicSpline1D.from_spline_coeff(
            g,
            degree = degree,
            delay = self._delay + 0.5 * (self._degree - degree)
        )

    #---------------
    def get_knots (
        self,
        domain: Interval = RR()
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:

        r"""
        .. _periodic_spline_1d-get_knots:

        Potential location of the knots of this spline over a domain.

        This function returns a ``numpy`` array of
        :ref:`uniform<def-uniform_split>` abscissa that belong to a given
        :ref:`interval<Interval>` and that are a proper subset of the
        :ref:`uniform split<def-uniform_split>`
        :math:`{\mathbb{F}}({\mathbf{S}})` of this
        :ref:`piecewise-polynomial<def-piecewise_polynomial>` spline.

        *   When ``domain`` is the empty interval
            :ref:`splinekit.interval.Empty<Empty>` notated
            :math:`\emptyset,` the returned array contains no element. The
            same happens when no element of :math:`{\mathbb{F}}({\mathbf{S}})`
            belongs to ``domain``.
        *   When ``domain`` is the real line 
            :ref:`splinekit.interval.RR<RR>` notated :math:`{\mathbb{R}},` the
            returned array contains only the splits that belong to
            :math:`[0,K),` where :math:`K` is the period of this spline.
        *   If ``domain`` is characterized by a threshold, such as
            intervals of the class :ref:`splinekit.interval.Above<Above>`,
            :ref:`splinekit.interval.NotBelow<NotBelow>`,
            :ref:`splinekit.interval.NotAbove<NotAbove>`, or
            :ref:`splinekit.interval.Below<Below>`, then the returned array
            contains only the splits that belong to an interval of
            :ref:`diameter<diameter>` :math:`K` for which the threshold is one
            of the bounds.
        *   The returned array will honor a ``domain`` of the class
            :ref:`splinekit.interval.Singleton<Singleton>`,
            :ref:`splinekit.interval.Open<Open>`,
            :ref:`splinekit.interval.OpenClosed<OpenClosed>`,
            :ref:`splinekit.interval.Closed<Closed>`, and
            :ref:`splinekit.interval.ClosedOpen<ClosedOpen>`.

        Parameters
        ----------
        domain : Interval
            Domain over which the returned splits belong.

        Returns
        -------
        np.ndarray[tuple[int], np.dtype[np.float64]]
            The array of splits between the pieces of this spline.

        Examples
        --------
        Load the libraries.
            >>> import numpy as np
            >>> import splinekit as sk
        Create an arbitrary cubic spline and find the knots over the main period.
            >>> c = np.array([1, 9, -7], dtype = float)
            >>> s = sk.PeriodicSpline1D.from_spline_coeff(c, degree = 3, delay = 8.4)
            >>> s.get_knots(sk.interval.RR())
            array([0.4, 1.4, 2.4])

        Notes
        -----
        The true :ref:`knots<def-knots>` of this spline are a subset of the
        returned array. For instance, a constant-valued spline is infinitely
        differentiable and thus has no knots. Yet, the returned array is still
        populated with the splits between the spline pieces.

        ----

        """

        (_, x0) = _divmod(0.5 * (self._degree + 1) + self._delay, 1)
        if isinstance(domain, Empty):
            return cast(
                np.ndarray[tuple[int], np.dtype[np.float64]],
                np.array([], dtype = float)
            )
        if isinstance(domain, RR):
            return cast(
                np.ndarray[tuple[int], np.dtype[np.float64]],
                np.linspace(x0, x0 + self._period - 1, num = self._period)
            )
        if isinstance(domain, Singleton):
            (_, x) = _divmod(domain.value, 1)
            return cast(
                np.ndarray[tuple[int], np.dtype[np.float64]],
                np.array([] if x != x0 else [domain.value], dtype = float)
            )
        if isinstance(domain, Above):
            (k, x) = _divmod(domain.threshold, 1)
            if x0 <= x + ulp(k):
                k += 1
            return cast(
                np.ndarray[tuple[int], np.dtype[np.float64]],
                np.linspace(
                    x0 + k,
                    x0 + k + self._period - 1,
                    num = self._period
                )
            )
        if isinstance(domain, NotBelow):
            (k, x) = _divmod(domain.threshold, 1)
            if x0 + ulp(k) < x:
                k += 1
            return cast(
                np.ndarray[tuple[int], np.dtype[np.float64]],
                np.linspace(
                    x0 + k,
                    x0 + k + self._period - 1,
                    num = self._period
                )
            )
        if isinstance(domain, NotAbove):
            (k, x) = _divmod(domain.threshold, 1)
            if x + ulp(k) < x0:
                k -= 1
            return cast(
                np.ndarray[tuple[int], np.dtype[np.float64]],
                np.linspace(
                    x0 + k - self._period + 1,
                    x0 + k,
                    num = self._period
                )
            )
        if isinstance(domain, Below):
            (k, x) = _divmod(domain.threshold, 1)
            if x <= x0 + ulp(k):
                k -= 1
            return cast(
                np.ndarray[tuple[int], np.dtype[np.float64]],
                np.linspace(
                    x0 + k - self._period + 1,
                    x0 + k,
                    num = self._period
                )
            )
        if isinstance(domain, Open):
            (k1, x1) = divmod(domain.infimum, 1)
            if x0 <= x1 + ulp(k1):
                k1 += 1
            (k2, x2) = divmod(domain.supremum, 1)
            if x2 <= x0 + ulp(k2):
                k2 -= 1
            if k1 <= k2:
                return cast(
                    np.ndarray[tuple[int], np.dtype[np.float64]],
                    np.linspace(
                        x0 + k1,
                        x0 + k2,
                        num = 1 + int(k2 - k1)
                    )
                )
            return cast(
                np.ndarray[tuple[int], np.dtype[np.float64]],
                np.array([], dtype = float)
            )
        if isinstance(domain, OpenClosed):
            (k1, x1) = divmod(domain.infimum, 1)
            if x0 <= x1 + ulp(k1):
                k1 += 1
            (k2, x2) = divmod(domain.supremum, 1)
            if x2 + ulp(k2) < x0:
                k2 -= 1
            if k1 <= k2:
                return cast(
                    np.ndarray[tuple[int], np.dtype[np.float64]],
                    np.linspace(
                        x0 + k1,
                        x0 + k2,
                        num = 1 + int(k2 - k1)
                    )
                )
            return cast(
                np.ndarray[tuple[int], np.dtype[np.float64]],
                np.array([], dtype = float)
            )
        if isinstance(domain, Closed):
            (k1, x1) = divmod(domain.infimum, 1)
            if x0 + ulp(k1) < x1:
                k1 += 1
            (k2, x2) = divmod(domain.supremum, 1)
            if x2 + ulp(k2) < x0:
                k2 -= 1
            if k1 <= k2:
                return cast(
                    np.ndarray[tuple[int], np.dtype[np.float64]],
                    np.linspace(
                        x0 + k1,
                        x0 + k2,
                        num = 1 + int(k2 - k1)
                    )
                )
            return cast(
                np.ndarray[tuple[int], np.dtype[np.float64]],
                np.array([], dtype = float)
            )
        if isinstance(domain, ClosedOpen):
            (k1, x1) = divmod(domain.infimum, 1)
            if x0 + ulp(k1) < x1:
                k1 += 1
            (k2, x2) = divmod(domain.supremum, 1)
            if x2 <= x0 + ulp(k2):
                k2 -= 1
            if k1 <= k2:
                return cast(
                    np.ndarray[tuple[int], np.dtype[np.float64]],
                    np.linspace(
                        x0 + k1,
                        x0 + k2,
                        num = 1 + int(k2 - k1)
                    )
                )
            return cast(
                np.ndarray[tuple[int], np.dtype[np.float64]],
                np.array([], dtype = float)
            )
        raise ValueError("Internal error (unexpected interval class)")

    #---------------
    def zeros (
        self
    ) -> List[Interval]:

        # TODO: docstring
        # TODO: tests
        r"""
        .. _periodic_spline_1d-zeros:


        ----

        """

        z = [
            s.domain
            for s in self.piecewise_signs().pieces
            if 0 == s.item
        ]
        if len(z) < 2:
            return z
        if isinstance(z[-1], ClosedOpen):
            if isinstance(z[0], Singleton):
                if self._period == z[-1].supremum and 0.0 == z[0].value:
                    z[0] = Closed((
                        z[-1].infimum - self._period,
                        0.0
                    ))
                    del z[-1]
            if isinstance(z[0], ClosedOpen):
                if self._period == z[-1].supremum and 0.0 == z[0].infimum:
                    z[0] = ClosedOpen((
                        z[-1].infimum - self._period,
                        z[0].supremum
                    ))
                    del z[-1]
            if isinstance(z[0], Closed):
                if self._period == z[-1].supremum and 0.0 == z[0].infimum:
                    z[0] = Closed((
                        z[-1].infimum - self._period,
                        z[0].supremum
                    ))
                    del z[-1]
        if isinstance(z[-1], Open):
            if isinstance(z[0], Singleton):
                if self._period == z[-1].supremum and 0.0 == z[0].value:
                    z[0] = OpenClosed((
                        z[-1].infimum - self._period,
                        0.0
                    ))
                    del z[-1]
            if isinstance(z[0], ClosedOpen):
                if self._period == z[-1].supremum and 0.0 == z[0].infimum:
                    z[0] = Open((
                        z[-1].infimum - self._period,
                        z[0].supremum
                    ))
                    del z[-1]
            if isinstance(z[0], Closed):
                if self._period == z[-1].supremum and 0.0 == z[0].infimum:
                    z[0] = OpenClosed((
                        z[-1].infimum - self._period,
                        z[0].supremum
                    ))
                    del z[-1]
        return z

    #---------------
    def zero_crossings (
        self
    ) -> List[List[Interval]]:

        r"""
        .. _periodic_spline_1d-zero_crossings:


        Zeros of odd multiplicity of this spline.

        This function returns a list of two items.

        *   The first item is a list of :ref:`intervals<Interval>`. This
            spline is :ref:`positive<def-positive>` to the left of each
            interval in this first list, while it is
            :ref:`negative<def-negative>` to the right. The intervals of this
            list thus correspond to the descending zero crossings of this
            spline.
        *   The second item is another list of :ref:`intervals<Interval>`.
            This spline is :ref:`negative<def-negative>` to the left of each
            interval in this second list, while it is
            :ref:`positive<def-positive>` to the right. The intervals of this
            list thus correspond to the ascending zero crossings of this
            spline.
        *   The class of every interval will be either one of
            :ref:`splinekit.interval.Singleton<Singleton>` or
            :ref:`splinekit.interval.Closed<Closed>`.
        *   The :ref:`diameter<diameter>` of the :ref:`enclosure<enclosure>`
            of all returned intervals is smaller than the period of this
            spline.

        Parameters
        ----------
        None.

        Returns
        -------
        list of list of Interval
            The intervals where this spline has zeros of odd multiplicity.

        Examples
        --------
        Load the libraries.
            >>> import numpy as np
            >>> import splinekit as sk
        Create an arbitrary cubic spline and find its zero-crossings.
            >>> c = np.array([1, 9, -7], dtype = float)
            >>> s = sk.PeriodicSpline1D.from_spline_coeff(c, degree = 3)
            >>> s.zero_crossings()
            [[Singleton(1.6008198378617022)], [Singleton(-0.12600019258625705)]]

        ----

        """

        sgn = [s for s in self.piecewise_signs().pieces if 0.0 != s.item]
        if len(sgn) < 2:
            return [[], []]
        s1 = sgn[-1]
        k = len(sgn) - 2
        while 0 <= k:
            s0 = sgn[k]
            if 0.0 < s0.item * s1.item:
                if not isinstance(
                    s1.domain,
                    (Open, OpenClosed, Closed, ClosedOpen)
                ):
                    raise ValueError(
                        "Internal error (unexpected interval subclass)"
                    )
                if isinstance(s0.domain, Singleton):
                    s0 = PeriodicNonuniformPiecewise.Piece(
                        Open((
                            s0.domain.value,
                            s1.domain.supremum
                        )),
                        s1.item
                    )
                elif isinstance(
                    s0.domain,
                    (Open, OpenClosed, Closed, ClosedOpen)
                ):
                    s0 = PeriodicNonuniformPiecewise.Piece(
                        Open((
                            s0.domain.infimum,
                            s1.domain.supremum
                        )),
                        s1.item
                    )
                else:
                    raise ValueError(
                        "Internal error (unexpected interval subclass)"
                    )
                sgn[k] = s0
                del sgn[k + 1]
            s1 = s0
            k -= 1
        if len(sgn) < 2:
            return [[], []]
        s1 = sgn[0]
        s0 = sgn[-1]
        if 0.0 < s0.item * s1.item:
            if not isinstance(
                s0.domain,
                (Open, OpenClosed, Closed, ClosedOpen)
            ):
                raise ValueError(
                    "Internal error (unexpected interval subclass)"
                )
            if isinstance(s1.domain, Singleton):
                s0 = PeriodicNonuniformPiecewise.Piece(
                    Open((
                        s0.domain.infimum - self._period,
                        s1.domain.value
                    )),
                    s1.item
                )
            elif isinstance(
                s1.domain,
                (Open, OpenClosed, Closed, ClosedOpen)
            ):
                s0 = PeriodicNonuniformPiecewise.Piece(
                    Open((
                        cast(Open, s0.domain).infimum - self._period,
                        s1.domain.supremum
                    )),
                    s1.item
                )
            else:
                raise ValueError(
                    "Internal error (unexpected interval subclass)"
                )
            sgn[0] = s0
            del sgn[-1]
        if len(sgn) < 2:
            return [[], []]
        s0 = sgn[-1]
        s1 = sgn[0]
        zc: List[List[Interval]] = [[], []]
        if (not isinstance(
            s0.domain,
            (Open, OpenClosed, Closed, ClosedOpen)
        )) or (not isinstance(
            s1.domain,
            (Open, OpenClosed, Closed, ClosedOpen)
        )):
            raise ValueError(
                "Internal error (unexpected interval subclass)"
            )
        if s0.domain.supremum - self._period == s1.domain.infimum:
            if s1.item < s0.item:
                zc[0].append(Singleton(s1.domain.infimum))
            else:
                zc[1].append(Singleton(s1.domain.infimum))
        else:
            if s1.item < s0.item:
                zc[0].append(Closed((
                    s0.domain.supremum - self._period,
                    s1.domain.infimum
                )))
            else:
                zc[1].append(Closed((
                    s0.domain.supremum - self._period,
                    s1.domain.infimum
                )))
        for k in range(1, len(sgn)):
            s0 = s1
            s1 = sgn[k]
            if (not isinstance(
                s0.domain,
                (Open, OpenClosed, Closed, ClosedOpen)
            )) or (not isinstance(
                s1.domain,
                (Open, OpenClosed, Closed, ClosedOpen)
            )):
                raise ValueError(
                    "Internal error (unexpected interval subclass)"
                )
            if s0.domain.supremum == s1.domain.infimum:
                if s1.item < s0.item:
                    zc[0].append(Singleton(s1.domain.infimum))
                else:
                    zc[1].append(Singleton(s1.domain.infimum))
            else:
                if s1.item < s0.item:
                    zc[0].append(Closed((
                        s0.domain.supremum,
                        s1.domain.infimum
                    )))
                else:
                    zc[1].append(Closed((
                        s0.domain.supremum,
                        s1.domain.infimum
                    )))
        zc[0].sort()
        zc[1].sort()
        return zc

    #---------------
    class Extremum (
        NamedTuple
    ):

        """
        .. _periodic_spline_1d-Extremum:

        Each ``Extremum`` object is a named tuple made of two fields. The
        first field is named ``domain`` and contains an
        :ref:`Interval<Interval>` object that describes an arbitrary domain
        (often, a :ref:`Singleton<Singleton>` object) over which the
        extremum extends. The second field is named ``value`` and contains the
        ``float`` value taken by the extremum.

        ====

        """

        domain: Interval

        r"""
        .. _periodic_spline_1d-Extremum-domain:

        the domain of this extremum.

        ----

        """

        value: float

        r"""
        .. _periodic_spline_1d-Extremum-value:

        The value of this extremum.

        ----

        """

    #---------------
    def extrema (
        self
    ) -> List[List[Extremum]]:

        # TODO: docstring
        # TODO: tests
        r"""
        .. _periodic_spline_1d-extrema:


        ----

        """

        if 0 < self._degree:
            zc = self.gradient().zero_crossings()
            return [
                [
                    PeriodicSpline1D.Extremum(
                        domain = ex,
                        value = self.at(ex.midpoint)
                    )
                    for ex in zc[1]
                ],
                [
                    PeriodicSpline1D.Extremum(
                        domain = ex,
                        value = self.at(ex.midpoint)
                    )
                    for ex in zc[0]
                ]
            ]
        zc = PeriodicSpline1D.from_spline_coeff(
            cast(
                np.ndarray[tuple[int], np.dtype[np.float64]],
                np.fromiter(
                    (
                        ck - self._spline_coeff[k - 1]
                        for (k, ck) in enumerate(self._spline_coeff)
                    ),
                    dtype = float,
                    count = self._period
                )
            ),
            degree = 0,
            delay = self._delay - 0.5
        ).zero_crossings()
        return [
            [
                PeriodicSpline1D.Extremum(
                    domain = Open((ex.value - 0.5, ex.value + 0.5))
                        if isinstance(ex, Singleton)
                        else Open((
                            cast(Closed, ex).infimum - 0.5,
                            cast(Closed, ex).supremum + 0.5
                        )),
                    value = self.at(ex.midpoint)
                )
                for ex in zc[1]
            ],
            [
                PeriodicSpline1D.Extremum(
                    domain = Open((ex.value - 0.5, ex.value + 0.5))
                        if isinstance(ex, Singleton)
                        else Open((
                            cast(Closed, ex).infimum - 0.5,
                            cast(Closed, ex).supremum + 0.5
                        )),
                    value = self.at(ex.midpoint)
                )
                for ex in zc[0]
            ]
        ]

    #---------------
    def plus (
        self,
        constant: float
    ) -> PeriodicSpline1D:

        r"""
        .. _periodic_spline_1d-plus:

        Add a constant to this spline.

        Letting this spline be
        :math:`s:{\mathbb{R}}\rightarrow{\mathbb{R}},x\mapsto s(x)` and
        letting ``constant`` be :math:`s_{0},` return the spline
        :math:`f:{\mathbb{R}}\rightarrow{\mathbb{R}},x\mapsto s(x)+s_{0}.`

        Parameters
        ----------
        constant : float
            The constant added to this spline.

        Returns
        -------
        PeriodicSpline1D
            This spline with the constant added.

        Examples
        --------
        Load the libraries.
            >>> import numpy as np
            >>> import splinekit as sk
        Create a cubic spline add some number.
            >>> c = np.array([1, 9, -7], dtype = float)
            >>> s = sk.PeriodicSpline1D.from_spline_coeff(c, degree = 3)
            >>> s.plus(5.0)
            PeriodicSpline1D([ 6. 14. -2.], degree = 3, delay = 0.0)

        ----

        """

        return PeriodicSpline1D.from_spline_coeff(
            cast(
                np.ndarray[tuple[int], np.dtype[np.float64]],
                np.add(constant, self._spline_coeff)
            ),
            degree = self._degree,
            delay = self._delay
        )

    #---------------
    def times (
        self,
        constant: float
    ) -> PeriodicSpline1D:

        r"""
        .. _periodic_spline_1d-times:

        Multiply this spline by a constant.

        Letting this spline be
        :math:`s:{\mathbb{R}}\rightarrow{\mathbb{R}},x\mapsto s(x)` and
        letting ``constant`` be :math:`\lambda_{0},` return the spline
        :math:`f:{\mathbb{R}}\rightarrow{\mathbb{R}},x\mapsto\lambda_{0}\,s(x).`

        Parameters
        ----------
        constant : float
            The constant that multiplies this spline.

        Returns
        -------
        PeriodicSpline1D
            This spline multiplied by the constant.

        Examples
        --------
        Load the libraries.
            >>> import numpy as np
            >>> import splinekit as sk
        Create a cubic spline and multiply it by some number.
            >>> c = np.array([1, 9, -7], dtype = float)
            >>> s = sk.PeriodicSpline1D.from_spline_coeff(c, degree = 3)
            >>> s.times(5.0)
            PeriodicSpline1D([  5.  45. -35.], degree = 3, delay = 0.0)

        ----

        """

        return PeriodicSpline1D.from_spline_coeff(
            cast(
                np.ndarray[tuple[int], np.dtype[np.float64]],
                np.multiply(constant, self._spline_coeff)
            ),
            degree = self._degree,
            delay = self._delay
        )

    #---------------
    def negated (
        self
    ) -> PeriodicSpline1D:

        r"""
        .. _periodic_spline_1d-negated:

        Multiply this spline by the constant ``-1.0``.

        Letting this spline be
        :math:`s:{\mathbb{R}}\rightarrow{\mathbb{R}},x\mapsto s(x),` return
        the spline
        :math:`f:{\mathbb{R}}\rightarrow{\mathbb{R}},x\mapsto -s(x).`

        Parameters
        ----------
        None

        Returns
        -------
        PeriodicSpline1D
            This spline multiplied by ``-1.0``.

        Examples
        --------
        Load the libraries.
            >>> import numpy as np
            >>> import splinekit as sk
        Create a cubic spline and flip its sign.
            >>> c = np.array([1, 9, -7], dtype = float)
            >>> s = sk.PeriodicSpline1D.from_spline_coeff(c, degree = 3)
            >>> s.negated()
            PeriodicSpline1D([-1. -9.  7.], degree = 3, delay = 0.0)

        ----

        """

        return PeriodicSpline1D.from_spline_coeff(
            cast(
                np.ndarray[tuple[int], np.dtype[np.float64]],
                np.negative(self._spline_coeff)
            ),
            degree = self._degree,
            delay = self._delay
        )

    #---------------
    def mirrored (
        self
    ) -> PeriodicSpline1D:

        r"""
        .. _periodic_spline_1d-mirrored:

        Reverse this spline around the origin.

        Letting this spline be
        :math:`s:{\mathbb{R}}\rightarrow{\mathbb{R}},x\mapsto s(x),` return
        the spline
        :math:`f:{\mathbb{R}}\rightarrow{\mathbb{R}},x\mapsto s(-x).`

        Parameters
        ----------
        None

        Returns
        -------
        PeriodicSpline1D
            This spline after mirroring around the origin.

        Examples
        --------
        Load the libraries.
            >>> import numpy as np
            >>> import splinekit as sk
        Create a cubic spline and mirror it.
            >>> c = np.array([1, 9, -7], dtype = float)
            >>> s = sk.PeriodicSpline1D.from_spline_coeff(c, degree = 3, delay = 0.71)
            >>> s.mirrored()
            PeriodicSpline1D([ 1. -7.  9.], degree = 3, delay = -0.71)

        ----

        """

        if 1 == self._period:
            return PeriodicSpline1D.from_spline_coeff(
                cast(
                    np.ndarray[tuple[int], np.dtype[np.float64]],
                    self._spline_coeff,
                ),
                degree = self._degree,
                delay = -self._delay
            )
        return PeriodicSpline1D.from_spline_coeff(
            cast(
                np.ndarray[tuple[int], np.dtype[np.float64]],
                np.concatenate((
                    [self._spline_coeff[0]],
                    np.flip(self._spline_coeff[1 : ])
                ))
            ),
            degree = self._degree,
            delay = -self._delay
        )

    #---------------
    def fractionalized_delay (
        self
    ) -> PeriodicSpline1D:

        r"""
        .. _periodic_spline_1d-fractionalized_delay:

        Reformat this spline to have a fractional delay.

        Let this spline be
        :math:`s:{\mathbb{R}}\rightarrow{\mathbb{R}},x\mapsto s(x),` as
        characterized by its period :math:`K,` its degree :math:`n,` its delay
        :math:`\delta x,` and its spline coefficients :math:`c.` The returned
        spline :math:`f` will be functionally indistinguishable from :math:`s,`
        with :math:`f:{\mathbb{R}}\rightarrow{\mathbb{R}},x\mapsto s(x).`
        However, the spline coefficients of :math:`f` will be modified so that
        the delay of :math:`f` will be :ref:`nonnegative<def-negative>` and
        smaller than :math:`1.`

        Parameters
        ----------
        None

        Returns
        -------
        PeriodicSpline1D
            This spline with a delay in :math:`[0,1).`

        Examples
        --------
        Load the libraries.
            >>> import numpy as np
            >>> import splinekit as sk
        Create a cubic spline and fractionalize its delay.
            >>> c = np.array([1, 9, -7], dtype = float)
            >>> s = sk.PeriodicSpline1D.from_spline_coeff(c, degree = 3, delay = -3.71)
            >>> s.fractionalized_delay()
            PeriodicSpline1D([ 9. -7.  1.], degree = 3, delay = 0.29000000000000004)

        ----

        """

        (k, d) = _divmod(self._delay, 1)
        k = (-k) % self._period
        return PeriodicSpline1D.from_spline_coeff(
            cast(
                np.ndarray[tuple[int], np.dtype[np.float64]],
                np.roll(self._spline_coeff, -k)
            ),
            degree = self._degree,
            delay = d
        )

    #---------------
    def delayed_by (
        self,
        dx: float
    ) -> PeriodicSpline1D:

        r"""
        .. _periodic_spline_1d-delayed_by:

        Delay this spline.

        Letting this spline be
        :math:`s:{\mathbb{R}}\rightarrow{\mathbb{R}},x\mapsto s(x),` return
        the spline
        :math:`f:{\mathbb{R}}\rightarrow{\mathbb{R}},x\mapsto s(x-\Delta x),`
        where the delay :math:`\Delta x` is ``dx``.

        Parameters
        ----------
        dx : float
            The additional delay applied to this spline.

        Returns
        -------
        PeriodicSpline1D
            This delayed spline.

        Examples
        --------
        Load the libraries.
            >>> import numpy as np
            >>> import splinekit as sk
        Create a cubic spline with an arbitrary delay, and delay it further.
            >>> c = np.array([1, 9, -7], dtype = float)
            >>> s = sk.PeriodicSpline1D.from_spline_coeff(c, degree = 3, delay = 0.71)
            >>> s.delayed_by(5.2)
            PeriodicSpline1D([ 1.  9. -7.], degree = 3, delay = 5.91)

        ----

        """

        return PeriodicSpline1D.from_spline_coeff(
            cast(
                np.ndarray[tuple[int], np.dtype[np.float64]],
                self._spline_coeff,
            ),
            degree = self._degree,
            delay = self._delay + dx
        )

    #---------------
    def differentiated (
        self,
        order: int
    ) -> PeriodicSpline1D:

        r"""
        .. _periodic_spline_1d-differentiated:

        This spline differentiated order times.

        Letting this spline be :math:`s` and letting :math:`m` be the
        ``order`` of differentiation, return the spline :math:`f` such that

        ..  math::

            f(x)=\frac{{\mathrm{d}}^{m}s(x)}{{\mathrm{d}}x^{m}}.

        Parameters
        ----------
        order : int
            The :ref:`nonnegative<def-negative>` order of differentiation.

        Returns
        -------
        PeriodicSpline1D
            The order-th derivative of this spline.

        Examples
        --------
        Load the libraries.
            >>> import numpy as np
            >>> import splinekit as sk
        Create a cubic spline and compute its third-order derivative.
            >>> c = np.array([1, 9, -7], dtype = float)
            >>> s = sk.PeriodicSpline1D.from_spline_coeff(c, degree = 3)
            >>> s.differentiated(3)
            PeriodicSpline1D([ 48. -24. -24.], degree = 0, delay = -1.5)

        Raises
        ------
        ValueError
            Raised when ``order`` is :ref:`negative<def-negative>`.
        ValueError
            Raised when ``order`` exceeds the degree of this spline.


        ----

        """

        if 0 > order:
            raise ValueError("Order must be nonnegative")
        if self._degree < order:
            raise ValueError(
                "The differentiation order must not exceed the spline degree"
            )
        if 0 == order:
            return self.copy()
        h = np.copy(mscale_filter(degree = order - 1, scale = 2))
        for (q, hq) in enumerate(h):
            h[q] = ((-1.0) ** q) * hq
        return PeriodicSpline1D.from_spline_coeff(
            cast(
                np.ndarray[tuple[int], np.dtype[np.float64]],
                np.fromiter(
                    (
                        fsum(
                            hq * self._spline_coeff[(k - q) % self._period]
                            for (q, hq) in enumerate(h)
                        )
                        for k in range(self._period)
                    ),
                    dtype = float,
                    count = self._period
                )
            ),
            degree = self._degree - order,
            delay = self._delay - 0.5 * order
        )

    #---------------
    def gradient (
        self
    ) -> PeriodicSpline1D:

        r"""
        .. _periodic_spline_1d-gradient:

        The gradient of this spline.

        Letting this spline be :math:`s,` return the spline :math:`f` such
        that

        ..  math::

            f(x)=\frac{{\mathrm{d}}s(x)}{{\mathrm{d}}x}.

        Parameters
        ----------
        None

        Returns
        -------
        PeriodicSpline1D
            The gradient of this spline.

        Examples
        --------
        Load the libraries.
            >>> import numpy as np
            >>> import splinekit as sk
        Create an undelayed cubic spline and compute its gradient.
            >>> c = np.array([1, 9, -7], dtype = float)
            >>> s = sk.PeriodicSpline1D.from_spline_coeff(c, degree = 3)
            >>> s.gradient()
            PeriodicSpline1D([  8.   8. -16.], degree = 2, delay = -0.5)

        Raises
        ------
        ValueError
            Raised when the degree of this spline is
            :ref:`nonpositive<def-positive>`.


        ----

        """

        if 1 > self._degree:
            raise ValueError("Degree must be positive")
        return PeriodicSpline1D.from_spline_coeff(
            cast(
                np.ndarray[tuple[int], np.dtype[np.float64]],
                np.fromiter(
                    (
                        ck - self._spline_coeff[k - 1]
                        for (k, ck) in enumerate(self._spline_coeff)
                    ),
                    dtype = float,
                    count = self._period
                )
            ),
            degree = self._degree - 1,
            delay = self._delay - 0.5
        )

    #---------------
    def anti_grad (
        self
    ) -> PeriodicSpline1D:

        r"""
        .. _periodic_spline_1d-anti_grad:

        The spline whose gradient is this spline, up to a constant.

        Letting this spline be :math:`s,` return the spline :math:`f` such
        that

        ..  math::

            \frac{{\mathrm{d}}f(x)}{{\mathrm{d}}x}=s(x)-E\{s\},

        where :math:`E\{s\}` is the :ref:`mean<periodic_spline_1d-mean>` of
        :math:`s.`

        Parameters
        ----------
        None

        Returns
        -------
        PeriodicSpline1D
            The anti-gradient of this spline.

        Examples
        --------
        Load the libraries.
            >>> import numpy as np
            >>> import splinekit as sk
        Create an undelayed cubic spline and compute its anti-gradient.
            >>> c = np.array([1, 9, -7], dtype = float)
            >>> s = sk.PeriodicSpline1D.from_spline_coeff(c, degree = 3)
            >>> s.anti_grad()
            PeriodicSpline1D([0. 8. 0.], degree = 4, delay = 0.5)

        ----

        """

        return PeriodicSpline1D.from_spline_coeff(
            cast(
                np.ndarray[tuple[int], np.dtype[np.float64]],
                np.cumsum(self.plus(-self.mean()).spline_coeff)
            ),
            degree = self._degree + 1,
            delay = self._delay + 0.5
        )

    #---------------
    def integrate (
        self,
        lower_bound: float,
        upper_bound: float
    ) -> float:

        r"""
        .. _periodic_spline_1d-integrate:

        The integration of this ``PeriodicSpline1D`` object between two bounds.

        This function returns the value :math:`F(x_{0},x_{1})` of the integral
        of this spline :math:`f` between a lower bound :math:`x_{0}` and an
        upper bound :math:`x_{1},` as defined by

        ..  math::

            F(x_{0},x_{1})=\int_{x_{0}}^{x_{1}}\,f(x)\,{\mathrm{d}}x.

        Parameters
        ----------
        lower_bound : float
            Lower bound of the integral.
        upper_bound : float
            Upper bound of the integral.

        Returns
        -------
        float
            The integral of this spline.

        Examples
        --------
        Load the libraries.
            >>> import numpy as np
            >>> import splinekit as sk
        Create a spline and compute its integral between two arbitrary bounds.
            >>> c = np.array([1, 9, -7], dtype = float)
            >>> s = sk.PeriodicSpline1D.from_spline_coeff(c, degree = 3)
            >>> s.integrate(lower_bound = -18.5, upper_bound = 7.3)
            28.7907

        ----

        """

        p0 = ceil(upper_bound - self._delay - 0.5 * (self._degree + 1))
        q0 = ceil(lower_bound - self._delay - 0.5 * (self._degree + 1))
        p = (p0 - 1) % self._period
        q = (q0 - 1) % self._period
        m = self.mean()
        intgrl = (upper_bound - lower_bound) * m
        if p < q:
            intgrl += (q - p) * m - fsum(self._spline_coeff[p + 1 : q + 1])
        elif q < p:
            intgrl += (q - p) * m + fsum(self._spline_coeff[q + 1 : p + 1])
        intgrl += fsum(
            (self._spline_coeff[k % self._period] - m) * (
                1.0 - integrated_b_spline(
                    k + self._delay - upper_bound,
                    self._degree
                )
            )
            for k in range(p0, p0 + self._degree + 2)
        )
        return intgrl - fsum(
            (self._spline_coeff[k % self._period] - m) * (
                1.0 - integrated_b_spline(
                    k + self._delay - lower_bound,
                    self._degree
                )
            )
            for k in range(q0, q0 + self._degree + 2)
        )

    #---------------
    def fourier_coeff (
        self,
        nu: int
    ) -> complex:

        r"""
        .. _periodic_spline_1d-fourier_coeff:

        The Fourier coefficient of index ``nu``.

        Letting this spline of period :math:`K` be
        :math:`f:{\mathbb{R}}\rightarrow{\mathbb{R}},x\mapsto f(x),` return
        the Fourier coefficient

        ..  math::

            F[\nu]=\frac{1}{K}\,\int_{0}^{K}\,f(x)\,{\mathrm{e}}^
            {-{\mathrm{j}\,\nu\,\frac{2\,\pi}{K}\,x}}\,{\mathrm{d}}x,

        with :math:`\nu\in{\mathbb{Z}}.` The infinite-length series of
        coefficients :math:`F` is such that

        ..  math::

            f(x)=\sum_{\nu\in{\mathbb{Z}}}\,F[\nu]\,{\mathrm{e}}^
            {{\mathrm{j}\,\nu\,\frac{2\,\pi}{K}\,x}}.

        Parameters
        ----------
        nu : int
            The frequency index.

        Returns
        -------
        complex
            The Fourier coefficient of index ``nu``.

        Examples
        --------
        Load the libraries.
            >>> import numpy as np
            >>> import splinekit as sk
        Create a cubic spline and compute its Fourier coefficient at some arbitrary index.
            >>> c = np.array([1, 9, -7], dtype = float)
            >>> s = sk.PeriodicSpline1D.from_spline_coeff(c, degree = 3, delay = 0.71)
            >>> s.fourier_coeff(-5)
            (0.003157821372112742-0.0014059526579449596j)

        ----

        """

        return complex(
            np.dot(
                self._spline_coeff,
                np.fromiter(
                    (
                        cmath.exp(-nu * 2j * np.pi * (k + self._delay) /
                            self._period)
                        for k in range(self._period)
                    ),
                    dtype = complex,
                    count = self._period
                )
            ) * (np.sinc(nu / self._period) ** (self._degree + 1)) /
                self._period
        )

    #---------------
    def piecewise_polynomials (
        self,
        *,
        symbol: str = "x"
    ) -> PeriodicNonuniformPiecewise:

        r"""
        .. _periodic_spline_1d-piecewise_polynomials:

        The list of all polynomial pieces over one period of this
        ``PeriodicSpline1D`` object.

        Assume that this spline :math:`f` is of period :math:`K,` degree
        :math:`n`, and delay :math:`\delta x.` Then, this function returns a
        :ref:`PeriodicNonuniformPiecewise<PeriodicNonuniformPiecewise>`
        object whose :ref:`Piece<Piece>` items contain a ``numpy``
        :ref:`polynomial<def-polynomial>` over each piece domain. For
        :math:`x\in{\mathbb{R}}` we write that

        ..  math::

            \begin{eqnarray*}
            f(x)&=&\sum_{k\in{\mathbb{Z}}}\,c[{k\bmod K}]\,
            \beta^{n}(x-\delta x-k)\\
            &=&{\mathbf{c}}^{{\mathsf{T}}}\,{\mathbf{W}}^{n}\,
            {\mathbf{v}}^{n}(\chi).
            \end{eqnarray*}

        There, :math:`c` are the :math:`K` spline coefficients,
        :math:`{\mathbf{c}}=\left(c[{\left(k-\Xi\right)\bmod
        K}]\right)_{k=0}^{n}\in{\mathbb{R}}^{n+1}` is a vector of
        :math:`\left(n+1\right)` spline coefficients, :math:`{\mathbf{W}}^{n}`
        is the :ref:`B-spline evaluation matrix<w_frac>`, and
        :math:`{\mathbf{v}}^{n}(\chi)` is the
        :ref:`Vandermonde vector<def-vandermonde_vector>` of argument
        :math:`\chi,` with :math:`\xi=\left(\frac{n-1}{2}-x+\delta x\right)
        \in{\mathbb{R}},` :math:`\Xi=\left\lceil\xi\right\rceil\in
        {\mathbb{Z}},` and :math:`\chi=\left(\Xi-\xi\right)\in[0,1).`

        With these notations and in the general case, each piece is a
        polynomial defined over an interval domain :math:`{\mathbb{X}}` such
        that :math:`\chi(n,x,\delta x)\in[0,1)` for :math:`x\in{\mathbb{X}}.`
        The corresponding ``window`` of the ``numpy.polynomial.Polynomial``
        object is then ``[0.0, 1.0]``, while its ``domain`` is set to the
        domain :math:`{\mathbb{X}}.` The polynomial is a weighted sum of the
        monomials found in the Vandermonde vector :math:`{\mathbf{v}}^{n},`
        the weights being given by the vector
        :math:`\left({\mathbf{W}}^{n}\right)^{{\mathsf{T}}}\,{\mathbf{c}}.`

        Exceptions to the general case arise.

        *   At the lower end and the upper end of the partition of the period
            :math:`[0, K),` the :ref:`diameter<diameter>` of the ``numpy``
            domain may be shortened. The ``window`` is adjusted accordingly.
        *   For :math:`n=0,` the values :math:`{\mathbb{F}}({\mathbb{S}})` at
            the :ref:`splitting points<def-piecewise_polynomial>` of a
            :ref:`piecewise-constant<def-spline>` spline are provided as a
            polynomial of degree zero and ``window = [-0.5, 0.5]``, with
            :math:`k`-th ``domain`` being :math:`[f_{k}-\frac{1}{2},
            f_{k}-\frac{1}{2}].`
        *   For :math:`K=0,` the delay :math:`\delta x` plays no role.

        Parameters
        ----------
        symbol : str
            Symbol used to represent the independent variable in string
            representations of the polynomial expression. The symbol must be
            a valid Python identifier.

        Returns
        -------
        PeriodicNonuniformPiecewise
            The polynomial pieces of this piecewise polynomial spline.

        Examples
        --------
        Load the libraries.
            >>> import numpy as np
            >>> import splinekit as sk
        Create a spline and get its polynomials.
            >>> c = np.array([1, 9, -7], dtype = float)
            >>> s = sk.PeriodicSpline1D.from_spline_coeff(c, degree = 3)
            >>> s.piecewise_polynomials()
            periodic_nonuniform_piecewise([PeriodicNonuniformPiecewise.Piece(domain=ClosedOpen((0.0, 1.0)), item=Polynomial([ 1.,  8.,  0., -4.], domain=[0., 1.], window=[0., 1.], symbol='x')), PeriodicNonuniformPiecewise.Piece(domain=ClosedOpen((1.0, 2.0)), item=Polynomial([  5.,  -4., -12.,   8.], domain=[1., 2.], window=[0., 1.], symbol='x')), PeriodicNonuniformPiecewise.Piece(domain=ClosedOpen((2.0, 3.0)), item=Polynomial([-3., -4., 12., -4.], domain=[2., 3.], window=[0., 1.], symbol='x'))], period = 3)
        All pieces are listed, even pieces of vanishing diameter.
            >>> c = np.array([-2, 5], dtype = float)
            >>> s = sk.PeriodicSpline1D.from_spline_coeff(c, degree = 0, delay = 0.03)
            >>> s.piecewise_polynomials()
            periodic_nonuniform_piecewise([PeriodicNonuniformPiecewise.Piece(domain=ClosedOpen((0.0, 0.53)), item=Polynomial([-2.], domain=[0.  , 0.53], window=[0.  , 0.53], symbol='x')), PeriodicNonuniformPiecewise.Piece(domain=Singleton(0.53), item=Polynomial([1.5], domain=[0.03, 1.03], window=[-0.5,  0.5], symbol='x')), PeriodicNonuniformPiecewise.Piece(domain=Open((0.53, 1.53)), item=Polynomial([5.], domain=[0.53, 1.53], window=[0., 1.], symbol='x')), PeriodicNonuniformPiecewise.Piece(domain=Singleton(1.53), item=Polynomial([1.5], domain=[1.03, 2.03], window=[-0.5,  0.5], symbol='x')), PeriodicNonuniformPiecewise.Piece(domain=Open((1.53, 2)), item=Polynomial([-2.], domain=[1.53, 2.  ], window=[0.  , 0.47], symbol='x'))], period = 2)

        Notes
        -----
        The polynomial variable :math:`x` appears with a negative sign in the
        expression of :math:`\xi,` but :math:`\xi` appears itself with a
        negative sign in the expression of :math:`\chi.` The two signs
        compensate each other so that :math:`\chi` grows like :math:`x.`

        ----

        """

        if 1 == self._period:
            return PeriodicNonuniformPiecewise(
                [PeriodicNonuniformPiecewise.Piece(
                    ClosedOpen((0.0, 1.0)),
                    np.polynomial.Polynomial(
                        [self._spline_coeff[0]],
                        domain = [0.0, 1.0],
                        window = [0.0, 1.0],
                        symbol = symbol
                    )
                )],
                period = 1
            )
        (_, b_first) = _divmod(
            self._delay + 0.5 * (self._degree - 1),
            self._period
        )
        b0 = b_first
        pp = []
        if 0 == self._degree:
            for k in range(self._period):
                b1 = b0 + 1.0
                pp.append(PeriodicNonuniformPiecewise.Piece(
                    Singleton(b0),
                    np.polynomial.Polynomial(
                        [
                            0.5 * (self._spline_coeff[k - 1] +
                                self._spline_coeff[k]
                            )
                        ],
                        domain = [b0 - 0.5, b0 + 0.5],
                        window = [-0.5, 0.5],
                        symbol = symbol
                    )
                ))
                if b1 < self._period:
                    pp.append(PeriodicNonuniformPiecewise.Piece(
                        Open((b0, b1)),
                        np.polynomial.Polynomial(
                            [self._spline_coeff[k]],
                            domain = [b0, b1],
                            window = [0.0, 1.0],
                            symbol = symbol
                        )
                    ))
                    b0 = b1
                elif self._period == b1:
                    pp.append(PeriodicNonuniformPiecewise.Piece(
                        Open((b0, b1)),
                        np.polynomial.Polynomial(
                            [self._spline_coeff[k]],
                            domain = [b0, b1],
                            window = [0.0, 1.0],
                            symbol = symbol
                        )
                    ))
                    (_, b0) = _divmod(b_first, 1)
                    if 0.0 != b0:
                        pp.append(PeriodicNonuniformPiecewise.Piece(
                            ClosedOpen((0.0, b0)),
                            np.polynomial.Polynomial(
                                [self._spline_coeff[k]],
                                domain = [0.0, b0],
                                window = [0.0, b0],
                                symbol = symbol
                            )
                        ))
                else:
                    pp.append(PeriodicNonuniformPiecewise.Piece(
                        Open((b0, self._period)),
                        np.polynomial.Polynomial(
                            [self._spline_coeff[k]],
                            domain = [b0, self._period],
                            window = [0.0, self._period - b0],
                            symbol = symbol
                        )
                    ))
                    (_, b0) = _divmod(b_first, 1)
                    pp.append(PeriodicNonuniformPiecewise.Piece(
                        ClosedOpen((0.0, b0)),
                        np.polynomial.Polynomial(
                            [self._spline_coeff[k]],
                            domain = [0.0, b0],
                            window = [0.0, b0],
                            symbol = symbol
                        )
                    ))
        else:
            for k in range(self._period):
                b1 = b0 + 1.0
                if b1 < self._period:
                    pp.append(PeriodicNonuniformPiecewise.Piece(
                        ClosedOpen((b0, b1)),
                        np.polynomial.Polynomial(
                            np.array([
                                self._spline_coeff[q % self._period]
                                for q in range(k, k + self._degree + 1)
                            ]) @ _w(self._degree),
                            domain = [b0, b1],
                            window = [0.0, 1.0],
                            symbol = symbol
                        )
                    ))
                    b0 = b1
                elif self._period == b1:
                    pp.append(PeriodicNonuniformPiecewise.Piece(
                        ClosedOpen((b0, b1)),
                        np.polynomial.Polynomial(
                            np.array([
                                self._spline_coeff[q % self._period]
                                for q in range(k, k + self._degree + 1)
                            ]) @ _w(self._degree),
                            domain = [b0, b1],
                            window = [0.0, 1.0],
                            symbol = symbol
                        )
                    ))
                    (_, b0) = _divmod(b_first, 1)
                    if 0.0 != b0:
                        pp.append(PeriodicNonuniformPiecewise.Piece(
                            ClosedOpen((0.0, b0)),
                            np.polynomial.Polynomial(
                                np.array([
                                    self._spline_coeff[q % self._period]
                                    for q in range(k, k + self._degree + 1)
                                ]) @ _w(self._degree),
                                domain = [0.0, b0],
                                window = [0.0, b0],
                                symbol = symbol
                            )
                        ))
                else:
                    pp.append(PeriodicNonuniformPiecewise.Piece(
                        ClosedOpen((b0, self._period)),
                        np.polynomial.Polynomial(
                            np.array([
                                self._spline_coeff[q % self._period]
                                for q in range(k, k + self._degree + 1)
                            ]) @ _w(self._degree),
                            domain = [b0, self._period],
                            window = [0.0, self._period - b0],
                            symbol = symbol
                        )
                    ))
                    (_, b0) = _divmod(b_first, 1)
                    pp.append(PeriodicNonuniformPiecewise.Piece(
                        ClosedOpen((0.0, b0)),
                        np.polynomial.Polynomial(
                            np.array([
                                self._spline_coeff[q % self._period]
                                for q in range(k, k + self._degree + 1)
                            ]) @ _w(self._degree),
                            domain = [0.0, b0],
                            window = [1.0 - b0, 1.0],
                            symbol = symbol
                        )
                    ))
        return PeriodicNonuniformPiecewise(pp, period = self._period)

    #---------------
    def piecewise_signs (
        self,
    ) -> PeriodicNonuniformPiecewise:

        # TODO: docstring
        # TODO: tests
        r"""
        .. _periodic_spline_1d-piecewise_signs:


        ----

        """

        if 1 == self._period:
            return PeriodicNonuniformPiecewise(
                [PeriodicNonuniformPiecewise.Piece(
                    ClosedOpen((0.0, 1.0)),
                    _sgn(self._spline_coeff[0])
                )],
                period = 1
            )
        (_, b_first) = _divmod(
            self._delay + 0.5 * (self._degree - 1),
            self._period
        )
        b0 = b_first
        ps = []
        if 0 == self._degree:
            for (k, ck) in enumerate(self._spline_coeff):
                b1 = b0 + 1.0
                ps.append(PeriodicNonuniformPiecewise.Piece(
                    Singleton(b0),
                    _sgn(0.5 * (self._spline_coeff[k - 1] + ck))
                ))
                if b1 < self._period:
                    ps.append(PeriodicNonuniformPiecewise.Piece(
                        Open((b0, b1)),
                        _sgn(ck)
                    ))
                    b0 = b1
                elif self._period == b1:
                    ps.append(PeriodicNonuniformPiecewise.Piece(
                        Open((b0, b1)),
                        _sgn(ck)
                    ))
                    (_, b0) = _divmod(b_first, 1)
                    if 0.0 != b0:
                        ps.append(PeriodicNonuniformPiecewise.Piece(
                            ClosedOpen((0.0, b0)),
                            _sgn(ck)
                        ))
                else:
                    ps.append(PeriodicNonuniformPiecewise.Piece(
                        Open((b0, self._period)),
                        _sgn(ck)
                    ))
                    (_, b0) = _divmod(b_first, 1)
                    ps.append(PeriodicNonuniformPiecewise.Piece(
                        ClosedOpen((0.0, b0)),
                        _sgn(ck)
                    ))
        else:
            for k in range(self._period):
                b1 = b0 + 1.0
                a0 = (
                    self._spline_coeff[k : k + self._degree + 1]
                    if (0 <= k and k + self._degree < self._period)
                    else np.fromiter(
                        (
                            self._spline_coeff[q % self._period]
                            for q in range(k, k + self._degree + 1)
                        ),
                        dtype = float,
                        count = self._degree + 1
                    )
                ) @ _w(self._degree)
                lo = float(a0[0])
                up = float(a0[0])
                for q in range(1, self._degree + 1):
                    lo += min(0.0, float(a0[q]))
                    up += max(0.0, float(a0[q]))
                if 0.0 < lo * up:
                    if b1 < self._period:
                        ps.append(PeriodicNonuniformPiecewise.Piece(
                            ClosedOpen((b0, b1)),
                            _sgn(lo + up)
                        ))
                        b0 = b1
                    elif self._period == b1:
                        ps.append(PeriodicNonuniformPiecewise.Piece(
                            ClosedOpen((b0, b1)),
                            _sgn(lo + up)
                        ))
                        (_, b0) = _divmod(b_first, 1)
                        if 0.0 != b0:
                            ps.append(PeriodicNonuniformPiecewise.Piece(
                                ClosedOpen((0.0, b0)),
                                _sgn(lo + up)
                            ))
                    else:
                        ps.append(PeriodicNonuniformPiecewise.Piece(
                            ClosedOpen((b0, self._period)),
                            _sgn(lo + up)
                        ))
                        (_, b0) = _divmod(b_first, 1)
                        ps.append(PeriodicNonuniformPiecewise.Piece(
                            ClosedOpen((0.0, b0)),
                            _sgn(lo + up)
                        ))
                    continue
                p = np.polynomial.Polynomial(
                    a0,
                    domain = [0.0, 1.0],
                    window = [0.0, 1.0]
                )
                complex_roots = p.roots()
                roots = list(set(map(
                    lambda r: b0 + float(r),
                    np.real(complex_roots[np.isreal(complex_roots), ...])
                )))
                roots.sort()
                if 0 == len(roots):
                    if b1 < self._period:
                        ps.append(PeriodicNonuniformPiecewise.Piece(
                            ClosedOpen((b0, b1)),
                            _sgn(float(p(0.5)))
                        ))
                        b0 = b1
                    elif self._period == b1:
                        ps.append(PeriodicNonuniformPiecewise.Piece(
                            ClosedOpen((b0, b1)),
                            _sgn(float(p(0.5)))
                        ))
                        (_, b0) = _divmod(b_first, 1)
                        if 0.0 != b0:
                            ps.append(PeriodicNonuniformPiecewise.Piece(
                                ClosedOpen((0.0, b0)),
                                _sgn(float(p(0.5)))
                            ))
                    else:
                        ps.append(PeriodicNonuniformPiecewise.Piece(
                            ClosedOpen((b0, self._period)),
                            float(_sgn(float(p(0.5))))
                        ))
                        (_, b0) = _divmod(b_first, 1)
                        ps.append(PeriodicNonuniformPiecewise.Piece(
                            ClosedOpen((0.0, b0)),
                            _sgn(float(p(0.5)))
                        ))
                    continue
                r0 = None
                for r in roots:
                    if r < b0 or b1 <= r:
                        continue
                    if b0 == r:
                        r0 = r
                        ps.append(PeriodicNonuniformPiecewise.Piece(
                            Singleton(r),
                            _sgn(0.0)
                        ))
                        continue
                    if r0 is None:
                        r0 = b0
                        ps.append(PeriodicNonuniformPiecewise.Piece(
                            Singleton(r0),
                            _sgn(float(p(0.0)))
                        ))
                    s = _sgn(float(p(0.5 * (r0 + r) - b0)))
                    if r < self._period:
                        ps.append(PeriodicNonuniformPiecewise.Piece(
                            Open((r0, r)),
                            s
                        ))
                        ps.append(PeriodicNonuniformPiecewise.Piece(
                            Singleton((r)),
                            _sgn(0.0)
                        ))
                    elif self._period == r:
                        ps.append(PeriodicNonuniformPiecewise.Piece(
                            Open((r0, r)),
                            s
                        ))
                        ps.append(PeriodicNonuniformPiecewise.Piece(
                            Singleton((0.0)),
                            _sgn(0.0)
                        ))
                    else:
                        if r0 < self._period:
                            ps.append(PeriodicNonuniformPiecewise.Piece(
                                Open((r0, self._period)),
                                s
                            ))
                            (_, wrapped_r) = _divmod(r, self._period)
                            if 0.0 == wrapped_r:
                                ps.append(PeriodicNonuniformPiecewise.Piece(
                                    Singleton(wrapped_r),
                                    s
                                ))
                            else:
                                ps.append(PeriodicNonuniformPiecewise.Piece(
                                    ClosedOpen((0.0, wrapped_r)),
                                    s
                                ))
                        elif self._period == r0:
                            ps.append(PeriodicNonuniformPiecewise.Piece(
                                Open((0.0, r % self._period)),
                                s
                            ))
                        else:
                            ps.append(PeriodicNonuniformPiecewise.Piece(
                                Open((r0 % self._period, r % self._period)),
                                s
                            ))
                        ps.append(PeriodicNonuniformPiecewise.Piece(
                            Singleton((r % self._period)),
                            _sgn(0.0)
                        ))
                    r0 = r
                if r0 is None:
                    if b1 < self._period:
                        ps.append(PeriodicNonuniformPiecewise.Piece(
                            ClosedOpen((b0, b1)),
                            _sgn(float(p(0.5)))
                        ))
                        b0 = b1
                    elif self._period == b1:
                        ps.append(PeriodicNonuniformPiecewise.Piece(
                            ClosedOpen((b0, b1)),
                            _sgn(float(p(0.5)))
                        ))
                        (_, b0) = _divmod(b_first, 1)
                        if 0.0 != b0:
                            ps.append(PeriodicNonuniformPiecewise.Piece(
                                ClosedOpen((0.0, b0)),
                                _sgn(float(p(0.5)))
                            ))
                    else:
                        ps.append(PeriodicNonuniformPiecewise.Piece(
                            ClosedOpen((b0, self._period)),
                            float(_sgn(float(p(0.5))))
                        ))
                        (_, b0) = _divmod(b_first, 1)
                        ps.append(PeriodicNonuniformPiecewise.Piece(
                            ClosedOpen((0.0, b0)),
                            _sgn(float(p(0.5)))
                        ))
                else:
                    s = _sgn(float(p(0.5 * (r0 + b1) - b0)))
                    if b1 < self._period:
                        ps.append(PeriodicNonuniformPiecewise.Piece(
                            Open((r0, b1)),
                            s
                        ))
                        b0 = b1
                    elif self._period == b1:
                        ps.append(PeriodicNonuniformPiecewise.Piece(
                            Open((r0, b1)),
                            s
                        ))
                        (_, b0) = _divmod(b_first, 1)
                        if 0.0 != b0:
                            ps.append(PeriodicNonuniformPiecewise.Piece(
                                ClosedOpen((0.0, b0)),
                                s
                            ))
                    else:
                        if r0 < self._period:
                            ps.append(PeriodicNonuniformPiecewise.Piece(
                                Open((r0, self._period)),
                                s
                            ))
                            (_, b0) = _divmod(b_first, 1)
                            ps.append(PeriodicNonuniformPiecewise.Piece(
                                ClosedOpen((0.0, b0)),
                                s
                            ))
                        elif self._period == r0:
                            (_, b0) = _divmod(b_first, 1)
                            if 0.0 != b0:
                                ps.append(PeriodicNonuniformPiecewise.Piece(
                                    Open((0.0, b0)),
                                    s
                                ))
                        else:
                            (_, b0) = _divmod(b_first, 1)
                            ps.append(PeriodicNonuniformPiecewise.Piece(
                                Open((r0 % self._period, b0)),
                                s
                            ))
        ps.sort(key = lambda p: p.domain.sortorder())
        k = 0
        while k < len(ps) - 1:
            if isinstance(ps[k].domain, Singleton):
                if isinstance(ps[k + 1].domain, Singleton):
                    if (cast(Singleton, ps[k].domain).value !=
                        cast(Singleton, ps[k + 1].domain).value
                    ):
                        raise ValueError(
                            "Internal error (sign pieces out of sequence)"
                        )
                    if ps[k].item != ps[k + 1].item:
                        raise ValueError(
                            "Internal error (inconsistent sign pieces)"
                        )
                    ps[k : k + 2] = [ps[k]]
                elif isinstance(ps[k + 1].domain, Open):
                    if (cast(Singleton, ps[k].domain).value !=
                        cast(Open, ps[k + 1].domain).infimum
                    ):
                        raise ValueError(
                            "Internal error (sign pieces out of sequence)"
                        )
                    if ps[k].item == ps[k + 1].item:
                        ps[k : k + 2] = [PeriodicNonuniformPiecewise.Piece(
                            ClosedOpen((
                                cast(Singleton, ps[k].domain).value,
                                cast(Open, ps[k + 1].domain).supremum
                            )),
                            ps[k].item
                        )]
                    else:
                        k += 1
                elif isinstance(ps[k + 1].domain, OpenClosed):
                    if (cast(Singleton, ps[k].domain).value !=
                        cast(OpenClosed, ps[k + 1].domain).infimum
                    ):
                        raise ValueError(
                            "Internal error (sign pieces out of sequence)"
                        )
                    if ps[k].item == ps[k + 1].item:
                        ps[k : k + 2] = [PeriodicNonuniformPiecewise.Piece(
                            Closed((
                                cast(Singleton, ps[k].domain).value,
                                cast(OpenClosed, ps[k + 1].domain).supremum
                            )),
                            ps[k].item
                        )]
                    else:
                        k += 1
                elif isinstance(ps[k + 1].domain, Closed):
                    if (cast(Singleton, ps[k].domain).value !=
                        cast(Closed, ps[k + 1].domain).infimum
                    ):
                        raise ValueError(
                            "Internal error (sign pieces out of sequence)"
                        )
                    if ps[k].item != ps[k + 1].item:
                        raise ValueError(
                            "Internal error (inconsistent sign pieces)"
                        )
                    ps[k : k + 2] = [ps[k + 1]]
                elif isinstance(ps[k + 1].domain, ClosedOpen):
                    if (cast(Singleton, ps[k].domain).value !=
                        cast(ClosedOpen, ps[k + 1].domain).infimum
                    ):
                        raise ValueError(
                            "Internal error (sign pieces out of sequence)"
                        )
                    if ps[k].item != ps[k + 1].item:
                        raise ValueError(
                            "Internal error (inconsistent sign pieces)"
                        )
                    ps[k : k + 2] = [ps[k + 1]]
            elif isinstance(ps[k].domain, Open):
                if isinstance(ps[k + 1].domain, Singleton):
                    if (cast(Open, ps[k].domain).supremum !=
                        cast(Singleton, ps[k + 1].domain).value
                    ):
                        raise ValueError(
                            "Internal error (sign pieces out of sequence)"
                        )
                    if ps[k].item == ps[k + 1].item:
                        ps[k : k + 2] = [PeriodicNonuniformPiecewise.Piece(
                            OpenClosed((
                                cast(Open, ps[k].domain).infimum,
                                cast(Singleton, ps[k + 1].domain).value
                            )),
                            ps[k].item
                        )]
                    else:
                        k += 1
                elif isinstance(ps[k + 1].domain, Open):
                    raise ValueError(
                        "Internal error (sign pieces out of sequence)"
                    )
                elif isinstance(ps[k + 1].domain, OpenClosed):
                    raise ValueError(
                        "Internal error (sign pieces out of sequence)"
                    )
                elif isinstance(ps[k + 1].domain, Closed):
                    if (cast(Open, ps[k].domain).supremum !=
                        cast(Closed, ps[k + 1].domain).infimum
                    ):
                        raise ValueError(
                            "Internal error (sign pieces out of sequence)"
                        )
                    if ps[k].item == ps[k + 1].item:
                        ps[k : k + 2] = [PeriodicNonuniformPiecewise.Piece(
                            OpenClosed((
                                cast(Open, ps[k].domain).infimum,
                                cast(Closed, ps[k + 1].domain).supremum
                            )),
                            ps[k].item
                        )]
                    else:
                        k += 1
                elif isinstance(ps[k + 1].domain, ClosedOpen):
                    if (cast(Open, ps[k].domain).supremum !=
                        cast(ClosedOpen, ps[k + 1].domain).infimum
                    ):
                        raise ValueError(
                            "Internal error (sign pieces out of sequence)"
                        )
                    if ps[k].item == ps[k + 1].item:
                        ps[k : k + 2] = [PeriodicNonuniformPiecewise.Piece(
                            Open((
                                cast(Open, ps[k].domain).infimum,
                                cast(ClosedOpen, ps[k + 1].domain).supremum
                            )),
                            ps[k].item
                        )]
                    else:
                        k += 1
            elif isinstance(ps[k].domain, OpenClosed):
                if isinstance(ps[k + 1].domain, Singleton):
                    if (cast(OpenClosed, ps[k].domain).supremum !=
                        cast(Singleton, ps[k + 1].domain).value
                    ):
                        raise ValueError(
                            "Internal error (sign pieces out of sequence)"
                        )
                    if ps[k].item != ps[k + 1].item:
                        raise ValueError(
                            "Internal error (inconsistent sign pieces)"
                        )
                    ps[k : k + 2] = [ps[k]]
                elif isinstance(ps[k + 1].domain, Open):
                    if (cast(OpenClosed, ps[k].domain).supremum !=
                        cast(Open, ps[k + 1].domain).infimum
                    ):
                        raise ValueError(
                            "Internal error (sign pieces out of sequence)"
                        )
                    if ps[k].item == ps[k + 1].item:
                        ps[k : k + 2] = [PeriodicNonuniformPiecewise.Piece(
                            Open((
                                cast(OpenClosed, ps[k].domain).infimum,
                                cast(Open, ps[k + 1].domain).supremum
                            )),
                            ps[k].item
                        )]
                    else:
                        k += 1
                elif isinstance(ps[k + 1].domain, OpenClosed):
                    if (cast(OpenClosed, ps[k].domain).supremum !=
                        cast(OpenClosed, ps[k + 1].domain).infimum
                    ):
                        raise ValueError(
                            "Internal error (sign pieces out of sequence)"
                        )
                    if ps[k].item == ps[k + 1].item:
                        ps[k : k + 2] = [PeriodicNonuniformPiecewise.Piece(
                            OpenClosed((
                                cast(OpenClosed, ps[k].domain).infimum,
                                cast(OpenClosed, ps[k + 1].domain).supremum
                            )),
                            ps[k].item
                        )]
                    else:
                        k += 1
                elif isinstance(ps[k + 1].domain, Closed):
                    if (cast(OpenClosed, ps[k].domain).supremum !=
                        cast(Closed, ps[k + 1].domain).infimum
                    ):
                        raise ValueError(
                            "Internal error (sign pieces out of sequence)"
                        )
                    if ps[k].item != ps[k + 1].item:
                        raise ValueError(
                            "Internal error (inconsistent sign pieces)"
                        )
                    if ps[k].item == ps[k + 1].item:
                        ps[k : k + 2] = [PeriodicNonuniformPiecewise.Piece(
                            OpenClosed((
                                cast(OpenClosed, ps[k].domain).infimum,
                                cast(Closed, ps[k + 1].domain).supremum
                            )),
                            ps[k].item
                        )]
                elif isinstance(ps[k + 1].domain, ClosedOpen):
                    if (cast(OpenClosed, ps[k].domain).supremum !=
                        cast(ClosedOpen, ps[k + 1].domain).infimum
                    ):
                        raise ValueError(
                            "Internal error (sign pieces out of sequence)"
                        )
                    if ps[k].item != ps[k + 1].item:
                        raise ValueError(
                            "Internal error (inconsistent sign pieces)"
                        )
                    if ps[k].item == ps[k + 1].item:
                        ps[k : k + 2] = [PeriodicNonuniformPiecewise.Piece(
                            Open((
                                cast(OpenClosed, ps[k].domain).infimum,
                                cast(ClosedOpen, ps[k + 1].domain).supremum
                            )),
                            ps[k].item
                        )]
            elif isinstance(ps[k].domain, Closed):
                if isinstance(ps[k + 1].domain, Singleton):
                    if (cast(Closed, ps[k].domain).supremum !=
                        cast(Singleton, ps[k + 1].domain).value
                    ):
                        raise ValueError(
                            "Internal error (sign pieces out of sequence)"
                        )
                    if ps[k].item != ps[k + 1].item:
                        raise ValueError(
                            "Internal error (inconsistent sign pieces)"
                        )
                    ps[k : k + 2] = [ps[k]]
                elif isinstance(ps[k + 1].domain, Open):
                    if (cast(Closed, ps[k].domain).supremum !=
                        cast(Open, ps[k + 1].domain).infimum
                    ):
                        raise ValueError(
                            "Internal error (sign pieces out of sequence)"
                        )
                    if ps[k].item == ps[k + 1].item:
                        ps[k : k + 2] = [PeriodicNonuniformPiecewise.Piece(
                            ClosedOpen((
                                cast(Closed, ps[k].domain).infimum,
                                cast(Open, ps[k + 1].domain).supremum
                            )),
                            ps[k].item
                        )]
                    else:
                        k += 1
                elif isinstance(ps[k + 1].domain, OpenClosed):
                    if (cast(Closed, ps[k].domain).supremum !=
                        cast(OpenClosed, ps[k + 1].domain).infimum
                    ):
                        raise ValueError(
                            "Internal error (sign pieces out of sequence)"
                        )
                    if ps[k].item == ps[k + 1].item:
                        ps[k : k + 2] = [PeriodicNonuniformPiecewise.Piece(
                            Closed((
                                cast(Closed, ps[k].domain).infimum,
                                cast(OpenClosed, ps[k + 1].domain).supremum
                            )),
                            ps[k].item
                        )]
                    else:
                        k += 1
                elif isinstance(ps[k + 1].domain, Closed):
                    if (cast(Closed, ps[k].domain).supremum !=
                        cast(Closed, ps[k + 1].domain).infimum
                    ):
                        raise ValueError(
                            "Internal error (sign pieces out of sequence)"
                        )
                    if ps[k].item != ps[k + 1].item:
                        raise ValueError(
                            "Internal error (inconsistent sign pieces)"
                        )
                    if ps[k].item == ps[k + 1].item:
                        ps[k : k + 2] = [PeriodicNonuniformPiecewise.Piece(
                            Closed((
                                cast(Closed, ps[k].domain).infimum,
                                cast(Closed, ps[k + 1].domain).supremum
                            )),
                            ps[k].item
                        )]
                elif isinstance(ps[k + 1].domain, ClosedOpen):
                    if (cast(Closed, ps[k].domain).supremum !=
                        cast(ClosedOpen, ps[k + 1].domain).infimum
                    ):
                        raise ValueError(
                            "Internal error (sign pieces out of sequence)"
                        )
                    if ps[k].item != ps[k + 1].item:
                        raise ValueError(
                            "Internal error (inconsistent sign pieces)"
                        )
                    if ps[k].item == ps[k + 1].item:
                        ps[k : k + 2] = [PeriodicNonuniformPiecewise.Piece(
                            ClosedOpen((
                                cast(Closed, ps[k].domain).infimum,
                                cast(ClosedOpen, ps[k + 1].domain).supremum
                            )),
                            ps[k].item
                        )]
            elif isinstance(ps[k].domain, ClosedOpen):
                if isinstance(ps[k + 1].domain, Singleton):
                    if (cast(ClosedOpen, ps[k].domain).supremum !=
                        cast(Singleton, ps[k + 1].domain).value
                    ):
                        raise ValueError(
                            "Internal error (sign pieces out of sequence)"
                        )
                    if ps[k].item == ps[k + 1].item:
                        ps[k : k + 2] = [PeriodicNonuniformPiecewise.Piece(
                            Closed((
                                cast(ClosedOpen, ps[k].domain).infimum,
                                cast(Singleton, ps[k + 1].domain).value
                            )),
                            ps[k].item
                        )]
                    else:
                        k += 1
                elif isinstance(ps[k + 1].domain, Open):
                    raise ValueError(
                        "Internal error (sign pieces out of sequence)"
                    )
                elif isinstance(ps[k + 1].domain, OpenClosed):
                    raise ValueError(
                        "Internal error (sign pieces out of sequence)"
                    )
                elif isinstance(ps[k + 1].domain, Closed):
                    if (cast(ClosedOpen, ps[k].domain).supremum !=
                        cast(Closed, ps[k + 1].domain).infimum
                    ):
                        raise ValueError(
                            "Internal error (sign pieces out of sequence)"
                        )
                    if ps[k].item == ps[k + 1].item:
                        ps[k : k + 2] = [PeriodicNonuniformPiecewise.Piece(
                            Closed((
                                cast(ClosedOpen, ps[k].domain).infimum,
                                cast(Closed, ps[k + 1].domain).supremum
                            )),
                            ps[k].item
                        )]
                    else:
                        k += 1
                elif isinstance(ps[k + 1].domain, ClosedOpen):
                    if (cast(ClosedOpen, ps[k].domain).supremum !=
                        cast(ClosedOpen, ps[k + 1].domain).infimum
                    ):
                        raise ValueError(
                            "Internal error (sign pieces out of sequence)"
                        )
                    if ps[k].item == ps[k + 1].item:
                        ps[k : k + 2] = [PeriodicNonuniformPiecewise.Piece(
                            ClosedOpen((
                                cast(ClosedOpen, ps[k].domain).infimum,
                                cast(ClosedOpen, ps[k + 1].domain).supremum
                            )),
                            ps[k].item
                        )]
                    else:
                        k += 1
        return PeriodicNonuniformPiecewise(ps, period = self._period)

    #---------------
    def projected (
        self,
        *,
        degree: int,
        delay: float = 0.0
    ) -> PeriodicSpline1D:

        r"""
        .. _periodic_spline_1d-projected:

        The spline of arbitrary degree and delay that best approximates this
        spline.

        Letting this spline be :math:`s,` return the spline :math:`f` of
        arbitrary degree ``degree`` and delay ``delay`` that minimizes the
        continuous least-squares criterion

        ..  math::

            J=\int_{0}^{K}\,\left(f(x)-s(x)\right)^{2}\,{\mathrm{d}}x.

        Parameters
        ----------
        degree : int
            The :ref:`nonnegative<def-negative>` degree of the approximating
            polynomial spline.
        delay : float
            The delay of the approximating spline.

        Returns
        -------
        PeriodicSpline1D
            The approximating spline.

        Examples
        --------
        Load the libraries.
            >>> import numpy as np
            >>> import splinekit as sk
        Create an undelayed linear spline and approximate it as a delayed cubic spline.
            >>> c = np.array([1, 5, -3], dtype = float)
            >>> s = sk.PeriodicSpline1D.from_spline_coeff(c, degree = 1)
            >>> s.projected(degree = 3, delay = 0.25)
            PeriodicSpline1D([ 4.27369157  4.45287093 -5.7265625 ], degree = 3, delay = 0.25)

        Raises
        ------
        ValueError
            Raised when ``degree`` is :ref:`negative<def-negative>`.


        ----

        """

        if 0 > degree:
            raise ValueError("Degree must be nonnegative")
        c = np.copy(self._spline_coeff)
        samples_to_coeff_p(
            cast(
                np.ndarray[tuple[int], np.dtype[np.float64]],
                c,
            ),
            degree = 2 * degree + 1
        )
        k0 = floor(delay - self._delay + 0.5 * (self._degree + degree + 2))
        x0 = delay - self._delay - k0
        h = np.fromiter(
            (
                b_spline(x0 + k, self._degree + degree + 1)
                for k in range(self._degree + degree + 1 + 1)
            ),
            dtype = float,
            count = self._degree + degree + 1 + 1
        )
        f = np.fromiter(
            (
                fsum(
                    hq * c[(k - q) % self._period]
                    for (q, hq) in enumerate(h)
                )
                for k in range(self._period)
            ),
            dtype = float,
            count = self._period
        )
        return PeriodicSpline1D.from_spline_coeff(
            cast(
                np.ndarray[tuple[int], np.dtype[np.float64]],
                np.roll(f, -k0)
            ),
            degree = degree,
            delay = delay
        )

    #---------------
    def upscaled (
        self,
        *,
        magnification: int
    ) -> PeriodicSpline1D:

        r"""
        .. _periodic_spline_1d-upscaled:

        The spline that is an upscaled version of this spline.

        Letting this spline be
        :math:`s:{\mathbb{R}}\rightarrow{\mathbb{R}},x\mapsto s(x),` return
        the spline
        :math:`f:{\mathbb{R}}\rightarrow{\mathbb{R}},x\mapsto s(\frac{x}{a}),`
        where :math:`a` is the ``magnification`` factor. The period of
        :math:`f` is :math:`\left(a\,K\right),` where :math:`K` is the period
        of :math:`s.`

        Parameters
        ----------
        magnification : int
            The :ref:`positive<def-positive>` upscaling factor.

        Returns
        -------
        PeriodicSpline1D
            The upscaled spline.

        Examples
        --------
        Load the libraries.
            >>> import numpy as np
            >>> import splinekit as sk
        Create a linear spline and upscale it by a factor three.
            >>> c = np.array([3, 15, -9], dtype = float)
            >>> s = sk.PeriodicSpline1D.from_spline_coeff(c, degree = 1, delay = -0.6)
            >>> s.upscaled(magnification = 3)
            PeriodicSpline1D([-5. -1.  3.  7. 11. 15.  7. -1. -9.], degree = 1, delay = -3.8)

        Raises
        ------
        ValueError
            Raised when ``magnification`` is :ref:`nonpositive<def-positive>`.


        ----

        """

        if 1 > magnification:
            raise ValueError("Magnification must be positive")
        if 1 == magnification:
            return self.copy()
        h = mscale_filter(degree = self._degree, scale = magnification)
        h0 = len(h)
        c = self._spline_coeff
        p0 = self._period
        return PeriodicSpline1D.from_spline_coeff(
            cast(
                np.ndarray[tuple[int], np.dtype[np.float64]],
                np.fromiter(
                    (
                        fsum(
                            h[k - magnification * p] * c[p % p0]
                            for p in range(
                                ceil((k - (h0 - 1)) / magnification),
                                floor(k / magnification) + 1
                            )
                        ) / magnification ** self._degree
                        for k in range(magnification * p0)
                    ),
                    dtype = float,
                    count = magnification * p0
                )
            ),
            degree = self._degree,
            delay = magnification * self._delay -
                0.5 * (magnification - 1) * (self._degree + 1)
        )

    #---------------
    def upscaled_projected (
        self,
        *,
        magnification: int,
        degree: int,
        delay: float = 0.0
    ) -> PeriodicSpline1D:

        r"""
        .. _periodic_spline_1d-upscaled_projected:

        The spline that best approximates an upscaled version of this spline.

        Letting this spline be :math:`s,` return the spline :math:`f` of
        arbitrary degree ``degree`` and delay ``delay`` that minimizes the
        continuous least-squares criterion

        ..  math::

            J=\int_{0}^{a\,K}\,\left(f(x)-s(\frac{x}{a})\right)^{2}\,
            {\mathrm{d}}x,

        where :math:`a` is the ``magnification`` factor. The period of
        :math:`f` is :math:`\left(a\,K\right),` where :math:`K` is the period
        of :math:`s.`

        Parameters
        ----------
        magnification : int
            The :ref:`positive<def-positive>` upscaling factor.
        degree : int
            The :ref:`nonnegative<def-negative>` degree of the approximating
            polynomial spline.
        delay : float
            The delay of the approximating spline.

        Returns
        -------
        PeriodicSpline1D
            The approximating upscaled spline.

        Examples
        --------
        Load the libraries.
            >>> import numpy as np
            >>> import splinekit as sk
        Upscale a linear spline by two and approximate it by a quadratic spline of coinciding knots.
            >>> c = np.array([1, 5, -3], dtype = float)
            >>> s = sk.PeriodicSpline1D.from_spline_coeff(c, degree = 1)
            >>> s.upscaled_projected(magnification = 2, degree = 2, delay = 0.5)
            PeriodicSpline1D([ 1.65934066  4.62637363  3.96703297 -1.96703297 -2.62637363  0.34065934], degree = 2, delay = 0.5)

        Raises
        ------
        ValueError
            Raised when ``magnification`` is :ref:`nonpositive<def-positive>`.
        ValueError
            Raised when ``degree`` is :ref:`negative<def-negative>`.


        ----

        """

        if 1 > magnification:
            raise ValueError("Magnification must be positive")
        if 0 > degree:
            raise ValueError("Degree must be nonnegative")
        if 1 == magnification:
            return self.projected(degree = degree, delay = delay)
        p0 = magnification * self._period
        h = mscale_filter(degree = self._degree, scale = magnification)
        h0 = len(h)
        x0 = delay - magnification * self._delay + 0.5 * (h0 - 1)
        n0 = self._degree + degree + 1
        k0 = -floor(x0 + 0.5 * (n0 + 1))
        c = cast(
            np.ndarray[tuple[int], np.dtype[np.float64]],
            np.fromiter(
                (
                    fsum(
                        h[k - magnification * q] *
                            self._spline_coeff[q % self._period]
                        for q in range(
                            ceil((k - h0 + 1) / magnification),
                            k // magnification + 1
                        )
                    ) / magnification ** self._degree
                    for k in range(p0)
                ),
                dtype = float,
                count = p0
            )
        )
        samples_to_coeff_p(c, degree = 2 * degree + 1)
        b = np.fromiter(
            (b_spline(x0 + k0 + k, n0) for k in range(n0 + 1)),
            dtype = float,
            count = n0 + 1
        )
        c = cast(
            np.ndarray[tuple[int], np.dtype[np.float64]],
            np.fromiter(
                (
                    fsum(bq * c[(k - q) % p0] for (q, bq) in enumerate(b))
                    for k in range(p0)
                ),
                dtype = float,
                count = p0
            )
        )
        return PeriodicSpline1D.from_spline_coeff(
            cast(
                np.ndarray[tuple[int], np.dtype[np.float64]],
                np.roll(c, k0)
            ),
            degree = degree,
            delay = delay
        )

    #---------------
    def downscaled_projected (
        self,
        *,
        minification: int,
        degree: int,
        delay: float = 0.0
    ) -> PeriodicSpline1D:

        r"""
        .. _periodic_spline_1d-downscaled_projected:

        The spline that best approximates a downscaled version of this spline.

        Letting this spline be :math:`s,` return the spline :math:`f` of
        arbitrary degree ``degree`` and delay ``delay`` that minimizes the
        continuous least-squares criterion

        ..  math::

            J=\int_{0}^{K/\gcd(a,K)}\,\left(f(x)-s(a\,x)\right)^{2}\,
            {\mathrm{d}}x,

        where :math:`a` is the ``minification`` factor. The period of
        :math:`f` is :math:`K/\gcd(a,K),` where :math:`K` is the period of
        :math:`s.` The period will shrink only when :math:`1\neq\gcd(a,K).`

        Parameters
        ----------
        minification : int
            The :ref:`positive<def-positive>` downscaling factor.
        degree : int
            The :ref:`nonnegative<def-negative>` degree of the approximating
            polynomial spline.
        delay : float
            The delay of the approximating spline.

        Returns
        -------
        PeriodicSpline1D
            The approximating downscaled spline.

        Examples
        --------
        Load the libraries.
            >>> import numpy as np
            >>> import splinekit as sk
        Downscale by two a periodized linear B-spline of odd period. The downscaled spline is made of two bumps, one centered at the origin, the other centered at half the (odd) period.
            >>> s = sk.PeriodicSpline1D.periodized_b_spline(degree = 1, period = 7)
            >>> s.downscaled_projected(minification = 2, degree = 1)
            PeriodicSpline1D([ 0.67073171 -0.09146341 -0.05487805  0.31097561  0.31097561 -0.05487805 -0.09146341], degree = 1, delay = 0.0)

        Raises
        ------
        ValueError
            Raised when ``minification`` is :ref:`nonpositive<def-positive>`.
        ValueError
            Raised when ``degree`` is :ref:`negative<def-negative>`.


        ----

        """

        if 1 > minification:
            raise ValueError("Minification must be positive")
        if 0 > degree:
            raise ValueError("Degree must be nonnegative")
        if 1 == minification:
            return self.projected(degree = degree, delay = delay)
        p0 = minification // gcd(minification, self._period)
        c = np.tile(self._spline_coeff, p0)
        p0 *= self._period
        h = mscale_filter(degree = degree, scale = minification)
        x0 = self._delay - minification * delay - 0.5 * (len(h) - 1)
        n0 = self._degree + degree + 1
        k0 = floor(x0 - 0.5 * (n0 - 1))
        c = cast(
            np.ndarray[tuple[int], np.dtype[np.float64]],
            np.fromiter(
                (
                    fsum(hq * c[(k - q) % p0] for (q, hq) in enumerate(h)) /
                        minification ** (degree + 1)
                    for k in range(p0)
                ),
                dtype = float,
                count = p0
            )
        )
        b = np.fromiter(
            (b_spline(k + k0 - x0, n0) for k in range(n0 + 1)),
            dtype = float,
            count = n0 + 1
        )
        c = cast(
            np.ndarray[tuple[int], np.dtype[np.float64]],
            np.fromiter(
                (
                    fsum(bq * c[(k - q) % p0] for (q, bq) in enumerate(b))
                    for k in range(p0)
                ),
                dtype = float,
                count = p0
            )
        )
        c = cast(
            np.ndarray[tuple[int], np.dtype[np.float64]],
            np.roll(c, k0)
        )
        c = cast(
            np.ndarray[tuple[int], np.dtype[np.float64]],
            np.fromiter(
                (c[k * minification] for k in range(p0 // minification)),
                dtype = float,
                count = p0 // minification
            )
        )
        samples_to_coeff_p(c, degree = 2 * degree + 1)
        return PeriodicSpline1D.from_spline_coeff(
            c,
            degree = degree,
            delay = delay
        )

    #---------------
    def rescaled_projected (
        self,
        *,
        period: int,
        degree: int,
        delay: float = 0.0
    ) -> PeriodicSpline1D:

        r"""
        .. _periodic_spline_1d-rescaled_projected:


        The spline that best approximates a rescaled version of this spline.

        Letting this spline of period :math:`P` be :math:`s,` return the spline
        :math:`f` of arbitrary ``period`` :math:`K,` degree ``degree``, and
        delay ``delay`` that minimizes the continuous least-squares criterion

        ..  math::

            J=\int_{0}^{K}\,\left(f(x)-s(\frac{P}{K}\,x)\right)^{2}\,
            {\mathrm{d}}x.

        Parameters
        ----------
        period : int
            The :ref:`positive<def-positive>` period of the approximating
            spline.
        degree : int
            The :ref:`nonnegative<def-negative>` degree of the approximating
            polynomial spline.
        delay : float
            The delay of the approximating spline.

        Returns
        -------
        PeriodicSpline1D
            The approximating rescaled spline.

        Examples
        --------
        Load the libraries.
            >>> import numpy as np
            >>> import splinekit as sk
        Rescale a periodized linear B-spline of period ``7`` to period ``4``.
            >>> s = sk.PeriodicSpline1D.periodized_b_spline(degree = 1, period = 7)
            >>> s.rescaled_projected(period = 4, degree = 1)
            PeriodicSpline1D([ 0.75510204 -0.12244898  0.06122449 -0.12244898], degree = 1, delay = 0.0)

        Raises
        ------
        ValueError
            Raised when ``period`` is :ref:`nonpositive<def-positive>`.
        ValueError
            Raised when ``degree`` is :ref:`negative<def-negative>`.
        """

        if 1 > period:
            raise ValueError("Period must be positive")
        if 0 > degree:
            raise ValueError("Degree must be nonnegative")
        if period == self._period:
            return self.projected(degree = degree, delay = delay)
        return self.upscaled(
            magnification = period // gcd(period, self._period)
        ).downscaled_projected(
            minification = self.period // gcd(period, self._period),
            degree = degree,
            delay = delay
        )
