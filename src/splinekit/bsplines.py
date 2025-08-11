"""
..  role:: raw-html(raw)
    :format: html
"""

#---------------
from typing import cast

#---------------
import functools

#---------------
from fractions import Fraction

#---------------
from math import ceil
from math import comb
from math import factorial
from math import fsum

#---------------
import numpy as np
import sympy

#---------------
from splinekit.interval import Closed
from splinekit.interval import Interval
from splinekit.interval import Open
from splinekit.spline_utilities import _alpha_frac
from splinekit.spline_utilities import _db
from splinekit.spline_utilities import _db_frac
from splinekit.spline_utilities import _iota
from splinekit.spline_utilities import _knots
from splinekit.spline_utilities import _pole
from splinekit.spline_utilities import _pse
from splinekit.spline_utilities import _sgn
from splinekit.spline_utilities import _w
from splinekit.spline_utilities import _wd
from splinekit.spline_utilities import _wint

#---------------
def polynomial_simple_element (
    x: float,
    n: int
) -> float:

    r"""
    .. _polynomial_simple_element:

    Polynomial simple element :math:`\varsigma^{n}.`

    Returns the value of a
    :ref:`polynomial simple element<def-polynomial_simple_element>` of integer
    degree :math:`n` evaluated at :math:`x.` It is defined as

    ..  math::

        \varsigma^{n}(x)=\left\{\begin{array}{ll}
        \delta^{\left(\left|n-1\right|\right)}(x),&
        n\in{\mathbb{Z}}\setminus{\mathbb{N}}\\
        \frac{1}{2\,\left(n!\right)}\,{\mathrm{sgn}}(x)\,x^{n},&
        n\in{\mathbb{N}},\end{array}\right.

    where :math:`\delta^{\left(m\right)}` is the :math:`m`-th derivative of
    the Dirac delta distribution and where :math:`{\mathrm{sgn}}` is the
    :ref:`signum<def-sgn>` function.

    Parameters
    ----------
    x : float
        Argument.
    n : int
        Degree of the polynomial simple element.

    Returns
    -------
    float
        The value of the polynomial simple element at :math:`x.`

    Examples
    --------
    Load the library.
        >>> import splinekit as sk
    Polynomial simple elements of positive degrees always vanish at ``0.0``.
        >>> sk.polynomial_simple_element(0.0, 2)
        0.0
        >>> sk.polynomial_simple_element(0.0, 1)
        0.0
    The polynomial simple element of degree zero vanishes at ``0.0`` as well.
        >>> sk.polynomial_simple_element(0.0, 0)
        0.0
    Polynomial simple elements of negative degrees are not real numbers at ``0.0``.
        >>> sk.polynomial_simple_element(0.0, -1)
        nan
    Polynomial simple elements of negative degrees vanish everywhere except at the origin.
        >>> sk.polynomial_simple_element(1.0, -5)
        0.0
    """

    return _pse(x, n)

#---------------
def b_spline (
    x: float,
    n: int
) -> float:

    r"""
    .. _b_spline:

    B-spline :math:`\beta^{n}.`

    Returns the value of the :ref:`polynomial B-spline<def-b_spline>`
    :math:`\beta^{n}` of :ref:`nonnegative<def-negative>` degree :math:`n`
    evaluated at the argument :math:`x.` For the degree :math:`n=0,` this
    function is defined as

    ..  math::

        \beta^{0}(x)=\varsigma^{0}(x+\frac{1}{2})-\varsigma^{0}(x-\frac{1}{2}),

    with :math:`\varsigma^{0}` a
    :ref:`polynomial simple element<polynomial_simple_element>` of degree
    :math:`0.` B-splines of :ref:`positive<def-positive>` degree :math:`n`
    are computed as

    ..  math::

        \beta^{n}(x)=\left\{\begin{array}{ll}\left(w^{n}[r][\cdot]\right)^
        {{\mathsf{T}}}\,{\mathbf{v}}^{n}(\chi),&\left|x\right|<\frac{n+1}{2}\\
        0,&\frac{n+1}{2}\leq\left|x\right|,\end{array}\right.

    with :math:`\xi=\left(\frac{n-1}{2}-x\right),`
    :math:`r=\left\lceil\xi\right\rceil\in{\mathbb{Z}},` and
    :math:`\chi=\left(r-\xi\right)\in[0,1).` Moreover,
    :math:`\left(w^{n}[r][\cdot]\right)^{{\mathsf{T}}}` is the
    :math:`(r+1)`-th row of the :ref:`B-spline evaluation matrix<w_frac>`, and
    :math:`{\mathbf{v}}^{n}(\chi)` is the
    :ref:`Vandermonde vector<def-vandermonde_vector>` of argument :math:`\chi`
    and degree :math:`n.`

    As computed above, the fact that the Vandermonde vector has the domain
    :math:`[0,1)` greatly favors numerical stability since the range of each
    of its components is :math:`[0,1].`

    Parameters
    ----------
    x : float
        Argument.
    n : int
        Nonnegative degree of the polynomial B-spline.

    Returns
    -------
    float
        The value of a B-spline.

    Examples
    --------
    Load the library.
        >>> import splinekit as sk
    Value of the cubic B-spline at the origin.
        >>> sk.b_spline(0.0, 3)
        0.6666666666666666
    The B-spline of degree ``0`` is even-symmetric, also at its discontinuity.
        >>> sk.b_spline(-0.5, 0) == sk.b_spline(0.5, 0)
        True
    The B-spline of degree ``0`` satisfies the partition of unity, also at its discontinuity.
        >>> 1.0 == sk.b_spline(-0.5, 0) + sk.b_spline(0.5, 0)
        True
    """

    if 0 > n:
        raise ValueError("The degree n must be nonnegative")
    if 0 == n:
        return 0.5 * (_sgn(x + 0.5) - _sgn(x - 0.5))
    if 0.5 * (n + 1.0) <= abs(x):
        return 0.0
    xi = 0.5 * (n - 1.0) - x
    xi0 = ceil(xi)
    return float(_w(n)[int(xi0)] @
        np.vander([xi0 - xi], n + 1, increasing = True)[0]
    )

#---------------
def b_spline_support (
    n: int
) -> Interval:

    r"""
    .. _b_spline_support:

    Support :math:`{\mathrm{supp}}\,\beta^{n}.`

    Returns the :ref:`support<def-support>` of the
    :ref:`polynomial B-spline<def-b_spline>` of
    :ref:`nonnegative<def-negative>` degree :math:`n.` For the degree
    :math:`n=0,` the support is the :ref:`closed interval<Closed>`

    ..  math::

        [-\frac{1}{2},\frac{1}{2}].

    B-splines of :ref:`positive<def-positive>` degree :math:`n` have as
    support the :ref:`open interval<Open>`

    ..  math::

        (-\frac{n+1}{2},\frac{n+1}{2}).

    Parameters
    ----------
    n : int
        Nonnegative degree of the polynomial B-spline.

    Returns
    -------
    Interval
        The support of a B-spline.

    Examples
    --------
    Load the library.
        >>> import splinekit as sk
    Support of the cubic B-spline.
        >>> sk.b_spline_support(3)
        Open((-2.0, 2.0))
    """

    if 0 > n:
        raise ValueError("The degree n must be nonnegative")
    if 0 == n:
        return Closed((-0.5, 0.5))
    return Open((-0.5 * (n + 1), 0.5 * (n + 1)))

#---------------
def b_spline_variance (
    n: int
) -> float:

    r"""
    .. _b_spline_variance:

    Variance :math:`\int\,\left(\cdot\right)^{2}\,\beta^{n}.`

    Returns the variance :math:`\sigma_{n}^{2}` of the
    :ref:`polynomial B-spline<def-b_spline>` :math:`\beta^{n}` of
    :ref:`nonnegative<def-negative>` degree :math:`n.` It is defined as

    ..  math::

        \sigma_{n}^{2}=\int_{-\infty}^{\infty}\,x^{2}\,\beta^{n}(x)\,
        {\mathrm{d}}x.

    Parameters
    ----------
    n : int
        Nonnegative degree of the polynomial B-spline.

    Returns
    -------
    float
        The variance of a B-spline.

    Examples
    --------
    Load the library.
        >>> import splinekit as sk
    Variance of the cubic B-spline.
        >>> sk.b_spline_variance(3)
        0.3333333333333333
    """

    if 0 > n:
        raise ValueError("The degree n must be nonnegative")
    return (n + 1) / 12.0

#---------------
def pole (
    n: int
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:

    r"""
    .. _pole:

    Poles of the B-spline :math:`\beta^{n}.`

    Returns the ``numpy`` vector :math:`{\mathbf{z}}^{n}\in
    {\mathbb{R}}^{\left\lfloor n/2\right\rfloor}` of the poles of the
    reciprocal of the z-transform of the sequence made of the samples at the
    integers of the :ref:`polynomial B-spline<db_frac>`
    :math:`\beta^{n}` of :ref:`nonnegative<def-negative>` degree :math:`n.`
    The components of :math:`{\mathbf{z}}^{n}` are sorted in increasing order
    (*i.e.*, in decreasing absolute order); only those poles that are in the
    interval :math:`(-1,0)` are provided. For every pole :math:`z,` it holds
    that

    ..  math::

        0=\sum_{k\in{\mathbb{Z}}}\,\beta^{n}(k)\,z^{-k}.

    Parameters
    ----------
    n : int
        Nonnegative degree of the B-spline.

    Returns
    -------
    np.ndarray[tuple[int], np.dtype[np.float64]]
        The poles.

    Examples
    --------
    Load the library.
        >>> import splinekit as sk
    The linear B-spline has no pole.
        >>> sk.pole(1)
        array([], dtype=float64)
    The B-spline of degree ``5`` has ``2`` poles.
        >>> sk.pole(5)
        array([-0.43057535, -0.04309629])

    Notes
    -----
    The results of this method are cached. If the returned results are
    mutated, the cache gets modified and the next call will return corrupted
    values.
    """

    return _pole(n)

#---------------
def integrated_b_spline (
    x: float,
    n: int
) -> float:

    r"""
    .. _integrated_b_spline:

    Integrated B-spline :math:`\int\beta^{n}.`

    Returns the value of the integral
    :math:`\int_{-\infty}^{x}\,\beta^{n}(y)\,{\mathrm{d}}y` of the
    :ref:`polynomial B-spline<def-b_spline>` :math:`\beta^{n}` of
    :ref:`nonnegative<def-negative>` degree :math:`n` evaluated at the
    argument :math:`x.` It is computed as

    ..  math::

        \int_{-\infty}^{x}\,\beta^{n}(y)\,{\mathrm{d}}y=
        \left\{\begin{array}{ll}0,&x\leq-\frac{n+1}{2}\\
        \left(w_{{\mathrm{int}}}^{n}[r][\cdot]\right)^
        {{\mathsf{T}}}\,{\mathbf{v}}^{n+1}(\chi),
        &-\frac{n+1}{2}<x<\frac{n+1}{2}\\
        1,&\frac{n+1}{2}\leq x,\end{array}\right.

    with :math:`\xi=\left(\frac{n-1}{2}-x\right),`
    :math:`r=\left\lceil\xi\right\rceil\in{\mathbb{Z}},` and
    :math:`\chi=\left(r-\xi\right)\in[0,1).` Moreover,
    :math:`\left(w_{{\mathrm{int}}}^{n}[r][\cdot]\right)^{{\mathsf{T}}}` is
    the :math:`(r+1)`-th row of the
    :ref:`integrated-B-spline evaluation matrix<wint>`, and
    :math:`{\mathbf{v}}^{n+1}(\chi)` is the
    :ref:`Vandermonde vector<def-vandermonde_vector>` of argument
    :math:`\chi` and degree :math:`n+1.`

    As computed above, the fact that the Vandermonde vector has the domain
    :math:`[0,1)` greatly favors numerical stability since the range of each
    of its components is :math:`[0,1].`

    Parameters
    ----------
    x : float
        Argument.
    n : int
        Nonnegative degree of the polynomial B-spline.

    Returns
    -------
    float
        The value of an integrated B-spline.

    Examples
    --------
    Load the library.
        >>> import splinekit as sk
    Value of the integrated B-spline of degree 11 at the origin.
        >>> sk.integrated_b_spline(0.0, 11)
        0.5
    """

    if 0 > n:
        raise ValueError("The degree n must be nonnegative")
    if x < -0.5 * (n + 1.0):
        return 0.0
    if x < 0.5 * (n + 1.0):
        xi = 0.5 * (n - 1.0) - x
        xi0 = ceil(xi)
        return float(_wint(n)[int(xi0)] @
            np.vander([xi0 - xi], n + 2, increasing = True)[0]
        )
    return 1.0

#---------------
def grad_b_spline (
    x: float,
    n: int
) -> float:

    r"""
    .. _grad_b_spline:

    Differentiated B-spline :math:`\dot{\beta}^{n}.`

    Returns the value :math:`\dot{\beta}^{n}(x)` of the gradient of the
    :ref:`polynomial B-spline<def-b_spline>` :math:`\beta^{n}` of
    :ref:`nonnegative<def-negative>` degree :math:`n` evaluated at the
    argument :math:`x.` It is computed as

    ..  math::

        \dot{\beta}^{n}(x)=\left\{\begin{array}{ll}
        \sum_{k=0}^{n+1}\,\left(-1\right)^{k}\,{n+1\choose k}\,
        \varsigma^{n-1}(x+\frac{n+1}{2}-k),&
        n\leq1\wedge\left|x\right|\leq\frac{n+1}{2}\\
        \left(w_{{\mathrm{d}}}^{n}[r][\cdot]\right)^
        {{\mathsf{T}}}\,{\mathbf{v}}^{n-1}(\chi),
        &1<n\wedge\left|x\right|<\frac{n+1}{2}\\
        0,&1<n\wedge\left|x\right|=\frac{n+1}{2}\\
        0,&\left|x\right|>\frac{n+1}{2},\end{array}\right.

    with :math:`\varsigma^{n-1}` a
    :ref:`polynomial simple element<polynomial_simple_element>` of degree
    :math:`\left(n-1\right),` with
    :math:`\xi=\left(\frac{n-1}{2}-x\right),`
    :math:`r=\left\lceil\xi\right\rceil\in{\mathbb{Z}},` and
    :math:`\chi=\left(r-\xi\right)\in[0,1).` Moreover,
    :math:`\left(w_{{\mathrm{d}}}^{n}[r][\cdot]\right)^{{\mathsf{T}}}` is
    the :math:`(r+1)`-th row of the
    :ref:`differentiated-B-spline evaluation matrix<wd>`, and
    :math:`{\mathbf{v}}^{n-1}(\chi)` is the
    :ref:`Vandermonde vector<def-vandermonde_vector>` of argument
    :math:`\chi` and degree :math:`\left(n-1\right).`

    As computed above, the fact that the Vandermonde vector has the domain
    :math:`[0,1)` greatly favors numerical stability since the range of each
    of its components is :math:`[0,1].`

    Parameters
    ----------
    x : float
        Argument.
    n : int
        Nonnegative degree of the polynomial B-spline.

    Returns
    -------
    float
        The value of the gradient of a B-spline.

    Examples
    --------
    Load the library.
        >>> import splinekit as sk
    Value of the differentiated linear B-spline at the origin.
        >>> sk.grad_b_spline(0.0, 1)
        0.0
    Value of the differentiated linear B-spline at ``-0.01``.
        >>> sk.grad_b_spline(-0.01, 1)
        1.0
    Value of the differentiated linear B-spline at ``0.01``.
        >>> sk.grad_b_spline(0.01, 1)
        -1.0
    Value of the differentiated B-spline of degree ``0`` at ``0.5``.
        >>> sk.grad_b_spline(0.5, 0)
        nan
    """

    if 0 > n:
        raise ValueError("The degree n must be nonnegative")
    if 0.5 * (n + 1.0) < abs(x):
        return 0.0
    if n <= 1:
        return fsum(
            (-1) ** q * comb(n + 1, q) * _pse( x + 0.5 * (n + 1.0) - q, n - 1)
            for q in range(n + 1 + 1)
        )
    if 0.5 * (n + 1.0) == abs(x):
        return 0.0
    xi = 0.5 * (n - 1.0) - x
    xi0 = ceil(xi)
    return float(
        np.array(
            [_wd(n)[int(xi0)][c] for c in range(n)],
            dtype = float
        ) @ np.vander([xi0 - xi], n, increasing = True)[0]
    )

#---------------
def diff_b_spline (
    x: float,
    *,
    degree: int,
    differentiation_order: int
) -> float:

    r"""
    .. _diff_b_spline:

    Differentiated B-spline :math:`\nabla^{d}\beta^{n}.`

    Returns the value
    :math:`\frac{{\mathrm{d}}^{d}{\beta}^{n}(x)}{{\mathrm{d}}x^{d}}` of the
    :math:`d`-th derivative of the :ref:`polynomial B-spline<def-b_spline>`
    :math:`\beta^{n}` of :ref:`nonnegative<def-negative>` degree :math:`n`
    evaluated at the argument :math:`x.` Its general case is computed as

    ..  math::

        \frac{{\mathrm{d}}^{d}{\beta}^{n}(x)}{{\mathrm{d}}x^{d}}=
        d!\,w^{n}[r][d]+\sum_{c=d+1}^{n}\,\frac{c!}{\left(c-d\right)!}\,
        w^{n}[r][c]\,\chi^{c-d},

    with :math:`\xi=\left(\frac{n-1}{2}-x\right),`
    :math:`r=\left\lceil\xi\right\rceil\in{\mathbb{Z}},` and
    :math:`\chi=\left(r-\xi\right)\in[0,1).` Moreover,
    :math:`w^{n}[r][c]` is the :math:`(r+1)`-th row and
    :math:`(c+1)`-th column component of the
    :ref:`B-spline evaluation matrix<w_frac>`.

    The :math:`d`-th derivative of a B-spline of :ref:`nonnegative<def-negative>` degree
        - is discontinuous for :math:`d=n;`
        - vanishes almost everywhere for :math:`d\geq n+1;`
        - is undefined at the :ref:`knots<def-knots>` of the B-spline for :math:`d\geq n+1.`

    The :math:`d`-th derivative of a B-spline of :ref:`positive<def-positive>` degree
        - is continuous and non-differentiable for :math:`d=\left(n-1\right).`

    The :math:`d`-th derivative of a B-spline of degree :math:`n\geq2`
        - is continuously differentiable for :math:`d\in[0\ldots n-2].`

    As computed above, the fact that the variable :math:`\chi` has the domain
    :math:`[0,1)` greatly favors numerical stability.

    Parameters
    ----------
    x : float
        Argument.
    degree : int
        Nonnegative degree of the polynomial B-spline.
    differentiation_order : int
        Nonnegative order of differentiation.

    Returns
    -------
    float
        The value of a differentiated B-spline.

    Examples
    --------
    Load the library.
        >>> import splinekit as sk
    Value of the Hessian of the linear B-spline at the origin.
        >>> sk.diff_b_spline(0.0, degree = 1, differentiation_order = 2)
        nan
    Value of the Hessian of the quadratic B-spline at the origin.
        >>> sk.diff_b_spline(0.0, degree = 2, differentiation_order = 2)
        -2.0
    Value of the Hessian of the cubic B-spline at the origin.
        >>> sk.diff_b_spline(0.0, degree = 3, differentiation_order = 2)
        -2.0
    Value of the Hessian of the quartic B-spline at the origin.
        >>> sk.diff_b_spline(0.0, degree = 4, differentiation_order = 2)
        -1.25
    """

    if 0 > degree:
        raise ValueError("Degree must be nonnegative")
    if 0 > differentiation_order:
        raise ValueError("Differentiation_order must be nonnegative")
    if 0 == differentiation_order:
        return b_spline(x, degree)
    if 1 == differentiation_order:
        return grad_b_spline(x, degree)
    if 0.5 * (degree + 1.0) < abs(x):
        return 0.0
    if differentiation_order < degree:
        xi = 0.5 * (degree - 1.0) - x
        xi0 = ceil(xi)
        if 0.5 * (degree + 1.0) == abs(x):
            return 0.0
        return float(
            np.array(
                [
                    _w(degree)[int(xi0)][c] *
                        factorial(c) / factorial(c - differentiation_order)
                    for c in range(differentiation_order, degree + 1)
                ],
                dtype = float
            ) @ np.vander(
                [xi0 - xi],
                degree - differentiation_order + 1,
                increasing = True
            )[0]
        )
    if (differentiation_order == degree) and not x in _knots(degree):
        xi = 0.5 * (degree - 1.0) - x
        xi0 = ceil(xi)
        return float(factorial(degree) * _w(degree)[int(xi0)][degree])
    return _db(x, degree, differentiation_order)

#---------------
@functools.lru_cache(maxsize = 64)
def mscale_filter (
    *,
    degree: int,
    scale: int
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:

    r"""
    .. _mscale_filter:

    M-scale filter of the B-spline :math:`\beta^{n}.`

    Returns a ``numpy`` one-dimensional array of the nonzero taps of the
    filter :math:`h^{n}_{M}` that appears in the
    :ref:`M-scale relation<def-m_scale_relation>` of B-splines expressed as

    ..  math::

        \beta^{n}(x)=\frac{1}{M^{n}}\,\sum_{k=0}^{\left(M-1\right)\,
        \left(n+1\right)}\,h^{n}_{M}[k]\,\beta^{n}(M\,x+
        \frac{\left(M-1\right)\,\left(n+1\right)}{2}-k),

    where :math:`\beta^{n}` is the :ref:`polynomial B-spline<db_frac>` of
    :ref:`nonnegative<def-negative>` degree :math:`n` and :math:`M` is the
    :ref:`positive<def-positive>` scale.

    Parameters
    ----------
    degree : int
        Nonnegative degree of the B-spline.
    scale : int
        Positive scale.

    Returns
    -------
    np.ndarray[tuple[int], np.dtype[np.float64]]
        The nonzero taps of the M-scale filter.

    Examples
    --------
    Load the library.
        >>> import splinekit as sk
    A linear B-spline is a sum of three linear B-splines of half the size.
        >>> sk.mscale_filter(degree = 1, scale = 2)
        array([1., 2., 1.])
    A quadratic B-spline is a sum of thirteen quadratic B-splines of fifth the size.
        >>> sk.mscale_filter(degree = 2, scale = 5)
        array([ 1., 3., 6., 10., 15., 18., 19., 18., 15., 10., 6., 3., 1.])

    Notes
    -----
    The results of this method are cached. If the returned results are
    mutated, the cache gets modified and the next call will return corrupted
    values.
    """

    if 0 > degree:
        raise ValueError("Degree must be nonnegative")
    if 1 > scale:
        raise ValueError("Scale must be positive")
    if 1 == scale:
        return cast(
            np.ndarray[tuple[int], np.dtype[np.float64]],
            np.array([1.0], dtype = float)
        )
    return cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.array(
            sympy.poly(
                sum(sympy.symbols("x") ** k for k in range(scale - 1 + 1)) **
                    (degree + 1),
                sympy.symbols("x")
            ).all_coeffs(),
            dtype = float
        )
    )

#---------------
@functools.lru_cache(maxsize = 128)
def ib_coeff (
    k: int,
    n: int
) -> float:

    r"""
    .. _ib_coeff:

    Inverse B-spline sequence :math:`\left(b^{n}\right)^{-1}.`

    Returns the :math:`k`-th component of the sequence
    :math:`\left(b^{n}\right)^{-1}` such that

    ..  math::

        {\mathbf{[\![}}q=0\,{\mathbf{]\!]}}=\sum_{k\in{\mathbb{Z}}}\,
        \left(b^{n}\right)^{-1}[k]\,\beta^{n}(q-k),

    where the notation :math:`{\mathbf{[\![}}\cdot\,{\mathbf{]\!]}}` is that
    of the :ref:`Iverson bracket<def-iverson>` and
    where :math:`\beta^{n}` is the :ref:`polynomial B-spline<db_frac>` of
    :ref:`nonnegative<def-negative>` degree :math:`n`.

    For :math:`n\in\{0,1\},` one has that

    ..  math::

        \left(b^{n}\right)^{-1}[k]={\mathbf{[\![}}k=0\,{\mathbf{]\!]}}.

    For :math:`n>1,` the inverse sequence :math:`\left(b^{n}\right)^{-1}` is
    computed as

    ..  math::

        \left(b^{n}\right)^{-1}[k]=
        \left({\mathbf{i}}^{n}\right)^{{\mathsf{T}}}\,
        \left({\mathbf{z}}^{n}\right)^{\left|k\right|},

    where the vector :math:`{\mathbf{i}}^{n}` is defined :ref:`here<iota>`,
    and where the absolute :math:`k`-th power
    :math:`\left({\mathbf{z}}^{n}\right)^{\left|k\right|}` of the vector
    :math:`{\mathbf{z}}^{n}` of :ref:`poles<pole>` must be understood
    component-wise.

    Parameters
    ----------
    k : int
        Index of the coefficient.
    n : int
        Nonnegative degree of the B-spline.

    Returns
    -------
    float
        A component of the inverse B-spline sequence.

    Examples
    --------
    Load the library.
        >>> import splinekit as sk
    The inverse quadratic B-spline sequence at the origin is the square root of two.
        >>> sk.ib_coeff(0, 2) * sk.ib_coeff(0, 2)
        2.0000000000000004

    Notes
    -----
    The results of this method are cached. If the returned results are
    mutated, the cache gets modified and the next call will return corrupted
    values.
    """

    if 0 > n:
        raise ValueError("The degree n must be nonnegative")
    if n <= 1:
        return 1.0 if 0 == k else 0.0
    return float(_iota(n) @ np.array([z ** abs(k) for z in pole(n)]))

#---------------
def interpolating_b_spline (
    x: float,
    degree: int
) -> float:

    r"""
    .. _interpolating_b_spline:

    Interpolating B-spline :math:`\eta^{n}.`

    Returns the value of the
    :ref:`interpolating polynomial B-spline<def-interpolating_b_spline>`
    :math:`\eta^{n}` of :ref:`nonnegative<def-negative>` degree :math:`n`
    evaluated at the argument :math:`x.` This function is defined as

    ..  math::

        \eta^{n}(x)=\sum_{k\in{\mathbb{Z}}}\,\left(b^{n}\right)^{-1}[k]\,
        \beta^{n}(x-k),

    with :math:`\left(b^{n}\right)^{-1}` a B-spline
    :ref:`inverse sequence<ib_coeff>` and :math:`\beta^{n}` a
    :ref:`polynomial B-spline<db_frac>`.

    Parameters
    ----------
    x : float
        Argument.
    n : int
        Nonnegative degree of the interpolating polynomial B-spline.

    Returns
    -------
    float
        The value of an interpolating B-spline.

    Examples
    --------
    Load the library.
        >>> import splinekit as sk
    Values of the interpolating quadratic B-spline at a few integers.
        >>> [sk.interpolating_b_spline(k, 2) for k in range(-3, 3)]
        [-3.230922474006803e-17,
         1.9949319973733282e-16,
         1.0000000000000002,
         1.9949319973733282e-16,
         -3.230922474006803e-17]
    Values of the interpolating quadratic B-spline at a few half integers.
        >>> [sk.interpolating_b_spline(k + 0.5, 2) for k in range(-3, 3)]
        [0.017243942703102966,
         -0.10050506338833456,
         0.5857864376269051,
         0.5857864376269051,
         -0.10050506338833456,
         0.017243942703102966]
    """

    if 0 > degree:
        raise ValueError("Degree must be nonnegative")
    k0 = ceil(x - 0.5 * (degree + 1))
    return fsum(
        ib_coeff(k, degree) * b_spline(x - k, degree)
        for k in range(k0, k0 + degree + 1)
    )

#---------------
def dual_b_spline (
    x: float,
    *,
    dual_degree: int,
    primal_degree: int
) -> float:

    r"""
    .. _dual_b_spline:

    Dual B-spline :math:`\mathring{\beta}^{m,n}.`

    Returns the value of the
    :ref:`polynomial dual B-spline<def-dual_b_spline>`
    :math:`\mathring{\beta}^{m,n}` of :ref:`nonnegative<def-negative>` dual
    degree :math:`m` and nonnegative primal degree :math:`n,` evaluated at the
    argument :math:`x.` This function is computed as

    ..  math::

        \mathring{\beta}^{m,n}(x)=\sum_{k\in{\mathbb{Z}}}\,
        \left(b^{m+n+1}\right)^{-1}[k]\,\beta^{m}(x-k),

    with :math:`\left(b^{m+n+1}\right)^{-1}` a B-spline
    :ref:`inverse sequence<ib_coeff>` and :math:`\beta^{m}` a
    :ref:`polynomial B-spline<db_frac>`.

    Parameters
    ----------
    x : float
        Argument.
    m : int
        Nonnegative dual degree of the polynomial dual B-spline.
    n : int
        Nonnegative primal degree of the polynomial dual B-spline.

    Returns
    -------
    float
        The value of a dual B-spline.

    Examples
    --------
    Load the library.
        >>> import splinekit as sk
    Value of the (dual-five, primal-two) B-spline at the origin.
        >>> sk.dual_b_spline(0.0, dual_degree = 5, primal_degree = 2)
        1.7451037781381014
    """

    if 0 > dual_degree:
        raise ValueError("The dual degree must be nonnegative")
    if 0 > primal_degree:
        raise ValueError("The primal degree must be nonnegative")
    k0 = ceil(x - 0.5 * (dual_degree + 1))
    return fsum(
        ib_coeff(k, dual_degree + primal_degree + 1) *
            b_spline(x - k, dual_degree)
        for k in range(k0, k0 + dual_degree + 1)
    )

#---------------
@functools.lru_cache(maxsize = 128)
def spline_polynomial (
    *,
    spline_degree: int,
    monomial_degree: int
) -> sympy.poly:

    r"""
    .. _spline_polynomial:

    Sum of monomial-weighted B-splines :math:`\sum_{k}\,k^{m}\beta^{n}.`

    Returns as a ``sympy.poly`` object the :ref:`polynomial<def-polynomial>`
    generated by the B-spline-weighted sum of the discrete
    :ref:`monomials<def-monomial>` of :ref:`nonnegative<def-negative>` degree
    :math:`m,` with :math:`n` the degree of the :ref:`B-spline<def-b_spline>` 
    such that :math:`n\geq m.`

    The :ref:`partition of unity<def-partition_of_unity>` is the sum of
    discrete constant monomials. It is

    ..  math::

        \sum_{k\in{\mathbb{Z}}}\,\beta^{n}(x-k)=\textcolor{green}{1}.

    B-spline-weighted sums of discrete monomials can be computed for positive
    :ref:`odd<def-odd>` monomial degree :math:`m` as

    ..  math::

        \sum_{k\in{\mathbb{Z}}}\,\beta^{n}(x-k)\,k^{m}=
        \textcolor{green}{\sum_{c=0}^{(m-1)/2}\,v^{n}[m-1][\frac{m-1}{2}-c]\,
        x^{2\,c+1}},

    ..  math::

    while B-spline-weighted sums of discrete monomials can be computed for
    positive :ref:`even<def-even>` monomial degree :math:`m` as

    ..  math::

        \sum_{k\in{\mathbb{Z}}}\,\beta^{n}(x-k)
        \,k^{m}=\textcolor{green}{v^{n}[m-1][\frac{m}{2}]+\sum_{c=1}^{m/2}\,
        v^{n}[m-1][\frac{m}{2}-c]\,x^{2\,c}}.

    There, :math:`v^{n}` is the lookup table described :ref:`here<v_frac>`.

    Parameters
    ----------
    spline_degree : int
        Nonnegative degree of the weighting polynomial B-spline.
    monomial_degree : int
        Nonnegative degree of the weighted monomials.

    Returns
    -------
    sympy.poly
        The polynomial in :math:`x` of the
        :raw-html:`<font color="green">right-hand side</font>` of the equations
        above.

    Examples
    --------
    Load the library.
        >>> import splinekit as sk
    Polynomial generated by weighting cubic discrete monomials by octic B-splines.
        >>> print(sk.spline_polynomial(spline_degree = 8, monomial_degree = 3))
        Poly(x**3 + 9/4*x, x, domain='QQ')

    Notes
    -----
    The results of this method are cached. If the returned results are
    mutated, the cache gets modified and the next call will return corrupted
    values.
    """

    if 0 > monomial_degree:
        raise ValueError("The monomial degree must be nonnegative")
    if spline_degree < monomial_degree:
        raise ValueError(
            "The monomial degree must not exceed the spline degree"
        )
    v = [
        1 if 0 == c else comb(monomial_degree, 2 * c)
            * sum(
                ((Fraction(spline_degree - 1, 2) - k) ** (2 * c)) *
                    _db_frac(
                        Fraction(spline_degree - 1, 2) - k,
                        spline_degree,
                        0
                    )
                for k in range(spline_degree - 1 + 1)
            )
        for c in range(spline_degree // 2 + 1)
    ]
    if 0 == monomial_degree % 2:
        return sympy.poly(
            np.ravel([[0, v[p]] for p in range(monomial_degree // 2 + 1)]) @
                np.vander([sympy.symbols("x")], monomial_degree + 2)[0],
            sympy.symbols("x"),
            domain = "QQ"
        )
    return sympy.poly(
        np.ravel([[v[p], 0] for p in range(monomial_degree // 2 + 1)]) @
            np.vander([sympy.symbols("x")], monomial_degree + 1)[0],
        sympy.symbols("x"),
        domain = "QQ"
    )

#---------------
@functools.lru_cache(maxsize = 128)
def partition_of_monomial (
    *,
    spline_degree: int,
    monomial_degree: int
) -> sympy.poly:

    r"""
    .. _partition_of_monomial:

    Coefficients :math:`\alpha` of
    :math:`x^{m}=\sum_{k}\,\alpha[k]\,\beta^{n}(x-k).`

    Returns as a ``sympy.poly`` object the expression of the
    discrete :ref:`polynomial<def-polynomial>` that, once weighted by a
    :ref:`polynomial B-spline<def-b_spline>` of
    :ref:`nonnegative<def-negative>` degree :math:`n` and summed, will
    represent a :ref:`monomial<def-monomial>` of nonnegative degree :math:`m.`
    The degree of the B-spline and of the monomial must be such that
    :math:`n\geq m.`

    The :ref:`partition of unity<def-partition_of_unity>` is the sum of
    discrete constant monomials. It is

    ..  math::

        1=\sum_{k\in{\mathbb{Z}}}\,\left(\textcolor{green}{1}\right)\,
        \beta^{n}(x-k).

    Monomials for positive odd :math:`m` can be reproduced as

    ..  math::

        x^{m}=\sum_{k\in{\mathbb{Z}}}\,
        \left(\textcolor{green}{\sum_{p=0}^{\left(m-1\right)/2}\,
        \alpha^{n}[m][2\,p+1]\,k^{2\,p+1}}\right)\,\beta^{n}(x-k),

    ..  math::

    while monomials for positive even :math:`m` can be reproduced as

    ..  math::

        x^{m}=\sum_{k\in{\mathbb{Z}}}\,
        \left(\textcolor{green}{\alpha^{n}[m][0]+\sum_{p=1}^{m/2}\,
        \alpha^{n}[m][2\,p]\,k^{2\,p}}\right)\,\beta^{n}(x-k).

    There, :math:`\alpha^{n}` is the lookup table described
    :ref:`here<alpha_frac>`.

    Parameters
    ----------
    spline_degree : int
        Nonnegative degree of the weighting polynomial B-spline.
    monomial_degree : int
        Nonnegative degree of the generated monomial.

    Returns
    -------
    sympy.poly
        The discrete polynomial in :math:`k` in the parenthesis of the
        :raw-html:`<font color="green">right-hand side</font>` of the equations
        above.

    Examples
    --------
    Load the library.
        >>> import splinekit as sk
    Discrete polynomial as the coefficient needed to generate a quadratic monomial from quartic B-splines.
        >>> print(sk.partition_of_monomial(spline_degree = 4, monomial_degree = 2))
        Poly(k**2 - 5/12, k, domain='QQ')

    Notes
    -----
    The results of this method are cached. If the returned results are
    mutated, the cache gets modified and the next call will return corrupted
    values.
    """

    if 0 > spline_degree:
        raise ValueError("the spline degree must be nonnegative")
    if 0 > monomial_degree:
        raise ValueError("the monomial degree must be nonnegative")
    if spline_degree < monomial_degree:
        raise ValueError(
            "The monomial degree must not exceed the spline degree"
        )
    if 0 == monomial_degree % 2:
        return sympy.poly(
            np.ravel([
                [
                    Fraction(0, 1),
                    _alpha_frac(
                        spline_degree,
                        monomial_degree
                    )[monomial_degree - p]
                ]
                for p in range(0, monomial_degree + 1, 2)
            ]) @ np.vander([sympy.symbols("k")], monomial_degree + 2)[0],
            sympy.symbols("k"),
            domain = "QQ"
        )
    return sympy.poly(
        np.ravel([
            [
                _alpha_frac(
                    spline_degree,
                    monomial_degree
                )[monomial_degree - p],
                Fraction(0, 1)
            ]
            for p in range(0, monomial_degree, 2)
        ]) @ np.vander([sympy.symbols("k")], monomial_degree + 1)[0],
        sympy.symbols("k"),
        domain = "QQ"
    )

#---------------
@functools.lru_cache(maxsize = 128)
def convolve_b_spline_monomial (
    *,
    spline_degree: int,
    spline_differentiation_order: int = 0,
    monomial_degree: int
) -> sympy.poly:

    r"""
    .. _convolve_b_spline_monomial:

    Convolution :math:`\nabla^{d}\beta^{n}*\left(\cdot\right)^{m}.`

    Returns  as a ``sympy.poly`` object the polynomial that results from the 
    convolution betweeen the :math:`d`-th derivative of a
    :ref:`polynomial B-spline<def-b_spline>` of
    :ref:`nonnegative<def-negative>` degree :math:`n` and a monomial of
    nonnegative degree :math:`m`. The order of differentiation must be such
    that :math:`d\in[0\ldots n+1+m].`

    ..  math::

        \begin{eqnarray*}
        \left(\left(\beta^{n}\right)^{(d)}*\left(\cdot\right)^{m}\right)(x)&=&
        \int_{{\mathbb{R}}}\,
        \frac{{\mathrm{d}}^{d}\beta^{n}(y)}{{\mathrm{d}}y^{d}}\,
        \left(x-y\right)^{m}\,{\mathrm{d}}y\\
        &=&\frac{m!}{\left(m+n+1-d\right)!}\,\sum_{k=0}^{n+1}\,
        \left(-1\right)^{k}\,{n+1\choose k}\,
        \left(x+\frac{n+1}{2}-k\right)^{m+n+1-d}\\
        &=&\textcolor{green}{\gamma_{m}^{n}[0]+
        \sum_{k=1}^{m}\,\gamma_{m}^{n}[k]\,x^{k}}.
        \end{eqnarray*}

    Parameters
    ----------
    spline_degree : int
        Nonnegative degree of the polynomial B-spline.
    spline_differentiation_order : int
        Nonnegative differentiation order of the polynomial B-spline (must not
        exceed :math:`m+n+1`).
    monomial_degree : int
        Nonnegative degree of the monomial.

    Returns
    -------
    sympy.poly
        The polynomial in :math:`x` in the
        :raw-html:`<font color="green">right-hand side</font>` of the equation
        above.

    Examples
    --------
    Load the library.
        >>> import splinekit as sk
    Polynomial obtained by convolving a linear B-spline with a cubic monomial.
        >>> print(sk.convolve_b_spline_monomial(spline_degree = 1, monomial_degree = 3))
        Poly(x**3 + 1/2*x, x, domain='QQ')
    Polynomial obtained by convolving the Hessian of a B-spline of degree zero with a quartic monomial.
        >>> print(sk.convolve_b_spline_monomial(spline_degree = 0, spline_differentiation_order = 2, monomial_degree = 4))
        Poly(12*x**2 + 1, x, domain='QQ')

    Notes
    -----
    The results of this method are cached. If the returned results are
    mutated, the cache gets modified and the next call will return corrupted
    values.
    """

    if 0 > spline_degree:
        raise ValueError("The spline degree must be nonnegative")
    if 0 > spline_differentiation_order:
        raise ValueError("The spline differentiation order must be nonnegative")
    if 0 > monomial_degree:
        raise ValueError("The monomial degree must be nonnegative")
    if spline_differentiation_order > spline_degree + 1 + monomial_degree:
        raise ValueError(
            "The differentiation order must not exceed (spline_degree + 1 + monomial_degree)"
        )
    return sympy.poly(
        Fraction(
            factorial(monomial_degree),
            factorial(monomial_degree + spline_degree + 1 -
                spline_differentiation_order)
        ) * sum(
            ((-1) ** k) * comb(spline_degree + 1, k) * ((sympy.symbols("x") +
                Fraction(spline_degree + 1, 2) - k) ** (monomial_degree +
                spline_degree + 1 - spline_differentiation_order))
            for k in range(spline_degree + 2)
        ),
        sympy.symbols("x"),
        domain = "QQ"
    )
