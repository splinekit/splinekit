"""
The :class:`splinekit.spline_utilities` module provides the building blocks
and low-level methods associated to the efficient and stable computation of
polynomial B-splines of arbitrary degree.
"""

#---------------
from __future__ import annotations

#---------------
from typing import cast
from typing import Dict
from typing import List
from typing import Tuple

#---------------
from fractions import Fraction

#---------------
from math import acos
from math import ceil
from math import comb
from math import cos
from math import factorial
from math import floor
from math import fsum
from math import sin
from math import sqrt
from math import ulp

#---------------
import functools

#---------------
import numpy as np
import sympy

#---------------
def _sgn_frac (
    x: Fraction
) -> int:

    r"""
    .. _sgn_frac:

    Signum function :math:`{\mathrm{sgn}}.`

    Returns as an integer the :ref:`signum<def-sgn>` of the rational argument
    :math:`x.`

    Parameters
    ----------
    x : Fraction
        Rational argument.

    Returns
    -------
    int
        The signum of :math:`x.`
    """

    return 1 if 0 < x else (-1 if 0 > x else 0)

#---------------
def _sgn (
    x: float
) -> float:

    r"""
    .. _sgn:

    Signum function :math:`{\mathrm{sgn}}.`

    Returns the :ref:`signum<def-sgn>` of the argument :math:`x.`

    Parameters
    ----------
    x : float
        Argument.

    Returns
    -------
    float
        The signum of :math:`x.`
    """

    return 1.0 if 0.0 < x else (-1.0 if 0.0 > x else 0.0)

#---------------
def _pse_frac (
    x: Fraction,
    n: int
) -> Fraction:

    r"""
    .. _pse_frac:

    Polynomial simple element :math:`\varsigma^{n}.`

    Returns the rational value of a
    :ref:`polynomial simple element<polynomial_simple_element>` of integer
    degree :math:`n` evaluated at the rational argument :math:`x.`

    Parameters
    ----------
    x : Fraction
        Rational argument.
    n : int
        Degree of the polynomial simple element.

    Returns
    -------
    Fraction
        The rational value of the polynomial simple element at :math:`x.`
    """

    if 0 == x:
        if 0 <= n:
            return Fraction(0, 1)
        return Fraction(0, 0)
    if n < 0:
        return Fraction(0, 1)
    if 0 == n:
        return Fraction(_sgn_frac(x), 2)
    return Fraction(_sgn_frac(x) * x ** n, 2 * factorial(n))

#---------------
def _pse (
    x: float,
    n: int
) -> float:

    r"""
    .. _pse:

    Polynomial simple element :math:`\varsigma^{n}.`

    See Also
    --------
    splinekit.bsplines.polynomial_simple_element : Polynomial simple element.
    """

    if 0.0 == x:
        if 0 <= n:
            return 0.0
        return float("nan")
    if n < 0:
        return 0.0
    if 0 == n:
        return 0.5 * _sgn(x)
    return _sgn(x) * x ** n / float(2 * factorial(n))

#---------------
def _db_frac (
    x: Fraction,
    n: int,
    d: int
) -> Fraction:

    r"""
    .. _db_frac:

    Differentiated B-spline :math:`\nabla^{d}\beta^{n}.`

    Returns the rational value of the :math:`d`-th derivative of a
    :ref:`polynomial B-spline<def-b_spline>` :math:`\beta^{n}` of
    :ref:`nonnegative<def-negative>` degree :math:`n` evaluated at the
    rational argument :math:`x,` with :math:`d` nonnegative. This function is
    computed as

    ..  math::

        \frac{{\mathrm{d}}^{d}\beta^{n}(x)}{{\mathrm{d}}x^{d}}=
        \sum_{k=0}^{n+1}\,\left(-1\right)^{k}\,{n+1\choose k}\,\varsigma^{n-d}
        (x+\frac{n+1}{2}-k),

    with :math:`\varsigma` a
    :ref:`polynomial simple element<polynomial_simple_element>`.

    Parameters
    ----------
    x : Fraction
        Rational argument.
    n : int
        Nonnegative degree of the polynomial B-spline.
    d : int
        Nonnegative order of differentiation.

    Returns
    -------
    Fraction
        The rational value of the :math:`d`-th derivative of a B-spline at
        rational :math:`x.`
    """

    return sum(
        (-1) ** q * comb(n + 1, q) *
            _pse_frac(x + Fraction(n + 1, 2) - q, n - d)
        for q in range(n + 1 + 1)
    )

#---------------
def _db (
    x: float,
    n: int,
    d: int
) -> float:

    r"""
    .. _db:

    Differentiated B-spline :math:`\nabla^{d}\beta^{n}.`

    Returns the value of the :math:`d`-th derivative of a
    :ref:`polynomial B-spline<db_frac>` of :ref:`nonnegative<def-negative>`
    degree :math:`n` evaluated at :math:`x,` with :math:`d` nonnegative.

    Parameters
    ----------
    x : float
        Argument.
    n : int
        Nonnegative degree of the polynomial B-spline.
    d : int
        Nonnegative order of differentiation.

    Returns
    -------
    float
        The value of the :math:`d`-th derivative of a B-spline at :math:`x.`
    """

    if 0.5 * (n + 1.0) < abs(x):
        return 0.0
    return fsum(
        (-1) ** q * comb(n + 1, q) *
            _pse(x + 0.5 * (n + 1) - q, n - d)
        for q in range(n + 1 + 1)
    )

#---------------
_bn: Dict[int, np.ndarray[tuple[int], np.dtype[np.float64]]] = {
    0: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.array([Fraction(1, 1)], dtype = float)
    ),
    1: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.array([Fraction(1, 1)], dtype = float)
    ),
    2: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.array(
            [Fraction(1, 8), Fraction(3, 4), Fraction(1, 8)],
            dtype = float
        )
    ),
    3: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.array(
            [Fraction(1, 6), Fraction(2, 3), Fraction(1, 6)],
            dtype = float
        )
    ),
    4: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.array(
            [
                Fraction(1, 384),
                Fraction(19, 96),
                Fraction(115, 192),
                Fraction(19, 96),
                Fraction(1, 384)
            ],
            dtype = float
        )
    ),
    5: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.array(
            [
                Fraction(1, 120),
                Fraction(13, 60),
                Fraction(11, 20),
                Fraction(13, 60),
                Fraction(1, 120)
            ],
            dtype = float
        )
    )
}

#---------------
def _b (
    n: int
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:

    r"""
    .. _b:

    B-spline discrete sequence :math:`{\mathrm{b}}^{n}.`

    Returns a ``numpy`` one-dimensional array of the nonvanishing values
    :math:`\beta^{n}(k)` of a :ref:`polynomial B-spline<db_frac>`
    :math:`\beta^{n}` of :ref:`nonnegative<def-negative>` degree :math:`n`
    sampled at the integers :math:`k\in[-\left\lfloor n/2\right\rfloor\ldots
    \left\lfloor n/2\right\rfloor].`

    Parameters
    ----------
    n : int
        Nonnegative degree of the polynomial B-spline.

    Returns
    -------
    np.ndarray[tuple[int], np.dtype[np.float64]]
        The samples of the B-spline.
    """

    if n in _bn:
        return _bn[n]
    _bn[n] = cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.array(
            [
                _db_frac(Fraction(k, 1), n, 0)
                for k in range(-(n // 2), n // 2 + 1)
            ],
            dtype = float
        )
    )
    return _bn[n]

#---------------
@functools.lru_cache(maxsize = 32)
def _b_dft_expression (
    n: int,
    *,
    fourier_idx: sympy.symbols = sympy.symbols("nu"),
    data_len: sympy.symbols = sympy.symbols("K")
) -> sympy.Function:

    r"""
    .. _b_dft_expression:

    Expression of the :math:`\nu`-th term of the DFT of the
    periodized :ref:`B-spline discrete sequence<b>` :math:`{\mathrm{b}}^{n}.`

    Returns a ``sympy`` expression of the :math:`\nu`-th term of the
    discrete Fourier transform of the :math:`K`-periodized one-dimensional
    array of the sampled values

    ..  math::

        b_{{\mathrm{per}},K}^{n}[k]=\sum_{p\in{\mathbb{Z}}}\,\beta^{n}(p\,K+k)

    of a :ref:`polynomial B-spline<db_frac>` :math:`\beta^{n}` of
    :ref:`nonnegative<def-negative>` degree :math:`n,` for
    :math:`\nu\in[1\ldots K-1].` The :math:`\nu`-th term is

    ..  math::

        \begin{eqnarray*}
        \hat{b}_{{\mathrm{per}},K}^{n}[\nu]&=&\sum_{k=0}^{K-1}\,
        b_{{\mathrm{per}},K}^{n}[k]\,
        {\mathrm{e}}^{{\mathrm{j}}\,\nu\,\frac{2\,\pi}{K}\,k}\\
        &=&\left\{\begin{array}{ll}1,&\nu=0\\
        \left.\frac{{\mathrm{d}}^{n}\csc\omega}{{\mathrm{d}}\omega^{n}}\right|
        _{\omega=\nu\,\frac{\pi}{K}}\,\frac{1}{n!}\,
        \sin^{n+1}(\nu\,\frac{\pi}{K}),&\nu\in[1\ldots K-1]\wedge
        n\in2\,{\mathbb{N}}\\
        \left.-\frac{{\mathrm{d}}^{n}\cot\omega}{{\mathrm{d}}\omega^{n}}\right|
        _{\omega=\nu\,\frac{\pi}{K}}\,\frac{1}{n!}\,
        \sin^{n+1}(\nu\,\frac{\pi}{K}),&\nu\in[1\ldots K-1]\wedge
        n\in2\,{\mathbb{N}}+1.\end{array}\right.
        \end{eqnarray*}

    Parameters
    ----------
    n : int
        Nonnegative degree of the polynomial B-spline.

    Returns
    -------
    sympy.Function
        The expression of the :math:`\nu`-th term of the DFT.

    Notes
    -----
    The results of this method are cached. If the returned results are
    mutated, the cache gets modified and the next call will return corrupted
    values.
    """

    omega = sympy.symbols("omega")
    term = (sympy.sin(omega) ** (n+1) / factorial(n))
    if 0 == n % 2:
        term *= sympy.diff(sympy.csc(omega), omega, n)
    else:
        term *= -sympy.diff(sympy.cot(omega), omega, n)
    return term.subs(omega, fourier_idx * sympy.pi / data_len)

#---------------
_poles: Dict[int, np.ndarray[tuple[int], np.dtype[np.float64]]] = {
    0: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.array([], dtype = float)
    ),
    1: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.array([], dtype = float)
    ),
    2: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray([-3.0 + sqrt(8.0)], dtype = float)
    ),
    3: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray([-2.0 + sqrt(3.0)], dtype = float)
    ),
    4: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -19.0 + sqrt(304.0) + sqrt(664.0 - sqrt(438976.0)),
                -19.0 - sqrt(304.0) + sqrt(664.0 + sqrt(438976.0))
            ],
            dtype = float
        )
    ),
    5: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -13.0 / 2.0 + sqrt(105.0 / 4.0) +
                    sqrt(135.0 / 2.0 - sqrt(17745.0 / 4.0)),
                -13.0 / 2.0 - sqrt(105.0 / 4.0) +
                    sqrt(135.0 / 2.0 + sqrt(17745.0 / 4.0))
            ],
            dtype = float
        )
    )
}

#---------------
def _pole (
    n: int
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:

    r"""
    Poles of the B-spline :math:`\beta^{n}.`

    See Also
    --------
    splinekit.bsplines.pole : Poles of polynomial B-splines.
    """

    if 0 > n:
        raise ValueError("The degree n must be nonnegative")
    if n in _poles:
        return _poles[n]
    if 6 == n:
        theta = acos(-668791.0 / sqrt(447872715451.0)) / 3.0
        c = sqrt(122416.0 / 9.0) * cos(theta)
        s = sqrt(122416.0 / 3.0) * sin(theta)
        lmbda1 = 361.0 / 3.0 - 2.0 * c
        lmbda2 = 361.0 / 3.0 + c - s
        lmbda3 = 361.0 / 3.0 + c + s
        _poles[n] = cast(
            np.ndarray[tuple[int], np.dtype[np.float64]],
            np.ascontiguousarray(
                [
                    sqrt(lmbda1 ** 2 - 1.0) - lmbda1,
                    sqrt(lmbda2 ** 2 - 1.0) - lmbda2,
                    sqrt(lmbda3 ** 2 - 1.0) - lmbda3
                ],
                dtype = float
            )
        )
        return _poles[n]
    if 7 == n:
        theta = acos(-738.0 / sqrt(556549.0)) / 3.0
        c = sqrt(301) * cos(theta)
        s = sqrt(903.0) * sin(theta)
        lmbda1 = 20.0 - 2.0 * c
        lmbda2 = 20.0 + c - s
        lmbda3 = 20.0 + c + s
        _poles[n] = cast(
            np.ndarray[tuple[int], np.dtype[np.float64]],
            np.ascontiguousarray(
                [
                    sqrt(lmbda1 ** 2 - 1.0) - lmbda1,
                    sqrt(lmbda2 ** 2 - 1.0) - lmbda2,
                    sqrt(lmbda3 ** 2 - 1.0) - lmbda3
                ],
                dtype = float
            )
        )
        return _poles[n]
    if 8 == n:
        theta = acos(sqrt(3191438329707.0 / 3435302785852.0)) / 3.0
        lmbda0 = sqrt(656944.0 + sqrt(106853376.0) * cos(theta))
        lmbda = 1300071.0 / 2.0 - lmbda0 ** 2
        rho = 1638.0 * lmbda0 - sqrt(4.0 * lmbda ** 2 - 250737.0)
        nu1 = sqrt(2641593.0 / 2.0 + lmbda - rho)
        nu2 = sqrt(2641593.0 / 2.0 + lmbda + rho)
        mu1 = -819.0 + lmbda0 + nu1
        mu2 = -819.0 + lmbda0 - nu1
        mu3 = -819.0 - lmbda0 + nu2
        mu4 = -819.0 - lmbda0 - nu2
        _poles[n] = cast(
            np.ndarray[tuple[int], np.dtype[np.float64]],
            np.ascontiguousarray(
                [
                    mu1 + sqrt(mu1 ** 2 - 1.0),
                    mu2 + sqrt(mu2 ** 2 - 1.0),
                    mu3 + sqrt(mu3 ** 2 - 1.0),
                    mu4 + sqrt(mu4 ** 2 - 1.0)
                ],
                dtype = float
            )
        )
        return _poles[n]
    if 9 == n:
        theta = acos(sqrt(607973645.0 / 699281408.0)) / 3.0
        lmbda0 = sqrt(53265.0 / 16.0 + sqrt(146160.0) * cos(theta))
        lmbda = 48397.0 / 16.0 - lmbda0 ** 2
        rho = (251.0 / 2.0) * lmbda0 - sqrt(4.0 * lmbda ** 2 - 7936.0)
        nu1 = sqrt(55699.0 / 8.0 + lmbda - rho)
        nu2 = sqrt(55699.0 / 8.0 + lmbda + rho)
        mu1 = -251.0 / 4.0 + lmbda0 + nu1
        mu2 = -251.0 / 4.0 + lmbda0 - nu1
        mu3 = -251.0 / 4.0 - lmbda0 + nu2
        mu4 = -251.0 / 4.0 - lmbda0 - nu2
        _poles[n] = cast(
            np.ndarray[tuple[int], np.dtype[np.float64]],
            np.ascontiguousarray(
                [
                    mu1 + sqrt(mu1 ** 2 - 1.0),
                    mu2 + sqrt(mu2 ** 2 - 1.0),
                    mu3 + sqrt(mu3 ** 2 - 1.0),
                    mu4 + sqrt(mu4 ** 2 - 1.0)
                ],
                dtype = float
            )
        )
        return _poles[n]
    roots = sympy.real_roots(_db_frac(Fraction(0, 1), n, 0) + 2 *
        sum(
            (-1) ** q * _db_frac(Fraction(2 * q), n, 0)
            for q in range(1, (n // 2) // 2 + 1)
        ) +
        sum(
            Fraction((-1) ** p, factorial(2 * p)) *
                sum(
                    (-1) ** q * Fraction(2 * q + 1, 2 * p + 1) *
                        Fraction(factorial(q + p), factorial(q - p)) *
                        _db_frac(Fraction(2 * q + 1, 1), n, 0)
                    for q in range(p, (n // 2 - 1) // 2 + 1)
                ) * sympy.symbols("x") ** (2 * p + 1)
            for p in range((n // 2) // 2 + 1)
        ) +
        sum(
            Fraction((-1) ** p, factorial(2 * p)) *
                sum(
                    (-1) ** q * Fraction(2 * q, q + p)
                        * Fraction(factorial(q + p), factorial(q - p))
                        * _db_frac(Fraction(2 * q, 1), n, 0)
                    for q in range(p, (n // 2) // 2 + 1)
                ) * sympy.symbols("x") ** (2 * p)
            for p in range(1, (n // 2) // 2 + 1)
        )
    )
    roots = [r.evalf() for r in roots]
    roots = sorted([float(2.0 / (r - sqrt(r * r - 4.0))) for r in roots])
    _poles[n] = cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(roots, dtype = float)
    )
    return _poles[n]

#---------------
_knotsn: Dict[int, np.ndarray[tuple[int], np.dtype[np.float64]]] = {
    0: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.array([Fraction(-1, 2), Fraction(1, 2)], dtype = "float")
    ),
    1: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.array(
            [Fraction(-1, 1), Fraction(0, 1), Fraction(1, 1)],
            dtype = "float"
        )
    ),
    2: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.array(
            [Fraction(-3, 2), Fraction(-1, 2), Fraction(1, 2), Fraction(3, 2)],
            dtype = "float"
        )
    ),
    3: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.array(
            [
                Fraction(-2, 1),
                Fraction(-1, 1),
                Fraction(0, 1),
                Fraction(1, 1),
                Fraction(2, 1)
            ],
            dtype = "float"
        )
    ),
    4: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.array(
            [
                Fraction(-5, 2),
                Fraction(-3, 2),
                Fraction(-1, 2),
                Fraction(1, 2),
                Fraction(3, 2),
                Fraction(5, 2)
            ],
            dtype = "float"
        )
    ),
    5: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.array(
            [
                Fraction(-3, 1),
                Fraction(-2, 1),
                Fraction(-1, 1),
                Fraction(0, 1),
                Fraction(1, 1),
                Fraction(2, 1),
                Fraction(3, 1)
            ],
            dtype = "float"
        )
    )
}

#---------------
def _knots (
    n: int
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:

    r"""
    .. _knots:

    Knots of the B-spline :math:`\beta^{n}.`

    Returns a ``numpy`` one-dimensional array of the :ref:`knots<def-knots>`
    of a :ref:`polynomial B-spline<db_frac>` of
    :ref:`nonnegative<def-negative>` degree :math:`n.`

    Parameters
    ----------
    n : int
        Nonnegative degree of the polynomial B-spline.

    Returns
    -------
    np.ndarray[tuple[int], np.dtype[np.float64]]
        The knots of the B-spline.
    """

    if n in _knotsn:
        return _knotsn[n]
    _knotsn[n] = cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.fromiter(
            (k - Fraction(n + 1, 2) for k in range(n + 1 + 1)),
            dtype = float,
            count = n + 1 + 1
        )
    )
    return _knotsn[n]

#---------------
@functools.lru_cache(maxsize = 32)
def _w_frac (
    n: int
) -> List[List[Fraction]]:

    r"""
    .. _w_frac:

    Spline evaluation matrix :math:`{\mathbf{W}}^{n}.`

    Returns the B-spline rational evaluation matrix
    :math:`{\mathbf{W}}^{n}\in{\mathbb{Q}}^{\left(n+1\right)\times
    \left(n+1\right)}` of a :ref:`polynomial B-spline<db_frac>`
    :math:`\beta^{n}` of :ref:`positive<def-positive>` degree :math:`n,` as
    defined by the rational component at its :math:`(r+1)`-th row and
    :math:`(c+1)`-th column

    ..  math::

        w_{r+1,c+1}^{n}=w^{n}[r][c]=\frac{1}{c!}\,
        \left(\left.\frac{{\mathrm{d}}^{c}\beta^{n}(x)}{{\mathrm{d}}x^{c}}
        \right|_{x=\frac{n-1}{2}-r}+\frac{1}{2}\,
        {\mathbf{[\![}}c=n\,{\mathbf{]\!]}}\,\left(-1\right)^{n-r}\,
        {n+1\choose r+1}\right),

    where the notation :math:`{\mathbf{[\![}}\cdot\,{\mathbf{]\!]}}` is that
    of the :ref:`Iverson bracket<def-iverson>`. The B-spline derivatives are
    computed as in :ref:`_db_frac <db_frac>`.

    Parameters
    ----------
    n : int
        Positive degree of the polynomial B-spline.

    Returns
    -------
    list of list of Fraction
        The B-spline rational evaluation matrix.

    Notes
    -----
    The results of this method are cached. If the returned results are
    mutated, the cache gets modified and the next call will return corrupted
    values.
    """

    return [
        [
            (_db_frac(Fraction(n - 1, 2) - r, n, c) +
                (Fraction(0, 1) if n != c else ((-1) ** (n - r) *
                Fraction(comb(n + 1, r + 1), 2)))
            ) * Fraction(1, factorial(c))
            for c in range(n + 1)
        ]
        for r in range(n + 1)
    ]

#---------------
_w0_frac: List[List[Fraction]] = [[Fraction(1, 1)]]

#---------------
_w1_frac: List[List[Fraction]] = [
    [Fraction(1, 1), Fraction(-1, 1)],
    [Fraction(0, 1), Fraction(1, 1)]
]

#---------------
_w2_frac: List[List[Fraction]] = [
    [Fraction(1, 2), Fraction(-1, 1), Fraction(1, 2)],
    [Fraction(1, 2), Fraction(1, 1), Fraction(-1, 1)],
    [Fraction(0, 1), Fraction(0, 1), Fraction(1, 2)]
]

#---------------
_w3_frac: List[List[Fraction]] = [
    [Fraction(1, 6), Fraction(-1, 2), Fraction(1, 2), Fraction(-1, 6)],
    [Fraction(2, 3), Fraction(0, 1), Fraction(-1, 1), Fraction(1, 2)],
    [Fraction(1, 6), Fraction(1, 2), Fraction(1, 2), Fraction(-1, 2)],
    [Fraction(0, 1), Fraction(0, 1), Fraction(0, 1), Fraction(1, 6)]
]

#---------------
_w4_frac: List[List[Fraction]] = [
    [
        Fraction(1, 24),
        Fraction(-1, 6),
        Fraction(1, 4),
        Fraction(-1, 6),
        Fraction(1, 24)
    ],
    [
        Fraction(11, 24),
        Fraction(-1, 2),
        Fraction(-1, 4),
        Fraction(1, 2),
        Fraction(-1, 6)
    ],
    [
        Fraction(11, 24),
        Fraction(1, 2),
        Fraction(-1, 4),
        Fraction(-1, 2),
        Fraction(1, 4)
    ],
    [
        Fraction(1, 24),
        Fraction(1, 6),
        Fraction(1, 4),
        Fraction(1, 6),
        Fraction(-1, 6)
    ],
    [
        Fraction(0, 1),
        Fraction(0, 1),
        Fraction(0, 1),
        Fraction(0, 1),
        Fraction(1, 24)
    ]
]

#---------------
_w5_frac: List[List[Fraction]] = [
    [
        Fraction(1, 120),
        Fraction(-1, 24),
        Fraction(1, 12),
        Fraction(-1, 12),
        Fraction(1, 24),
        Fraction(-1, 120)
    ],
    [
        Fraction(13, 60),
        Fraction(-5, 12),
        Fraction(1, 6),
        Fraction(1, 6),
        Fraction(-1, 6),
        Fraction(1, 24)
    ],
    [
        Fraction(11, 20),
        Fraction(0, 1),
        Fraction(-1, 2),
        Fraction(0, 1),
        Fraction(1, 4),
        Fraction(-1, 12)
    ],
    [
        Fraction(13, 60),
        Fraction(5, 12),
        Fraction(1, 6),
        Fraction(-1, 6),
        Fraction(-1, 6),
        Fraction(1, 12)
    ],
    [
        Fraction(1, 120),
        Fraction(1, 24),
        Fraction(1, 12),
        Fraction(1, 12),
        Fraction(1, 24),
        Fraction(-1, 24)
    ],
    [
        Fraction(0, 1),
        Fraction(0, 1),
        Fraction(0, 1),
        Fraction(0, 1),
        Fraction(0, 1),
        Fraction(1, 120)
    ]
]

#---------------
_wn: Dict[int, np.ndarray[tuple[int, int], np.dtype[np.float64]]] = {
    0: cast(
        np.ndarray[tuple[int, int], np.dtype[np.float64]],
        np.array(_w0_frac, dtype = float)
    ),
    1: cast(
        np.ndarray[tuple[int, int], np.dtype[np.float64]],
        np.array(_w1_frac, dtype = float)
    ),
    2: cast(
        np.ndarray[tuple[int, int], np.dtype[np.float64]],
        np.array(_w2_frac, dtype = float)
    ),
    3: cast(
        np.ndarray[tuple[int, int], np.dtype[np.float64]],
        np.array(_w3_frac, dtype = float)
    ),
    4: cast(
        np.ndarray[tuple[int, int], np.dtype[np.float64]],
        np.array(_w4_frac, dtype = float)
    ),
    5: cast(
        np.ndarray[tuple[int, int], np.dtype[np.float64]],
        np.array(_w5_frac, dtype = float)
    )
}

#---------------
def _w (
    n: int
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:

    r"""
    .. _w:

    Spline evaluation matrix :math:`{\mathbf{W}}^{n}.`

    Returns the :ref:`B-spline evaluation matrix<w_frac>`
    :math:`{\mathbf{W}}^{n}\in{\mathbb{R}}^{\left(n+1\right)\times
    \left(n+1\right)}`
    of a :ref:`polynomial B-spline<db_frac>` of :ref:`positive<def-positive>`
    degree :math:`n.`

    Parameters
    ----------
    n : int
        Positive degree of the polynomial B-spline.

    Returns
    -------
    np.ndarray[tuple[int, int], np.dtype[np.float64]]
        The B-spline evaluation matrix.
    Notes
    -----
    The results of this method are cached. If the returned results are
    mutated, the cache gets modified and the next call will return corrupted
    values.
    """

    if n in _wn:
        return _wn[n]
    _wn[n] = cast(
        np.ndarray[tuple[int, int], np.dtype[np.float64]],
        np.array(_w_frac(n), dtype = float)
    )
    return _wn[n]

#---------------
_iwn: Dict[int, np.ndarray[tuple[int, int], np.dtype[np.float64]]] = {
    0: np.linalg.inv(np.array(_w0_frac, dtype = float)),
    1: np.linalg.inv(np.array(_w1_frac, dtype = float)),
    2: np.linalg.inv(np.array(_w2_frac, dtype = float)),
    3: np.linalg.inv(np.array(_w3_frac, dtype = float)),
    4: np.linalg.inv(np.array(_w4_frac, dtype = float)),
    5: np.linalg.inv(np.array(_w5_frac, dtype = float))
}

#---------------
def _inv_w (
    n: int
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:

    r"""
    .. _inv_w:

    Spline evaluation matrix inverse
    :math:`\left({\mathbf{W}}^{n}\right)^{-1}.`

    Returns the inverse
    :math:`\left({\mathbf{W}}^{n}\right)^{-1}
    \in{\mathbb{R}}^{\left(n+1\right)\times\left(n+1\right)}`
    of the :ref:`B-spline evaluation matrix<w_frac>` of
    :ref:`positive<def-positive>` degree :math:`n.`

    Parameters
    ----------
    n : int
        Positive degree of the polynomial B-spline.

    Returns
    -------
    np.ndarray[tuple[int, int], np.dtype[np.float64]]
        The inverse of the B-spline evaluation matrix.

    Notes
    -----
    The results of this method are cached. If the returned results are
    mutated, the cache gets modified and the next call will return corrupted
    values.
    """

    if n in _iwn:
        return _iwn[n]
    _iwn[n] = np.linalg.inv(np.array(_w_frac(n), dtype = float))
    return _iwn[n]

#---------------
_wint0_frac: List[List[Fraction]] = [[Fraction(0, 1), Fraction(1, 1)]]

#---------------
_wint1_frac: List[List[Fraction]] = [
    [Fraction(1, 2), Fraction(1, 1), Fraction(-1, 2)],
    [Fraction(0, 1), Fraction(0, 1), Fraction(1, 2)]
]

#---------------
_wint2_frac: List[List[Fraction]] = [
    [Fraction(5, 6), Fraction(1, 2), Fraction(-1, 2), Fraction(1, 6)],
    [Fraction(1, 6), Fraction(1, 2), Fraction(1, 2), Fraction(-1, 3)],
    [Fraction(0, 1), Fraction(0, 1), Fraction(0, 1), Fraction(1, 6)]
]

#---------------
_wint3_frac: List[List[Fraction]] = [
    [
        Fraction(23, 24),
        Fraction(1, 6),
        Fraction(-1, 4),
        Fraction(1, 6),
        Fraction(-1, 24)
    ],
    [
        Fraction(1, 2),
        Fraction(2, 3),
        Fraction(0, 1),
        Fraction(-1, 3),
        Fraction(1, 8)
    ],
    [
        Fraction(1, 24),
        Fraction(1, 6),
        Fraction(1, 4),
        Fraction(1, 6),
        Fraction(-1, 8)
    ],
    [
        Fraction(0, 1),
        Fraction(0, 1),
        Fraction(0, 1),
        Fraction(0, 1),
        Fraction(1, 24)
    ]
]

#---------------
_wint4_frac: List[List[Fraction]] = [
    [
        Fraction(119, 120),
        Fraction(1, 24),
        Fraction(-1, 12),
        Fraction(1, 12),
        Fraction(-1, 24),
        Fraction(1, 120)
    ],
    [
        Fraction(31, 40),
        Fraction(11, 24),
        Fraction(-1, 4),
        Fraction(-1, 12),
        Fraction(1, 8),
        Fraction(-1, 30)
    ],
    [
        Fraction(9, 40),
        Fraction(11, 24),
        Fraction(1, 4),
        Fraction(-1, 12),
        Fraction(-1, 8),
        Fraction(1, 20)
    ],
    [
        Fraction(1, 120),
        Fraction(1, 24),
        Fraction(1, 12),
        Fraction(1, 12),
        Fraction(1, 24),
        Fraction(-1, 30)
    ],
    [
        Fraction(0, 1),
        Fraction(0, 1),
        Fraction(0, 1),
        Fraction(0, 1),
        Fraction(0, 1),
        Fraction(1, 120)
    ]
]

#---------------
_wint5_frac: List[List[Fraction]] = [
    [
        Fraction(719, 720),
        Fraction(1, 120),
        Fraction(-1, 48),
        Fraction(1, 36),
        Fraction(-1, 48),
        Fraction(1, 120),
        Fraction(-1, 720)
    ],
    [
        Fraction(331, 360),
        Fraction(13, 60),
        Fraction(-5, 24),
        Fraction(1, 18),
        Fraction(1, 24),
        Fraction(-1, 30),
        Fraction(1, 144)
    ],
    [
        Fraction(1, 2),
        Fraction(11, 20),
        Fraction(0, 1),
        Fraction(-1, 6),
        Fraction(0, 1),
        Fraction(1, 20),
        Fraction(-1, 72)
    ],
    [
        Fraction(29, 360),
        Fraction(13, 60),
        Fraction(5, 24),
        Fraction(1, 18),
        Fraction(-1, 24),
        Fraction(-1, 30),
        Fraction(1, 72)
    ],
    [
        Fraction(1, 720),
        Fraction(1, 120),
        Fraction(1, 48),
        Fraction(1, 36),
        Fraction(1, 48),
        Fraction(1, 120),
        Fraction(-1, 144)
    ],
    [
        Fraction(0, 1),
        Fraction(0, 1),
        Fraction(0, 1),
        Fraction(0, 1),
        Fraction(0, 1),
        Fraction(0, 1),
        Fraction(1, 720)
    ]
]

#---------------
_wintn: Dict[int, np.ndarray[tuple[int, int], np.dtype[np.float64]]] = {
    0: cast(
        np.ndarray[tuple[int, int], np.dtype[np.float64]],
        np.array(_wint0_frac, dtype = float)
    ),
    1: cast(
        np.ndarray[tuple[int, int], np.dtype[np.float64]],
        np.array(_wint1_frac, dtype = float)
    ),
    2: cast(
        np.ndarray[tuple[int, int], np.dtype[np.float64]],
        np.array(_wint2_frac, dtype = float)
    ),
    3: cast(
        np.ndarray[tuple[int, int], np.dtype[np.float64]],
        np.array(_wint3_frac, dtype = float)
    ),
    4: cast(
        np.ndarray[tuple[int, int], np.dtype[np.float64]],
        np.array(_wint4_frac, dtype = float)
    ),
    5: cast(
        np.ndarray[tuple[int, int], np.dtype[np.float64]],
        np.array(_wint5_frac, dtype = float)
    )
}

#---------------
def _wint (
    n: int
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:

    r"""
    .. _wint:

    Integrated-B-spline evaluation matrix
    :math:`{\mathbf{W}}_{{\mathrm{int}}}^{n}.`

    Returns the integrated-B-spline evaluation matrix
    :math:`{\mathbf{W}}_{{\mathrm{int}}}^{n}
    \in{\mathbb{R}}^{\left(n+1\right)\times\left(n+2\right)}`
    of an integrated :ref:`polynomial B-spline<db_frac>` of
    :ref:`nonnegative<def-negative>` degree :math:`n,` as defined by the
    component at its :math:`(r+1)`-th row and :math:`(c+1)`-th column

    ..  math::

        \left[{\mathbf{W}}^{n}_{{\mathrm{int}}}\right]_{r+1,c+1}=
        w_{{\mathrm{int}}}^{n}[r][c]=\left\{\begin{array}{ll}\sum_{q=0}^{n}\,
        \frac{1}{q+1}\sum_{p=r+1}^{n}\,w^{n}[p][q],&c=0\\
        \frac{1}{c}\,w^{n}[r][c-1],&c\in[1\ldots n+1],\end{array}\right.

    where :math:`w^{n}` is a component of the
    :ref:`B-spline evaluation matrix<w_frac>`.

    Parameters
    ----------
    n : int
        Nonnegative degree of the polynomial B-spline.

    Returns
    -------
    np.ndarray[tuple[int, int], np.dtype[np.float64]]
        The integrated-B-spline evaluation matrix.

    Notes
    -----
    The results of this method are cached. If the returned results are
    mutated, the cache gets modified and the next call will return corrupted
    values.
    """

    if n in _wintn:
        return _wintn[n]
    w = _w_frac(n)
    _wintn[n] = cast(
        np.ndarray[tuple[int, int], np.dtype[np.float64]],
        np.array(
            [
                [
                    sum(
                        sum(
                            Fraction(wpq, q + 1)
                            for (q, wpq) in enumerate(w[p])
                        )
                        for p in range(r + 1, n + 1)
                    ) if 0 == c else Fraction(w[r][c - 1], c)
                    for c in range(n + 2)
                ]
                for r in range(n + 1)
            ],
            dtype = float
        )
    )
    return _wintn[n]

#---------------
_wd0_frac: List[List[Fraction]] = [[]]

#---------------
_wd1_frac: List[List[Fraction]] = [
    [Fraction(-1, 1)],
    [Fraction(1, 1)]
]

#---------------
_wd2_frac: List[List[Fraction]] = [
    [Fraction(-1, 1), Fraction(1, 1)],
    [Fraction(1, 1), Fraction(-2, 1)],
    [Fraction(0, 1), Fraction(1, 1)]
]

#---------------
_wd3_frac: List[List[Fraction]] = [
    [Fraction(-1, 2), Fraction(1, 1), Fraction(-1, 2)],
    [Fraction(0, 1), Fraction(-2, 1), Fraction(3, 2)],
    [Fraction(1, 2), Fraction(1, 1), Fraction(-3, 2)],
    [Fraction(0, 1), Fraction(0, 1), Fraction(1, 2)]
]

#---------------
_wd4_frac: List[List[Fraction]] = [
    [Fraction(-1, 6), Fraction(1, 2), Fraction(-1, 2), Fraction(1, 6)],
    [Fraction(-1, 2), Fraction(-1, 2), Fraction(3, 2), Fraction(-2, 3)],
    [Fraction(1, 2), Fraction(-1, 2), Fraction(-3, 2), Fraction(1, 1)],
    [Fraction(1, 6), Fraction(1, 2), Fraction(1, 2), Fraction(-2, 3)],
    [Fraction(0, 1), Fraction(0, 1), Fraction(0, 1), Fraction(1, 6)]
]

#---------------
_wd5_frac: List[List[Fraction]] = [
    [
        Fraction(-1, 24),
        Fraction(1, 6),
        Fraction(-1, 4),
        Fraction(1, 6),
        Fraction(-1, 24)
    ],
    [
        Fraction(-5, 12),
        Fraction(1, 3),
        Fraction(1, 2),
        Fraction(-2, 3),
        Fraction(5, 24)
    ],
    [
        Fraction(0, 1),
        Fraction(-1, 1),
        Fraction(0, 1),
        Fraction(1, 1),
        Fraction(-5, 12)
    ],
    [
        Fraction(5, 12),
        Fraction(1, 3),
        Fraction(-1, 2),
        Fraction(-2, 3),
        Fraction(5, 12)
    ],
    [
        Fraction(1, 24),
        Fraction(1, 6),
        Fraction(1, 4),
        Fraction(1, 6),
        Fraction(-5, 24)
    ],
    [
        Fraction(0, 1),
        Fraction(0, 1),
        Fraction(0, 1),
        Fraction(0, 1),
        Fraction(1, 24)
    ]
]

#---------------
_wdn: Dict[int, np.ndarray[tuple[int, int], np.dtype[np.float64]]] = {
    0: cast(
        np.ndarray[tuple[int, int], np.dtype[np.float64]],
        np.array(_wd0_frac, dtype = float)
    ),
    1: cast(
        np.ndarray[tuple[int, int], np.dtype[np.float64]],
        np.array(_wd1_frac, dtype = float)
    ),
    2: cast(
        np.ndarray[tuple[int, int], np.dtype[np.float64]],
        np.array(_wd2_frac, dtype = float)
    ),
    3: cast(
        np.ndarray[tuple[int, int], np.dtype[np.float64]],
        np.array(_wd3_frac, dtype = float)
    ),
    4: cast(
        np.ndarray[tuple[int, int], np.dtype[np.float64]],
        np.array(_wd4_frac, dtype = float)
    ),
    5: cast(
        np.ndarray[tuple[int, int], np.dtype[np.float64]],
        np.array(_wd5_frac, dtype = float)
    )
}

#---------------
def _wd (
    n: int
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:

    r"""
    .. _wd:

    Differentiated-B-spline evaluation matrix
    :math:`{\mathbf{W}}_{{\mathrm{d}}}^{n}.`

    Returns the differentiated-B-spline evaluation matrix
    :math:`{\mathbf{W}}_{{\mathrm{d}}}^{n}
    \in{\mathbb{R}}^{\left(n+1\right)\times n}`
    of a differentiated :ref:`polynomial B-spline<db_frac>` of degree
    :math:`n>1,` as defined by the component at its :math:`(r+1)`-th row and
    :math:`(c+1)`-th column

    ..  math::

        \left[{\mathbf{W}}^{n}_{{\mathrm{d}}}\right]_{r+1,c+1}=
        w_{{\mathrm{d}}}^{n}[r][c]=\left(c+1\right)\,w^{n}[r][c+1],

    where :math:`w^{n}` is a component of the
    :ref:`B-spline evaluation matrix<w_frac>`.

    Parameters
    ----------
    n : int
        Positive degree of the polynomial B-spline.

    Returns
    -------
    np.ndarray[tuple[int, int], np.dtype[np.float64]]
        The differentiated-B-pline evaluation matrix.

    Notes
    -----
    The results of this method are cached. If the returned results are
    mutated, the cache gets modified and the next call will return corrupted
    values.
    """

    if n in _wdn:
        return _wdn[n]
    w = _w_frac(n)
    _wdn[n] = cast(
        np.ndarray[tuple[int, int], np.dtype[np.float64]],
        np.array(
            [[wrc * c for (c, wrc) in enumerate(wr) if 0 < c] for wr in w],
            dtype = float
        )
    )
    return _wdn[n]

#---------------
_iotan: Dict[int, np.ndarray[tuple[int], np.dtype[np.float64]]] = {
}

#---------------
def _iota (
    n: int
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:

    r"""
    .. _iota:

    Returns the vector :math:`{\mathbf{i}}^{n}` of the pole coefficients used
    in the computation of :ref:`inverse B-spline sequences<ib_coeff>`. For a
    :ref:`polynomial B-spline<db_frac>` of degree :math:`n>1,` this vector is
    given by
    
    ..  math::

        {\mathbf{i}}^{n}=\left(\frac{1}
        {\beta^{n}(\left\lfloor\frac{n}{2}\right\rfloor)\,
        \left(\left[{\mathbf{z}}^{n}\right]_{c+1}-
        \frac{1}{\left[{\mathbf{z}}^{n}\right]_{c+1}}\right)\,
        \prod_{p\in[0\ldots\left\lfloor\frac{n}{2}\right\rfloor-1]\setminus
        \{c\}}\,\left(\left[{\mathbf{z}}^{n}\right]_{c+1}-
        \left[{\mathbf{z}}^{n}\right]_{p+1}+
        \frac{1}{\left[{\mathbf{z}}^{n}\right]_{c+1}}-
        \frac{1}{\left[{\mathbf{z}}^{n}\right]_{p+1}}\right)}\right)
        _{c=0}^{\left\lfloor\frac{n}{2}\right\rfloor-1},

    where :math:`{\mathbf{z}}^{n}` is the vector of :ref:`poles <pole>`.
    Equivalently, it is also given by the first column of the inverse of the
    matrix :math:`{\mathbf{Z}}^{n},` as in

    ..  math::

        {\mathbf{i}}^{n}=\left[\left({\mathbf{Z}}^{n}\right)^{-1}\right]
        _{\cdot,1}.

    There, the component at the :math:`(r+1)`-th row and :math:`(c+1)`-th
    column of :math:`{\mathbf{Z}}^{n}\in{\mathbf{R}}^
    {\left\lfloor n/2\right\rfloor\times\left\lfloor n/2\right\rfloor}` is
    given by

    ..  math::

        z_{r+1,c+1}^{n}=z^{n}[r][c]=\sum_{q=-\left\lfloor n/2\right\rfloor}^
        {\left\lfloor n/2\right\rfloor}\,
        \left(\left[{\mathbf{z}}^{n}\right]_{c+1}\right)^{\left|r-q\right|}\,
        \beta^{n}(q).

    Parameters
    ----------
    n : int
        Degree of the polynomial B-spline, with :math:`n > 1`.

    Returns
    -------
    np.ndarray[tuple[int], np.dtype[np.float64]]:
        The vector :math:`{\mathbf{i}}^{n}.`

    Notes
    -----
    The results of this method are cached. If the returned results are
    mutated, the cache gets modified and the next call will return corrupted
    values.
    """

    if n in _iotan:
        return _iotan[n]
    _iotan[n] = np.linalg.lstsq(
        (_w(n) @ np.vander(
            np.array([0.5 * ((n + 1) % 2)]),
            n + 1,
            increasing = True
        )[0]) @ np.array(
            [
                [
                    [p ** abs(c + floor(r - 0.5 * (n - 1))) for p in _pole(n)]
                    for c in range(n + 1)
                ]
                for r in range(n // 2 - 1 + 1)
            ],
            dtype = float
        ),
        np.concatenate((np.array([1]), np.zeros(n // 2 - 1)))
    )[0]
    return _iotan[n]

#---------------
_vn_frac: Dict[int, List[List[Fraction]]] = {
    1: [[Fraction(1, 1)]],
    2: [
        [Fraction(1, 1), Fraction(0, 1)],
        [Fraction(1, 1), Fraction(1, 4)]
    ],
    3: [
        [Fraction(1, 1), Fraction(0, 1)],
        [Fraction(1, 1), Fraction(1, 3)],
        [Fraction(1, 1), Fraction(1, 1)]
    ],
    4: [
        [Fraction(1, 1), Fraction(0, 1), Fraction(0, 1)],
        [Fraction(1, 1), Fraction(5, 12), Fraction(0, 1)],
        [Fraction(1, 1), Fraction(5, 4), Fraction(0, 1)],
        [Fraction(1, 1), Fraction(5, 2), Fraction(23, 48)]
    ],
    5: [
        [Fraction(1, 1), Fraction(0, 1), Fraction(0, 1)],
        [Fraction(1, 1), Fraction(1, 2), Fraction(0, 1)],
        [Fraction(1, 1), Fraction(3, 2), Fraction(0, 1)],
        [Fraction(1, 1), Fraction(3, 1), Fraction(7, 10)],
        [Fraction(1, 1), Fraction(5, 1), Fraction(7, 2)]
    ]
}

#---------------
def _v_frac (
    n: int
) -> List[List[Fraction]]:

    r"""
    .. _v_frac:

    Returns the rational lookup table :math:`{\mathbf{V}}^{n}
    \in{\mathbb{Q}}^{n\times\left(\left\lfloor n/2\right\rfloor+1\right)}`
    that contains the polynomial coefficients of a B-spline-weighted sum of
    discrete monomials. For B-splines :math:`\beta^{n}` of
    :ref:`positive<def-positive>` degree :math:`n,` it is given by the
    component at its :math:`(r+1)`-th row and :math:`(c+1)`-th column

    ..  math::

        \left[{\mathbf{V}}^{n}\right]_{r+1,c+1}=
        v^{n}[r][c]=\left\{\begin{array}{ll}1,&c=0\\{r+1\choose 2\,c}\,
        \sum_{k=0}^{n-1}\,\left(\frac{n-1}{2}-k\right)^{2\,c}\,
        \beta^{n}(\frac{n-1}{2}-k),&c>0.\end{array}\right.

    Parameters
    ----------
    n : int
        Positive degree of the polynomial B-spline.

    Returns
    -------
    list of list of Fraction
        The lookup table :math:`{\mathbf{V}}^{n}.`

    Notes
    -----
    The results of this method are cached. If the returned results are
    mutated, the cache gets modified and the next call will return corrupted
    values.
    """

    if n in _vn_frac:
        return _vn_frac[n]
    _vn_frac[n] = [
        [
            Fraction(1, 1) if 0 == c else Fraction(comb(r + 1, 2 * c) *
                sum(
                    ((Fraction(n - 1, 2) - k) ** (2 * c)) *
                        _db_frac(Fraction(n - 1, 2) - k, n, 0)
                    for k in range(n - 1 + 1)
                )
            )
            for c in range(n // 2 + 1)
        ]
        for r in range(n)
    ]
    return _vn_frac[n]

#---------------
@functools.lru_cache(maxsize = 512)
def _alpha_frac (
    n: int,
    m: int
) -> List[Fraction]:

    r"""
    .. _alpha_frac:

    Returns the lookup table :math:`{\mathbf{A}}^{n}` that contains the spline
    coefficients of a monomial-reproducing spline. For a B-spline
    :math:`\beta^{n}` of :ref:`nonnegative<def-negative>` degree :math:`n`
    and a monomial of nonnegative degree :math:`m` not greater than :math:`n,`
    its component at :math:`p\in[0\ldots m]` is defined by the recursion

    ..  math::

        \alpha^{n}[m][p]=\left\{\begin{array}{ll}
        -\sum_{q=1}^{\left\lfloor\left(m-p\right)/2\right\rfloor}
        \,v^{n}[m-1][q]\,\alpha^{n}[m-2\,q][p],&0\leq p<m-1\\0,&p=m-1\\1,&p=m,
        \end{array}\right.

    with :math:`v^{n}[m-1][q]` the component at row :math:`m` and column
    :math:`(q+1)` of the the lookup table :math:`{\mathbf{V}}^{n}` in
    :ref:`_v_frac <v_frac>`.

    Parameters
    ----------
    n : int
        Nonnegative degree of the polynomial B-spline.
    m : int
        Nonnegative degree of the monomial.

    Returns
    -------
    list of list of Fraction
        The lookup table :math:`{\mathbf{A}}^{n}.`

    Notes
    -----
    The results of this method are cached. If the returned results are
    mutated, the cache gets modified and the next call will return corrupted
    values.
    """

    return [
        Fraction(1, 1) if m == p
            else Fraction(0, 1) if m - 1 == p
            else Fraction(-sum(
                _v_frac(n)[m - 1][q] * _alpha_frac(n, m - 2 * q)[p]
                for q in range(1, (m - p) // 2 + 1)
            ))
        for p in range(m + 1)
    ]

#---------------
def _divmod (
    x: float,
    m: int
) -> Tuple[int, float]:

    r"""
    .. _divmod:

    Quotient and remainder from dividend and divisor.

    A function similar to the built-in ``divmod`` function, with the added
    guarantee that the remainder part of result lies in the closed-open
    interval :math:`[0,m),` while the built-in version only guarantees the
    closed interval :math:`[0,m].` The tipping point where :math:`m` is
    included or not mimicks the ``Mod`` function of the *Wolfram* framework.

    Parameters
    ----------
    x : float
        Dividend whose quotient and remainder are sought.
    m : int
        Nonzero divisor.

    Returns
    -------
    tuple (int, float):
        A tuple ``(q, r)`` such that ``x == m * q + r`` up to numerical
        inaccuracies, with the guarantee that ``r < m``.

    Examples
    --------
    Load the libraries.
        >>> import splinekit.spline_utilities as sku
        >>> import math
    Here is an argument at which this function agrees with the built-in one.
        >>> sku._divmod(-math.ulp(1.0), 1) == divmod(-math.ulp(1.0), 1)
        True
    Here is an argument at which this function disagrees with the built-in one.
        >>> sku._divmod(-math.ulp(0.1), 1) == divmod(-math.ulp(0.1), 1)
        False
    """

    if 0.0 <= x:
        (div, mod) = divmod(x, m)
        return (int(div), mod)
    if 1 == m:
        if x < -ulp(0.25):
            (div, mod) = divmod(x, m)
            return (int(div), mod)
        return(0, 0.0)
    if x < -ulp((m - 3) / 4 + ceil(m / 4)):
        (div, mod) = divmod(x, m)
        return (int(div), mod)
    return(0, 0.0)
