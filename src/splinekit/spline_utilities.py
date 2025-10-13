"""
The :class:`splinekit.spline_utilities` module provides the building blocks
and low-level methods associated to the efficient and stable computation of
polynomial B-splines of arbitrary degree.

====

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


    ----

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


    ----

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


    ----

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


    ----

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

    Differentiated B-spline :math:`{\mathrm{D}}^{d}\beta^{n}.`

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


    ----

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

    Differentiated B-spline :math:`{\mathrm{D}}^{d}\beta^{n}.`

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


    ----

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

    Notes
    -----
    The results of this method are cached. If the returned results are
    mutated, the cache gets modified and the next call will return corrupted
    values.

    ----

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

# Discarded
# #---------------
# @functools.lru_cache(maxsize = 32)
# def _b_dft_expression (
#     n: int,
#     *,
#     fourier_idx: sympy.symbols = sympy.symbols("nu"),
#     data_len: sympy.symbols = sympy.symbols("K")
# ) -> sympy.Function:

#     r"""
#     .. _b_dft_expression:

#     Expression of the :math:`\nu`-th term of the DFT of the
#     periodized :ref:`B-spline discrete sequence<b>` :math:`{\mathrm{b}}^{n}.`

#     Returns a ``sympy`` expression of the :math:`\nu`-th term of the
#     discrete Fourier transform of the :math:`K`-periodized one-dimensional
#     array of the sampled values

#     ..  math::

#         b_{{\mathrm{per}},K}^{n}[k]=\sum_{p\in{\mathbb{Z}}}\,\beta^{n}(p\,K+k)

#     of a :ref:`polynomial B-spline<db_frac>` :math:`\beta^{n}` of
#     :ref:`nonnegative<def-negative>` degree :math:`n,` for
#     :math:`\nu\in[1\ldots K-1].` The :math:`\nu`-th term is

#     ..  math::

#         \begin{eqnarray*}
#         B_{{\mathrm{per}},K}^{n}[\nu]&=&\sum_{k=0}^{K-1}\,
#         b_{{\mathrm{per}},K}^{n}[k]\,
#         {\mathrm{e}}^{-{\mathrm{j}}\,\nu\,\frac{2\,\pi}{K}\,k}\\
#         &=&\left\{\begin{array}{ll}1,&\nu=0\\
#         \left.\frac{{\mathrm{d}}^{n}\csc\omega}{{\mathrm{d}}\omega^{n}}\right|
#         _{\omega=\nu\,\frac{\pi}{K}}\,\frac{1}{n!}\,
#         \sin^{n+1}(\nu\,\frac{\pi}{K}),&\nu\in[1\ldots K-1]\wedge
#         n\in2\,{\mathbb{N}}\\
#         \left.-\frac{{\mathrm{d}}^{n}\cot\omega}{{\mathrm{d}}\omega^{n}}\right|
#         _{\omega=\nu\,\frac{\pi}{K}}\,\frac{1}{n!}\,
#         \sin^{n+1}(\nu\,\frac{\pi}{K}),&\nu\in[1\ldots K-1]\wedge
#         n\in2\,{\mathbb{N}}+1.\end{array}\right.
#         \end{eqnarray*}

#     Parameters
#     ----------
#     n : int
#         Nonnegative degree of the polynomial B-spline.

#     Returns
#     -------
#     sympy.Function
#         The expression of the :math:`\nu`-th term of the DFT.

#     Notes
#     -----
#     The results of this method are cached. If the returned results are
#     mutated, the cache gets modified and the next call will return corrupted
#     values.

#     ----

#     """

#     omega = sympy.symbols("omega")
#     term = (sympy.sin(omega) ** (n+1) / factorial(n))
#     if 0 == n % 2:
#         term *= sympy.diff(sympy.csc(omega), omega, n)
#     else:
#         term *= -sympy.diff(sympy.cot(omega), omega, n)
#     return term.subs(omega, fourier_idx * sympy.pi / data_len)

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
        np.ascontiguousarray(
            [
                -0.1715728752538097
            ],
            dtype = float
        )
    ),
    3: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.2679491924311228
            ],
            dtype = float
        )
    ),
    4: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.36134122590025997,
                -0.01372542929733811
            ],
            dtype = float
        )
    ),
    5: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.43057534709997025,
                -0.04309628820326239
            ],
            dtype = float
        )
    ),
    6: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.4882945893030738,
                -0.08167927107623996,
                -0.0014141518082908533
            ],
            dtype = float
        )
    ),
    7: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.5352804307964426,
                -0.12255461519232824,
                -0.009148694809614
            ],
            dtype = float
        )
    ),
    8: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.5746869092485949,
                -0.16303526929728873,
                -0.023632294694834854,
                -0.0001538213107323827
            ],
            dtype = float
        )
    ),
    9: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.6079973891690234,
                -0.20175052019312423,
                -0.043222608540474994,
                -0.002121306903163145
            ],
            dtype = float
        )
    ),
    10: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.6365506639694238,
                -0.23818279837757328,
                -0.06572703322830856,
                -0.007528194675548689,
                -0.000016982762823274665
            ],
            dtype = float
        )
    ),
    11: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.6612660689007348,
                -0.27218034929478585,
                -0.0897595997937133,
                -0.016669627366234657,
                -0.000510557534446502
            ],
            dtype = float
        )
    ),
    12: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.6828648841977232,
                -0.30378079328825414,
                -0.11435052002713589,
                -0.028836190198663805,
                -0.0025161662172613355,
                -1.8833056450639026e-6
            ],
            dtype = float
        )
    ),
    13: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.7018942518168079,
                -0.33310723293062355,
                -0.13890111319431944,
                -0.04321386674036367,
                -0.006738031415244913,
                -0.00012510011321441872
            ],
            dtype = float
        )
    ),
    14: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.7187837872399445,
                -0.36031907191696105,
                -0.1630335147992987,
                -0.059089482194831025,
                -0.013246756734847914,
                -0.0008640240409533377,
                -2.091309677527533e-7
            ],
            dtype = float
        )
    ),
    15: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.7338725716848373,
                -0.38558573427843523,
                -0.18652010845096437,
                -0.0759075920476682,
                -0.02175206579654047,
                -0.0028011514820764547,
                -0.000030935680451474424
            ],
            dtype = float
        )
    ),
    16: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.7474323877664687,
                -0.40907360475725096,
                -0.20922871933953968,
                -0.09325471898024063,
                -0.0318677061204539,
                -0.006258406785125984,
                -0.0003015653633069596,
                -2.3232486364212317e-8
            ],
            dtype = float
        )
    ),
    17: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.7596832240719325,
                -0.4309396531803966,
                -0.23108984359927184,
                -0.11082899331624724,
                -0.043213911456684156,
                -0.011258183689471601,
                -0.0011859331251521767,
                -7.687562581254683e-6
            ],
            dtype = float
        )
    ),
    18: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.7708050514072082,
                -0.4513287333778175,
                -0.2520745746826387,
                -0.12841283679671456,
                -0.05546296713852202,
                -0.017662377684785218,
                -0.00301193072899483,
                -0.00010633735588713665,
                -2.5812403962571556e-9
            ],
            dtype = float
        )
    ),
    19: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.7809464448517323,
                -0.4703728194676429,
                -0.2721803762830345,
                -0.14585089375756452,
                -0.06834590612488047,
                -0.025265073344855598,
                -0.005936659591082969,
                -0.0005084101946808165,
                -1.9154786562122476e-6
            ],
            dtype = float
        )
    ),
    20: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.7902311174807248,
                -0.4881912606391278,
                -0.2914216014747826,
                -0.1630335348502631,
                -0.08164811561958894,
                -0.03384947955391196,
                -0.009973029020058727,
                -0.0014683217571043405,
                -0.000037746573197519025,
                -2.8679944881501054e-10
            ],
            dtype = float
        )
    ),
    21: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.7987628856657313,
                -0.5048915374483703,
                -0.30982319641579037,
                -0.17988466679853585,
                -0.09520081246073121,
                -0.04321391844071574,
                -0.015045499987294554,
                -0.003172003963884172,
                -0.00021990295763163234,
                -4.779764689425363e-7
            ],
            dtype = float
        )
    ),
    22: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.8066294994690704,
                -0.5205702359699174,
                -0.3274164741105745,
                -0.19635282612751534,
                -0.10887245198407758,
                -0.05318160458588925,
                -0.0210356609306781,
                -0.005706613645975126,
                -0.0007225479650742529,
                -0.000013458154983304882,
                -3.186643260385933e-11
            ],
            dtype = float
        )
    ),
    23: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.8139056235618003,
                -0.5353140836684857,
                -0.3442362769193781,
                -0.21240466054019305,
                -0.12256116099193873,
                -0.06360248015411367,
                -0.02781166203794072,
                -0.009079595335295149,
                -0.0017112714467817038,
                -0.00009573394350073983,
                -1.1936918816066495e-7
            ],
            dtype = float
        )
    ),
    24: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.8206551810729534,
                -0.5492009639573298,
                -0.3603190713767428,
                -0.22802014722790687,
                -0.1361885000680127,
                -0.0743514972773694,
                -0.03524412666583197,
                -0.013246375325300808,
                -0.003297682623486455,
                -0.0003580715441086799,
                -4.812675563310483e-6,
                -3.5407088072319184e-12
            ],
            dtype = float
        )
    ),
    25: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.826933212634977,
                -0.56230086768001,
                -0.375701677009631,
                -0.243189068415228,
                -0.14969446439099,
                -0.0853256075621658,
                -0.04321391826796003,
                -0.018134360805584098,
                -0.005534084441487331,
                -0.0009299873306092405,
                -0.00004187875052377163,
                -2.982478285796373e-8
            ],
            dtype = float
        )
    ),
    26: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.8327873634529699,
                -0.5746767631908143,
                -0.39042042836203417,
                -0.2579084013447034,
                -0.16303353482162078,
                -0.09644049675858785,
                -0.05161516247228333,
                -0.02365917403431693,
                -0.008424934725813555,
                -0.0019200118363432769,
                -0.0001784141661193633,
                -1.724530392210464e-6,
                -3.934118864515372e-13
            ],
            dtype = float
        )
    ),
    27: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.8382590821014602,
                -0.5863853797684231,
                -0.4045106347246844,
                -0.272180376229601,
                -0.17617157759323776,
                -0.10762752439060491,
                -0.060355763096686506,
                -0.029734603746786607,
                -0.01194239633191651,
                -0.003398965898172872,
                -0.0005082738135987046,
                -0.000018386371708328325,
                -7.453737087428511e-9
            ],
            dtype = float
        )
    ),
    28: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.8433845940335701,
                -0.5974779061705426,
                -0.4180062448978234,
                -0.28601102519176763,
                -0.18908341496245726,
                -0.1188310346590514,
                -0.06935667734278621,
                -0.03627817198644647,
                -0.01603960247911934,
                -0.0053999391510447066,
                -0.0011244739688530765,
                -0.0000892772318754022,
                -6.188192321062352e-7,
                -4.371242485810544e-14
            ],
            dtype = float
        )
    ),
    29: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.8481956975631376,
                -0.608000609732787,
                -0.43093965319217575,
                -0.2994090942789881,
                -0.20175092056381402,
                -0.13000607173353435,
                -0.0785506554711269,
                -0.043213918263483336,
                -0.02066024751435231,
                -0.007925828213695597,
                -0.002100199856436504,
                -0.0002790426068554639,
                -8.09458244198441e-6,
                -1.863088870511326e-9
            ],
            dtype = float
        )
    ),
    30: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.852720418887378,
                -0.6179953830117888,
                -0.44334160241059595,
                -0.3123852256265622,
                -0.21416152332065827,
                -0.14111647387019208,
                -0.08788081801854614,
                -0.050473516811469804,
                -0.025744933850994408,
                -0.010958396292871727,
                -0.00348228866906904,
                -0.0006616482677279397,
                -0.00004482596596998705,
                -2.2226905133536694e-7,
                -4.856935856324464e-15
            ],
            dtype = float
        )
    ),
    31: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.8569835543745783,
                -0.6275002254374089,
                -0.4552411529474755,
                -0.32495134176930657,
                -0.22630702921250065,
                -0.15213329957130273,
                -0.0972992587802864,
                -0.057996442426399446,
                -0.031235083453329888,
                -0.01446609949032905,
                -0.005292679268065068,
                -0.0013039731838981044,
                -0.00015375114935857996,
                -3.5711513081322674e-6,
                -4.657236728778484e-10
            ],
            dtype = float
        )
    ),
    32: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.8610071220775077,
                -0.6365496672844415,
                -0.46666569639255456,
                -0.337120182028479,
                -0.23818269066358583,
                -0.16303353482158023,
                -0.10676576019127221,
                -0.06572962575614473,
                -0.0370751794716569,
                -0.01841002303519408,
                -0.007533364010056748,
                -0.002256727280342519,
                -0.00039079751052768254,
                -0.00002256895223014066,
                -7.988927839078505e-8,
                -5.396595313947431e-16
            ],
            dtype = float
        )
    ),
    33: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.8648107396863689,
                -0.6451751428015156,
                -0.47764099846751634,
                -0.3489049538475771,
                -0.24978646932981421,
                -0.17379903374120698,
                -0.11624665358132276,
                -0.07362686063059345,
                -0.043213918263772834,
                -0.022748040786974878,
                -0.01019192053500616,
                -0.003552018463170819,
                -0.0008127973866455559,
                -0.00008496814678289396,
                -1.578071702213532e-6,
                -1.1642409378627733e-10
            ],
            dtype = float
        )
    ),
    34: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.8684119425100919,
                -0.6534053187195727,
                -0.48819126063884694,
                -0.36031907137690444,
                -0.2611184504267236,
                -0.18441565100741045,
                -0.12571382892454094,
                -0.08164811550866749,
                -0.049604678350979994,
                -0.027437567631765556,
                -0.01324637525145965,
                -0.0052049747739034685,
                -0.0014683963396586999,
                -0.00023154232259873006,
                -0.000011388454884531738,
                -2.8727897849853297e-8,
                -5.996216987387189e-17
            ],
            dtype = float
        )
    ),
    35: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.8718264522846042,
                -0.6612663837047974,
                -0.49833919293373846,
                -0.3713759605149176,
                -0.27218037622960944,
                -0.19487253092794063,
                -0.13514388528824783,
                -0.08975883392391287,
                -0.056205577426153924,
                -0.032437284236098346,
                -0.016669021915311406,
                -0.007217023436407387,
                -0.002393632153391935,
                -0.0005082814595413964,
                -0.00004707210567991254,
                -6.982167361691686e-7,
                -2.910506393324535e-11
            ],
            dtype = float
        )
    ),
    36: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.8750684054438953,
                -0.6687823036766544,
                -0.5081060927404221,
                -0.3820889147120882,
                -0.2829752736425094,
                -0.20516152414182076,
                -0.14451740754635553,
                -0.09792926875377747,
                -0.0629792929483648,
                -0.03770814705533075,
                -0.020429222487409172,
                -0.009579426581121988,
                -0.0036112201471467545,
                -0.0009586466637195034,
                -0.00013754232032761284,
                -5.757252845898399e-6,
                -1.033388279267918e-8,
                -6.662463309679195e-18
            ],
            dtype = float
        )
    ),
    37: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.8781505478013883,
                -0.675975047309533,
                -0.5175119259768467,
                -0.3924709896295194,
                -0.293507156298858,
                -0.2152767082291065,
                -0.15381835335187108,
                -0.10613387166264764,
                -0.06989275733426102,
                -0.043213918263772605,
                -0.02449536318833238,
                -0.01227642795169496,
                -0.005131933739471126,
                -0.0016185404500439666,
                -0.00031871545526048633,
                -0.000026131692895771312,
                -3.0922668854510665e-7,
                -7.276131065451552e-12
            ],
            dtype = float
        )
    ),
    38: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.88108440126276,
                -0.6828647854893555,
                -0.5265754081532769,
                -0.4025349275748516,
                -0.30378078592941193,
                -0.225213992968174,
                -0.16303353482158045,
                -0.11435074562522658,
                -0.07691679639577582,
                -0.048921382752319394,
                -0.028836162217098546,
                -0.015287794002170486,
                -0.0069568303357382556,
                -0.0025141897503566965,
                -0.0006276117067541021,
                -0.0000818825638348434,
                -2.9148971463249454e-6,
                -3.718130253557185e-9,
                -7.402737007448796e-19
            ],
            dtype = float
        )
    ),
    39: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.8838804071427819,
                -0.6894700680083403,
                -0.5353140836684926,
                -0.412293104766146,
                -0.3138014810230074,
                -0.2349707946459765,
                -0.1721521807622996,
                -0.12256116099394455,
                -0.084025752356603,
                -0.05480036756601221,
                -0.03342150293797134,
                -0.01859074722483924,
                -0.009079595195884275,
                -0.0036621043788701386,
                -0.0010975868567375147,
                -0.00020030688794487426,
                -0.000014532131624790515,
                -1.370545029060038e-7,
                -1.819013794625788e-12
            ],
            dtype = float
        )
    ),
    40: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.8865480498249418,
                -0.6958079803507161,
                -0.5437444022594512,
                -0.4217574960836136,
                -0.3235749633481037,
                -0.24454576680792411,
                -0.18116556697156275,
                -0.13074913168332145,
                -0.09119711540837702,
                -0.060823641667355696,
                -0.03822293082901785,
                -0.02216137502477028,
                -0.011488648242654397,
                -0.005070168042357454,
                -0.0017555630443931954,
                -0.00041186505462278376,
                -0.000048837498693133175,
                -1.4776640683288942e-6,
                -1.3380026581391422e-9,
                -8.225263340476075e-20
            ],
            dtype = float
        )
    ),
    41: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.8890959638357833,
                -0.7018942830458227,
                -0.5518817919367212,
                -0.43093965319217514,
                -0.3331072348755986,
                -0.2539385772414767,
                -0.19006670387369587,
                -0.13890104622536414,
                -0.09841117686633294,
                -0.06696674766202966,
                -0.043213918263772244,
                -0.025975619480793903,
                -0.014168885396777246,
                -0.006739186449046985,
                -0.002621041808219292,
                -0.0007461328733157923,
                -0.000126135088806702,
                -8.093502330497814e-6,
                -6.078089086680649e-8,
                -4.547507808420939e-13
            ],
            dtype = float
        )
    ),
    42: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.8915320268698523,
                -0.7077435357376589,
                -0.5597407280298026,
                -0.43985069285205436,
                -0.34240447918016464,
                -0.26314972292489897,
                -0.19885007236072622,
                -0.14700534781386743,
                -0.10565071024794828,
                -0.07320779911941623,
                -0.04836997283028642,
                -0.030009947252512277,
                -0.017103043468257726,
                -0.008664489321413532,
                -0.0037063049253603713,
                -0.0012289135655289393,
                -0.00027083250359544376,
                -0.00002917476017494444,
                -7.49860936947263e-7,
                -4.815479655866042e-10,
                -9.139181489029075e-21
            ],
            dtype = float
        )
    ),
    43: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.8938634408678929,
                -0.7133692078362501,
                -0.5673347981712786,
                -0.4485012929513466,
                -0.3514729825996738,
                -0.27218037622960944,
                -0.20751140013780547,
                -0.15505225747088705,
                -0.1129006825209351,
                -0.07952726494797825,
                -0.05366864301808978,
                -0.03424178287301239,
                -0.020272729691552235,
                -0.010837383147015173,
                -0.005017257442672954,
                -0.0018807162220058793,
                -0.0005082812655905479,
                -0.0000795617466383835,
                -4.513325250611669e-6,
                -2.6967598171126342e-8,
                -1.1368732005595934e-13
            ],
            dtype = float
        )
    ),
    44: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.8960968028992142,
                -0.7187837773708822,
                -0.5746767631908141,
                -0.4569016943466848,
                -0.36031907137690444,
                -0.2810322569159788,
                -0.21604747211082126,
                -0.16303353482158045,
                -0.12014799537449801,
                -0.08590775414808884,
                -0.05908945890328844,
                -0.038649771098940744,
                -0.023659174021309768,
                -0.013246375251690057,
                -0.006554527808208168,
                -0.002716322197289291,
                -0.0008621057366132465,
                -0.0001784049129824875,
                -0.000017452554019181007,
                -3.808577626111631e-7,
                -1.7332351664980852e-10,
                -1.0154646098787866e-21
            ],
            dtype = float
        )
    ),
    45: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.8982381673121013,
                -0.7239988194534053,
                -0.5817786139867799,
                -0.46506170702509514,
                -0.3689490617529449,
                -0.289707525469325,
                -0.22445596940784804,
                -0.17094227147090993,
                -0.12738125503313347,
                -0.09233380888114755,
                -0.0646138337576593,
                -0.043213918263772244,
                -0.027243762343032312,
                -0.015878157235604606,
                -0.008314592328776581,
                -0.0037450549054819982,
                -0.0013524654180154746,
                -0.00034687984034379994,
                -0.00005025781166635491,
                -2.519607608499221e-6,
                -1.1969483659831502e-8,
                -2.8421777258420245e-14
            ],
            dtype = float
        )
    ),
    46: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.9002931003810235,
                -0.7290250855770061,
                -0.5886516245074035,
                -0.47299071943143234,
                -0.3773692205719937,
                -0.29820869413441126,
                -0.23273533251406422,
                -0.1787727125329917,
                -0.13459056845392445,
                -0.09879171027189003,
                -0.0702249441891338,
                -0.047915649714721904,
                -0.031008401728534796,
                -0.018718368210865994,
                -0.010290802730601551,
                -0.004971427346080922,
                -0.001995218786783391,
                -0.0006059115211361685,
                -0.00011769902040814798,
                -0.000010452752565634948,
                -1.9357966158860966e-7,
                -6.238789827562169e-11,
                -1.1282940109718572e-22
            ],
            dtype = float
        )
    ),
    47: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.9022667284857858,
                -0.7338725748180573,
                -0.595306401013988,
                -0.48069771006358936,
                -0.38558573442663546,
                -0.3065385526622772,
                -0.240884644743001,
                -0.1865201024138841,
                -0.14176736347020788,
                -0.10526929908897724,
                -0.07590760057817472,
                -0.05273781039104739,
                -0.03493575979111422,
                -0.02175217069495486,
                -0.012474266075319755,
                -0.006395939485227589,
                -0.002801762884498413,
                -0.0009744447470603107,
                -0.00023710406543300783,
                -0.00003178725446064709,
                -1.4079294006267424e-6,
                -5.314134848086204e-9,
                -7.105436895895931e-15
            ],
            dtype = float
        )
    ),
    48: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.9041637806978998,
                -0.738550597872473,
                -0.6017529278226239,
                -0.48819126063884694,
                -0.3936046857473073,
                -0.31470010631404544,
                -0.24890353288661873,
                -0.194180551461978,
                -0.14890423040414438,
                -0.11175581201141857,
                -0.0816481155086671,
                -0.05766462802162182,
                -0.039009411385609864,
                -0.024964674020065632,
                -0.014854566526072104,
                -0.008015886037162408,
                -0.0037793022467360513,
                -0.001468396390468777,
                -0.000426544185620036,
                -0.00007775273934995433,
                -6.266965407354385e-6,
                -9.845095966270365e-8,
                -2.245746804180422e-11,
                -1.2536600121893846e-23
            ],
            dtype = float
        )
    ),
    49: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.9059886265166821,
                -0.7430678347403393,
                -0.608000609732787,
                -0.49547957029202966,
                -0.4014320345397496,
                -0.3226965240973023,
                -0.25679208240215856,
                -0.20175092056381488,
                -0.15599478276566547,
                -0.11824173330413186,
                -0.0874341750863989,
                -0.0626816527967349,
                -0.043213918263772244,
                -0.028341237271078422,
                -0.017420339050793876,
                -0.00982609958672001,
                -0.004931347166588485,
                -0.0021001999049303005,
                -0.00070325410137834,
                -0.00016229267074085614,
                -0.0000201272871401716,
                -7.873829485288934e-7,
                -2.359865249843014e-9,
                -1.7763581807214033e-15
            ],
            dtype = float
        )
    ),
    50: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.907745309387148,
                -0.7474323867708736,
                -0.6140583113548245,
                -0.502570470389379,
                -0.4090736047154368,
                -0.3305310955615144,
                -0.26455076492210594,
                -0.2092287211733016,
                -0.16303353482158045,
                -0.12471866120585644,
                -0.09325471611918149,
                -0.06777568325308847,
                -0.04753486106030658,
                -0.03186767908781832,
                -0.02015971401621062,
                -0.011819595737879343,
                -0.006258302173730027,
                -0.00287877850854815,
                -0.0010825178936698337,
                -0.0003007036187496243,
                -0.00005142397812177874,
                -3.760832804314673e-6,
                -5.009572941249599e-8,
                -8.084136104198476e-12,
                -1.3929555690987785e-24
            ],
            dtype = float
        )
    ),
    51: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.9094375765394267,
                -0.751651823691739,
                -0.6199343935462047,
                -0.5094714396387324,
                -0.4165350741544562,
                -0.33820719476539035,
                -0.27218037622960944,
                -0.21661202862018325,
                -0.17001579401699726,
                -0.13117918804984938,
                -0.0990998108333827,
                -0.07293468512752384,
                -0.05195883806985232,
                -0.03553041669335946,
                -0.023060653998570046,
                -0.013988111734484944,
                -0.0077580582592267815,
                -0.0038097874729552442,
                -0.0015770350594634715,
                -0.000508281265017397,
                -0.00011122114601066585,
                -0.000012756851207498162,
                -4.4065818769292273e-7,
                -1.0481381842440522e-9,
                -4.440893984732253e-16
            ],
            dtype = float
        )
    ),
    52: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.9110689056118992,
                -0.7557332261704166,
                -0.6256367471603685,
                -0.5161896192525214,
                -0.4238219677976189,
                -0.3457282502636366,
                -0.27968198313773185,
                -0.22389940685308296,
                -0.17693756643802805,
                -0.13761679300749677,
                -0.10496055993341141,
                -0.07814770778953226,
                -0.056473441487644986,
                -0.039316551940529076,
                -0.02611120321346857,
                -0.01632254337066926,
                -0.009426540896188955,
                -0.004895999754957772,
                -0.002196678923324891,
                -0.0007992415694400763,
                -0.00021225620346178325,
                -0.00003404583069892401,
                -2.2587294154172846e-6,
                -2.550153644265239e-8,
                -2.910148713662225e-12,
                -1.5477284101095694e-25
            ],
            dtype = float
        )
    ),
    53: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.9126425284551897,
                -0.7596832243891832,
                -0.6311728243034963,
                -0.5227318279797359,
                -0.43093965319217514,
                -0.35309772015310864,
                -0.28705687795732837,
                -0.23108984303835214,
                -0.18379547370634794,
                -0.14402574631068013,
                -0.11082899424079518,
                -0.0834048013305367,
                -0.061067218912751776,
                -0.043213918263772244,
                -0.029299667592936656,
                -0.018813291303395922,
                -0.011258191452027798,
                -0.006137747478094043,
                -0.002948532523563765,
                -0.0011860021245487843,
                -0.0003678400905426574,
                -0.00007630340724378297,
                -8.092442111683355e-6,
                -2.4676785850432055e-7,
                -4.655971182029792e-10,
                -1.1102232898763806e-16
            ],
            dtype = float
        )
    ),
    54: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.9141614524590332,
                -0.7635080330576853,
                -0.6365496672844415,
                -0.5291045768698532,
                -0.4378933380193009,
                -0.36031907137690444,
                -0.2943065394437468,
                -0.23818269066358583,
                -0.1905866798851154,
                -0.1504010238330278,
                -0.11669798478498002,
                -0.08869693630159932,
                -0.06572962575613937,
                -0.047211099223880594,
                -0.03261474071876812,
                -0.021450530050601917,
                -0.01324637525169032,
                -0.007533364015681437,
                -0.0038370973959960864,
                -0.00167879840986626,
                -0.0005908781938764372,
                -0.00014999181357509128,
                -0.000022561111100311378,
                -1.357556320403412e-6,
                -1.2986351369732129e-8,
                -1.0476179083692744e-12,
                -1.7196982334550138e-26
            ],
            dtype = float
        )
    ),
    55: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.9156284796974643,
                -0.7672134832370651,
                -0.6417739354321905,
                -0.5353140836684926,
                -0.44468806921615006,
                -0.367395762615825,
                -0.30143259928403154,
                -0.24517761998897047,
                -0.19730882714849232,
                -0.15673823096844988,
                -0.12256116099394455,
                -0.09401592732695248,
                -0.07045097259089789,
                -0.0512974267804351,
                -0.036045588066102294,
                -0.024224412891807597,
                -0.015383718409154342,
                -0.009079595195918425,
                -0.004864594311165972,
                -0.002285563105367284,
                -0.0008931374813681356,
                -0.00026651062565955733,
                -0.00005239828717115822,
                -5.137506906754919e-6,
                -1.382649109414373e-7,
                -2.0684718841850474e-10,
                -2.7755579345723475e-17
            ],
            dtype = float
        )
    ),
    56: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.9170462241482575,
                -0.7708050513059739,
                -0.6468519299440145,
                -0.5413662867733801,
                -0.45132873337444446,
                -0.3743312302049634,
                -0.3084368133297448,
                -0.2520745748552699,
                -0.20395997912230565,
                -0.16303353482158045,
                -0.12841283650395421,
                -0.09935436128818419,
                -0.07522237029565018,
                -0.05546296542661709,
                -0.03958189954871109,
                -0.02712522472698159,
                -0.017662380021222595,
                -0.010771964234270091,
                -0.006031299437761151,
                -0.0030119885714329953,
                -0.0012847815775502006,
                -0.0004373523623084323,
                -0.00010609806458790573,
                -0.000014962788142418203,
                -8.164516198321918e-7,
                -6.615128110762502e-9,
                -3.7713339783130447e-13,
                -1.9107758149499938e-27
            ],
            dtype = float
        )
    ),
    57: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.9184171272087063,
                -0.7742878853607295,
                -0.6517896169162875,
                -0.5472668587017261,
                -0.4578200581542426,
                -0.38112887660308536,
                -0.31532103690070534,
                -0.258873734998347,
                -0.21053857094143835,
                -0.1692836038086689,
                -0.13424794204287988,
                -0.10470553040733276,
                -0.08003567497057287,
                -0.05969848677346948,
                -0.043213918263772244,
                -0.030143493353930777,
                -0.020074268129448576,
                -0.012605085156715688,
                -0.007335879689380135,
                -0.003861702975161963,
                -0.0017741396063933933,
                -0.0006734046735467812,
                -0.00019329334683286398,
                -0.00003601327883503014,
                -3.2638225133693485e-6,
                -7.75072575750066e-8,
                -9.190232267972575e-11,
                -6.938894428451715e-18
            ],
            dtype = float
        )
    ),
    58: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.9197434717011816,
                -0.7776668293086058,
                -0.6565926486990374,
                -0.5530212190376891,
                -0.46416661449803953,
                -0.3877920610172556,
                -0.32208720358585674,
                -0.265575483140291,
                -0.21704336519126946,
                -0.17548555385263867,
                -0.140061964818732,
                -0.11006337030900197,
                -0.08488343396880615,
                -0.06399543798443126,
                -0.04693245160867293,
                -0.033270067954755,
                -0.022611208140933797,
                -0.014572925454811205,
                -0.008775706081530214,
                -0.004836506785160474,
                -0.00236765380593292,
                -0.000984452564564986,
                -0.0003240591671052822,
                -0.00007511641048071271,
                -9.930776956915244e-6,
                -4.913059439419576e-7,
                -3.3705352612482757e-9,
                -1.3576572476897792e-13,
                -2.123084238833319e-28
            ],
            dtype = float
        )
    ),
    59: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.9210273945370914,
                -0.780946444884102,
                -0.6612663837047974,
                -0.5586345468422007,
                -0.4703728194686114,
                -0.39432409184753064,
                -0.32873730705237764,
                -0.27218037622960944,
                -0.22347341300721962,
                -0.18163690043857977,
                -0.14585089384941718,
                -0.11542240297757862,
                -0.08975883392391298,
                -0.06834590655818809,
                -0.050728869550097394,
                -0.03649617201913728,
                -0.0252650718760034,
                -0.016669021915272694,
                -0.010347134315021645,
                -0.005936632224107085,
                -0.003069950365331242,
                -0.0013788768764416604,
                -0.0005082812653176181,
                -0.0001403201479143101,
                -0.000024770942860376483,
                -2.0747763733627365e-6,
                -4.34664391567354e-8,
                -4.08350369379458e-11,
                -1.7347235497408708e-18
            ],
            dtype = float
        )
    ),
    60: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.9222708981868007,
                -0.7841310317924899,
                -0.6658159047923511,
                -0.5641117925179371,
                -0.47644293956551564,
                -0.40072822066866814,
                -0.33527338544571306,
                -0.2786891202902296,
                -0.22982801969968272,
                -0.1877355158736361,
                -0.1516111706909334,
                -0.12077768442099635,
                -0.09465565132105408,
                -0.07274258327488013,
                -0.05459509371831686,
                -0.03981343655467195,
                -0.028027874516061434,
                -0.018886655016848365,
                -0.012045748538277952,
                -0.007161001337990534,
                -0.0038839865274134487,
                -0.0018635168878164408,
                -0.000755161595840687,
                -0.00024034197389327682,
                -0.00005322436758928623,
                -6.595393313904605e-6,
                -2.9579875736118245e-7,
                -1.7177168033392493e-9,
                -4.8875077107171076e-14,
                -2.358982487592574e-29
            ],
            dtype = float
        )
    ),
    61: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.9234758610847342,
                -0.7872246461625116,
                -0.6702460363365698,
                -0.5694576891302343,
                -0.4823810944011633,
                -0.4070076375094646,
                -0.34169750802366106,
                -0.28510254841354815,
                -0.23610671435282315,
                -0.19377959116696938,
                -0.15733964505228187,
                -0.1261247567894946,
                -0.09956820592174331,
                -0.07717872460666438,
                -0.05852358012817192,
                -0.043213918263772244,
                -0.030891845710366517,
                -0.021218987687129015,
                -0.013866568344366514,
                -0.008507468130428993,
                -0.004811238155284079,
                -0.0024436561575537016,
                -0.0010728795716081214,
                -0.00038402091775539067,
                -0.00010194922782065828,
                -0.000017049971487078265,
                -1.3196537818027081e-6,
                -2.4385108646727007e-8,
                -1.8145242344514563e-11,
                -4.336808793672729e-19
            ],
            dtype = float
        )
    ),
    62: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.9246440470832125,
                -0.7902311174703596,
                -0.67456136008673,
                -0.5746767631908141,
                -0.48819126063884694,
                -0.4131654672269919,
                -0.34801176371882186,
                -0.29142160149139734,
                -0.24230922291631948,
                -0.19976760201078173,
                -0.16303353482158045,
                -0.13145960466373172,
                -0.10449131718418751,
                -0.0816481155086671,
                -0.06250729764791062,
                -0.04669010639559142,
                -0.03384948010350252,
                -0.023659174021309768,
                -0.0158042213660128,
                -0.009973037076130776,
                -0.005851902460744381,
                -0.003123092751020668,
                -0.001468396390454024,
                -0.0005798516070203425,
                -0.00017840489903763428,
                -0.000037739878662370455,
                -4.382855899509381e-6,
                -1.78172397554411e-7,
                -8.755515165697274e-10,
                -1.759487946759481e-14,
                -2.6210916528806367e-30
            ],
            dtype = float
        )
    ),
    63: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.9257771140549149,
                -0.7931540640796975,
                -0.6787662299075237,
                -0.5797733449156959,
                -0.4938772761128658,
                -0.4192047668038275,
                -0.3542182513671092,
                -0.29764731134222616,
                -0.2484354443704526,
                -0.20569827840226937,
                -0.16869039006609063,
                -0.13677861521232657,
                -0.10942026370455886,
                -0.08614503321956303,
                -0.0665397038092314,
                -0.05023492118919319,
                -0.03689357163405768,
                -0.026200443073712772,
                -0.017853084969113246,
                -0.011554054885177998,
                -0.00700509919807898,
                -0.0039042616219389452,
                -0.001947375584049369,
                -0.0008356343140306815,
                -0.00029039377683615146,
                -0.00007412655752652698,
                -0.00001174294992498076,
                -8.397872968930662e-7,
                -1.3684683806787502e-8,
                -8.063266723078353e-12,
                -1.0842021870726353e-19
            ],
            dtype = float
        )
    ),
    64: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.9268766217320613,
                -0.7959969075271369,
                -0.6828647854893555,
                -0.5847515779719666,
                -0.4994428440655861,
                -0.425128523422359,
                -0.36031907137690444,
                -0.30378078592941193,
                -0.2544854295981636,
                -0.2115705774991035,
                -0.17430806060709111,
                -0.14207854191794025,
                -0.11435074562522658,
                -0.09066421248858227,
                -0.07061471914450802,
                -0.053841706185707425,
                -0.04001723515712169,
                -0.028836162217098456,
                -0.020007400953418374,
                -0.01324637525169032,
                -0.00826906012728915,
                -0.004788384629201518,
                -0.0025141897473873642,
                -0.001158243834670933,
                -0.0004456399864462366,
                -0.0001325318731478263,
                -0.00002677781240733567,
                -2.9141267371700047e-6,
                -1.0736530248313076e-7,
                -4.463515233457091e-10,
                -6.3341189411285774e-15,
                -2.912324058756263e-31
            ],
            dtype = float
        )
    ),
    65: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.9279440388601419,
                -0.7987628856690556,
                -0.686860965107505,
                -0.5896154287303924,
                -0.5048915374484522,
                -0.43093965319217514,
                -0.36631631864482367,
                -0.3098231964106005,
                -0.2604593626439014,
                -0.21738365934800682,
                -0.17988466680747983,
                -0.14735647157875656,
                -0.11927884990373686,
                -0.09520081248947698,
                -0.07472670092419377,
                -0.057504216177187255,
                -0.043213918263772244,
                -0.031559883926409966,
                -0.02226136714642051,
                -0.015045497997790041,
                -0.009641301182126574,
                -0.005775630735684152,
                -0.003171986019617984,
                -0.0015535028240725908,
                -0.0006514476504332299,
                -0.00021976849353276965,
                -0.000053933582983624585,
                -8.09241153448564e-6,
                -5.346611235149531e-7,
                -7.68187522923712e-9,
                -3.5832229445198317e-12,
                -2.710505451726914e-20
            ],
            dtype = float
        )
    ),
    66: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.9289807497351108,
                -0.8014550647936335,
                -0.6907585175032586,
                -0.5943686950424314,
                -0.5102268032440502,
                -0.43664100042510956,
                -0.3722120765508517,
                -0.3157757657913194,
                -0.2663575440791824,
                -0.22313686516767522,
                -0.18541857324490088,
                -0.15260979430544128,
                -0.12420101830362021,
                -0.09975038556799876,
                -0.07887041692797532,
                -0.06121660215529754,
                -0.04647740560314562,
                -0.03436537923790451,
                -0.024609209517892243,
                -0.016946684907322866,
                -0.011118774918424256,
                -0.0068652747165336295,
                -0.0039227878526243674,
                -0.0020261382656597457,
                -0.0009144529778441629,
                -0.00034277206027482797,
                -0.00009852337121110816,
                -0.000019011143554262923,
                -1.938532584285646e-6,
                -6.472155501373304e-8,
                -2.275763890987391e-10,
                -2.2802732509267065e-15,
                -3.235915620840292e-32
            ],
            dtype = float
        )
    ),
    67: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.9299880601850998,
                -0.8040763507913813,
                -0.694561012954136,
                -0.5990150145611707,
                -0.5154519667746886,
                -0.4422353373682158,
                -0.37800841188847484,
                -0.3216397589856498,
                -0.27218037622960944,
                -0.22882969790418548,
                -0.19090836497567984,
                -0.15783617625007362,
                -0.1291140179493511,
                -0.10430884788721467,
                -0.083041019700892,
                -0.06497339430378041,
                -0.04980181754501701,
                -0.037246660588333085,
                -0.027045238071059182,
                -0.01894505491605461,
                -0.012698002843863458,
                -0.008055847107441354,
                -0.004767615975192454,
                -0.002579799043400562,
                -0.0012404662517279232,
                -0.0005082812653032449,
                -0.00016644000666314276,
                -0.00003926571632517079,
                -5.579608379006211e-6,
                -3.4054125460637395e-7,
                -4.31327591739565e-9,
                -1.5923851689455433e-12,
                -6.776263606881024e-21
            ],
            dtype = float
        )
    ),
    68: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.9309672030509799,
                -0.8066294994680026,
                -0.6982718535948786,
                -0.6035578726262254,
                -0.5205702359698935,
                -0.4477253643182878,
                -0.3837073706050338,
                -0.32741647411220187,
                -0.27792835004836064,
                -0.23446180480961895,
                -0.19635282612472624,
                -0.16303353482158045,
                -0.1340149142789065,
                -0.10887245197653109,
                -0.08723402160740454,
                -0.06876948382799264,
                -0.053181604639360586,
                -0.04019799626173427,
                -0.029563889356914744,
                -0.021035661382529856,
                -0.014375188460706323,
                -0.009345271306090842,
                -0.005706616296799008,
                -0.003217115053028041,
                -0.0016343904403624925,
                -0.0007225877464084826,
                -0.00026384491076829314,
                -0.00007328856987865137,
                -0.00001350445263171644,
                -1.2901267515459787e-6,
                -3.90282796152472e-8,
                -1.1604423770396207e-10,
                -8.208959399737821e-16,
                -3.595461800933658e-33
            ],
            dtype = float
        )
    ),
    69: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.9319193432140204,
                -0.8091171260750613,
                -0.7018942830458227,
                -0.608000609732787,
                -0.5255847055709587,
                -0.45311371005286627,
                -0.3893109742443116,
                -0.3331072348755986,
                -0.28360203344747464,
                -0.2400329618233787,
                -0.20175092056381488,
                -0.16820001616079114,
                -0.13890104622536414,
                -0.11343776114956021,
                -0.09144527089218696,
                -0.07260010421876946,
                -0.056611539020617,
                -0.043213918263772244,
                -0.032159758054026474,
                -0.023213554038733843,
                -0.01614631253706751,
                -0.010730985939618774,
                -0.006739186449048859,
                -0.003939781962028146,
                -0.0021001999049433747,
                -0.0009913601078032545,
                -0.00039687867620964007,
                -0.00012613522859676417,
                -0.00002860309284519784,
                -3.848884513438656e-6,
                -2.169834020865414e-7,
                -2.422376761306991e-9,
                -7.076711494277493e-13,
                -1.694065898565157e-21
            ],
            dtype = float
        )
    ),
    70: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.9328455822137345,
                -0.8115417141264629,
                -0.7054313954006726,
                -0.612346428604891,
                -0.5304983612554501,
                -0.4584029325223086,
                -0.39482121699779965,
                -0.33871338390265476,
                -0.28920206092117,
                -0.24554305956101713,
                -0.2071017744653578,
                -0.17333397466664976,
                -0.1437700034616689,
                -0.11800162572838853,
                -0.09567092887836394,
                -0.07646081239512423,
                -0.060086703656012395,
                -0.046289225100643344,
                -0.0348276196805752,
                -0.02547382799860206,
                -0.01800721244414704,
                -0.012210052064396718,
                -0.007864095699354085,
                -0.0047486595096031025,
                -0.002640963005575213,
                -0.0013195358099949539,
                -0.0005714196425089323,
                -0.00020322910214813457,
                -0.000054549045504936085,
                -9.597564227935256e-6,
                -8.589533563311568e-7,
                -2.3541876998239512e-8,
                -5.917780675716349e-11,
                -2.9552192104649525e-16,
                -3.994957556592953e-34
            ],
            dtype = float
        )
    ),
    71: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.9337469624943455,
                -0.8139056235621438,
                -0.7088861436214755,
                -0.6165984008926458,
                -0.5353140836684926,
                -0.4635955197557124,
                -0.40024006328348805,
                -0.344236276918866,
                -0.294729124315393,
                -0.2509920907377216,
                -0.2124046605410638,
                -0.17843395438291243,
                -0.1486196055484827,
                -0.12256116099394455,
                -0.09990744837593148,
                -0.08034747005172799,
                -0.06360248013842429,
                -0.04941898065440764,
                -0.03756244616908695,
                -0.02781166194000626,
                -0.019953647487742487,
                -0.013779245675216389,
                -0.009079595195918423,
                -0.005643874598555006,
                -0.003258894498401575,
                -0.0017112696403317243,
                -0.0007929008144967314,
                -0.00031010732626540397,
                -0.00009564821533073803,
                -0.000020846674740964612,
                -2.656159007317023e-6,
                -1.3830393460638646e-7,
                -1.360689895815997e-9,
                -3.145009940164318e-13,
                -4.235164741976034e-22
            ],
            dtype = float
        )
    ),
    72: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.9346244713142269,
                -0.8162110983144033,
                -0.7122613473847169,
                -0.6207594735126902,
                -0.5400346523509303,
                -0.46869389094048325,
                -0.40556944578172666,
                -0.34967727766506174,
                -0.30018396461519686,
                -0.25638013887327005,
                -0.21765898379491538,
                -0.18349867207153026,
                -0.15344788283277397,
                -0.12711372677142355,
                -0.10415155333046623,
                -0.08425622544544807,
                -0.06715453556574953,
                -0.0525985101114943,
                -0.04035941574068414,
                -0.03022234730543179,
                -0.021981352113070868,
                -0.01543513653057383,
                -0.010383517067452087,
                -0.006624923181227694,
                -0.00395542632558478,
                -0.002169926091243567,
                -0.0010661846710345794,
                -0.00045219563829418955,
                -0.0001566361456900833,
                -0.00004062286862867069,
                -6.82405222224955e-6,
                -5.720972541067221e-7,
                -1.4204345511170045e-8,
                -3.0180534437464106e-11,
                -1.0638773476258269e-16,
                -4.438841729547726e-35
            ],
            dtype = float
        )
    ),
    73: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.9354790443491272,
                -0.8184602733270596,
                -0.7155597004189185,
                -0.6248324746505527,
                -0.5446627495571188,
                -0.47370039764131494,
                -0.4108112638669199,
                -0.3550377534664347,
                -0.30556736463675804,
                -0.2617073681425995,
                -0.2228642686378536,
                -0.18852700181472193,
                -0.15825305895367842,
                -0.13165690855415885,
                -0.108400219711311,
                -0.08818349578467126,
                -0.07073880892278446,
                -0.05582339370859082,
                -0.043213918263772244,
                -0.0327013101073506,
                -0.02408607873356099,
                -0.017174154571158704,
                -0.011773361957269907,
                -0.007690767070537606,
                -0.004731288090542393,
                -0.002698103672803587,
                -0.001395489298957652,
                -0.0006346266694923304,
                -0.00024246084902087154,
                -0.00007257003155847636,
                -0.000015200796057831279,
                -1.8337740550314983e-6,
                -8.818219030103214e-8,
                -7.644509170239256e-10,
                -1.3977136738627003e-13,
                -1.0587911848700753e-22
            ],
            dtype = float
        )
    ),
    74: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.936311569016803,
                -0.8206551810728424,
                -0.7187837773708822,
                -0.6288201194429273,
                -0.5492009639573276,
                -0.4786173251294592,
                -0.415967382381795,
                -0.36031907137690444,
                -0.31088014252411833,
                -0.2669740142514,
                -0.22802014722763456,
                -0.19351796100239052,
                -0.16303353482158045,
                -0.13618850006747318,
                -0.11265065761641133,
                -0.09212595033111586,
                -0.07435149728191234,
                -0.05908945890328844,
                -0.04612155706873222,
                -0.03524412668639184,
                -0.026263631766834255,
                -0.018992645308724573,
                -0.01324637525169032,
                -0.008839923386062229,
                -0.005586590880342169,
                -0.0032976808754246138,
                -0.0017843566260290111,
                -0.0008621057382932476,
                -0.0003580788699682607,
                -0.00012079384346691925,
                -0.000030266868892558904,
                -4.854057006314838e-6,
                -3.8117110322205556e-7,
                -8.572517958441022e-9,
                -1.5392976308840345e-11,
                -3.829954468142509e-17,
                -4.93204636616414e-36
            ],
            dtype = float
        )
    ),
    75: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.9371228875478413,
                -0.8227977576102216,
                -0.7219360402347508,
                -0.6327250153571523,
                -0.5536517942215277,
                -0.48344689379753286,
                -0.4210396307078791,
                -0.36552259483112853,
                -0.31612314596237484,
                -0.27218037622960944,
                -0.23312634891168277,
                -0.1984706975750291,
                -0.16778787394593128,
                -0.14070648717480405,
                -0.1169002945557122,
                -0.09608049428273208,
                -0.07798904206020539,
                -0.062392771449656784,
                -0.04907814801124865,
                -0.03784653455822445,
                -0.028509894279948326,
                -0.020886915555180935,
                -0.014799612673578582,
                -0.010070545473290449,
                -0.006520910039634074,
                -0.0039698754984384245,
                -0.002235653070551355,
                -0.0011388216380953217,
                -0.0005082812653035224,
                -0.00018968110079006488,
                -0.000055088160012730454,
                -0.0000110888127697753,
                -1.2664741649030264e-6,
                -5.624111800285509e-8,
                -4.295405960328379e-10,
                -6.211820188612614e-14,
                -2.646977961297782e-23
            ],
            dtype = float
        )
    ),
    76: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.937913799824977,
                -0.8248898482170532,
                -0.7250188443753437,
                -0.6365496672844415,
                -0.5580176524828263,
                -0.48819126063884694,
                -0.4260298020917955,
                -0.3706496807446234,
                -0.3212972470292256,
                -0.2773268090475632,
                -0.23818269066358583,
                -0.20338447840460913,
                -0.17251478899674336,
                -0.14520903303096094,
                -0.12114675986484279,
                -0.1000442534752535,
                -0.0816481155086671,
                -0.06572962575613937,
                -0.052079716427938814,
                -0.04050443929955315,
                -0.030820848466562748,
                -0.022853270790489887,
                -0.016429996135460142,
                -0.011380494910810045,
                -0.00753336401568144,
                -0.004715310995725254,
                -0.002751593697154667,
                -0.0014683963904539292,
                -0.0006975569031229516,
                -0.00028371964461306495,
                -0.0000932018507154767,
                -0.00002256113081947866,
                -3.4540949091447026e-6,
                -2.5404345622373725e-7,
                -5.1747854729844296e-9,
                -7.851299514600243e-12,
                -1.378782596710295e-17,
                -5.480051517960155e-37
            ],
            dtype = float
        )
    ),
    77: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.9386850660110747,
                -0.8269332126350126,
                -0.7280344441747316,
                -0.6402964823626514,
                -0.5623008676800109,
                -0.49285252077344793,
                -0.43093965319217514,
                -0.37570167700957985,
                -0.3264033376157164,
                -0.28241371697003304,
                -0.2431890684153133,
                -0.2082586787072019,
                -0.17721312949380566,
                -0.14969446439113673,
                -0.12538787019444322,
                -0.10401455991676022,
                -0.08532560756085665,
                -0.06909653482038225,
                -0.05512249250466449,
                -0.043213918263772244,
                -0.03319259100981146,
                -0.024888045362078413,
                -0.018134360843808175,
                -0.01276740473275999,
                -0.008622687530767262,
                -0.0055340851829702,
                -0.0033337823005653647,
                -0.001853866575184715,
                -0.0009299939001212747,
                -0.0004073364597678398,
                -0.0001484701878774761,
                -0.00004183717394070694,
                -8.092414417421647e-6,
                -8.749701438224565e-7,
                -3.587924187089961e-8,
                -2.413877604681891e-10,
                -2.7607244991755772e-14,
                -6.617444902010602e-24
            ],
            dtype = float
        )
    ),
    78: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.9394374089837942,
                -0.8289295299557259,
                -0.7309849983287506,
                -0.6439677745435994,
                -0.5665036887796315,
                -0.4974327090058076,
                -0.4357709038164786,
                -0.38067992034017606,
                -0.331442325354889,
                -0.28744154757265344,
                -0.24814544919743944,
                -0.2130927723913125,
                -0.18188187052632848,
                -0.15416125898881333,
                -0.12962161601694464,
                -0.1079889381524868,
                -0.08901861313031217,
                -0.07249021996913861,
                -0.05820290547636302,
                -0.04597122178055875,
                -0.035621344230845196,
                -0.026987626587222993,
                -0.019909494660974525,
                -0.014228734313810132,
                -0.009787298162297921,
                -0.006425838042661158,
                -0.003983261321712213,
                -0.0022976888365604763,
                -0.0012092160027128566,
                -0.00056476355998133,
                -0.000224925233189313,
                -0.00007194712999811506,
                -0.000016824239779719968,
                -2.4587685818225375e-6,
                -1.693648576307353e-7,
                -3.1243798942110355e-9,
                -4.004793250540924e-12,
                -4.9636147779762344e-18,
                -6.08894613106684e-38
            ],
            dtype = float
        )
    ),
    79: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.9401715165934433,
                -0.8308804031767374,
                -0.7338725748180573,
                -0.6475657689192028,
                -0.5706282878788406,
                -0.5019338014014463,
                -0.44052523682094125,
                -0.38558573442663546,
                -0.33641513000392725,
                -0.2924107863534194,
                -0.2530518640087444,
                -0.21788632325517238,
                -0.1865201024138841,
                -0.15860803390064954,
                -0.1338461490911528,
                -0.11196509244468177,
                -0.09272441991682576,
                -0.07590760057817472,
                -0.061317576993922006,
                -0.04877277237849888,
                -0.03810346378602321,
                -0.029148473705427412,
                -0.021752170694954852,
                -0.015761816547152616,
                -0.011025355988352269,
                -0.00738981645636397,
                -0.004700566907595812,
                -0.0028017628844972707,
                -0.0015383487038848165,
                -0.0007599350639808006,
                -0.0003266219297871001,
                -0.00011627041200159543,
                -0.00003178732123737228,
                -5.90788744628185e-6,
                -6.046797230362033e-7,
                -2.2894931567267092e-8,
                -1.3566752188824737e-10,
                -1.2269590052413782e-14,
                -1.6543612253291403e-24
            ],
            dtype = float
        )
    ),
    80: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.9408880437587027,
                -0.8327873634529586,
                -0.7366991555764287,
                -0.6510926058199559,
                -0.5746767631908141,
                -0.506357716871799,
                -0.44520429815026247,
                -0.3904204283620504,
                -0.3413226802314788,
                -0.29732195187918226,
                -0.25790840134467663,
                -0.2226389769546214,
                -0.19112702122712502,
                -0.16303353482158045,
                -0.13805977082503273,
                -0.11594089474352129,
                -0.09644049675896507,
                -0.07934578390462355,
                -0.06446331392549638,
                -0.05161516247310297,
                -0.04063544355571373,
                -0.03136713250759018,
                -0.023659174021309768,
                -0.017363898034798944,
                -0.01233481634588283,
                -0.008424934522887631,
                -0.005485785652010685,
                -0.0033674663481670524,
                -0.001920009239251039,
                -0.0009964146780020926,
                -0.0004575103591855595,
                -0.0001784048991019287,
                -0.00005556428135729614,
                -0.000012550977551931001,
                -1.7508279229222369e-6,
                -1.129421849056652e-7,
                -1.886750775886922e-9,
                -2.0428439102873864e-12,
                -1.786900667205543e-18,
                -6.765495701185377e-39
            ],
            dtype = float
        )
    ),
    81: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.941587614413733,
                -0.8346518740670261,
                -0.7394666408772448,
                -0.6545503446985452,
                -0.5786511419150768,
                -0.510706318758358,
                -0.4498096969966236,
                -0.39518529531017355,
                -0.34616591076718556,
                -0.30217559141348954,
                -0.2627152013213892,
                -0.2273504536707813,
                -0.19570192009382767,
                -0.1674366261785252,
                -0.1422609214782181,
                -0.11991437341905577,
                -0.10016448255288105,
                -0.08280205513043146,
                -0.06763710080412551,
                -0.054495150881740326,
                -0.043213918263772244,
                -0.03364024635690846,
                -0.0256273233606593,
                -0.01903217303507292,
                -0.01371347599892277,
                -0.009529828730051048,
                -0.0063386105641958825,
                -0.003995696944678743,
                -0.002356314789126424,
                -0.0012773503619144776,
                -0.0006213324796373401,
                -0.0002620365304153678,
                -0.00009109567162513759,
                -0.000024161227848243922,
                -4.314548525852306e-6,
                -4.18006302418534e-7,
                -1.4612806430793983e-8,
                -7.625708061966683e-11,
                -5.4530469836876426e-15,
                -4.135903063078852e-25
            ],
            dtype = float
        )
    ),
    82: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.9422708233187396,
                -0.8364753341399276,
                -0.742176853457505,
                -0.6579409678107142,
                -0.5825533829954261,
                -0.5149814164086162,
                -0.4543430060602154,
                -0.3998816113860608,
                -0.3509457598751797,
                -0.3069722769777963,
                -0.26747245033851624,
                -0.23202054141355122,
                -0.20024418122233,
                -0.17181628201617619,
                -0.1464481701476368,
                -0.12388370272026546,
                -0.10389417574427147,
                -0.08627386768874377,
                -0.07083609208834933,
                -0.05740965845978671,
                -0.04583566427766605,
                -0.035964564215037194,
                -0.027653488448791463,
                -0.020763811897088154,
                -0.015159013163606197,
                -0.010702907681491713,
                -0.00725839461168445,
                -0.0046869184590665304,
                -0.002848903837645714,
                -0.001605451822266848,
                -0.0008215445224040328,
                -0.0003708205951350556,
                -0.00014157246428842579,
                -0.000042929604599644914,
                -9.366445875841599e-6,
                -1.247099936342178e-6,
                -7.53351966874322e-8,
                -1.1395593666469787e-9,
                -1.042087535136695e-12,
                -6.432840743558172e-19,
                -7.517217445761529e-40
            ],
            dtype = float
        )
    ),
    83: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.9429382377451239,
                -0.8382590821014633,
                -0.7448315423972451,
                -0.6612663837047974,
                -0.5863853797684231,
                -0.5191847667376046,
                -0.45880576189571226,
                -0.40451063472467935,
                -0.35566316711747875,
                -0.31171260180312416,
                -0.27218037622960944,
                -0.23664908990311478,
                -0.20475326858042922,
                -0.17617157759324895,
                -0.15062020548201335,
                -0.12784719292500704,
                -0.10762752439049594,
                -0.08975883392391298,
                -0.07405760436590109,
                -0.06035576309653652,
                -0.04849759896334731,
                -0.03833694619670618,
                -0.02973460375063806,
                -0.022555984672119847,
                -0.016669021915272694,
                -0.01194239637829077,
                -0.0082442007956384,
                -0.0054412079295129995,
                -0.0033989664454857184,
                -0.001982986210587484,
                -0.0010612635620828464,
                -0.0005082812653035308,
                -0.00021032271892000478,
                -0.000071401981900544,
                -0.00001837151146113408,
                -3.151931207226061e-6,
                -2.890390025723752e-7,
                -9.328639094682778e-9,
                -4.2866974788468516e-11,
                -2.4235398654603244e-15,
                -1.0339757657354005e-25
            ],
            dtype = float
        )
    ),
    84: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.9435903990451184,
                -0.8400043989384233,
                -0.7474323867708736,
                -0.6645284305307229,
                -0.5901489625056404,
                -0.5233180757699113,
                -0.4631994653311067,
                -0.4090736047154368,
                -0.36031907137690444,
                -0.3163971771337224,
                -0.27683924385429926,
                -0.24123600497719505,
                -0.2092287211733016,
                -0.18050168163225416,
                -0.15477582707368298,
                -0.13180328114354786,
                -0.11136261678255059,
                -0.09325471611918149,
                -0.07729910860093556,
                -0.06333069426336559,
                -0.05119677890474913,
                -0.040754367096656995,
                -0.03186767908781832,
                -0.024405880538151577,
                -0.018241041532691123,
                -0.01324637525169032,
                -0.00929484817214799,
                -0.0062583021737298685,
                -0.0040072800220017244,
                -0.0024117876592652295,
                -0.0013432353014765123,
                -0.0006777320668527781,
                -0.0003007036189351296,
                -0.00011239304150033956,
                -0.00003318058927438572,
                -6.992236632308364e-6,
                -8.885494357645786e-7,
                -5.026209559316715e-8,
                -6.88373768589226e-10,
                -5.315999891610714e-13,
                -2.315822246425805e-19,
                -8.352463828623921e-41
            ],
            dtype = float
        )
    ),
    85: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.9442278241151626,
                -0.8417125112368709,
                -0.7499809990857109,
                -0.6677288791786562,
                -0.5938459008529889,
                -0.5273830001580015,
                -0.46752558194704286,
                -0.4135717413831068,
                -0.3649144091124196,
                -0.3210266293482695,
                -0.28144935109083485,
                -0.2457812434767895,
                -0.21367014686903232,
                -0.1848058491702828,
                -0.1589139374790053,
                -0.13575052273819616,
                -0.11509767261135032,
                -0.09675941791309638,
                -0.08055822250136178,
                -0.06633182726866019,
                -0.05393039724365547,
                -0.043213918263772244,
                -0.03404980767414662,
                -0.026310723613957174,
                -0.019872581331890002,
                -0.014612814264856687,
                -0.0104089535619141,
                -0.0071376423834967955,
                -0.004674247949436733,
                -0.002893276809094729,
                -0.0016698194904465114,
                -0.0008822172713134204,
                -0.0004160061347268737,
                -0.0001688897891454568,
                -0.0000559878259053218,
                -0.000013973980675767636,
                -2.3032789377471366e-6,
                -1.9991162786669165e-7,
                -5.956432302460386e-9,
                -2.4099012667664283e-11,
                -1.0771159870270187e-15,
                -2.58493941429025e-26
            ],
            dtype = float
        )
    ),
    86: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.9448510067612127,
                -0.8433845940335679,
                -0.7524789285218672,
                -0.6708694362568748,
                -0.5974779061705426,
                -0.5313811486734646,
                -0.471785542606305,
                -0.418006244897825,
                -0.36945011282267126,
                -0.3256015973676943,
                -0.28601102519176497,
                -0.25028480856761787,
                -0.2180772167249303,
                -0.18908341496245412,
                -0.16303353482158045,
                -0.13968758332200534,
                -0.11883103465908301,
                -0.10027097611563089,
                -0.08383270306363907,
                -0.06935667734281006,
                -0.05669578035026253,
                -0.04571280813775999,
                -0.03627817198537529,
                -0.02826778567776887,
                -0.021561141519045625,
                -0.016039602469581294,
                -0.011584968921041425,
                -0.008078415971204818,
                -0.005399939076669846,
                -0.0034284871067288233,
                -0.0020429893349672205,
                -0.0011244725036385983,
                -0.0005593761144573936,
                -0.00024395527417076461,
                -0.00008926378175031317,
                -0.000025654621113552353,
                -5.221452275683917e-6,
                -6.332510570785836e-7,
                -3.354107335473843e-8,
                -4.15882346996396e-10,
                -2.711912218496679e-13,
                -8.336959017078111e-20,
                -9.280515365137692e-42
            ],
            dtype = float
        )
    ),
    87: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.9454604189736517,
                -0.8450217734903117,
                -0.7549276639865644,
                -0.6739517469179084,
                -0.6010466337763279,
                -0.5353140836684926,
                -0.4759807440244282,
                -0.4223782951987917,
                -0.3739271096961021,
                -0.3301227303218548,
                -0.2905246194691923,
                -0.2547467454585661,
                -0.22244965977198328,
                -0.19333378739355966,
                -0.167133705935387,
                -0.14361323130047104,
                -0.12256116099394455,
                -0.10378755292723374,
                -0.08712043933699969,
                -0.07240289365095648,
                -0.059490383996732094,
                -0.04824836171274531,
                -0.0385500478287332,
                -0.030274396245981418,
                -0.023304230556725175,
                -0.01752457343408459,
                -0.012821214500796475,
                -0.009079595195918423,
                -0.00618412665125575,
                -0.004018095276792999,
                -0.0024643415163550964,
                -0.0014069019750037569,
                -0.0007337529721256157,
                -0.00034064002752079664,
                -0.00013567501796417033,
                -0.00004391731779482696,
                -0.000010632474875091728,
                -1.6835882648428745e-6,
                -1.3829936470451606e-7,
                -3.803915710693874e-9,
                -1.3548941005150998e-11,
                -4.787137069492593e-16,
                -6.46234853565777e-27
            ],
            dtype = float
        )
    ),
    88: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.9460565121186315,
                -0.8466251294038676,
                -0.757328636995025,
                -0.6769773975414586,
                -0.6045536850975592,
                -0.5391833225054868,
                -0.48011254937357306,
                -0.4266890517180425,
                -0.3783463204282718,
                -0.33459068545012266,
                -0.2949905102793249,
                -0.2591671374820629,
                -0.22678725821861342,
                -0.19755644285704518,
                -0.17121362000781717,
                -0.14752633092140655,
                -0.12628661764504948,
                -0.10730742855813191,
                -0.0904194454367715,
                -0.07546825331023212,
                -0.06231178917431661,
                -0.05081801914861918,
                -0.040862806924329066,
                -0.032327950411292744,
                -0.025099379496317627,
                -0.019065526962716527,
                -0.0141159080232362,
                -0.010139972348196387,
                -0.007026325700477402,
                -0.004662453947800795,
                -0.002935113868081053,
                -0.0017315694261607862,
                -0.0009418253574579362,
                -0.0004619057533063441,
                -0.00019799981881378633,
                -0.00007092089732701259,
                -0.00001984224403943626,
                -3.900239122549892e-6,
                -4.514154984136776e-7,
                -2.238723324784235e-8,
                -2.512869518178057e-10,
                -1.3834855924169295e-13,
                -3.001304974337203e-20,
                -1.0311683739041878e-42
            ],
            dtype = float
        )
    ),
    89: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.9466397180522276,
                -0.8481956975631387,
                -0.7596832243891832,
                -0.6799479182821073,
                -0.608000609732787,
                -0.5429903389531829,
                -0.4841822889128085,
                -0.43093965319217514,
                -0.38270865818905947,
                -0.3390061262134321,
                -0.29940909427898893,
                -0.26354610250460847,
                -0.23108984303835214,
                -0.20175092056381488,
                -0.17527252268534757,
                -0.15142583579966973,
                -0.13000607173352513,
                -0.11082899424079518,
                -0.09372785382670915,
                -0.07855065547112446,
                -0.06515769766823047,
                -0.05341933371404852,
                -0.043213918263772244,
                -0.034425914788171995,
                -0.026944153685643752,
                -0.0206602475161583,
                -0.015467190157711428,
                -0.011258191452027798,
                -0.007925828216699436,
                -0.00536162490848918,
                -0.003456208241866604,
                -0.0021001999049433742,
                -0.0011860021245564086,
                -0.0006105633603668626,
                -0.00027904746432342713,
                -0.00010903435326931606,
                -0.000034460853572423245,
                -8.092414640311792e-6,
                -1.230939504142446e-6,
                -9.569649670984529e-8,
                -2.429664823584637e-9,
                -7.617940288683483e-12,
                -2.1276006400019295e-16,
                -1.6155871339049005e-27
            ],
            dtype = float
        )
    ),
    90: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.9472104501631277,
                -0.8497344719642702,
                -0.7619927509046526,
                -0.6828647854893555,
                -0.6113889074283955,
                -0.5467365645481168,
                -0.48819126063884694,
                -0.43513121755127177,
                -0.38701502772422475,
                -0.34336972059759513,
                -0.30378078592941193,
                -0.2678837896386431,
                -0.2353572899092023,
                -0.20591681774643167,
                -0.17930973060722893,
                -0.1553107828850344,
                -0.13371828503555583,
                -0.11435074562522658,
                -0.09704390888252432,
                -0.0816481155086671,
                -0.06802592748251626,
                -0.056049969213783704,
                -0.045600948470845594,
                -0.036565831866995184,
                -0.028836162217098456,
                -0.022306519718098845,
                -0.016873146612091394,
                -0.012432776565324271,
                -0.008881735764259447,
                -0.006115411885102687,
                -0.0040282165714651715,
                -0.002514189747387362,
                -0.001468396390453929,
                -0.0007892256310976392,
                -0.0003815860013381835,
                -0.00016076441713481669,
                -0.000056367036433226164,
                -0.000015351469931598,
                -2.9141206269017037e-6,
                -3.218667681510578e-7,
                -1.4945311014055955e-8,
                -1.5185100046433117e-10,
                -7.057984379364979e-14,
                -1.0804697217170925e-20,
                -1.1457426376713199e-43
            ],
            dtype = float
        )
    ),
    91: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.947769104349198,
                -0.8512424068935798,
                -0.7642584915956343,
                -0.6857294240070897,
                -0.614720029972818,
                -0.5504233899206104,
                -0.49214073095205496,
                -0.43926484187545706,
                -0.39126632457741267,
                -0.3476821395896842,
                -0.30810601522494985,
                -0.27218037622960944,
                -0.23958951547530766,
                -0.2100537852271275,
                -0.1833246263350889,
                -0.15918028684319124,
                -0.13742210795331486,
                -0.11787127654442651,
                -0.10036596074274994,
                -0.084758759357857,
                -0.07091440818898208,
                -0.05870769702651425,
                -0.04802156135411948,
                -0.038745323036644935,
                -0.03077306543986716,
                -0.024002141305806537,
                -0.01833182716102176,
                -0.013662156841128714,
                -0.009892989319407826,
                -0.00692339206401555,
                -0.004651448576676985,
                -0.0029746224782509338,
                -0.001790820322814181,
                -0.0010002736517415512,
                -0.00050828126530353,
                -0.0002286832435683975,
                -0.00008765618760896801,
                -0.000027049197896200025,
                -6.1608810343638195e-6,
                -9.002054308642426e-7,
                -6.623070950193766e-8,
                -1.5521268084486473e-9,
                -4.2834405263459e-12,
                -9.455947237632098e-17,
                -4.038967834748833e-28
            ],
            dtype = float
        )
    ),
    92: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.9483160599327017,
                -0.852720418887378,
                -0.7664816741267577,
                -0.6885432093591589,
                -0.6179953830117888,
                -0.5540521660847638,
                -0.49603193533324846,
                -0.4433416024105961,
                -0.39546343442011994,
                -0.3519440558110826,
                -0.31238522562656185,
                -0.2764360650944762,
                -0.24378647390414995,
                -0.2141615233206581,
                -0.18731665364871206,
                -0.16303353482158045,
                -0.14111647387019477,
                -0.12138927313579251,
                -0.10369245944835805,
                -0.08788081801854582,
                -0.07382117625926102,
                -0.06139039285731511,
                -0.050473516811393,
                -0.04096209049865445,
                -0.03275258082072291,
                -0.025744933850683008,
                -0.0198412619282487,
                -0.014944688561005507,
                -0.010958396295097254,
                -0.007784945846478852,
                -0.005325959926055934,
                -0.0034822887235703417,
                -0.0021547883801531307,
                -0.0012458354551488683,
                -0.0006616481861368485,
                -0.0003153613032894974,
                -0.00013057928050932638,
                -0.000044814459353177,
                -0.000011880489198312997,
                -2.1778730517949532e-6,
                -2.2954533889146416e-7,
                -9.978953902717602e-9,
                -9.177176990561426e-11,
                -3.600747602843039e-14,
                -3.889690822798e-21,
                -1.2730473751903554e-44
            ],
            dtype = float
        )
    ),
    93: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.9488516805186871,
                -0.8541693885770896,
                -0.7686634809402186,
                -0.6913074698273419,
                -0.6212163277878519,
                -0.5576242056922056,
                -0.4998660790273849,
                -0.4473625546355739,
                -0.399607232478422,
                -0.3561561422924012,
                -0.3166188721817443,
                -0.2806510819901608,
                -0.24794815371484738,
                -0.21823977804543424,
                -0.19128531318048547,
                -0.16686978157346344,
                -0.14480039386743745,
                -0.12490350830320171,
                -0.10702194936892123,
                -0.09101262224767868,
                -0.07674437042664503,
                -0.06409603329007633,
                -0.05295466922018516,
                -0.043213918263772244,
                -0.0347724874021418,
                -0.027532751542897593,
                -0.021399475225888902,
                -0.016278674378551756,
                -0.01207665480058203,
                -0.008699284523726971,
                -0.0060515799905886845,
                -0.004037708593178507,
                -0.002561526945251438,
                -0.0015277750221088788,
                -0.0008440141980177809,
                -0.0004233086771918867,
                -0.0001874791563531894,
                -0.00007049321953025361,
                -0.000021237914890423332,
                -4.6915959039850334e-6,
                -6.584820156789982e-7,
                -4.58463328971739e-8,
                -9.916727559291787e-10,
                -2.408619042852334e-12,
                -4.2026236899083623e-17,
                -1.0097419586853213e-28
            ],
            dtype = float
        )
    ),
    94: cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        np.ascontiguousarray(
            [
                -0.9493763148005799,
                -0.8555901624273985,
                -0.7708050513059739,
                -0.694023488427622,
                -0.6243841828072738,
                -0.561140784249561,
                -0.5036443377307944,
                -0.45132873337444446,
                -0.4036985830464006,
                -0.36031907137690444,
                -0.320807419814355,
                -0.2848256732922525,
                -0.2520745748552699,
                -0.22228833761856984,
                -0.19523015836307875,
                -0.17068834491531193,
                -0.14847295178010833,
                -0.12841283650395421,
                -0.11035306391034967,
                -0.09415259745083325,
                -0.07968222711413828,
                -0.06682269220967961,
                -0.05546296542661709,
                -0.045498672393750904,
                -0.036830629074152105,
                -0.02936348830295137,
                -0.023004497234022654,
                -0.017662380021222595,
                -0.01324637525169032,
                -0.00966547571180895,
                -0.006827938575228681,
                -0.0046411553221080695,
                -0.003011988571433112,
                -0.001847689732435545,
                -0.0010574938028939742,
                -0.0005549263520467101,
                -0.0002607292110804812,
                -0.00010609806309865674,
                -0.00003564051026191885,
                -9.196780048361693e-6,
                -1.6280186407429953e-6,
                -1.6373727834044476e-7,
                -6.664007980961474e-9,
                -5.546770274585565e-11,
                -1.8370021829170875e-14,
                -1.4002886516570633e-21,
                -1.4144970835448392e-45
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
    Poles of the reciprocal of the z-transform of a B-spline :math:`\beta^{n}.`

    See Also
    --------
    splinekit.bsplines.pole : Poles of polynomial B-splines.

    Notes
    -----
    The results of this method are cached. If the returned results are
    mutated, the cache gets modified and the next call will return corrupted
    values.

    ----

    """

    if 0 > n:
        raise ValueError("The degree n must be nonnegative")
    if n in _poles:
        return _poles[n]
    if 0 == n:
        _poles[n] = cast(
            np.ndarray[tuple[int], np.dtype[np.float64]],
            np.array([], dtype = float)
        )
        return _poles[n]
    if 1 == n:
        _poles[n] = cast(
            np.ndarray[tuple[int], np.dtype[np.float64]],
            np.array([], dtype = float)
        )
        return _poles[n]
    if 2 == n:
        _poles[n] = cast(
            np.ndarray[tuple[int], np.dtype[np.float64]],
            np.ascontiguousarray([-3.0 + sqrt(8.0)], dtype = float)
        )
        return _poles[n]
    if 3 == n:
        _poles[n] = cast(
            np.ndarray[tuple[int], np.dtype[np.float64]],
            np.ascontiguousarray([-2.0 + sqrt(3.0)], dtype = float)
        )
        return _poles[n]
    if 4 == n:
        _poles[n] = cast(
            np.ndarray[tuple[int], np.dtype[np.float64]],
            np.ascontiguousarray(
                [
                    -19.0 + sqrt(304.0) + sqrt(664.0 - sqrt(438976.0)),
                    -19.0 - sqrt(304.0) + sqrt(664.0 + sqrt(438976.0))
                ],
                dtype = float
            )
        )
        return _poles[n]
    if 5 == n:
        _poles[n] = cast(
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

    Notes
    -----
    The results of this method are cached. If the returned results are
    mutated, the cache gets modified and the next call will return corrupted
    values.

    ----

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

    ----

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

    ----

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

    ----

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

    ----

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

    ----

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

    ----

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

    ----

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

    ----

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
