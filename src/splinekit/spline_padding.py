"""
..  role:: raw-html(raw)
    :format: html
"""

#---------------
from typing import cast
from typing import Dict
from typing import Tuple

#---------------
from math import floor

#---------------
import numpy as np

#---------------
import scipy

#---------------
from splinekit.bases import Bases

#---------------
from splinekit import pole

#---------------
from splinekit.spline_utilities import _b
from splinekit.spline_utilities import _sgn

#---------------
def pad_p (
    data: np.ndarray[tuple[int], np.dtype[np.float64]],
    *,
    at: int
) -> float:

    r"""
    .. _pad_p:

    Periodic padding.

    Returns the value of the :math:`k`-th data sample after the data have been
    extended by periodic padding. The padded data :math:`f` satisfy that

    ..  math::

        \forall k\in{\mathbb{Z}}:f[k]=f[k+K],

    where :math:`K` is the length of the unpadded data.

    :raw-html:`<TABLE border="1" frame="hsides" rules="groups" align="center">
    <CAPTION>Periodic padding of <FONT color="blue">data samples</FONT></CAPTION>
    <COLGROUP align="right">
    <COLGROUP align="center" span="7">
    <TR><TH>k<TH>&#160;&#160;&#160;<TH>&#402;<SUB>1</SUB>[k]<TH>&#402;<SUB>2</SUB>[k]<TH>&#402;<SUB>3</SUB>[k]<TH>&#402;<SUB>4</SUB>[k]<TH>&#402;<SUB>5</SUB>[k]<TH>&#402;<SUB>6</SUB>[k]
    <TBODY>
    <TR><TD>&#8722;20<TD><TD>a<TD>a<TD>b<TD>a<TD>a<TD>e
    <TR><TD>&#8722;19<TD><TD>a<TD>b<TD>c<TD>b<TD>b<TD>f
    <TR><TD>&#8722;18<TD><TD>a<TD>a<TD>a<TD>c<TD>c<TD>a
    <TR><TD>&#8722;17<TD><TD>a<TD>b<TD>b<TD>d<TD>d<TD>b
    <TR><TD>&#8722;16<TD><TD>a<TD>a<TD>c<TD>a<TD>e<TD>c
    <TR><TD>&#8722;15<TD><TD>a<TD>b<TD>a<TD>b<TD>a<TD>d
    <TR><TD>&#8722;14<TD><TD>a<TD>a<TD>b<TD>c<TD>b<TD>e
    <TR><TD>&#8722;13<TD><TD>a<TD>b<TD>c<TD>d<TD>c<TD>f
    <TR><TD>&#8722;12<TD><TD>a<TD>a<TD>a<TD>a<TD>d<TD>a
    <TR><TD>&#8722;11<TD><TD>a<TD>b<TD>b<TD>b<TD>e<TD>b
    <TR><TD>&#8722;10<TD><TD>a<TD>a<TD>c<TD>c<TD>a<TD>c
    <TR><TD>&#8722;9<TD><TD>a<TD>b<TD>a<TD>d<TD>b<TD>d
    <TR><TD>&#8722;8<TD><TD>a<TD>a<TD>b<TD>a<TD>c<TD>e
    <TR><TD>&#8722;7<TD><TD>a<TD>b<TD>c<TD>b<TD>d<TD>f
    <TR><TD>&#8722;6<TD><TD>a<TD>a<TD>a<TD>c<TD>e<TD>a
    <TR><TD>&#8722;5<TD><TD>a<TD>b<TD>b<TD>d<TD>a<TD>b
    <TR><TD>&#8722;4<TD><TD>a<TD>a<TD>c<TD>a<TD>b<TD>c
    <TR><TD>&#8722;3<TD><TD>a<TD>b<TD>a<TD>b<TD>c<TD>d
    <TR><TD>&#8722;2<TD><TD>a<TD>a<TD>b<TD>c<TD>d<TD>e
    <TR><TD>&#8722;1<TD><TD>a<TD>b<TD>c<TD>d<TD>e<TD>f
    <TR><TD>0<TD><TD><FONT color="blue"><B>A</B></FONT><TD><FONT color="blue"><B>A</B></FONT><TD><FONT color="blue"><B>A</B></FONT><TD><FONT color="blue"><B>A</B></FONT><TD><FONT color="blue"><B>A</B></FONT><TD><FONT color="blue"><B>A</B></FONT>
    <TR><TD>1<TD><TD>a<TD><FONT color="blue"><B>B</B></FONT><TD><FONT color="blue"><B>B</B></FONT><TD><FONT color="blue"><B>B</B></FONT><TD><FONT color="blue"><B>B</B></FONT><TD><FONT color="blue"><B>B</B></FONT>
    <TR><TD>2<TD><TD>a<TD>a<TD><FONT color="blue"><B>C</B></FONT><TD><FONT color="blue"><B>C</B></FONT><TD><FONT color="blue"><B>C</B></FONT><TD><FONT color="blue"><B>C</B></FONT>
    <TR><TD>3<TD><TD>a<TD>b<TD>a<TD><FONT color="blue"><B>D</B></FONT><TD><FONT color="blue"><B>D</B></FONT><TD><FONT color="blue"><B>D</B></FONT>
    <TR><TD>4<TD><TD>a<TD>a<TD>b<TD>a<TD><FONT color="blue"><B>E</B></FONT><TD><FONT color="blue"><B>E</B></FONT>
    <TR><TD>5<TD><TD>a<TD>b<TD>c<TD>b<TD>a<TD><FONT color="blue"><B>F</B></FONT>
    <TR><TD>6<TD><TD>a<TD>a<TD>a<TD>c<TD>b<TD>a
    <TR><TD>7<TD><TD>a<TD>b<TD>b<TD>d<TD>c<TD>b
    <TR><TD>8<TD><TD>a<TD>a<TD>c<TD>a<TD>d<TD>c
    <TR><TD>9<TD><TD>a<TD>b<TD>a<TD>b<TD>e<TD>d
    <TR><TD>10<TD><TD>a<TD>a<TD>b<TD>c<TD>a<TD>e
    <TR><TD>11<TD><TD>a<TD>b<TD>c<TD>d<TD>b<TD>f
    <TR><TD>12<TD><TD>a<TD>a<TD>a<TD>a<TD>c<TD>a
    <TR><TD>13<TD><TD>a<TD>b<TD>b<TD>b<TD>d<TD>b
    <TR><TD>14<TD><TD>a<TD>a<TD>c<TD>c<TD>e<TD>c
    <TR><TD>15<TD><TD>a<TD>b<TD>a<TD>d<TD>a<TD>d
    <TR><TD>16<TD><TD>a<TD>a<TD>b<TD>a<TD>b<TD>e
    <TR><TD>17<TD><TD>a<TD>b<TD>c<TD>b<TD>c<TD>f
    <TR><TD>18<TD><TD>a<TD>a<TD>a<TD>c<TD>d<TD>a
    <TR><TD>19<TD><TD>a<TD>b<TD>b<TD>d<TD>e<TD>b
    <TR><TD>20<TD><TD>a<TD>a<TD>c<TD>a<TD>a<TD>c
    <TR><TD>21<TD><TD>a<TD>b<TD>a<TD>b<TD>b<TD>d
    <TR><TD>22<TD><TD>a<TD>a<TD>b<TD>c<TD>c<TD>e
    <TR><TD>23<TD><TD>a<TD>b<TD>c<TD>d<TD>d<TD>f
    <TR><TD>24<TD><TD>a<TD>a<TD>a<TD>a<TD>e<TD>a
    <TR><TD>25<TD><TD>a<TD>b<TD>b<TD>b<TD>a<TD>b
    <TR><TD>26<TD><TD>a<TD>a<TD>c<TD>c<TD>b<TD>c
    <TR><TD>27<TD><TD>a<TD>b<TD>a<TD>d<TD>c<TD>d
    <TR><TD>28<TD><TD>a<TD>a<TD>b<TD>a<TD>d<TD>e
    <TR><TD>29<TD><TD>a<TD>b<TD>c<TD>b<TD>e<TD>f
    </TABLE>`

    Parameters
    ----------
    data : np.ndarray[tuple[int], np.dtype[np.float64]],
        Data to extend.
    at : int
        Arbitrary index.

    Returns
    -------
    float
        The value of the padded data at the requested index.

    Examples
    --------
    Load the library.
        >>> import splinekit as sk
    Short data at large index ``-42``.
        >>> sk.pad_p([1, 5, -3], at = -42)
        1

    ----

    """

    return data[at % len(data)]

#---------------
def change_basis_p (
    data: np.ndarray[tuple[int], np.dtype[np.float64]],
    *,
    degree: int,
    source_basis: Bases,
    target_basis: Bases
) -> None:

    r"""
    .. _change_basis_p:

    In-place conversion of data coefficients from one spline basis to another,
    under periodic padding.

    At input time, ``data`` is a one-dimensional ``numpy.ndarray`` of
    coefficients that is expressing the
    :ref:`uniform spline<def-uniform_spline>` :math:`f` of
    :ref:`nonnegative<def-negative>` degree :math:`n` as a weighted sum of
    integer-shifted bases of the type ``source_basis``.

    At output time, ``data`` is a one-dimensional ``numpy.ndarray`` of
    coefficients that is expressing the same spline :math:`f` as a weighted
    sum of integer-shifted bases of the type ``target_basis``.

    Parameters
    ----------
    data : np.ndarray[tuple[int], np.dtype[np.float64]],
        Data to convert from one basis to another.
    degree : int
        Nonnegative degree of the polynomial spline.
    source_basis : int
        Type of the basis at input time.
    target_basis : int
        Type of the basis at output time.

    Returns
    -------
    None :
        The returned coefficients overwrite ``data``.

    Examples
    --------
    Load the libraries.
        >>> import numpy as np
        >>> import splinekit as sk
    Arbitrary data samples with periodic padding.
        >>> f = np.array([1, 5, -3], dtype = "float")
    Cubic coefficients.
        >>> c = f.copy()
        >>> sk.change_basis_p(c, degree = 3, source_basis = sk.Bases.CARDINAL, target_basis = sk.Bases.BASIC)
        >>> print(c)
        [ 1.  9. -7.]

    Notes
    -----
    The computations are performed in-place.

    ----

    """

    if 0 == len(data):
        raise ValueError("Data must contain at least one element")
    if 0 > degree:
        raise ValueError("Degree must be nonnegative")
    if target_basis == Bases.BASIC:
        if source_basis == Bases.BASIC:
            return
        if source_basis == Bases.CARDINAL:
            samples_to_coeff_p(data, degree = degree)
            return
        if source_basis == Bases.DUAL:
            samples_to_coeff_p(data, degree = 2 * degree + 1)
            return
        if source_basis == Bases.ORTHONORMAL:
            ortho_to_coeff_p(data, degree = degree)
            return
        raise ValueError(
            "The source basis must be one of [CARDINAL, BASIC, ORTHONORMAL, DUAL]"
        )
    if target_basis == Bases.CARDINAL:
        if source_basis == Bases.BASIC:
            coeff_to_samples_p(data, degree = degree)
            return
        if source_basis == Bases.CARDINAL:
            return
        if source_basis == Bases.DUAL:
            samples_to_coeff_p(data, degree = 2 * degree + 1)
            coeff_to_samples_p(data, degree = degree)
            return
        if source_basis == Bases.ORTHONORMAL:
            ortho_to_coeff_p(data, degree = degree)
            coeff_to_samples_p(data, degree = degree)
            return
        raise ValueError(
            "The source basis must be one of [CARDINAL, BASIC, ORTHONORMAL, DUAL]"
        )
    if target_basis == Bases.DUAL:
        if source_basis == Bases.BASIC:
            coeff_to_samples_p(data, degree = 2 * degree + 1)
            return
        if source_basis == Bases.CARDINAL:
            samples_to_coeff_p(data, degree = degree)
            coeff_to_samples_p(data, degree = 2 * degree + 1)
            return
        if source_basis == Bases.DUAL:
            return
        if source_basis == Bases.ORTHONORMAL:
            coeff_to_ortho_p(data, degree = degree)
            return
        raise ValueError(
            "The source basis must be one of [CARDINAL, BASIC, ORTHONORMAL, DUAL]"
        )
    if target_basis == Bases.ORTHONORMAL:
        if source_basis == Bases.BASIC:
            coeff_to_ortho_p(data, degree = degree)
            return
        if source_basis == Bases.CARDINAL:
            samples_to_coeff_p(data, degree = degree)
            coeff_to_ortho_p(data, degree = degree)
            return
        if source_basis == Bases.DUAL:
            ortho_to_coeff_p(data, degree = degree)
            return
        if source_basis == Bases.ORTHONORMAL:
            return
        raise ValueError(
            "The source basis must be one of [CARDINAL, BASIC, ORTHONORMAL, DUAL]"
        )
    raise ValueError(
        "The source basis must be one of [CARDINAL, BASIC, ORTHONORMAL, DUAL]"
    )

#---------------
def samples_to_coeff_p (
    data: np.ndarray[tuple[int], np.dtype[np.float64]],
    *,
    degree: int
) -> None:

    r"""
    .. _samples_to_coeff_p:

    In-place conversion of data samples into spline coefficients under
    periodic padding.

    Replaces a one-dimensional ``numpy.ndarray`` of data samples :math:`f` by
    spline coefficients :math:`c` such that

    ..  math::

        \forall k\in[0\ldots K-1]:f[k]=\sum_{q\in{\mathbb{Z}}}\,c[q]\,
        \beta^{n}(k-q),

    where :math:`K` is the length of the provided data and :math:`\beta^{n}`
    is a :ref:`polynomial B-spline<def-b_spline>` of
    :ref:`nonnegative<def-negative>` degree :math:`n.` The data samples
    are assumed to conform to a :ref:`periodic padding<pad_p>`.

    Parameters
    ----------
    data : np.ndarray[tuple[int], np.dtype[np.float64]],
        Data to convert from samples to coefficients.
    degree : int
        Nonnegative degree of the polynomial B-spline.

    Returns
    -------
    None :
        The returned coefficients overwrite ``data``.

    Examples
    --------
    Load the libraries.
        >>> import numpy as np
        >>> import splinekit as sk
    Arbitrary data samples with periodic padding.
        >>> f = np.array([1, 5, -3], dtype = "float")
    Cubic coefficients.
        >>> c = f.copy()
        >>> sk.samples_to_coeff_p(c, degree = 3)
        >>> print(c)
        [ 1.  9. -7.]

    Notes
    -----
    The computations are performed in-place.

    ----

    """

    if np.ndarray != type(data):
        raise ValueError("Data must be a numpy array")
    if 1 != len(data.shape):
        raise ValueError("Data must be a one-dimensional numpy array")
    if float != data.dtype:
        raise ValueError(
            "Data must be np.ndarray[tuple[int], np.dtype[np.float64]]"
        )
    if 0 == len(data):
        raise ValueError("Data must contain at least one element")
    if 0 > degree:
        raise ValueError("Degree must be nonnegative")
    p = pole(degree)
    p0 = len(data)
    if (0 == len(p)) or (1 == p0):
        return
    for z in p:
        sigma = data[0]
        zeta = z
        k = 1
        while (k < p0) and (0.0 != zeta):
            sigma += zeta * data[-k]
            zeta *= z
            k += 1
        data[0] = sigma / (1.0 - zeta)
        for k in range(1, p0):
            data[k] += z * data[k - 1]
        sigma = data[-1]
        zeta = z
        k = 0
        while (k < p0 - 1) and (0.0 != zeta):
            sigma += zeta * data[k]
            zeta *= z
            k += 1
        z12 = (1.0 - z) ** 2
        data[-1] = z12 * sigma / (1.0 - zeta)
        for k in range(1, p0):
            data[-1 - k] = z * data[-k] + z12 * data[-1 - k]

#---------------
def coeff_to_samples_p (
    data: np.ndarray[tuple[int], np.dtype[np.float64]],
    *,
    degree: int
) -> None:

    r"""
    .. _coeff_to_samples_p:

    In-place conversion of spline coefficients into data samples under
    periodic padding.

    Replaces a one-dimensional ``numpy.ndarray`` of spline coefficietnts
    :math:`c` by data samples :math:`f` such that

    ..  math::

        \forall k\in[0\ldots K-1]:\sum_{q\in{\mathbb{Z}}}\,c[q]\,
        \beta^{n}(k-q)=f[k],

    where :math:`K` is the length of the provided data and :math:`\beta^{n}`
    is a :ref:`polynomial B-spline<def-b_spline>` of
    :ref:`nonnegative<def-negative>` degree :math:`n.` The spline coefficients
    are assumed to conform to a :ref:`periodic padding<pad_p>`.

    Parameters
    ----------
    data : np.ndarray[tuple[int], np.dtype[np.float64]],
        Data to convert from coefficients to samples.
    degree : int
        Nonnegative degree of the polynomial B-spline.

    Returns
    -------
    None :
        The returned data samples overwrite ``data``.

    Examples
    --------
    Load the libraries.
        >>> import numpy as np
        >>> import splinekit as sk
    Arbitrary spline coefficients with periodic padding.
        >>> c = np.array([1, 9, -7], dtype = "float")
    Data samples.
        >>> f = c.copy()
        >>> sk.coeff_to_samples_p(f, degree = 3)
        >>> print(f)
        [ 1.  5. -3.]

    Notes
    -----
    The computations are performed in-place.

    ----

    """

    if np.ndarray != type(data):
        raise ValueError("Data must be a numpy array")
    if 1 != len(data.shape):
        raise ValueError("Data must be a one-dimensional numpy array")
    if float != data.dtype:
        raise ValueError(
            "Data must be np.ndarray[tuple[int], np.dtype[np.float64]]"
        )
    if 0 == len(data):
        raise ValueError("Data must contain at least one element")
    if 0 > degree:
        raise ValueError("Degree must be nonnegative")
    if 1 >= degree:
        return
    p0 = len(data)
    periodized_b = np.zeros(p0, dtype = float)
    for (x, bx) in enumerate(_b(degree)):
        periodized_b[(x - (degree // 2)) % p0] += bx
    mat = scipy.linalg.toeplitz(periodized_b, periodized_b)
    np.matmul(mat, data, out = data)

#---------------
_sqrtdftbn: Dict[
    Tuple[int, int],
    np.ndarray[tuple[int], np.dtype[np.complex128]]
] = {
}

#---------------
def coeff_to_ortho_p (
    data: np.ndarray[tuple[int], np.dtype[np.float64]],
    *,
    degree: int
) -> None:

    r"""
    .. _coeff_to_ortho_p:

    In-place conversion of B-spline coefficients into orthonormal-spline
    coefficients under periodic padding.

    Replaces a one-dimensional ``numpy.ndarray`` of spline coefficients
    :math:`c` by orthonormal-spline coefficients :math:`g` such that

    ..  math::

        \forall x\in{\mathbb{R}}:\sum_{q\in{\mathbb{Z}}}\,c[q]\,
        \beta^{n}(x-q)=\sum_{q\in{\mathbb{Z}}}\,g[q]\,\phi^{n}(x-q),

    where :math:`K` is the length of the provided data, :math:`\beta^{n}` is
    is a :ref:`polynomial B-spline<def-b_spline>` of
    :ref:`nonnegative<def-negative>` degree :math:`n,` and :math:`\phi^{n}`
    is an :ref:`orthonormal polynomial spline<def-orthonormal_b_spline>`.
    The spline coefficients are assumed to conform to a
    :ref:`periodic padding<pad_p>` of period :math:`K.`

    Parameters
    ----------
    data : np.ndarray[tuple[int], np.dtype[np.float64]],
        Data to convert from spline coefficients to orthonormal coefficients.
    degree : int
        Nonnegative degree of the polynomial B-spline.

    Returns
    -------
    None :
        The returned coefficients overwrite ``data``.

    Examples
    --------
    Load the libraries.
        >>> import numpy as np
        >>> import splinekit as sk
    Arbitrary cubic dual-spline coefficients with periodic padding.
        >>> a1 = np.array([1, 2.75714286, -0.75714286], dtype = "float")
    Spline coefficients.
        >>> c = a1.copy()
        >>> sk.samples_to_coeff_p(c, degree = 2 * 3 + 1)
    Orthonormal coefficients.
        >>> g = c.copy()
        >>> sk.coeff_to_ortho_p(g, degree = 3)
    The re-application of ``coeff_to_ortho_p`` yields back the dual coefficients, up to numerical accuracy.
        >>> a2 = g.copy()
        >>> sk.coeff_to_ortho_p(a2, degree = 3)
        >>> print(a1 - a2)
        [-2.22044605e-16 -1.06581410e-14  1.18793864e-14]

    Notes
    -----
    The computations are performed in-place.

    ----

    """

    if np.ndarray != type(data):
        raise ValueError("Data must be a numpy array")
    if 1 != len(data.shape):
        raise ValueError("Data must be a one-dimensional numpy array")
    if float != data.dtype:
        raise ValueError(
            "Data must be np.ndarray[tuple[int], np.dtype[np.float64]]"
        )
    if 0 == len(data):
        raise ValueError("Data must contain at least one element")
    if 0 > degree:
        raise ValueError("Degree must be nonnegative")
    p0 = len(data)
    if not (degree, p0) in _sqrtdftbn:
        periodized_b = np.zeros(p0, dtype = float)
        for (x, bx) in enumerate(_b(2 * degree + 1)):
            periodized_b[(x - degree) % p0] += bx
        _sqrtdftbn[(degree, p0)] = cast(
            np.ndarray[tuple[int], np.dtype[np.complex128]],
            np.sqrt(np.fft.rfft(periodized_b))
        )
    np.fft.irfft(
        np.fft.rfft(data) * _sqrtdftbn[(degree, p0)],
        n = p0,
        out = data
    )

#---------------
def ortho_to_coeff_p (
    data: np.ndarray[tuple[int], np.dtype[np.float64]],
    *,
    degree: int
) -> None:

    r"""
    .. _ortho_to_coeff_p:

    In-place conversion of orthonormal-spline coefficients into B-spline
    coefficients under periodic padding.

    Replaces a one-dimensional ``numpy.ndarray`` of orthonormal spline
    coefficients :math:`g` by B-spline coefficients :math:`c` such that

    ..  math::

        \forall x\in{\mathbb{R}}:\sum_{q\in{\mathbb{Z}}}\,g[q]\,
        \phi^{n}(x-q)=\sum_{q\in{\mathbb{Z}}}\,c[q]\,\beta^{n}(x-q),

    where :math:`K` is the length of the provided data, :math:`\phi^{n}`
    is an :ref:`orthonormal polynomial spline<def-orthonormal_b_spline>` of
    :ref:`nonnegative<def-negative>` degree :math:`n,` and :math:`\beta^{n}`
    is a :ref:`polynomial B-spline<def-b_spline>`. The orthonormal spline
    coefficients are assumed to conform to a :ref:`periodic padding<pad_p>`
    of period :math:`K.`

    Parameters
    ----------
    data : np.ndarray[tuple[int], np.dtype[np.float64]],
        Data to convert from orthonormal coefficients to spline coefficients.
    degree : int
        Nonnegative degree of the polynomial orthonormal-spline.

    Returns
    -------
    None :
        The returned coefficients overwrite ``data``.

    Examples
    --------
    Load the libraries.
        >>> import numpy as np
        >>> import splinekit as sk
    Arbitrary cubic spline coefficients with periodic padding.
        >>> c1 = np.array([1, 9, -7], dtype = "float")
    Dual-spline coefficients.
        >>> a = c1.copy()
        >>> sk.coeff_to_samples_p(a, degree = 2 * 3 + 1)
    Orthonormal coefficients.
        >>> g1 = a.copy()
        >>> sk.ortho_to_coeff_p(g1, degree = 3)
    The re-application of ``ortho_to_coeff_p`` yields back the spline coefficients, up to numerical accuracy.
        >>> g2 = g1.copy()
        >>> sk.ortho_to_coeff_p(g2, degree = 3)
        >>> print(c1 - g2)
        [ 3.33066907e-16  3.55271368e-15 -2.66453526e-15]

    Notes
    -----
    The computations are performed in-place.

    ----

    """

    if np.ndarray != type(data):
        raise ValueError("Data must be a numpy array")
    if 1 != len(data.shape):
        raise ValueError("Data must be a one-dimensional numpy array")
    if float != data.dtype:
        raise ValueError(
            "Data must be np.ndarray[tuple[int], np.dtype[np.float64]]"
        )
    if 0 == len(data):
        raise ValueError("Data must contain at least one element")
    if 0 > degree:
        raise ValueError("Degree must be nonnegative")
    p0 = len(data)
    if not (degree, p0) in _sqrtdftbn:
        periodized_b = np.zeros(p0, dtype = float)
        for (x, bx) in enumerate(_b(2 * degree + 1)):
            periodized_b[(x - degree) % p0] += bx
        _sqrtdftbn[(degree, p0)] = cast(
            np.ndarray[tuple[int], np.dtype[np.complex128]],
            np.sqrt(np.fft.rfft(periodized_b))
        )
    np.fft.irfft(
        np.fft.rfft(data) / _sqrtdftbn[(degree, p0)],
        n = p0,
        out = data
    )

#---------------
def pad_n (
    data: np.ndarray[tuple[int], np.dtype[np.float64]],
    *,
    at: int
) -> float:

    r"""
    .. _pad_n:

    Narrow-mirror padding.

    Returns the value of the :math:`k`-th data sample after the data have been
    extended by narrow-mirror padding. The padded data :math:`f` satisfy that

    ..  math::

        \forall k\in{\mathbb{Z}}:f[k]=\left\{\begin{array}{rcl}f[k]&=&f[-k]\\
        f[k+K-1]&=&f[K-1-k],\end{array}\right.

    where :math:`K` is the length of the unpadded data. The conditions of
    narrow-mirror padding imply the periodicity

    ..  math::

        \forall k\in{\mathbb{Z}}:f[k]=f[k+2\,K-2].

    :raw-html:`<TABLE border="1" frame="hsides" rules="groups" align="center">
    <CAPTION>Narrow-mirror padding of <FONT color="blue">data samples</FONT></CAPTION>
    <COLGROUP align="right">
    <COLGROUP align="center" span="7">
    <TR><TH>k<TH>&#160;&#160;&#160;<TH>&#402;<SUB>1</SUB>[k]<TH>&#402;<SUB>2</SUB>[k]<TH>&#402;<SUB>3</SUB>[k]<TH>&#402;<SUB>4</SUB>[k]<TH>&#402;<SUB>5</SUB>[k]<TH>&#402;<SUB>6</SUB>[k]
    <TBODY>
    <TR><TD>&#8722;20<TD><TD>a<TD>a<TD>a<TD>c<TD>e<TD>a
    <TR><TD>&#8722;19<TD><TD>a<TD>b<TD>b<TD>b<TD>d<TD>b
    <TR><TD>&#8722;18<TD><TD>a<TD>a<TD>c<TD>a<TD>c<TD>c
    <TR><TD>&#8722;17<TD><TD>a<TD>b<TD>b<TD>b<TD>b<TD>d
    <TR><TD>&#8722;16<TD><TD>a<TD>a<TD>a<TD>c<TD>a<TD>e
    <TR><TD>&#8722;15<TD><TD>a<TD>b<TD>b<TD>d<TD>b<TD>f
    <TR><TD>&#8722;14<TD><TD>a<TD>a<TD>c<TD>c<TD>c<TD>e
    <TR><TD>&#8722;13<TD><TD>a<TD>b<TD>b<TD>b<TD>d<TD>d
    <TR><TD>&#8722;12<TD><TD>a<TD>a<TD>a<TD>a<TD>e<TD>c
    <TR><TD>&#8722;11<TD><TD>a<TD>b<TD>b<TD>b<TD>d<TD>b
    <TR><TD>&#8722;10<TD><TD>a<TD>a<TD>c<TD>c<TD>c<TD>a
    <TR><TD>&#8722;9<TD><TD>a<TD>b<TD>b<TD>d<TD>b<TD>b
    <TR><TD>&#8722;8<TD><TD>a<TD>a<TD>a<TD>c<TD>a<TD>c
    <TR><TD>&#8722;7<TD><TD>a<TD>b<TD>b<TD>b<TD>b<TD>d
    <TR><TD>&#8722;6<TD><TD>a<TD>a<TD>c<TD>a<TD>c<TD>e
    <TR><TD>&#8722;5<TD><TD>a<TD>b<TD>b<TD>b<TD>d<TD>f
    <TR><TD>&#8722;4<TD><TD>a<TD>a<TD>a<TD>c<TD>e<TD>e
    <TR><TD>&#8722;3<TD><TD>a<TD>b<TD>b<TD>d<TD>d<TD>d
    <TR><TD>&#8722;2<TD><TD>a<TD>a<TD>c<TD>c<TD>c<TD>c
    <TR><TD>&#8722;1<TD><TD>a<TD>b<TD>b<TD>b<TD>b<TD>b
    <TR><TD>0<TD><TD><FONT color="blue"><B>A</B></FONT><TD><FONT color="blue"><B>A</B></FONT><TD><FONT color="blue"><B>A</B></FONT><TD><FONT color="blue"><B>A</B></FONT><TD><FONT color="blue"><B>A</B></FONT><TD><FONT color="blue"><B>A</B></FONT>
    <TR><TD>1<TD><TD>a<TD><FONT color="blue"><B>B</B></FONT><TD><FONT color="blue"><B>B</B></FONT><TD><FONT color="blue"><B>B</B></FONT><TD><FONT color="blue"><B>B</B></FONT><TD><FONT color="blue"><B>B</B></FONT>
    <TR><TD>2<TD><TD>a<TD>a<TD><FONT color="blue"><B>C</B></FONT><TD><FONT color="blue"><B>C</B></FONT><TD><FONT color="blue"><B>C</B></FONT><TD><FONT color="blue"><B>C</B></FONT>
    <TR><TD>3<TD><TD>a<TD>b<TD><FONT color="purple"><B>b</B></FONT><TD><FONT color="blue"><B>D</B></FONT><TD><FONT color="blue"><B>D</B></FONT><TD><FONT color="blue"><B>D</B></FONT>
    <TR><TD>4<TD><TD>a<TD>a<TD>a<TD><FONT color="purple"><B>c</B></FONT><TD><FONT color="blue"><B>E</B></FONT><TD><FONT color="blue"><B>E</B></FONT>
    <TR><TD>5<TD><TD>a<TD>b<TD>b<TD><FONT color="purple"><B>b</B></FONT><TD><FONT color="purple"><B>d</B></FONT><TD><FONT color="blue"><B>F</B></FONT>
    <TR><TD>6<TD><TD>a<TD>a<TD>c<TD>a<TD><FONT color="purple"><B>c</B></FONT><TD><FONT color="purple"><B>e</B></FONT>
    <TR><TD>7<TD><TD>a<TD>b<TD>b<TD>b<TD><FONT color="purple"><B>b</B></FONT><TD><FONT color="purple"><B>d</B></FONT>
    <TR><TD>8<TD><TD>a<TD>a<TD>a<TD>c<TD>a<TD><FONT color="purple"><B>c</B></FONT>
    <TR><TD>9<TD><TD>a<TD>b<TD>b<TD>d<TD>b<TD><FONT color="purple"><B>b</B></FONT>
    <TR><TD>10<TD><TD>a<TD>a<TD>c<TD>c<TD>c<TD>a
    <TR><TD>11<TD><TD>a<TD>b<TD>b<TD>b<TD>d<TD>b
    <TR><TD>12<TD><TD>a<TD>a<TD>a<TD>a<TD>e<TD>c
    <TR><TD>13<TD><TD>a<TD>b<TD>b<TD>b<TD>d<TD>d
    <TR><TD>14<TD><TD>a<TD>a<TD>c<TD>c<TD>c<TD>e
    <TR><TD>15<TD><TD>a<TD>b<TD>b<TD>d<TD>b<TD>f
    <TR><TD>16<TD><TD>a<TD>a<TD>a<TD>c<TD>a<TD>e
    <TR><TD>17<TD><TD>a<TD>b<TD>b<TD>b<TD>b<TD>d
    <TR><TD>18<TD><TD>a<TD>a<TD>c<TD>a<TD>c<TD>c
    <TR><TD>19<TD><TD>a<TD>b<TD>b<TD>b<TD>d<TD>b
    <TR><TD>20<TD><TD>a<TD>a<TD>a<TD>c<TD>e<TD>a
    <TR><TD>21<TD><TD>a<TD>b<TD>b<TD>d<TD>d<TD>b
    <TR><TD>22<TD><TD>a<TD>a<TD>c<TD>c<TD>c<TD>c
    <TR><TD>23<TD><TD>a<TD>b<TD>b<TD>b<TD>b<TD>d
    <TR><TD>24<TD><TD>a<TD>a<TD>a<TD>a<TD>a<TD>e
    <TR><TD>25<TD><TD>a<TD>b<TD>b<TD>b<TD>b<TD>f
    <TR><TD>26<TD><TD>a<TD>a<TD>c<TD>c<TD>c<TD>e
    <TR><TD>27<TD><TD>a<TD>b<TD>b<TD>d<TD>d<TD>d
    <TR><TD>28<TD><TD>a<TD>a<TD>a<TD>c<TD>e<TD>c
    <TR><TD>29<TD><TD>a<TD>b<TD>b<TD>b<TD>d<TD>b
    </TABLE>`

    Parameters
    ----------
    data : np.ndarray[tuple[int], np.dtype[np.float64]],
        Data to extend.
    at : int
        Arbitrary index.

    Returns
    -------
    float
        The value of the padded data at the requested index.

    Examples
    --------
    Load the library.
        >>> import splinekit as sk
    Short data at large index ``-42``.
        >>> sk.pad_n([1, 5, -3], at = -42)
        -3

    ----

    """

    k0 = len(data)
    if 1 == k0:
        return data[0]
    p0 = 2 * k0 - 2
    q = abs(at) % p0
    if q < k0:
        return data[q]
    return data[p0 - q]

#---------------
def samples_to_coeff_n (
    data: np.ndarray[tuple[int], np.dtype[np.float64]],
    *,
    degree: int
) -> None:

    r"""
    .. _samples_to_coeff_n:

    In-place conversion of data samples into spline coefficients under
    narrow-mirror padding.

    Replaces a one-dimensional ``numpy.ndarray`` of data samples :math:`f` by
    spline coefficients :math:`c` such that

    ..  math::

        \forall k\in[0\ldots K-1]:f[k]=\sum_{q\in{\mathbb{Z}}}\,c[q]\,
        \beta^{n}(k-q),

    where :math:`K` is the length of the provided data and :math:`\beta^{n}`
    is a :ref:`polynomial B-spline<def-b_spline>` of
    :ref:`nonnegative<def-negative>` degree :math:`n.` The spline coefficients
    are assumed to conform to a :ref:`narrow-mirror padding<pad_n>`.

    Parameters
    ----------
    data : np.ndarray[tuple[int], np.dtype[np.float64]],
        Data to convert from samples to coefficients.
    degree : int
        Nonnegative degree of the polynomial B-spline.

    Returns
    -------
    None :
        The returned coefficients overwrite ``data``.

    Examples
    --------
    Load the libraries.
        >>> import numpy as np
        >>> import splinekit as sk
    Cubic coefficients of arbitrary data with narrow-mirror padding.
        >>> f = np.array([1, 5, -3], dtype = "float")
        >>> c = f.copy()
        >>> sk.samples_to_coeff_n(c, degree = 3)
        >>> print(c)
        [ -4.  11. -10.]

    Notes
    -----
    The computations are performed in-place.

    ----

    """

    if np.ndarray != type(data):
        raise ValueError("Data must be a numpy array")
    if 1 != len(data.shape):
        raise ValueError("Data must be a one-dimensional numpy array")
    if float != data.dtype:
        raise ValueError(
            "Data must be np.ndarray[tuple[int], np.dtype[np.float64]]"
        )
    if 0 == len(data):
        raise ValueError("Data must contain at least one element")
    if 0 > degree:
        raise ValueError("Degree must be nonnegative")
    p = pole(degree)
    k0 = len(data)
    if (0 == len(p)) or (1 == k0):
        return
    for z in p:
        sigma1 = data[0]
        sigma2 = data[-1]
        zeta = z
        k = 1
        while (k < k0 - 1) and (0.0 != zeta):
            sigma1 += zeta * data[k]
            sigma2 += zeta * data[-1 - k]
            zeta *= z
            k += 1
        data[0] = (sigma1 + zeta * sigma2) / (1.0 - zeta ** 2)
        for k in range(1, k0):
            data[k] += z * data[k - 1]
        z12 = (1.0 - z) ** 2
        data[-1] = z12 * (z * data[-2] + data[-1]) / (1.0 - z ** 2)
        for k in range(1, k0):
            data[-1 - k] = z * data[-k] + z12 * data[-1 - k]

#---------------
def pad_w (
    data: np.ndarray[tuple[int], np.dtype[np.float64]],
    *,
    at: int
) -> float:

    r"""
    .. _pad_w:

    Wide-mirror padding.

    Returns the value of the :math:`k`-th data sample after the data have been
    extended by wide-mirror padding. The padded data :math:`f` satisfy that

    ..  math::

        \forall k\in{\mathbb{Z}}:\left\{\begin{array}{rcl}f[x]&=&f[-1-x]\\
        f[x+K]&=&f[K-1-x],\end{array}\right.

    where :math:`K` is the length of the unpadded data. The conditions of
    wide-mirror padding imply the periodicity

    ..  math::

        \forall k\in{\mathbb{Z}}:f[k]=f[k+2\,K].

    :raw-html:`<TABLE border="1" frame="hsides" rules="groups" align="center">
    <CAPTION>Wide-mirror padding of <FONT color="blue">data samples</FONT></CAPTION>
    <COLGROUP align="right">
    <COLGROUP align="center" span="7">
    <TR><TH>k<TH>&#160;&#160;&#160;<TH>&#402;<SUB>1</SUB>[k]<TH>&#402;<SUB>2</SUB>[k]<TH>&#402;<SUB>3</SUB>[k]<TH>&#402;<SUB>4</SUB>[k]<TH>&#402;<SUB>5</SUB>[k]<TH>&#402;<SUB>6</SUB>[k]
    <TBODY>
    <TR><TD>&#8722;20<TD><TD>a<TD>a<TD>b<TD>d<TD>a<TD>e
    <TR><TD>&#8722;19<TD><TD>a<TD>b<TD>a<TD>c<TD>b<TD>f
    <TR><TD>&#8722;18<TD><TD>a<TD>b<TD>a<TD>b<TD>c<TD>f
    <TR><TD>&#8722;17<TD><TD>a<TD>a<TD>b<TD>a<TD>d<TD>e
    <TR><TD>&#8722;16<TD><TD>a<TD>a<TD>c<TD>a<TD>e<TD>d
    <TR><TD>&#8722;15<TD><TD>a<TD>b<TD>c<TD>b<TD>e<TD>c
    <TR><TD>&#8722;14<TD><TD>a<TD>b<TD>b<TD>c<TD>d<TD>b
    <TR><TD>&#8722;13<TD><TD>a<TD>a<TD>a<TD>d<TD>c<TD>a
    <TR><TD>&#8722;12<TD><TD>a<TD>a<TD>a<TD>d<TD>b<TD>a
    <TR><TD>&#8722;11<TD><TD>a<TD>b<TD>b<TD>c<TD>a<TD>b
    <TR><TD>&#8722;10<TD><TD>a<TD>b<TD>c<TD>b<TD>a<TD>c
    <TR><TD>&#8722;9<TD><TD>a<TD>a<TD>c<TD>a<TD>b<TD>d
    <TR><TD>&#8722;8<TD><TD>a<TD>a<TD>b<TD>a<TD>c<TD>e
    <TR><TD>&#8722;7<TD><TD>a<TD>b<TD>a<TD>b<TD>d<TD>f
    <TR><TD>&#8722;6<TD><TD>a<TD>b<TD>a<TD>c<TD>e<TD>f
    <TR><TD>&#8722;5<TD><TD>a<TD>a<TD>b<TD>d<TD>e<TD>e
    <TR><TD>&#8722;4<TD><TD>a<TD>a<TD>c<TD>d<TD>d<TD>d
    <TR><TD>&#8722;3<TD><TD>a<TD>b<TD>c<TD>c<TD>c<TD>c
    <TR><TD>&#8722;2<TD><TD>a<TD>b<TD>b<TD>b<TD>b<TD>b
    <TR><TD>&#8722;1<TD><TD>a<TD>a<TD>a<TD>a<TD>a<TD>a
    <TR><TD>0<TD><TD><FONT color="blue"><B>A</B></FONT><TD><FONT color="blue"><B>A</B></FONT><TD><FONT color="blue"><B>A</B></FONT><TD><FONT color="blue"><B>A</B></FONT><TD><FONT color="blue"><B>A</B></FONT><TD><FONT color="blue"><B>A</B></FONT>
    <TR><TD>1<TD><TD><FONT color="purple"><B>a</B></FONT><TD><FONT color="blue"><B>B</B></FONT><TD><FONT color="blue"><B>B</B></FONT><TD><FONT color="blue"><B>B</B></FONT><TD><FONT color="blue"><B>B</B></FONT><TD><FONT color="blue"><B>B</B></FONT>
    <TR><TD>2<TD><TD>a<TD><FONT color="purple"><B>b</B></FONT><TD><FONT color="blue"><B>C</B></FONT><TD><FONT color="blue"><B>C</B></FONT><TD><FONT color="blue"><B>C</B></FONT><TD><FONT color="blue"><B>C</B></FONT>
    <TR><TD>3<TD><TD>a<TD><FONT color="purple"><B>a</B></FONT><TD><FONT color="purple"><B>c</B></FONT><TD><FONT color="blue"><B>D</B></FONT><TD><FONT color="blue"><B>D</B></FONT><TD><FONT color="blue"><B>D</B></FONT>
    <TR><TD>4<TD><TD>a<TD>a<TD><FONT color="purple"><B>b</B></FONT><TD><FONT color="purple"><B>d</B></FONT><TD><FONT color="blue"><B>E</B></FONT><TD><FONT color="blue"><B>E</B></FONT>
    <TR><TD>5<TD><TD>a<TD>b<TD><FONT color="purple"><B>a</B></FONT><TD><FONT color="purple"><B>c</B></FONT><TD><FONT color="purple"><B>e</B></FONT><TD><FONT color="blue"><B>F</B></FONT>
    <TR><TD>6<TD><TD>a<TD>b<TD>a<TD><FONT color="purple"><B>b</B></FONT><TD><FONT color="purple"><B>d</B></FONT><TD><FONT color="purple"><B>f</B></FONT>
    <TR><TD>7<TD><TD>a<TD>a<TD>b<TD><FONT color="purple"><B>a</B></FONT><TD><FONT color="purple"><B>c</B></FONT><TD><FONT color="purple"><B>e</B></FONT>
    <TR><TD>8<TD><TD>a<TD>a<TD>c<TD>a<TD><FONT color="purple"><B>b</B></FONT><TD><FONT color="purple"><B>d</B></FONT>
    <TR><TD>9<TD><TD>a<TD>b<TD>c<TD>b<TD><FONT color="purple"><B>a</B></FONT><TD><FONT color="purple"><B>c</B></FONT>
    <TR><TD>10<TD><TD>a<TD>b<TD>b<TD>c<TD>a<TD><FONT color="purple"><B>b</B></FONT>
    <TR><TD>11<TD><TD>a<TD>a<TD>a<TD>d<TD>b<TD><FONT color="purple"><B>a</B></FONT>
    <TR><TD>12<TD><TD>a<TD>a<TD>a<TD>d<TD>c<TD>a
    <TR><TD>13<TD><TD>a<TD>b<TD>b<TD>c<TD>d<TD>b
    <TR><TD>14<TD><TD>a<TD>b<TD>c<TD>b<TD>e<TD>c
    <TR><TD>15<TD><TD>a<TD>a<TD>c<TD>a<TD>e<TD>d
    <TR><TD>16<TD><TD>a<TD>a<TD>b<TD>a<TD>d<TD>e
    <TR><TD>17<TD><TD>a<TD>b<TD>a<TD>b<TD>c<TD>f
    <TR><TD>18<TD><TD>a<TD>b<TD>a<TD>c<TD>b<TD>f
    <TR><TD>19<TD><TD>a<TD>a<TD>b<TD>d<TD>a<TD>e
    <TR><TD>20<TD><TD>a<TD>a<TD>c<TD>d<TD>a<TD>d
    <TR><TD>21<TD><TD>a<TD>b<TD>c<TD>c<TD>b<TD>c
    <TR><TD>22<TD><TD>a<TD>b<TD>b<TD>b<TD>c<TD>b
    <TR><TD>23<TD><TD>a<TD>a<TD>a<TD>a<TD>d<TD>a
    <TR><TD>24<TD><TD>a<TD>a<TD>a<TD>a<TD>e<TD>a
    <TR><TD>25<TD><TD>a<TD>b<TD>b<TD>b<TD>e<TD>b
    <TR><TD>26<TD><TD>a<TD>b<TD>c<TD>c<TD>d<TD>c
    <TR><TD>27<TD><TD>a<TD>a<TD>c<TD>d<TD>c<TD>d
    <TR><TD>28<TD><TD>a<TD>a<TD>b<TD>d<TD>b<TD>e
    <TR><TD>29<TD><TD>a<TD>b<TD>a<TD>c<TD>a<TD>f
    </TABLE>`

    Parameters
    ----------
    data : np.ndarray[tuple[int], np.dtype[np.float64]],
        Data to extend.
    at : int
        Arbitrary index.

    Returns
    -------
    float
        The value of the padded data at the requested index.

    Examples
    --------
    Load the library.
        >>> import splinekit as sk
    Short data at large index ``-42``.
        >>> sk.pad_w([1, 5, -3], at = -42)
        1

    ----

    """

    k0 = len(data)
    p0 = 2 * k0
    q = at % p0
    if q < k0:
        return data[q]
    return data[p0 - 1 - q]

#---------------
def samples_to_coeff_w (
    data: np.ndarray[tuple[int], np.dtype[np.float64]],
    *,
    degree: int
) -> None:

    r"""
    .. _samples_to_coeff_w:

    In-place conversion of data samples into spline coefficients under
    wide-mirror padding.

    Replaces a one-dimensional ``numpy.ndarray`` of data samples :math:`f` by
    spline coefficients :math:`c` such that

    ..  math::

        \forall k\in[0\ldots K-1]:f[k]=\sum_{q\in{\mathbb{Z}}}\,c[q]\,
        \beta^{n}(k-q),

    where :math:`K` is the length of the provided data and :math:`\beta^{n}`
    is a :ref:`polynomial B-spline<def-b_spline>` of
    :ref:`nonnegative<def-negative>` degree :math:`n.` The spline coefficients
    are assumed to conform to a :ref:`wide-mirror padding<pad_w>`.

    Parameters
    ----------
    data : np.ndarray[tuple[int], np.dtype[np.float64]],
        Data to convert from samples to coefficients.
    degree : int
        Nonnegative degree of the polynomial B-spline.

    Returns
    -------
    None :
        The returned coefficients overwrite ``data``.

    Examples
    --------
    Load the libraries.
        >>> import numpy as np
        >>> import splinekit as sk
    Cubic coefficients of arbitrary data with wide-mirror padding.
        >>> f = np.array([1, 5, -3], dtype = "float")
        >>> c = f.copy()
        >>> sk.samples_to_coeff_w(c, degree = 3)
        >>> print(c)
        [-0.6  9.  -5.4]

    Notes
    -----
    The computations are performed in-place.

    ----

    """

    if np.ndarray != type(data):
        raise ValueError("Data must be a numpy array")
    if 1 != len(data.shape):
        raise ValueError("Data must be a one-dimensional numpy array")
    if float != data.dtype:
        raise ValueError(
            "Data must be np.ndarray[tuple[int], np.dtype[np.float64]]"
        )
    if 0 == len(data):
        raise ValueError("Data must contain at least one element")
    if 0 > degree:
        raise ValueError("Degree must be nonnegative")
    p = pole(degree)
    k0 = len(data)
    if (0 == len(p)) or (1 == k0):
        return
    for z in p:
        sigma1 = 0.0
        sigma2 = 0.0
        zeta = 1.0
        k = 0
        while (k < k0) and (0.0 != zeta):
            sigma1 += zeta * data[k]
            sigma2 += zeta * data[-1 - k]
            zeta *= z
            k += 1
        data[0] += z * (sigma1 + zeta * sigma2) / (1.0 - zeta ** 2)
        for k in range(1, k0):
            data[k] += z * data[k - 1]
        z12 = (1.0 - z) ** 2
        data[-1] *= 1.0 - z
        for k in range(1, k0):
            data[-1 - k] = z * data[-k] + z12 * data[-1 - k]

#---------------
def pad_a (
    data: np.ndarray[tuple[int], np.dtype[np.float64]],
    *,
    at: int
) -> float:

    r"""
    .. _pad_a:

    Anti-mirror padding.

    Returns the value of the :math:`k`-th data sample after the data have been
    extended by anti-mirror padding. The padded data :math:`f` satisfy that

    ..  math::

        \forall k\in{\mathbb{Z}}:\left\{\begin{array}{rcl}
        f[k]-f[0]&=&f[0]-f[-k]\\
        f[k+K-1]-f[K-1]&=&f[K-1]-f[K-1-k]\end{array}\right.

    where :math:`K` is the length of the unpadded data. The conditions of
    anti-mirror padding imply the pseudo periodicity

    ..  math::

        \forall k\in{\mathbb{Z}}:f[k]=f[k+2\,K-2]-2\,\left(f[K-1]-f[0]\right).

    :raw-html:`<TABLE border="1" frame="hsides" rules="groups" align="center" width="720px">
    <CAPTION>Anti-mirror padding of <FONT color="blue">data samples</FONT></CAPTION>
    <COLGROUP align="right">
    <COLGROUP align="center" span="7">
    <TR><TH>k<TH>&#160;&#160;&#160;<TH>&#402;<SUB>1</SUB>[k]<TH>&#402;<SUB>2</SUB>[k]<TH>&#402;<SUB>3</SUB>[k]<TH>&#402;<SUB>4</SUB>[k]<TH>&#402;<SUB>5</SUB>[k]<TH>&#402;<SUB>6</SUB>[k]
    <TBODY>
    <TR><TD>&#8722;20<TD><TD>a<TD>(21 a &#8722; 20 b)<TD>(11 a &#8722; 10 c)<TD>(8 a &#8722; c &#8722; 6 d)<TD>(6 a &#8722; 5 e)<TD>(5 a &#8722; 4 f)
    <TR><TD>&#8722;19<TD><TD>a<TD>(20 a &#8722; 19 b)<TD>(10 a + b &#8722; 10 c)<TD>(8 a &#8722; b &#8722; 6 d)<TD>(6 a &#8722; d &#8722; 4 e)<TD>(4 a + b &#8722; 4 f)
    <TR><TD>&#8722;18<TD><TD>a<TD>(19 a &#8722; 18 b)<TD>(10 a &#8722; 9 c)<TD>(7 a &#8722; 6 d)<TD>(6 a &#8722; c &#8722; 4 e)<TD>(4 a + c &#8722; 4 f)
    <TR><TD>&#8722;17<TD><TD>a<TD>(18 a &#8722; 17 b)<TD>(10 a &#8722; b &#8722; 8 c)<TD>(6 a + b &#8722; 6 d)<TD>(6 a &#8722; b &#8722; 4 e)<TD>(4 a + d &#8722; 4 f)
    <TR><TD>&#8722;16<TD><TD>a<TD>(17 a &#8722; 16 b)<TD>(9 a &#8722; 8 c)<TD>(6 a + c &#8722; 6 d)<TD>(5 a &#8722; 4 e)<TD>(4 a + e &#8722; 4 f)
    <TR><TD>&#8722;15<TD><TD>a<TD>(16 a &#8722; 15 b)<TD>(8 a + b &#8722; 8 c)<TD>(6 a &#8722; 5 d)<TD>(4 a + b &#8722; 4 e)<TD>(4 a &#8722; 3 f)
    <TR><TD>&#8722;14<TD><TD>a<TD>(15 a &#8722; 14 b)<TD>(8 a &#8722; 7 c)<TD>(6 a &#8722; c &#8722; 4 d)<TD>(4 a + c &#8722; 4 e)<TD>(4 a &#8722; e &#8722; 2 f)
    <TR><TD>&#8722;13<TD><TD>a<TD>(14 a &#8722; 13 b)<TD>(8 a &#8722; b &#8722; 6 c)<TD>(6 a &#8722; b &#8722; 4 d)<TD>(4 a + d &#8722; 4 e)<TD>(4 a &#8722; d &#8722; 2 f)
    <TR><TD>&#8722;12<TD><TD>a<TD>(13 a &#8722; 12 b)<TD>(7 a &#8722; 6 c)<TD>(5 a &#8722; 4 d)<TD>(4 a &#8722; 3 e)<TD>(4 a &#8722; c &#8722; 2 f)
    <TR><TD>&#8722;11<TD><TD>a<TD>(12 a &#8722; 11 b)<TD>(6 a + b &#8722; 6 c)<TD>(4 a + b &#8722; 4 d)<TD>(4 a &#8722; d &#8722; 2 e)<TD>(4 a &#8722; b &#8722; 2 f)
    <TR><TD>&#8722;10<TD><TD>a<TD>(11 a &#8722; 10 b)<TD>(6 a &#8722; 5 c)<TD>(4 a + c &#8722; 4 d)<TD>(4 a &#8722; c &#8722; 2 e)<TD>(3 a &#8722; 2 f)
    <TR><TD>&#8722;9<TD><TD>a<TD>(10 a &#8722; 9 b)<TD>(6 a &#8722; b &#8722; 4 c)<TD>(4 a &#8722; 3 d)<TD>(4 a &#8722; b &#8722; 2 e)<TD>(2 a + b &#8722; 2 f)
    <TR><TD>&#8722;8<TD><TD>a<TD>(9 a &#8722; 8 b)<TD>(5 a &#8722; 4 c)<TD>(4 a &#8722; c &#8722; 2 d)<TD>(3 a &#8722; 2 e)<TD>(2 a + c &#8722; 2 f)
    <TR><TD>&#8722;7<TD><TD>a<TD>(8 a &#8722; 7 b)<TD>(4 a + b &#8722; 4 c)<TD>(4 a &#8722; b &#8722; 2 d)<TD>(2 a + b &#8722; 2 e)<TD>(2 a + d &#8722; 2 f)
    <TR><TD>&#8722;6<TD><TD>a<TD>(7 a &#8722; 6 b)<TD>(4 a &#8722; 3 c)<TD>(3 a &#8722; 2 d)<TD>(2 a + c &#8722; 2 e)<TD>(2 a + e &#8722; 2 f)
    <TR><TD>&#8722;5<TD><TD>a<TD>(6 a &#8722; 5 b)<TD>(4 a &#8722; b &#8722; 2 c)<TD>(2 a + b &#8722; 2 d)<TD>(2 a + d &#8722; 2 e)<TD>(2 a &#8722; f)
    <TR><TD>&#8722;4<TD><TD>a<TD>(5 a &#8722; 4 b)<TD>(3 a &#8722; 2 c)<TD>(2 a + c &#8722; 2 d)<TD>(2 a &#8722; e)<TD>(2 a &#8722; e)
    <TR><TD>&#8722;3<TD><TD>a<TD>(4 a &#8722; 3 b)<TD>(2 a + b &#8722; 2 c)<TD>(2 a &#8722; d)<TD>(2 a &#8722; d)<TD>(2 a &#8722; d)
    <TR><TD>&#8722;2<TD><TD>a<TD>(3 a &#8722; 2 b)<TD>(2 a &#8722; c)<TD>(2 a &#8722; c)<TD>(2 a &#8722; c)<TD>(2 a &#8722; c)
    <TR><TD>&#8722;1<TD><TD>a<TD>(2 a &#8722; b)<TD>(2 a &#8722; b)<TD>(2 a &#8722; b)<TD>(2 a &#8722; b)<TD>(2 a &#8722; b)
    <TR><TD>0<TD><TD><FONT color="blue"><B>A</B></FONT><TD><FONT color="blue"><B>A</B></FONT><TD><FONT color="blue"><B>A</B></FONT><TD><FONT color="blue"><B>A</B></FONT><TD><FONT color="blue"><B>A</B></FONT><TD><FONT color="blue"><B>A</B></FONT>
    <TR><TD>1<TD><TD>a<TD><FONT color="blue"><B>B</B></FONT><TD><FONT color="blue"><B>B</B></FONT><TD><FONT color="blue"><B>B</B></FONT><TD><FONT color="blue"><B>B</B></FONT><TD><FONT color="blue"><B>B</B></FONT>
    <TR><TD>2<TD><TD>a<TD>(&#8722;a + 2 b)<TD><FONT color="blue"><B>C</B></FONT><TD><FONT color="blue"><B>C</B></FONT><TD><FONT color="blue"><B>C</B></FONT><TD><FONT color="blue"><B>C</B></FONT>
    <TR><TD>3<TD><TD>a<TD>(&#8722;2 a + 3 b)<TD>(&#8722;b + 2 c)<TD><FONT color="blue"><B>D</B></FONT><TD><FONT color="blue"><B>D</B></FONT><TD><FONT color="blue"><B>D</B></FONT>
    <TR><TD>4<TD><TD>a<TD>(&#8722;3 a + 4 b)<TD>(&#8722;a + 2 c)<TD>(&#8722;c + 2 d)<TD><FONT color="blue"><B>E</B></FONT><TD><FONT color="blue"><B>E</B></FONT>
    <TR><TD>5<TD><TD>a<TD>(&#8722;4 a + 5 b)<TD>(&#8722;2 a + b + 2 c)<TD>(&#8722;b + 2 d)<TD>(&#8722;d + 2 e)<TD><FONT color="blue"><B>F</B></FONT>
    <TR><TD>6<TD><TD>a<TD>(&#8722;5 a + 6 b)<TD>(&#8722;2 a + 3 c)<TD>(&#8722;a + 2 d)<TD>(&#8722;c + 2 e)<TD>(&#8722;e + 2 f)
    <TR><TD>7<TD><TD>a<TD>(&#8722;6 a + 7 b)<TD>(&#8722;2 a &#8722; b + 4 c)<TD>(&#8722;2 a + b + 2 d)<TD>(&#8722;b + 2 e)<TD>(&#8722;d + 2 f)
    <TR><TD>8<TD><TD>a<TD>(&#8722;7 a + 8 b)<TD>(&#8722;3 a + 4 c)<TD>(&#8722;2 a + c + 2 d)<TD>(&#8722;a + 2 e)<TD>(&#8722;c + 2 f)
    <TR><TD>9<TD><TD>a<TD>(&#8722;8 a + 9 b)<TD>(&#8722;4 a + b + 4 c)<TD>(&#8722;2 a + 3 d)<TD>(&#8722;2 a + b + 2 e)<TD>(&#8722;b + 2 f)
    <TR><TD>10<TD><TD>a<TD>(&#8722;9 a + 10 b)<TD>(&#8722;4 a + 5 c)<TD>(&#8722;2 a &#8722; c + 4 d)<TD>(&#8722;2 a + c + 2 e)<TD>(&#8722;a + 2 f)
    <TR><TD>11<TD><TD>a<TD>(&#8722;10 a + 11 b)<TD>(&#8722;4 a &#8722; b + 6 c)<TD>(&#8722;2 a &#8722; b + 4 d)<TD>(&#8722;2 a + d + 2 e)<TD>(&#8722;2 a + b + 2 f)
    <TR><TD>12<TD><TD>a<TD>(&#8722;11 a + 12 b)<TD>(&#8722;5 a + 6 c)<TD>(&#8722;3 a + 4 d)<TD>(&#8722;2 a + 3 e)<TD>(&#8722;2 a + c + 2 f)
    <TR><TD>13<TD><TD>a<TD>(&#8722;12 a + 13 b)<TD>(&#8722;6 a + b + 6 c)<TD>(&#8722;4 a + b + 4 d)<TD>(&#8722;2 a &#8722; d + 4 e)<TD>(&#8722;2 a + d + 2 f)
    <TR><TD>14<TD><TD>a<TD>(&#8722;13 a + 14 b)<TD>(&#8722;6 a + 7 c)<TD>(&#8722;4 a + c + 4 d)<TD>(&#8722;2 a &#8722; c + 4 e)<TD>(&#8722;2 a + e + 2 f)
    <TR><TD>15<TD><TD>a<TD>(&#8722;14 a + 15 b)<TD>(&#8722;6 a &#8722; b + 8 c)<TD>(&#8722;4 a + 5 d)<TD>(&#8722;2 a &#8722; b + 4 e)<TD>(&#8722;2 a + 3 f)
    <TR><TD>16<TD><TD>a<TD>(&#8722;15 a + 16 b)<TD>(&#8722;7 a + 8 c)<TD>(&#8722;4 a &#8722; c + 6 d)<TD>(&#8722;3 a + 4 e)<TD>(&#8722;2 a &#8722; e + 4 f)
    <TR><TD>17<TD><TD>a<TD>(&#8722;16 a + 17 b)<TD>(&#8722;8 a + b + 8 c)<TD>(&#8722;4 a &#8722; b + 6 d)<TD>(&#8722;4 a + b + 4 e)<TD>(&#8722;2 a &#8722; d + 4 f)
    <TR><TD>18<TD><TD>a<TD>(&#8722;17 a + 18 b)<TD>(&#8722;8 a + 9 c)<TD>(&#8722;5 a + 6 d)<TD>(&#8722;4 a + c + 4 e)<TD>(&#8722;2 a &#8722; c + 4 f)
    <TR><TD>19<TD><TD>a<TD>(&#8722;18 a + 19 b)<TD>(&#8722;8 a &#8722; b + 10 c)<TD>(&#8722;6 a + b + 6 d)<TD>(&#8722;4 a + d + 4 e)<TD>(&#8722;2 a &#8722; b + 4 f)
    <TR><TD>20<TD><TD>a<TD>(&#8722;19 a + 20 b)<TD>(&#8722;9 a + 10 c)<TD>(&#8722;6 a + c + 6 d)<TD>(&#8722;4 a + 5 e)<TD>(&#8722;3 a + 4 f)
    <TR><TD>21<TD><TD>a<TD>(&#8722;20 a + 21 b)<TD>(&#8722;10 a + b + 10 c)<TD>(&#8722;6 a + 7 d)<TD>(&#8722;4 a &#8722; d + 6 e)<TD>(&#8722;4 a + b + 4 f)
    <TR><TD>22<TD><TD>a<TD>(&#8722;21 a + 22 b)<TD>(&#8722;10 a + 11 c)<TD>(&#8722;6 a &#8722; c + 8 d)<TD>(&#8722;4 a &#8722; c + 6 e)<TD>(&#8722;4 a + c + 4 f)
    <TR><TD>23<TD><TD>a<TD>(&#8722;22 a + 23 b)<TD>(&#8722;10 a &#8722; b + 12 c)<TD>(&#8722;6 a &#8722; b + 8 d)<TD>(&#8722;4 a &#8722; b + 6 e)<TD>(&#8722;4 a + d + 4 f)
    <TR><TD>24<TD><TD>a<TD>(&#8722;23 a + 24 b)<TD>(&#8722;11 a + 12 c)<TD>(&#8722;7 a + 8 d)<TD>(&#8722;5 a + 6 e)<TD>(&#8722;4 a + e + 4 f)
    <TR><TD>25<TD><TD>a<TD>(&#8722;24 a + 25 b)<TD>(&#8722;12 a + b + 12 c)<TD>(&#8722;8 a + b + 8 d)<TD>(&#8722;6 a + b + 6 e)<TD>(&#8722;4 a + 5 f)
    <TR><TD>26<TD><TD>a<TD>(&#8722;25 a + 26 b)<TD>(&#8722;12 a + 13 c)<TD>(&#8722;8 a + c + 8 d)<TD>(&#8722;6 a + c + 6 e)<TD>(&#8722;4 a &#8722; e + 6 f)
    <TR><TD>27<TD><TD>a<TD>(&#8722;26 a + 27 b)<TD>(&#8722;12 a &#8722; b + 14 c)<TD>(&#8722;8 a + 9 d)<TD>(&#8722;6 a + d + 6 e)<TD>(&#8722;4 a &#8722; d + 6 f)
    <TR><TD>28<TD><TD>a<TD>(&#8722;27 a + 28 b)<TD>(&#8722;13 a + 14 c)<TD>(&#8722;8 a &#8722; c + 10 d)<TD>(&#8722;6 a + 7 e)<TD>(&#8722;4 a &#8722; c + 6 f)
    <TR><TD>29<TD><TD>a<TD>(&#8722;28 a + 29 b)<TD>(&#8722;14 a + b + 14 c)<TD>(&#8722;8 a &#8722; b + 10 d)<TD>(&#8722;6 a &#8722; d + 8 e)<TD>(&#8722;4 a &#8722; b + 6 f)
    </TABLE>`

    Parameters
    ----------
    data : np.ndarray[tuple[int], np.dtype[np.float64]],
        Data to extend.
    at : int
        Arbitrary index.

    Returns
    -------
    float
        The value of the padded data at the requested index.

    Examples
    --------
    Load the library.
        >>> import splinekit as sk
    Short data at large index ``-42``.
        >>> sk.pad_a([1, 5, -3], at = -42)
        85

    ----

    """

    k0 = len(data)
    if 1 == k0:
        return data[0]
    k2 = 2 * k0 - 2
    p = (at - 1) // k2
    q = at - k2 * p
    return (2 * p * (data[-1] - data[0]) +
        (data[q] if q < k0 else 2 * data[-1] - data[k2 - q])
    )

#---------------
def samples_to_coeff_a (
    data: np.ndarray[tuple[int], np.dtype[np.float64]],
    *,
    degree: int
) -> None:

    r"""
    .. _samples_to_coeff_a:

    In-place conversion of data samples into spline coefficients under
    anti-mirror padding.

    Replaces a one-dimensional ``numpy.ndarray`` of data samples :math:`f` by
    spline coefficients :math:`c` such that

    ..  math::

        \forall k\in[0\ldots K-1]:f[k]=\sum_{q\in{\mathbb{Z}}}\,c[q]\,
        \beta^{n}(k-q),

    where :math:`K` is the length of the provided data and :math:`\beta^{n}`
    is a :ref:`polynomial B-spline<def-b_spline>` of
    :ref:`nonnegative<def-negative>` degree :math:`n.` The spline coefficients
    are assumed to conform to a :ref:`anti-mirror padding<pad_a>`.

    Parameters
    ----------
    data : np.ndarray[tuple[int], np.dtype[np.float64]],
        Data to convert from samples to coefficients.
    degree : int
        Nonnegative degree of the polynomial B-spline.

    Returns
    -------
    None :
        The returned coefficients overwrite ``data``.

    Examples
    --------
    Load the libraries.
        >>> import numpy as np
        >>> import splinekit as sk
    Cubic coefficients of arbitrary data with anti-mirror padding.
        >>> f = np.array([1, 5, -3], dtype = "float")
        >>> c = f.copy()
        >>> sk.samples_to_coeff_a(c, degree = 3)
        >>> print(c)
        [ 1.  8. -3.]

    Notes
    -----
    The computations are performed in-place.

    ----

    """

    if np.ndarray != type(data):
        raise ValueError("Data must be a numpy array")
    if 1 != len(data.shape):
        raise ValueError("Data must be a one-dimensional numpy array")
    if float != data.dtype:
        raise ValueError(
            "Data must be np.ndarray[tuple[int], np.dtype[np.float64]]"
        )
    if 0 == len(data):
        raise ValueError("Data must contain at least one element")
    if 0 > degree:
        raise ValueError("Degree must be nonnegative")
    p = pole(degree)
    k0 = len(data)
    if (0 == len(p)) or (1 == k0):
        return
    for z in p:
        sigma1 = 0.0
        sigma2 = 0.0
        zeta = z
        k = 1
        while (k < k0 - 1) and (0.0 != zeta):
            sigma1 += zeta * data[k]
            sigma2 += zeta * data[-1 - k]
            zeta *= z
            k += 1
        data[0] = ((data[0] - zeta * data[-1]) * (1.0 + z) / (1.0 - z) -
            sigma1 + zeta * sigma2) / (1.0 - zeta ** 2)
        for k in range(1, k0):
            data[k] += z * data[k - 1]
        z12 = (1.0 - z) ** 2
        data[-1] -= z * data[-2]
        for k in range(1, k0):
            data[-1 - k] = z * data[-k] + z12 * data[-1 - k]

#---------------
def pad_np (
    data: np.ndarray[tuple[int], np.dtype[np.float64]],
    *,
    at: int
) -> float:

    r"""
    .. _pad_np:

    Nega-periodic padding.

    Returns the value of the :math:`k`-th data sample after the data have been
    extended by nega-periodic padding. The padded data :math:`f` satisfy that

    ..  math::

        \forall k\in{\mathbb{Z}}:f[k+K]=-f[k],

    where :math:`K` is the length of the unpadded data. The conditions of
    nega-periodic padding imply the periodicity

    ..  math::

        \forall k\in{\mathbb{Z}}:f[k]=f[k+2\,K].

    :raw-html:`<TABLE border="1" frame="hsides" rules="groups" align="center">
    <CAPTION>Nega-periodic padding of <FONT color="blue">data samples</FONT></CAPTION>
    <COLGROUP align="right">
    <COLGROUP align="center" span="7">
    <TR><TH>k<TH>&#160;&#160;&#160;<TH>&#402;<SUB>1</SUB>[k]<TH>&#402;<SUB>2</SUB>[k]<TH>&#402;<SUB>3</SUB>[k]<TH>&#402;<SUB>4</SUB>[k]<TH>&#402;<SUB>5</SUB>[k]<TH>&#402;<SUB>6</SUB>[k]
    <TBODY>
    <TR><TD>&#8722;20<TD><TD>a<TD>a<TD>&#8722;b<TD>&#8722;a<TD>a<TD>e
    <TR><TD>&#8722;19<TD><TD>&#8722;a<TD>b<TD>&#8722;c<TD>&#8722;b<TD>b<TD>f
    <TR><TD>&#8722;18<TD><TD>a<TD>&#8722;a<TD>a<TD>&#8722;c<TD>c<TD>&#8722;a
    <TR><TD>&#8722;17<TD><TD>&#8722;a<TD>&#8722;b<TD>b<TD>&#8722;d<TD>d<TD>&#8722;b
    <TR><TD>&#8722;16<TD><TD>a<TD>a<TD>c<TD>a<TD>e<TD>&#8722;c
    <TR><TD>&#8722;15<TD><TD>&#8722;a<TD>b<TD>&#8722;a<TD>b<TD>&#8722;a<TD>&#8722;d
    <TR><TD>&#8722;14<TD><TD>a<TD>&#8722;a<TD>&#8722;b<TD>c<TD>&#8722;b<TD>&#8722;e
    <TR><TD>&#8722;13<TD><TD>&#8722;a<TD>&#8722;b<TD>&#8722;c<TD>d<TD>&#8722;c<TD>&#8722;f
    <TR><TD>&#8722;12<TD><TD>a<TD>a<TD>a<TD>&#8722;a<TD>&#8722;d<TD>a
    <TR><TD>&#8722;11<TD><TD>&#8722;a<TD>b<TD>b<TD>&#8722;b<TD>&#8722;e<TD>b
    <TR><TD>&#8722;10<TD><TD>a<TD>&#8722;a<TD>c<TD>&#8722;c<TD>a<TD>c
    <TR><TD>&#8722;9<TD><TD>&#8722;a<TD>&#8722;b<TD>&#8722;a<TD>&#8722;d<TD>b<TD>d
    <TR><TD>&#8722;8<TD><TD>a<TD>a<TD>&#8722;b<TD>a<TD>c<TD>e
    <TR><TD>&#8722;7<TD><TD>&#8722;a<TD>b<TD>&#8722;c<TD>b<TD>d<TD>f
    <TR><TD>&#8722;6<TD><TD>a<TD>&#8722;a<TD>a<TD>c<TD>e<TD>&#8722;a
    <TR><TD>&#8722;5<TD><TD>&#8722;a<TD>&#8722;b<TD>b<TD>d<TD>&#8722;a<TD>&#8722;b
    <TR><TD>&#8722;4<TD><TD>a<TD>a<TD>c<TD>&#8722;a<TD>&#8722;b<TD>&#8722;c
    <TR><TD>&#8722;3<TD><TD>&#8722;a<TD>b<TD>&#8722;a<TD>&#8722;b<TD>&#8722;c<TD>&#8722;d
    <TR><TD>&#8722;2<TD><TD>a<TD>&#8722;a<TD>&#8722;b<TD>&#8722;c<TD>&#8722;d<TD>&#8722;e
    <TR><TD>&#8722;1<TD><TD>&#8722;a<TD>&#8722;b<TD>&#8722;c<TD>&#8722;d<TD>&#8722;e<TD>&#8722;f
    <TR><TD>0<TD><TD><FONT color="blue"><B>A</B></FONT><TD><FONT color="blue"><B>A</B></FONT><TD><FONT color="blue"><B>A</B></FONT><TD><FONT color="blue"><B>A</B></FONT><TD><FONT color="blue"><B>A</B></FONT><TD><FONT color="blue"><B>A</B></FONT>
    <TR><TD>1<TD><TD><FONT color="purple"><B>&#8722;a</B></FONT><TD><FONT color="blue"><B>B</B></FONT><TD><FONT color="blue"><B>B</B></FONT><TD><FONT color="blue"><B>B</B></FONT><TD><FONT color="blue"><B>B</B></FONT><TD><FONT color="blue"><B>B</B></FONT>
    <TR><TD>2<TD><TD>a<TD><FONT color="purple"><B>&#8722;a</B></FONT><TD><FONT color="blue"><B>C</B></FONT><TD><FONT color="blue"><B>C</B></FONT><TD><FONT color="blue"><B>C</B></FONT><TD><FONT color="blue"><B>C</B></FONT>
    <TR><TD>3<TD><TD>&#8722;a<TD><FONT color="purple"><B>&#8722;b</B></FONT><TD><FONT color="purple"><B>&#8722;a</B></FONT><TD><FONT color="blue"><B>D</B></FONT><TD><FONT color="blue"><B>D</B></FONT><TD><FONT color="blue"><B>D</B></FONT>
    <TR><TD>4<TD><TD>a<TD>a<TD><FONT color="purple"><B>&#8722;b</B></FONT><TD><FONT color="purple"><B>&#8722;a</B></FONT><TD><FONT color="blue"><B>E</B></FONT><TD><FONT color="blue"><B>E</B></FONT>
    <TR><TD>5<TD><TD>&#8722;a<TD>b<TD><FONT color="purple"><B>&#8722;c</B></FONT><TD><FONT color="purple"><B>&#8722;b</B></FONT><TD><FONT color="purple"><B>&#8722;a</B></FONT><TD><FONT color="blue"><B>F</B></FONT>
    <TR><TD>6<TD><TD>a<TD>&#8722;a<TD>a<TD><FONT color="purple"><B>&#8722;c</B></FONT><TD><FONT color="purple"><B>&#8722;b</B></FONT><TD><FONT color="purple"><B>&#8722;a</B></FONT>
    <TR><TD>7<TD><TD>&#8722;a<TD>&#8722;b<TD>b<TD><FONT color="purple"><B>&#8722;d</B></FONT><TD><FONT color="purple"><B>&#8722;c</B></FONT><TD><FONT color="purple"><B>&#8722;b</B></FONT>
    <TR><TD>8<TD><TD>a<TD>a<TD>c<TD>a<TD><FONT color="purple"><B>&#8722;d</B></FONT><TD><FONT color="purple"><B>&#8722;c</B></FONT>
    <TR><TD>9<TD><TD>&#8722;a<TD>b<TD>&#8722;a<TD>b<TD><FONT color="purple"><B>&#8722;e</B></FONT><TD><FONT color="purple"><B>&#8722;d</B></FONT>
    <TR><TD>10<TD><TD>a<TD>&#8722;a<TD>&#8722;b<TD>c<TD>a<TD><FONT color="purple"><B>&#8722;e</B></FONT>
    <TR><TD>11<TD><TD>&#8722;a<TD>&#8722;b<TD>&#8722;c<TD>d<TD>b<TD><FONT color="purple"><B>&#8722;f</B></FONT>
    <TR><TD>12<TD><TD>a<TD>a<TD>a<TD>&#8722;a<TD>c<TD>a
    <TR><TD>13<TD><TD>&#8722;a<TD>b<TD>b<TD>&#8722;b<TD>d<TD>b
    <TR><TD>14<TD><TD>a<TD>&#8722;a<TD>c<TD>&#8722;c<TD>e<TD>c
    <TR><TD>15<TD><TD>&#8722;a<TD>&#8722;b<TD>&#8722;a<TD>&#8722;d<TD>&#8722;a<TD>d
    <TR><TD>16<TD><TD>a<TD>a<TD>&#8722;b<TD>a<TD>&#8722;b<TD>e
    <TR><TD>17<TD><TD>&#8722;a<TD>b<TD>&#8722;c<TD>b<TD>&#8722;c<TD>f
    <TR><TD>18<TD><TD>a<TD>&#8722;a<TD>a<TD>c<TD>&#8722;d<TD>&#8722;a
    <TR><TD>19<TD><TD>&#8722;a<TD>&#8722;b<TD>b<TD>d<TD>&#8722;e<TD>&#8722;b
    <TR><TD>20<TD><TD>a<TD>a<TD>c<TD>&#8722;a<TD>a<TD>&#8722;c
    <TR><TD>21<TD><TD>&#8722;a<TD>b<TD>&#8722;a<TD>&#8722;b<TD>b<TD>&#8722;d
    <TR><TD>22<TD><TD>a<TD>&#8722;a<TD>&#8722;b<TD>&#8722;c<TD>c<TD>&#8722;e
    <TR><TD>23<TD><TD>&#8722;a<TD>&#8722;b<TD>&#8722;c<TD>&#8722;d<TD>d<TD>&#8722;f
    <TR><TD>24<TD><TD>a<TD>a<TD>a<TD>a<TD>e<TD>a
    <TR><TD>25<TD><TD>&#8722;a<TD>b<TD>b<TD>b<TD>&#8722;a<TD>b
    <TR><TD>26<TD><TD>a<TD>&#8722;a<TD>c<TD>c<TD>&#8722;b<TD>c
    <TR><TD>27<TD><TD>&#8722;a<TD>&#8722;b<TD>&#8722;a<TD>d<TD>&#8722;c<TD>d
    <TR><TD>28<TD><TD>a<TD>a<TD>&#8722;b<TD>&#8722;a<TD>&#8722;d<TD>e
    <TR><TD>29<TD><TD>&#8722;a<TD>b<TD>&#8722;c<TD>&#8722;b<TD>&#8722;e<TD>f
    </TABLE>`

    Parameters
    ----------
    data : np.ndarray[tuple[int], np.dtype[np.float64]],
        Data to extend.
    at : int
        Arbitrary index.

    Returns
    -------
    float
        The value of the padded data at the requested index.

    Examples
    --------
    Load the library.
        >>> import splinekit as sk
    Short data at large index ``-42``.
        >>> sk.pad_np([1, 5, -3], at = -42)
        1

    ----

    """

    k0 = len(data)
    q = at % (2 * k0)
    if 0 <= q < k0:
        return data[q]
    return -data[q - k0]

#---------------
def samples_to_coeff_np (
    data: np.ndarray[tuple[int], np.dtype[np.float64]],
    *,
    degree: int
) -> None:

    r"""
    .. _samples_to_coeff_np:

    In-place conversion of data samples into spline coefficients under
    nega-periodic padding.

    Replaces a one-dimensional ``numpy.ndarray`` of data samples :math:`f` by
    spline coefficients :math:`c` such that

    ..  math::

        \forall k\in[0\ldots K-1]:f[k]=\sum_{q\in{\mathbb{Z}}}\,c[q]\,
        \beta^{n}(k-q),

    where :math:`K` is the length of the provided data and :math:`\beta^{n}`
    is a :ref:`polynomial B-spline<def-b_spline>` of
    :ref:`nonnegative<def-negative>` degree :math:`n.` The spline coefficients
    are assumed to conform to a :ref:`nega-periodic padding<pad_np>`.

    Parameters
    ----------
    data : np.ndarray[tuple[int], np.dtype[np.float64]],
        Data to convert from samples to coefficients.
    degree : int
        Nonnegative degree of the polynomial B-spline.

    Returns
    -------
    None :
        The returned coefficients overwrite ``data``.

    Examples
    --------
    Load the libraries.
        >>> import numpy as np
        >>> import splinekit as sk
    Cubic coefficients of arbitrary data with nega-periodic padding.
        >>> f = np.array([1, 5, -3], dtype = "float")
        >>> c = f.copy()
        >>> sk.samples_to_coeff_np(c, degree = 3)
        >>> print(c)
        [-3.  10.2 -7.8]

    Notes
    -----
    The computations are performed in-place.

    ----

    """

    if np.ndarray != type(data):
        raise ValueError("Data must be a numpy array")
    if 1 != len(data.shape):
        raise ValueError("Data must be a one-dimensional numpy array")
    if float != data.dtype:
        raise ValueError(
            "Data must be np.ndarray[tuple[int], np.dtype[np.float64]]"
        )
    if 0 == len(data):
        raise ValueError("Data must contain at least one element")
    if 0 > degree:
        raise ValueError("Degree must be nonnegative")
    p = pole(degree)
    k0 = len(data)
    if 0 == len(p):
        return
    for z in p:
        sigma1 = data[0]
        sigma2 = data[-1]
        zeta = z
        k = 1
        while k < k0 and (0.0 != zeta):
            sigma1 += zeta * data[k]
            sigma2 += zeta * data[-1 - k]
            zeta *= z
            k += 1
        zz = z / (1.0 + zeta)
        sigma1 *= zz
        sigma2 *= zz
        data[0] -= sigma2
        for k in range(1, k0):
            data[k] += z * data[k - 1]
        zeta *= zeta
        z12 = (1.0 - z) ** 2
        zz = (1.0 - z) / (1.0 + z)
        data[-1] *= (1.0 + zeta) * zz
        data[-1] -= (sigma2 * zeta / z + sigma1) * zz
        for k in range(1, k0):
            data[-1 - k] = z * data[-k] + z12 * data[-1 - k]

#---------------
def pad_nn (
    data: np.ndarray[tuple[int], np.dtype[np.float64]],
    *,
    at: int
) -> float:

    r"""
    .. _pad_nn:

    Nega-narrow-mirror padding.

    Returns the value of the :math:`k`-th data sample after the data have been
    extended by nega-narrow-mirror padding. The padded data :math:`f` satisfy
    that

    ..  math::

        \forall k\in{\mathbb{Z}}:\left\{\begin{array}{rcl}f[k-1]&=&-f[-1-k]\\
        f[k+K+1]&=&-f[K-1-k],\end{array}\right.

    where :math:`K` is the length of the unpadded data. The conditions of
    nega-narrow-mirror padding imply the periodicity

    ..  math::

        \forall k\in{\mathbb{Z}}:\left\{\begin{array}{rcl}f[k]&=&f[k+2\,K+2]\\
        f[k\,\left(K+1\right)-1]&=&0.\end{array}\right.

    :raw-html:`<TABLE border="1" frame="hsides" rules="groups" align="center">
    <CAPTION>Nega-narrow-mirror padding of <FONT color="blue">data samples</FONT></CAPTION>
    <COLGROUP align="right">
    <COLGROUP align="center" span="7">
    <TR><TH>k<TH>&#160;&#160;&#160;<TH>&#402;<SUB>1</SUB>[k]<TH>&#402;<SUB>2</SUB>[k]<TH>&#402;<SUB>3</SUB>[k]<TH>&#402;<SUB>4</SUB>[k]<TH>&#402;<SUB>5</SUB>[k]<TH>&#402;<SUB>6</SUB>[k]
    <TBODY>
    <TR><TD>&#8722;20<TD><TD>a<TD>&#8722;a<TD>&#8722;c<TD>a<TD>e<TD>&#8722;e
    <TR><TD>&#8722;19<TD><TD>0<TD>0<TD>&#8722;b<TD>b<TD>0<TD>&#8722;d
    <TR><TD>&#8722;18<TD><TD>&#8722;a<TD>a<TD>&#8722;a<TD>c<TD>&#8722;e<TD>&#8722;c
    <TR><TD>&#8722;17<TD><TD>0<TD>b<TD>0<TD>d<TD>&#8722;d<TD>&#8722;b
    <TR><TD>&#8722;16<TD><TD>a<TD>0<TD>a<TD>0<TD>&#8722;c<TD>&#8722;a
    <TR><TD>&#8722;15<TD><TD>0<TD>&#8722;b<TD>b<TD>&#8722;d<TD>&#8722;b<TD>0
    <TR><TD>&#8722;14<TD><TD>&#8722;a<TD>&#8722;a<TD>c<TD>&#8722;c<TD>&#8722;a<TD>a
    <TR><TD>&#8722;13<TD><TD>0<TD>0<TD>0<TD>&#8722;b<TD>0<TD>b
    <TR><TD>&#8722;12<TD><TD>a<TD>a<TD>&#8722;c<TD>&#8722;a<TD>a<TD>c
    <TR><TD>&#8722;11<TD><TD>0<TD>b<TD>&#8722;b<TD>0<TD>b<TD>d
    <TR><TD>&#8722;10<TD><TD>&#8722;a<TD>0<TD>&#8722;a<TD>a<TD>c<TD>e
    <TR><TD>&#8722;9<TD><TD>0<TD>&#8722;b<TD>0<TD>b<TD>d<TD>f
    <TR><TD>&#8722;8<TD><TD>a<TD>&#8722;a<TD>a<TD>c<TD>e<TD>0
    <TR><TD>&#8722;7<TD><TD>0<TD>0<TD>b<TD>d<TD>0<TD>&#8722;f
    <TR><TD>&#8722;6<TD><TD>&#8722;a<TD>a<TD>c<TD>0<TD>&#8722;e<TD>&#8722;e
    <TR><TD>&#8722;5<TD><TD>0<TD>b<TD>0<TD>&#8722;d<TD>&#8722;d<TD>&#8722;d
    <TR><TD>&#8722;4<TD><TD>a<TD>0<TD>&#8722;c<TD>&#8722;c<TD>&#8722;c<TD>&#8722;c
    <TR><TD>&#8722;3<TD><TD>0<TD>&#8722;b<TD>&#8722;b<TD>&#8722;b<TD>&#8722;b<TD>&#8722;b
    <TR><TD>&#8722;2<TD><TD>&#8722;a<TD>&#8722;a<TD>&#8722;a<TD>&#8722;a<TD>&#8722;a<TD>&#8722;a
    <TR><TD>&#8722;1<TD><TD>0<TD>0<TD>0<TD>0<TD>0<TD>0
    <TR><TD>0<TD><TD><FONT color="blue"><B>A</B></FONT><TD><FONT color="blue"><B>A</B></FONT><TD><FONT color="blue"><B>A</B></FONT><TD><FONT color="blue"><B>A</B></FONT><TD><FONT color="blue"><B>A</B></FONT><TD><FONT color="blue"><B>A</B></FONT>
    <TR><TD>1<TD><TD><FONT color="purple"><B>0</B></FONT><TD><FONT color="blue"><B>B</B></FONT><TD><FONT color="blue"><B>B</B></FONT><TD><FONT color="blue"><B>B</B></FONT><TD><FONT color="blue"><B>B</B></FONT><TD><FONT color="blue"><B>B</B></FONT>
    <TR><TD>2<TD><TD><FONT color="purple"><B>&#8722;a</B></FONT><TD><FONT color="purple"><B>0</B></FONT><TD><FONT color="blue"><B>C</B></FONT><TD><FONT color="blue"><B>C</B></FONT><TD><FONT color="blue"><B>C</B></FONT><TD><FONT color="blue"><B>C</B></FONT>
    <TR><TD>3<TD><TD><FONT color="purple"><B>0</B></FONT><TD><FONT color="purple"><B>&#8722;b</B></FONT><TD><FONT color="purple"><B>0</B></FONT><TD><FONT color="blue"><B>D</B></FONT><TD><FONT color="blue"><B>D</B></FONT><TD><FONT color="blue"><B>D</B></FONT>
    <TR><TD>4<TD><TD>a<TD><FONT color="purple"><B>&#8722;a</B></FONT><TD><FONT color="purple"><B>&#8722;c</B></FONT><TD><FONT color="purple"><B>0</B></FONT><TD><FONT color="blue"><B>E</B></FONT><TD><FONT color="blue"><B>E</B></FONT>
    <TR><TD>5<TD><TD>0<TD><FONT color="purple"><B>0</B></FONT><TD><FONT color="purple"><B>&#8722;b</B></FONT><TD><FONT color="purple"><B>&#8722;d</B></FONT><TD><FONT color="purple"><B>0</B></FONT><TD><FONT color="blue"><B>F</B></FONT>
    <TR><TD>6<TD><TD>&#8722;a<TD>a<TD><FONT color="purple"><B>&#8722;a</B></FONT><TD><FONT color="purple"><B>&#8722;c</B></FONT><TD><FONT color="purple"><B>&#8722;e</B></FONT><TD><FONT color="purple"><B>0</B></FONT>
    <TR><TD>7<TD><TD>0<TD>b<TD><FONT color="purple"><B>0</B></FONT><TD><FONT color="purple"><B>&#8722;b</B></FONT><TD><FONT color="purple"><B>&#8722;d</B></FONT><TD><FONT color="purple"><B>&#8722;f</B></FONT>
    <TR><TD>8<TD><TD>a<TD>0<TD>a<TD><FONT color="purple"><B>&#8722;a</B></FONT><TD><FONT color="purple"><B>&#8722;c</B></FONT><TD><FONT color="purple"><B>&#8722;e</B></FONT>
    <TR><TD>9<TD><TD>0<TD>&#8722;b<TD>b<TD><FONT color="purple"><B>0</B></FONT><TD><FONT color="purple"><B>&#8722;b</B></FONT><TD><FONT color="purple"><B>&#8722;d</B></FONT>
    <TR><TD>10<TD><TD>&#8722;a<TD>&#8722;a<TD>c<TD>a<TD><FONT color="purple"><B>&#8722;a</B><TD><FONT color="purple"><B>&#8722;c</B></FONT>
    <TR><TD>11<TD><TD>0<TD>0<TD>0<TD>b<TD><FONT color="purple"><B>0</B><TD><FONT color="purple"><B>&#8722;b</B></FONT>
    <TR><TD>12<TD><TD>a<TD>a<TD>&#8722;c<TD>c<TD>a<TD><FONT color="purple"><B>&#8722;a</B></FONT>
    <TR><TD>13<TD><TD>0<TD>b<TD>&#8722;b<TD>d<TD>b<TD><FONT color="purple"><B>0</B></FONT>
    <TR><TD>14<TD><TD>&#8722;a<TD>0<TD>&#8722;a<TD>0<TD>c<TD>a
    <TR><TD>15<TD><TD>0<TD>&#8722;b<TD>0<TD>&#8722;d<TD>d<TD>b
    <TR><TD>16<TD><TD>a<TD>&#8722;a<TD>a<TD>&#8722;c<TD>e<TD>c
    <TR><TD>17<TD><TD>0<TD>0<TD>b<TD>&#8722;b<TD>0<TD>d
    <TR><TD>18<TD><TD>&#8722;a<TD>a<TD>c<TD>&#8722;a<TD>&#8722;e<TD>e
    <TR><TD>19<TD><TD>0<TD>b<TD>0<TD>0<TD>&#8722;d<TD>f
    <TR><TD>20<TD><TD>a<TD>0<TD>&#8722;c<TD>a<TD>&#8722;c<TD>0
    <TR><TD>21<TD><TD>0<TD>&#8722;b<TD>&#8722;b<TD>b<TD>&#8722;b<TD>&#8722;f
    <TR><TD>22<TD><TD>&#8722;a<TD>&#8722;a<TD>&#8722;a<TD>c<TD>&#8722;a<TD>&#8722;e
    <TR><TD>23<TD><TD>0<TD>0<TD>0<TD>d<TD>0<TD>&#8722;d
    <TR><TD>24<TD><TD>a<TD>a<TD>a<TD>0<TD>a<TD>&#8722;c
    <TR><TD>25<TD><TD>0<TD>b<TD>b<TD>&#8722;d<TD>b<TD>&#8722;b
    <TR><TD>26<TD><TD>&#8722;a<TD>0<TD>c<TD>&#8722;c<TD>c<TD>&#8722;a
    <TR><TD>27<TD><TD>0<TD>&#8722;b<TD>0<TD>&#8722;b<TD>d<TD>0
    <TR><TD>28<TD><TD>a<TD>&#8722;a<TD>&#8722;c<TD>&#8722;a<TD>e<TD>a
    <TR><TD>29<TD><TD>0<TD>0<TD>&#8722;b<TD>0<TD>0<TD>b
    </TABLE>`

    Parameters
    ----------
    data : np.ndarray[tuple[int], np.dtype[np.float64]],
        Data to extend.
    at : int
        Arbitrary index.

    Returns
    -------
    float
        The value of the padded data at the requested index.

    Examples
    --------
    Load the library.
        >>> import splinekit as sk
    Short data at large index ``-42``.
        >>> sk.pad_nn([1, 5, -3], at = -42)
        -1

    ----

    """

    k0 = len(data)
    q = at - (2 * k0 + 2) * (at // (2 * k0 + 2))
    if q < k0:
        return data[q]
    if q == k0:
        return 0 * data[0]
    if q <= 2 * k0:
        return -data[2 * k0 - q]
    return 0 * data[0]

#---------------
def samples_to_coeff_nn (
    data: np.ndarray[tuple[int], np.dtype[np.float64]],
    *,
    degree: int
) -> None:

    r"""
    .. _samples_to_coeff_nn:

    In-place conversion of data samples into spline coefficients under
    nega-narrow-mirror padding.

    Replaces a one-dimensional ``numpy.ndarray`` of data samples :math:`f` by
    spline coefficients :math:`c` such that

    ..  math::

        \forall k\in[0\ldots K-1]:f[k]=\sum_{q\in{\mathbb{Z}}}\,c[q]\,
        \beta^{n}(k-q),

    where :math:`K` is the length of the provided data and :math:`\beta^{n}`
    is a :ref:`polynomial B-spline<def-b_spline>` of
    :ref:`nonnegative<def-negative>` degree :math:`n.` The spline coefficients
    are assumed to conform to a :ref:`nega-narrow-mirror padding<pad_nn>`.

    Parameters
    ----------
    data : np.ndarray[tuple[int], np.dtype[np.float64]],
        Data to convert from samples to coefficients.
    degree : int
        Nonnegative degree of the polynomial B-spline.

    Returns
    -------
    None :
        The returned coefficients overwrite ``data``.

    Examples
    --------
    Load the libraries.
        >>> import numpy as np
        >>> import splinekit as sk
    Cubic coefficients of arbitrary data with nega-narrow-mirror padding.
        >>> f = np.array([1, 5, -3], dtype = "float")
        >>> c = f.copy()
        >>> sk.samples_to_coeff_nn(c, degree = 3)
        >>> print(c)
        [-0.85714286  9.42857143 -6.85714286]

    Notes
    -----
    The computations are performed in-place.

    ----

    """

    if np.ndarray != type(data):
        raise ValueError("Data must be a numpy array")
    if 1 != len(data.shape):
        raise ValueError("Data must be a one-dimensional numpy array")
    if float != data.dtype:
        raise ValueError(
            "Data must be np.ndarray[tuple[int], np.dtype[np.float64]]"
        )
    if 0 == len(data):
        raise ValueError("Data must contain at least one element")
    if 0 > degree:
        raise ValueError("Degree must be nonnegative")
    p = pole(degree)
    k0 = len(data)
    if 0 == len(p):
        return
    for z in p:
        sigma1 = 0.0
        sigma2 = 0.0
        zeta = 1.0
        k = 0
        while k < k0 and (0.0 != zeta):
            sigma1 += zeta * data[k]
            sigma2 += zeta * data[-1 - k]
            zeta *= z
            k += 1
        zeta *= z
        data[0] -= (sigma1 - zeta * sigma2) * (z ** 2) / (1.0 - zeta ** 2)
        for k in range(1, k0):
            data[k] += z * data[k - 1]
        z12 = (1.0 - z) ** 2
        data[-1] *= z12
        for k in range(1, k0):
            data[-1 - k] = z * data[-k] + z12 * data[-1 - k]

#---------------
def pad_nw (
    data: np.ndarray[tuple[int], np.dtype[np.float64]],
    *,
    at: int
) -> float:

    r"""
    .. _pad_nw:

    Nega-wide-mirror padding.

    Returns the value of the :math:`k`-th data sample after the data have been
    extended by nega-wide-mirror padding. The padded data :math:`f` satisfy
    that

    ..  math::

        \forall k\in{\mathbb{Z}}:\left\{\begin{array}{rcl}f[k]&=&-f[-1-k]\\
        f[k+K]&=&-f[K-1-k],\end{array}\right.

    where :math:`K` is the length of the unpadded data. The conditions of
    nega-wide-mirror padding imply the periodicity

    ..  math::

        \forall k\in{\mathbb{Z}}:f[k]=f[k+2\,K].

    :raw-html:`<TABLE border="1" frame="hsides" rules="groups" align="center">
    <CAPTION>Nega-wide-mirror padding of <FONT color="blue">data samples</FONT></CAPTION>
    <COLGROUP align="right">
    <COLGROUP align="center" span="7">
    <TR><TH>k<TH>&#160;&#160;&#160;<TH>&#402;<SUB>1</SUB>[k]<TH>&#402;<SUB>2</SUB>[k]<TH>&#402;<SUB>3</SUB>[k]<TH>&#402;<SUB>4</SUB>[k]<TH>&#402;<SUB>5</SUB>[k]<TH>&#402;<SUB>6</SUB>[k]
    <TBODY>
    <TR><TD>&#8722;20<TD><TD>a<TD>a<TD>&#8722;b<TD>&#8722;d<TD>a<TD>e
    <TR><TD>&#8722;19<TD><TD>&#8722;a<TD>b<TD>&#8722;a<TD>&#8722;c<TD>b<TD>f
    <TR><TD>&#8722;18<TD><TD>a<TD>&#8722;b<TD>a<TD>&#8722;b<TD>c<TD>&#8722;f
    <TR><TD>&#8722;17<TD><TD>&#8722;a<TD>&#8722;a<TD>b<TD>&#8722;a<TD>d<TD>&#8722;e
    <TR><TD>&#8722;16<TD><TD>a<TD>a<TD>c<TD>a<TD>e<TD>&#8722;d
    <TR><TD>&#8722;15<TD><TD>&#8722;a<TD>b<TD>&#8722;c<TD>b<TD>&#8722;e<TD>&#8722;c
    <TR><TD>&#8722;14<TD><TD>a<TD>&#8722;b<TD>&#8722;b<TD>c<TD>&#8722;d<TD>&#8722;b
    <TR><TD>&#8722;13<TD><TD>&#8722;a<TD>&#8722;a<TD>&#8722;a<TD>d<TD>&#8722;c<TD>&#8722;a
    <TR><TD>&#8722;12<TD><TD>a<TD>a<TD>a<TD>&#8722;d<TD>&#8722;b<TD>a
    <TR><TD>&#8722;11<TD><TD>&#8722;a<TD>b<TD>b<TD>&#8722;c<TD>&#8722;a<TD>b
    <TR><TD>&#8722;10<TD><TD>a<TD>&#8722;b<TD>c<TD>&#8722;b<TD>a<TD>c
    <TR><TD>&#8722;9<TD><TD>&#8722;a<TD>&#8722;a<TD>&#8722;c<TD>&#8722;a<TD>b<TD>d
    <TR><TD>&#8722;8<TD><TD>a<TD>a<TD>&#8722;b<TD>a<TD>c<TD>e
    <TR><TD>&#8722;7<TD><TD>&#8722;a<TD>b<TD>&#8722;a<TD>b<TD>d<TD>f
    <TR><TD>&#8722;6<TD><TD>a<TD>&#8722;b<TD>a<TD>c<TD>e<TD>&#8722;f
    <TR><TD>&#8722;5<TD><TD>&#8722;a<TD>&#8722;a<TD>b<TD>d<TD>&#8722;e<TD>&#8722;e
    <TR><TD>&#8722;4<TD><TD>a<TD>a<TD>c<TD>&#8722;d<TD>&#8722;d<TD>&#8722;d
    <TR><TD>&#8722;3<TD><TD>&#8722;a<TD>b<TD>&#8722;c<TD>&#8722;c<TD>&#8722;c<TD>&#8722;c
    <TR><TD>&#8722;2<TD><TD>a<TD>&#8722;b<TD>&#8722;b<TD>&#8722;b<TD>&#8722;b<TD>&#8722;b
    <TR><TD>&#8722;1<TD><TD>&#8722;a<TD>&#8722;a<TD>&#8722;a<TD>&#8722;a<TD>&#8722;a<TD>&#8722;a
    <TR><TD>0<TD><TD><FONT color="blue"><B>A</B></FONT><TD><FONT color="blue"><B>A</B></FONT><TD><FONT color="blue"><B>A</B></FONT><TD><FONT color="blue"><B>A</B></FONT><TD><FONT color="blue"><B>A</B></FONT><TD><FONT color="blue"><B>A</B></FONT>
    <TR><TD>1<TD><TD><FONT color="purple"><B>&#8722;a</B></FONT><TD><FONT color="blue"><B>B</B></FONT><TD><FONT color="blue"><B>B</B></FONT><TD><FONT color="blue"><B>B</B></FONT><TD><FONT color="blue"><B>B</B></FONT><TD><FONT color="blue"><B>B</B></FONT>
    <TR><TD>2<TD><TD>a<TD><FONT color="purple"><B>&#8722;b</B></FONT><TD><FONT color="blue"><B>C</B></FONT><TD><FONT color="blue"><B>C</B></FONT><TD><FONT color="blue"><B>C</B></FONT><TD><FONT color="blue"><B>C</B></FONT>
    <TR><TD>3<TD><TD>&#8722;a<TD><FONT color="purple"><B>&#8722;a</B></FONT><TD><FONT color="purple"><B>&#8722;c</B></FONT><TD><FONT color="blue"><B>D</B></FONT><TD><FONT color="blue"><B>D</B></FONT><TD><FONT color="blue"><B>D</B></FONT>
    <TR><TD>4<TD><TD>a<TD>a<TD><FONT color="purple"><B>&#8722;b</B></FONT><TD><FONT color="purple"><B>&#8722;d</B></FONT><TD><FONT color="blue"><B>E</B></FONT><TD><FONT color="blue"><B>E</B></FONT>
    <TR><TD>5<TD><TD>&#8722;a<TD>b<TD><FONT color="purple"><B>&#8722;a</B></FONT><TD><FONT color="purple"><B>&#8722;c</B></FONT><TD><FONT color="purple"><B>&#8722;e</B></FONT><TD><FONT color="blue"><B>F</B></FONT>
    <TR><TD>6<TD><TD>a<TD>&#8722;b<TD>a<TD><FONT color="purple"><B>&#8722;b</B></FONT><TD><FONT color="purple"><B>&#8722;d</B></FONT><TD><FONT color="purple"><B>&#8722;f</B></FONT>
    <TR><TD>7<TD><TD>&#8722;a<TD>&#8722;a<TD>b<TD><FONT color="purple"><B>&#8722;a</B></FONT><TD><FONT color="purple"><B>&#8722;c</B></FONT><TD><FONT color="purple"><B>&#8722;e</B></FONT>
    <TR><TD>8<TD><TD>a<TD>a<TD>c<TD>a<TD><FONT color="purple"><B>&#8722;b</B></FONT><TD><FONT color="purple"><B>&#8722;d</B></FONT>
    <TR><TD>9<TD><TD>&#8722;a<TD>b<TD>&#8722;c<TD>b<TD><FONT color="purple"><B>&#8722;a</B></FONT><TD><FONT color="purple"><B>&#8722;c</B></FONT>
    <TR><TD>10<TD><TD>a<TD>&#8722;b<TD>&#8722;b<TD>c<TD>a<TD><FONT color="purple"><B>&#8722;b</B></FONT>
    <TR><TD>11<TD><TD>&#8722;a<TD>&#8722;a<TD>&#8722;a<TD>d<TD>b<TD><FONT color="purple"><B>&#8722;a</B></FONT>
    <TR><TD>12<TD><TD>a<TD>a<TD>a<TD>&#8722;d<TD>c<TD>a
    <TR><TD>13<TD><TD>&#8722;a<TD>b<TD>b<TD>&#8722;c<TD>d<TD>b
    <TR><TD>14<TD><TD>a<TD>&#8722;b<TD>c<TD>&#8722;b<TD>e<TD>c
    <TR><TD>15<TD><TD>&#8722;a<TD>&#8722;a<TD>&#8722;c<TD>&#8722;a<TD>&#8722;e<TD>d
    <TR><TD>16<TD><TD>a<TD>a<TD>&#8722;b<TD>a<TD>&#8722;d<TD>e
    <TR><TD>17<TD><TD>&#8722;a<TD>b<TD>&#8722;a<TD>b<TD>&#8722;c<TD>f
    <TR><TD>18<TD><TD>a<TD>&#8722;b<TD>a<TD>c<TD>&#8722;b<TD>&#8722;f
    <TR><TD>19<TD><TD>&#8722;a<TD>&#8722;a<TD>b<TD>d<TD>&#8722;a<TD>&#8722;e
    <TR><TD>20<TD><TD>a<TD>a<TD>c<TD>&#8722;d<TD>a<TD>&#8722;d
    <TR><TD>21<TD><TD>&#8722;a<TD>b<TD>&#8722;c<TD>&#8722;c<TD>b<TD>&#8722;c
    <TR><TD>22<TD><TD>a<TD>&#8722;b<TD>&#8722;b<TD>&#8722;b<TD>c<TD>&#8722;b
    <TR><TD>23<TD><TD>&#8722;a<TD>&#8722;a<TD>&#8722;a<TD>&#8722;a<TD>d<TD>&#8722;a
    <TR><TD>24<TD><TD>a<TD>a<TD>a<TD>a<TD>e<TD>a
    <TR><TD>25<TD><TD>&#8722;a<TD>b<TD>b<TD>b<TD>&#8722;e<TD>b
    <TR><TD>26<TD><TD>a<TD>&#8722;b<TD>c<TD>c<TD>&#8722;d<TD>c
    <TR><TD>27<TD><TD>&#8722;a<TD>&#8722;a<TD>&#8722;c<TD>d<TD>&#8722;c<TD>d
    <TR><TD>28<TD><TD>a<TD>a<TD>&#8722;b<TD>&#8722;d<TD>&#8722;b<TD>e
    <TR><TD>29<TD><TD>&#8722;a<TD>b<TD>&#8722;a<TD>&#8722;c<TD>&#8722;a<TD>f
    </TABLE>`

    Parameters
    ----------
    data : np.ndarray[tuple[int], np.dtype[np.float64]],
        Data to extend.
    at : int
        Arbitrary index.

    Returns
    -------
    float
        The value of the padded data at the requested index.

    Examples
    --------
    Load the library.
        >>> import splinekit as sk
    Short data at large index ``-42``.
        >>> sk.pad_nw([1, 5, -3], at = -42)
        1

    ----

    """

    k0 = len(data)
    q = floor(abs(0.5 + at)) % (2 * k0)
    return int(_sgn(0.5 + at)) * (data[q] if q < k0 else -data[2 * k0 - 1 - q])

#---------------
def samples_to_coeff_nw (
    data: np.ndarray[tuple[int], np.dtype[np.float64]],
    *,
    degree: int
) -> None:

    r"""
    .. _samples_to_coeff_nw:

    In-place conversion of data samples into spline coefficients under
    nega-wide-mirror padding.

    Replaces a one-dimensional ``numpy.ndarray`` of data samples :math:`f` by
    spline coefficients :math:`c` such that

    ..  math::

        \forall k\in[0\ldots K-1]:f[k]=\sum_{q\in{\mathbb{Z}}}\,c[q]\,
        \beta^{n}(k-q),

    where :math:`K` is the length of the provided data and :math:`\beta^{n}`
    is a :ref:`polynomial B-spline<def-b_spline>` of
    :ref:`nonnegative<def-negative>` degree :math:`n.` The spline coefficients
    are assumed to conform to a :ref:`nega-wide-mirror padding<pad_nw>`.

    Parameters
    ----------
    data : np.ndarray[tuple[int], np.dtype[np.float64]],
        Data to convert from samples to coefficients.
    degree : int
        Nonnegative degree of the polynomial B-spline.

    Returns
    -------
    None :
        The returned coefficients overwrite ``data``.

    Examples
    --------
    Load the libraries.
        >>> import numpy as np
        >>> import splinekit as sk
    Cubic coefficients of arbitrary data with nega-wide-mirror padding.
        >>> f = np.array([1, 5, -3], dtype = "float")
        >>> c = f.copy()
        >>> sk.samples_to_coeff_nw(c, degree = 3)
        >>> print(c)
        [-1.4 10.2 -9.4]

    Notes
    -----
    The computations are performed in-place.
    """

    if np.ndarray != type(data):
        raise ValueError("Data must be a numpy array")
    if 1 != len(data.shape):
        raise ValueError("Data must be a one-dimensional numpy array")
    if float != data.dtype:
        raise ValueError(
            "Data must be np.ndarray[tuple[int], np.dtype[np.float64]]"
        )
    if 0 == len(data):
        raise ValueError("Data must contain at least one element")
    if 0 > degree:
        raise ValueError("Degree must be nonnegative")
    p = pole(degree)
    k0 = len(data)
    if 0 == len(p):
        return
    for z in p:
        sigma1 = 0.0
        sigma2 = 0.0
        zeta = 1.0
        k = 0
        while k < k0 and (0.0 != zeta):
            sigma1 += zeta * data[k]
            sigma2 += zeta * data[-1 - k]
            zeta *= z
            k += 1
        data[0] -= (sigma1 - zeta * sigma2) * z / (1.0 - zeta ** 2)
        for k in range(1, k0):
            data[k] += z * data[k - 1]
        z12 = (1.0 - z) ** 2
        data[-1] *= z12 / (1.0 + z)
        for k in range(1, k0):
            data[-1 - k] = z * data[-k] + z12 * data[-1 - k]
