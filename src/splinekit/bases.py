#---------------
class Bases:

    r"""
    .. _Bases:

    The class that codifies the standard bases of splines.

    The :ref:`uniform splines<def-uniform_spline>` are 
    :ref:`piecewise-polynomial<def-piecewise_polynomial>` functions of some
    :ref:`nonnegative<def-negative>` degree :math:`n.` They can be written as
    the weighted sum of the integer shifts of some basis function. In linear
    algebra, several systems of coordinates can span the same space and
    the actual values of the coordinates of a fixed vector depend on the
    actual system of coordinates. Likewise, the expression of a fixed spline
    :math:`f` depends on the actual basis. With common bases, the same spline
    can be expressed indifferently as

    ..  math::

        \begin{eqnarray*}
        f(x)&=&\sum_{k\in{\mathbb{Z}}}\,c[k]\,\beta^{n}(x-k)\\
        &=&\sum_{k\in{\mathbb{Z}}}\,f(k)\,\eta^{n}(x-k)\\
        &=&\sum_{k\in{\mathbb{Z}}}\,g[k]\,\mathring{\beta}^{n,n}(x-k)\\
        &=&\sum_{k\in{\mathbb{Z}}}\,a[k]\,\phi^{n}(x-k).
        \end{eqnarray*}

    In general, while the spline :math:`f` remains the same, the coefficients
    are mutually different, with
    :math:`f\neq c\neq g\neq a\wedge c\neq a\neq f\neq g.`

    *   For ``splinekit.Bases.BASIC``, the coefficients are :math:`c` and the
        basis is the :ref:`polynomial B-spline<def-b_spline>`
        :math:`\beta^{n},` so that

        ..  math::

            f(x)=\sum_{k\in{\mathbb{Z}}}\,c[k]\,\beta^{n}(x-k).

    *   For ``splinekit.Bases.CARDINAL``, the coefficients are :math:`f` and
        the basis is the :ref:`cardinal spline<def-cardinal_b_spline>`
        :math:`\eta^{n},` so that

        ..  math::

            f(x)=\sum_{k\in{\mathbb{Z}}}\,f(k)\,\eta^{n}(x-k).

    *   For ``splinekit.Bases.DUAL``, the coefficients are :math:`g` and the
        basis is the :ref:`polynomial dual b-spline<def-dual_b_spline>`
        :math:`\mathring{\beta}^{n,n},` so that

        ..  math::

            f(x)=\sum_{k\in{\mathbb{Z}}}\,g[k]\,\mathring{\beta}^{n,n}(x-k).

    *   For ``splinekit.Bases.ORTHONORMAL``, the coefficients are :math:`a`
        and the basis is the
        :ref:`polynomial orthonormal b-spline<def-orthonormal_b_spline>`
        :math:`\phi^{n},` so that

        ..  math::

            f(x)=\sum_{k\in{\mathbb{Z}}}\,a[k]\,\phi^{n}(x-k).

    See Also
    --------
    splinekit.bsplines.b_spline : B-spline basis (BASIC).
    splinekit.bsplines.cardinal_b_spline : Cardinal-spline basis (CARDINAL).
    splinekit.bsplines.dual_b_spline : Dual-spline basis (DUAL).
    splinekit.periodic_spline_1d.PeriodicSpline1D.periodized_b_spline : Periodized B-spline basis.
    splinekit.periodic_spline_1d.PeriodicSpline1D.periodized_cardinal_b_spline : Periodized cardinal-spline basis.
    splinekit.periodic_spline_1d.PeriodicSpline1D.periodized_dual_b_spline : Periodized dual-spline basis.
    splinekit.periodic_spline_1d.PeriodicSpline1D.periodized_orthonormal_b_spline : Periodized orthonormal-spline basis.
    """

    (
        BASIC,
        CARDINAL,
        DUAL,
        ORTHONORMAL
    ) = range(4)
