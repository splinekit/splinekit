.. _glossary-splines:

Glossary—Splines
================

.. _def-spline:

**Polynomial Spline** For the degree :math:`n=0,` the :ref:`piecewise polynomial<def-piecewise_polynomial>` :math:`p_{{\mathbb{S}},{\mathbb{F}},{\mathbb{P}}}^{0}` is called a piecewise-constant spline when :math:`{\mathbb{F}}({\mathbb{S}})` is such that :math:`\forall k\in{\mathbb{Z}}:f_{k}=\frac{p_{{\mathbf{a}}_{k-1}^{0}}((s_{k-1}+s_{k})/2)+p_{{\mathbf{a}}_{k}^{0}}((s_{k}+s_{k+1})/2)}{2}.` For a :ref:`positive<def-positive>` integer degree :math:`n,` the :ref:`piecewise polynomial<def-piecewise_polynomial>` :math:`p_{{\mathbb{S}},{\mathbb{F}},{\mathbb{P}}}^{n}` is called a spline when it happens to be of differentiability class :math:`{\mathcal{C}}^{n-1}` over :math:`{\mathbb{R}}.`

.. _def-uniform_spline:

**Uniform Polynomial Spline** A :ref:`polynomial spline<def-spline>` is said to be uniform when it admits the :ref:`uniform split<def-uniform_split>` :math:`{\mathbb{S}}(s_{0})` as its :ref:`split of the real numbers<def-split>`.

.. _def-b_spline:

**Polynomial B-Spline** A polynomial B-spline of :ref:`nonnegative<def-negative>` integer degree is uniquely defined as a :ref:`uniform spline<def-uniform_spline>` of minimal :ref:`support<def-support>` that is :ref:`pointwise even-symmetric<def-even_symmetry>` and that satisfies the :ref:`pointwise partition of unity<def-partition_of_unity>`. It is notated :math:`\beta^{n}:{\mathbb{R}}\rightarrow{\mathbb{R}},` where :math:`n` is the degree.

.. _def-cardinal_b_spline:

**Cardinal Polynomial B-Spline** A cardinal polynomial B-spline of :ref:`nonnegative<def-negative>` integer degree is uniquely defined as a :ref:`uniform spline<def-uniform_spline>` that is :ref:`interpolating<def-interpolating_function>`, :ref:`pointwise even-symmetric<def-even_symmetry>`, and that satisfies the :ref:`pointwise partition of unity<def-partition_of_unity>`. It is notated :math:`\eta^{n}:{\mathbb{R}}\rightarrow{\mathbb{R}},` where :math:`n` is the degree.

.. _def-dual_b_spline:

**Polynomial Dual B-Spline** A polynomial dual B-spline is notated :math:`\mathring{\beta}^{m,n}.` It is a real function indexed by a :ref:`nonnegative<def-negative>` integer dual degree :math:`m` and a :ref:`nonnegative<def-negative>` integer primal degree :math:`n.` It is uniquely defined as the spline of dual degree :math:`m` that is :ref:`integer-shift-orthonormal<def-integer-shift_orthonormal>` to a :ref:`polynomial B-spline<def-b_spline>` of primal degree :math:`n,` so that :math:`{\mathbf{[\![}}k=0\,{\mathbf{]\!]}}=\int_{-\infty}^{\infty}\,\mathring{\beta}^{m,n}(x)\,\beta^{n}(x+k)\,{\mathrm{d}}x,` where the notation :math:`{\mathbf{[\![}}P\,{\mathbf{]\!]}}` is that of the :ref:`Iverson bracket<def-iverson>`.

.. _def-orthonormal_b_spline:

**Polynomial Orthonormal B-Spline** A polynomial orthonormal B-spline is notated :math:`\phi^{n}.` It is a real function indexed by a :ref:`nonnegative<def-negative>` integer degree :math:`n.` It is uniquely defined as the :ref:`uniform spline<def-uniform_spline>` of degree :math:`n` that is :ref:`integer-shift-orthonormal<def-integer-shift_orthonormal>` to itself, so that :math:`{\mathbf{[\![}}k=0\,{\mathbf{]\!]}}=\int_{-\infty}^{\infty}\,\phi^{n}(x)\,\phi^{n}(x+k)\,{\mathrm{d}}x,` along with :math:`\phi^{n}(x)=\sum_{k\in{\mathbb{Z}}}\,p[k]\,\beta^{n}(x-k)` for some well-chosen sequence :math:`p,` where the notation :math:`{\mathbf{[\![}}P\,{\mathbf{]\!]}}` is that of the :ref:`Iverson bracket<def-iverson>`.

.. _def-knots:

**Knots** The knots of a :ref:`polynomial spline<def-spline>` :math:`f` of :ref:`nonnegative<def-negative>` integer degree :math:`n` are those arguments :math:`x` at which :math:`\frac{{\mathrm{d}}^{n+1}f(x)}{{\mathrm{d}}x^{n+1}}\not\in{\mathbb{R}}.` When the spline is :ref:`uniform<def-uniform_spline>`, they are a finite subset of the :ref:`uniform split<def-uniform_split>` :math:`{\mathbb{S}}(s_{0}+\frac{n+1}{2})` associated to the spline.

.. _def-m_scale_relation:

**M-Scale Relation** The M-scale relation expresses a :ref:`B-spline<def-b_spline>` of :ref:`nonnegative<def-negative>` integer degree as a sum of translated and rescaled (minified) B-splines of same degree.

