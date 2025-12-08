.. _glossary-polynomials:

Glossary—Polynomials
====================

.. _def-vandermonde_vector:

**Vandermonde Vector** The Vandermonde vector :math:`{\mathbf{v}}^{n}(x)\in{\mathbb{R}}^{n+1}` of argument :math:`x\in{\mathbb{R}}` and :ref:`nonnegative<def-negative>` integer degree :math:`n\in{\mathbb{N}}` is defined by its :math:`(r+1)`-th row component :math:`v_{r+1}^{n}(x)=v^{n}[r](x)=\left\{\begin{array}{ll}1,&r=0\\x^{r},&r\in[1\ldots n].\end{array}\right.`

.. _def-polynomial:

**Polynomial** A real polynomial is characterized by an integer degree :math:`n\in\{-1\}\cup{\mathbb{N}}.` By convention, the polynomial of degree :math:`n=\left(-1\right)` is the parameterless zero-valued polynomial :math:`p_{\emptyset}^{-1}(x)=0.` For :ref:`nonnegative<def-negative>` degrees, a real polynomial is characterized by a vector :math:`{\mathbf{a}}\in{\mathbb{R}}^{n+1}` of real polynomial coefficients. It is a function :math:`p_{{\mathbf{a}}}^{n}:{\mathbb{R}}\rightarrow{\mathbb{R}}` such that :math:`x\mapsto p_{{\mathbf{a}}}^{n}(x)={\mathbf{a}}^{{\mathsf{T}}}\,{\mathbf{v}}^{n}(x),` with :math:`{\mathbf{v}}^{n}` a :ref:`Vandermonde vector<def-vandermonde_vector>`. The polynomials of degree :math:`n` form a subset of the polynomials of degree :math:`n+1.`

.. _def-monomial:

**Monomial** A monomial is a :ref:`polynomial<def-polynomial>` of :ref:`nonnegative<def-negative>` degree such that every component of its defining vector of coefficients is zero, except the component of highest dimension. When this component takes value one, the monomial is said to be canonical.

.. _def-polynomial_simple_element:

**Polynomial Simple Element** A polynomial simple element of integer degree :math:`n\in{\mathbb{Z}}` is a real function :math:`{\mathbb{R}}\rightarrow{\mathbb{R}}` for :ref:`nonnegative<def-negative>` degrees and a distribution otherwise. It is notated :math:`\varsigma^{n}.` For :math:`n\in{\mathbb{N}},` it is one of the Green's functions of the differentiation operator of order :math:`n + 1.` It is chosen to be :ref:`even-symmetric<def-even_symmetry>` for :ref:`odd<def-odd>` degree, and :ref:`odd-symmetric<def-odd_symmetry>` for :ref:`even<def-even>` degree.

.. _def-split:

**Split of the Real Numbers** The increasing infinite list :math:`{\mathbb{S}}=[s_{k}\in{\mathbb{R}}|s_{k}<s_{k+1}]_{k\in{\mathbb{Z}}}` is called a split of the real numbers.

.. _def-uniform_split:

**Uniform Split** The :ref:`split<def-split>` :math:`{\mathbb{S}}` of the real numbers is said to be uniform when :math:`\forall k\in{\mathbb{Z}}:s_{k+1}=s_{k}+1.` An equivalent formulation is :math:`\forall k\in{\mathbb{Z}}:s_{k}=s_{0}+k,` which leads one to notate a uniform split as :math:`{\mathbb{S}}(s_{0}).`

.. _def-piecewise_polynomial:

**Piecewise Polynomial** A real function is said to be a piecewise polynomial when it admits a description by a :ref:`nonnegative<def-negative>` integer degree :math:`n\in{\mathbb{N}},` a :ref:`split<def-split>` :math:`{\mathbb{S}}` of the real numbers, an infinite list :math:`{\mathbb{F}}({\mathbb{S}})=[f_{k}\in{\mathbb{R}}]_{k\in{\mathbb{Z}}}` of values at the splitting points, and an infinite list :math:`{\mathbb{P}}({\mathbb{S}})=[p_{{\mathbf{a}}_{k}}^{n}:(s_{k},s_{k+1})\rightarrow{\mathbb{R}}]_{k\in{\mathbb{Z}}}` of :ref:`polynomials<def-polynomial>` whose open domains lie between the splitting points. Such a function is notated :math:`p_{{\mathbb{S}},{\mathbb{F}},{\mathbb{P}}}^{n}:{\mathbb{R}}\rightarrow{\mathbb{R}}` and is such that :math:`x\mapsto p_{{\mathbb{S}},{\mathbb{F}},{\mathbb{P}}}^{n}(x)=\left\{\begin{array}{ll}f_{k},&x=s_{k}\in{\mathbb{S}}\\p_{{\mathbf{a}}_{k}}^{n}(x),&{\mathbb{S}}\ni s_{k}<x<s_{k+1}\in{\mathbb{S}}.\end{array}\right.`

