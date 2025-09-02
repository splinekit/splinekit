.. _glossary:

Glossary
========

.. _def-j:

**Imaginary Basis** The basis of the imaginary numbers is :math:`{\mathrm{j}},` with :math:`{\mathrm{j}}^{2}=-1.` This notation follows the engineering conventions.

.. _def-sgn:

**Sgn Function** Let :math:`{\mathrm{j}}` be the :ref:`basis of the imaginary numbers<def-j>`. Then, the signum of a number is defined as the complex function :math:`{\mathrm{sgn}}:{\mathbb{C}}\rightarrow{\mathbb{C}}`. It is such that :math:`z\mapsto{\mathrm{sgn}}\,z=\left\{\begin{array}{ll}0,&z=0\\{\mathrm{e}}^{{\mathrm{j}}\,\arg z},&z\neq0.\end{array}\right.`

.. _def-negative:

**Negative Numbers** The complex number :math:`z\in{\mathbb{C}}` is said to be negative when :math:`{\mathrm{sgn}}\,z=-1.` It is said to be nonnegative otherwise. When the domain of the :ref:`signum function<def-sgn>` is restricted to the real numbers, negative numbers are written :math:`\{x\in{\mathbb{R}}|x<0\}` in set notation, :math:`(-\infty,0)` in interval notation, and :math:`{\mathbb{R}}_{<0}` for short. When the domain is restricted to the integers, the negative integers are written :math:`{\mathbb{Z}}\setminus{\mathbb{N}}={\mathbb{Z}}_{<0}=\left(-{\mathbb{N}}\right)\setminus\{0\}` while the nonnegative integers are written :math:`{\mathbb{N}}={\mathbb{Z}}_{\geq0}.` The number zero is neither :ref:`negative<def-negative>` nor :ref:`positive<def-positive>`.

.. _def-positive:

**Positive Numbers** The complex number :math:`z\in{\mathbb{C}}` is said to be positive when :math:`{\mathrm{sgn}}\,z=1.` It is said to be nonpositive otherwise. When the domain of the :ref:`signum function<def-sgn>` is restricted to the real numbers, positive numbers are written :math:`\{x\in{\mathbb{R}}|x>0\}` in set notation, :math:`(0,\infty)` in interval notation, and :math:`{\mathbb{R}}_{>0}` for short. When the domain is restricted to the integers, the positive integers are written :math:`{\mathbb{N}}\setminus\{0\}=\left({\mathbb{N}}+1\right)={\mathbb{Z}}_{>0}` while the nonpositive integers are written :math:`\left(-{\mathbb{N}}\right)={\mathbb{Z}}_{\leq0}=\left({\mathbb{Z}}\setminus{\mathbb{N}}\right)\cup\{0\}={\mathbb{Z}}\setminus\left({\mathbb{N}}\setminus\{0\}\right).` The number zero is neither :ref:`negative<def-negative>` nor :ref:`positive<def-positive>`.

.. _def-even:

**Even Numbers** The even numbers are notated :math:`\left(2\,{\mathbb{Z}}\right).` The :ref:`nonnegative<def-negative>` even numbers are notated :math:`\left(2\,{\mathbb{N}}\right).` The :ref:`positive<def-positive>` even numbers are notated :math:`\left(2\,{\mathbb{N}}+2\right).`

.. _def-odd:

**Odd Numbers** The odd numbers are notated :math:`\left(2\,{\mathbb{Z}}+1\right).` The :ref:`positive<def-positive>` odd numbers are notated :math:`\left(2\,{\mathbb{N}}+1\right).`

.. _def-vandermonde_vector:

**Vandermonde Vector** The Vandermonde vector :math:`{\mathbf{v}}^{n}(x)\in{\mathbb{R}}^{n+1}` of argument :math:`x\in{\mathbb{R}}` and :ref:`nonnegative<def-negative>` integer degree :math:`n\in{\mathbb{N}}` is defined by its :math:`(r+1)`-th row component :math:`v_{r+1}^{n}(x)=v^{n}[r](x)=\left\{\begin{array}{ll}1,&r=0\\x^{r},&r\in[1\ldots n].\end{array}\right.`

.. _def-polynomial:

**Polynomial** A real polynomial is characterized by an integer degree :math:`n\in\{-1\}\cup{\mathbb{N}}.` By convention, the polynomial of degree :math:`n=\left(-1\right)` is the parameterless zero-valued polynomial :math:`p_{\emptyset}^{-1}(x)=0.` For :ref:`nonnegative<def-negative>` degrees, a real polynomial is characterized by a vector :math:`{\mathbf{a}}\in{\mathbb{R}}^{n+1}` of real polynomial coefficients. It is a function :math:`p_{{\mathbf{a}}}^{n}:{\mathbb{R}}\rightarrow{\mathbb{R}}` such that :math:`x\mapsto p_{{\mathbf{a}}}^{n}(x)={\mathbf{a}}^{{\mathsf{T}}}\,{\mathbf{v}}^{n}(x),` with :math:`{\mathbf{v}}^{n}` a :ref:`Vandermonde vector<def-vandermonde_vector>`. The polynomials of degree :math:`n` form a subset of the polynomials of degree :math:`n+1.`

.. _def-monomial:

**Monomial** A monomial is a :ref:`polynomial<def-polynomial>` of :ref:`nonnegative<def-negative>` degree such that every component of its defining vector of coefficients is zero, except the component of highest dimension which takes value one.

.. _def-polynomial_simple_element:

**Polynomial Simple Element** A polynomial simple element of integer degree :math:`n\in{\mathbb{Z}}` is a real function :math:`{\mathbb{R}}\rightarrow{\mathbb{R}}` for :ref:`nonnegative<def-negative>` degrees and a distribution otherwise. It is notated :math:`\varsigma^{n}.` For :math:`n\in{\mathbb{N}},` it is one of the Green's functions of the differentiation operator of order :math:`n + 1.` It is chosen to be :ref:`even-symmetric<def-even_symmetry>` for :ref:`odd<def-odd>` degree, and :ref:`odd-symmetric<def-odd_symmetry>` for :ref:`even<def-even>` degree.

.. _def-split:

**Split of the Real Numbers** The increasing infinite list :math:`{\mathbb{S}}=[s_{k}\in{\mathbb{R}}|s_{k}<s_{k+1}]_{k\in{\mathbb{Z}}}` is called a split of the real numbers.

.. _def-uniform_split:

**Uniform Split** The :ref:`split<def-split>` :math:`{\mathbb{S}}` of the real numbers is said to be uniform when :math:`\forall k\in{\mathbb{Z}}:s_{k+1}=s_{k}+1.` An equivalent formulation is :math:`\forall k\in{\mathbb{Z}}:s_{k}=s_{0}+k,` which leads one to notate a uniform split as :math:`{\mathbb{S}}(s_{0}).`

.. _def-piecewise_polynomial:

**Piecewise Polynomial** A real function is said to be a piecewise polynomial when it admits a description by a :ref:`nonnegative<def-negative>` integer degree :math:`n\in{\mathbb{N}},` a :ref:`split<def-split>` :math:`{\mathbb{S}}` of the real numbers, an infinite list :math:`{\mathbb{F}}({\mathbb{S}})=[f_{k}\in{\mathbb{R}}]_{k\in{\mathbb{Z}}}` of values at the splitting points, and an infinite list :math:`{\mathbb{P}}({\mathbb{S}})=[p_{{\mathbf{a}}_{k}}^{n}:(s_{k},s_{k+1})\rightarrow{\mathbb{R}}]_{k\in{\mathbb{Z}}}` of :ref:`polynomials<def-polynomial>` whose open domains lie between the splitting points. Such a function is notated :math:`p_{{\mathbb{S}},{\mathbb{F}},{\mathbb{P}}}^{n}:{\mathbb{R}}\rightarrow{\mathbb{R}}` and is such that :math:`x\mapsto p_{{\mathbb{S}},{\mathbb{F}},{\mathbb{P}}}^{n}(x)=\left\{\begin{array}{ll}f_{k},&x=s_{k}\in{\mathbb{S}}\\p_{{\mathbf{a}}_{k}}^{n}(x),&{\mathbb{S}}\ni s_{k}<x<s_{k+1}\in{\mathbb{S}}.\end{array}\right.`

.. _def-spline:

**Spline** For the degree :math:`n=0,` the :ref:`piecewise polynomial<def-piecewise_polynomial>` :math:`p_{{\mathbb{S}},{\mathbb{F}},{\mathbb{P}}}^{0}` is called a piecewise-constant spline when :math:`{\mathbb{F}}({\mathbb{S}})` is such that :math:`\forall k\in{\mathbb{Z}}:f_{k}=\frac{p_{{\mathbf{a}}_{k-1}^{0}}((s_{k-1}+s_{k})/2)+p_{{\mathbf{a}}_{k}^{0}}((s_{k}+s_{k+1})/2))}{2}.` For a :ref:`positive<def-positive>` integer degree :math:`n,` the :ref:`piecewise polynomial<def-piecewise_polynomial>` :math:`p_{{\mathbb{S}},{\mathbb{F}},{\mathbb{P}}}^{n}` is called a spline when it happens to be of differentiability class :math:`{\mathcal{C}}^{n-1}` over :math:`{\mathbb{R}}.`

.. _def-uniform_spline:

**Uniform Spline** A :ref:`spline<def-spline>` is said to be uniform when it admits the :ref:`uniform split<def-uniform_split>` :math:`{\mathbb{S}}(s_{0})` as its :ref:`split of the real numbers<def-split>`.

.. _def-even_symmetry:

**Even Symmetry** A real function :math:`f:{\mathbb{R}}\rightarrow{\mathbb{R}}` is said to be pointwise even-symmetric when :math:`\forall x\in{\mathbb{R}}:f(x)=f(-x).`

.. _def-odd_symmetry:

**Odd Symmetry** A real function :math:`f:{\mathbb{R}}\rightarrow{\mathbb{R}}` is said to be pointwise odd-symmetric when :math:`\forall x\in{\mathbb{R}}:f(x)=-f(-x).`

.. _def-partition_of_unity:

**Partition of Unity** A real function :math:`f:{\mathbb{R}}\rightarrow{\mathbb{R}}` is said to satisfy the pointwise partition-of-unity condition when it is such that :math:`\forall x\in{\mathbb{R}}:\sum_{k\in{\mathbb{Z}}}\,f(x-k)=1.`

.. _def-support:

**Support** The support of a real function is the set of points of its domain where the function does not vanish.

.. _def-b_spline:

**Polynomial B-Spline** A polynomial B-spline of :ref:`nonnegative<def-negative>` integer degree is uniquely defined as a :ref:`uniform spline<def-uniform_spline>` of minimal :ref:`support<def-support>` that is :ref:`pointwise even-symmetric<def-even_symmetry>` and that satisfies the :ref:`pointwise partition of unity<def-partition_of_unity>`. It is notated :math:`\beta^{n}:{\mathbb{R}}\rightarrow{\mathbb{R}},` where :math:`n` is the degree.

.. _def-interpolating_function:

**Interpolating Function** A real function :math:`f:{\mathbb{R}}\rightarrow{\mathbb{R}}` is said to be interpolating when its restriction to the integers is such that :math:`\forall k\in{\mathbb{Z}}:f(k)={\mathbf{[\![}}k=0\,{\mathbf{]\!]}},` where the notation :math:`{\mathbf{[\![}}P\,{\mathbf{]\!]}}` is that of the :ref:`Iverson bracket<def-iverson>`.

.. _def-cardinal_b_spline:

**Cardinal Polynomial B-Spline** A cardinal polynomial B-spline of :ref:`nonnegative<def-negative>` integer degree is uniquely defined as a :ref:`uniform spline<def-uniform_spline>` that is :ref:`interpolating<def-interpolating_function>`, :ref:`pointwise even-symmetric<def-even_symmetry>`, and that satisfies the :ref:`pointwise partition of unity<def-partition_of_unity>`. It is notated :math:`\eta^{n}:{\mathbb{R}}\rightarrow{\mathbb{R}},` where :math:`n` is the degree.

.. _def-integer-shift_orthogonal:

**Integer-Shift Orthogonality** The two functions :math:`f:{\mathbb{R}}\rightarrow{\mathbb{C}}` and  :math:`g:{\mathbb{R}}\rightarrow{\mathbb{C}}` are said to be mutually integer-shift-orthogonal in terms of the integer shift :math:`k` if :math:`\forall k\in{\mathbb{Z}}\setminus\{0\}:0=\int_{-\infty}^{\infty}\,f^{*}(x)\,g(x+k)\,{\mathrm{d}}x.` As a special case, the zero function :math:`x\mapsto f(x)=0` is integer-shift-orthogonal to every function :math:`g.`

.. _def-integer-shift_orthonormal:

**Integer-Shift Orthonormality** The two functions :math:`f:{\mathbb{R}}\rightarrow{\mathbb{C}}` and  :math:`g:{\mathbb{R}}\rightarrow{\mathbb{C}}` are said to be mutually integer-shift-orthonormal if they are :ref:`integer-shift-orthogonal<def-integer-shift_orthogonal>` and if :math:`1=\int_{-\infty}^{\infty}\,f^{*}(x)\,g(x)\,{\mathrm{d}}x`.

.. _def-dual_b_spline:

**Polynomial Dual B-Spline** A polynomial dual B-spline is notated :math:`\mathring{\beta}^{m,n}.` It is a real function indexed by a :ref:`nonnegative<def-negative>` integer dual degree :math:`m` and a :ref:`nonnegative<def-negative>` integer primal degree :math:`n.` It is uniquely defined as the spline of dual degree :math:`m` that is :ref:`integer-shift-orthonormal<def-integer-shift_orthonormal>` to a :ref:`polynomial B-spline<def-b_spline>` of primal degree :math:`n,` so that :math:`{\mathbf{[\![}}k=0\,{\mathbf{]\!]}}=\int_{-\infty}^{\infty}\,\mathring{\beta}^{m,n}(x)\,\beta^{n}(x+k)\,{\mathrm{d}}x,` where the notation :math:`{\mathbf{[\![}}P\,{\mathbf{]\!]}}` is that of the :ref:`Iverson bracket<def-iverson>`.

.. _def-orthonormal_b_spline:

**Polynomial Orthonormal B-Spline** A polynomial orthonormal B-spline is notated :math:`\phi^{n}.` It is a real function indexed by a :ref:`nonnegative<def-negative>` integer degree :math:`n.` It is uniquely defined as the :ref:`uniform spline<def-uniform_spline>` of degree :math:`n` that is :ref:`integer-shift-orthonormal<def-integer-shift_orthonormal>` to itself, so that :math:`{\mathbf{[\![}}k=0\,{\mathbf{]\!]}}=\int_{-\infty}^{\infty}\,\phi^{n}(x)\,\phi^{n}(x+k)\,{\mathrm{d}}x,` along with :math:`\phi^{n}(x)=\sum_{k\in{\mathbb{Z}}}\,p[k]\,\beta^{n}(x-k)` for some well-chosen sequence :math:`p,` where the notation :math:`{\mathbf{[\![}}P\,{\mathbf{]\!]}}` is that of the :ref:`Iverson bracket<def-iverson>`.

.. _def-knots:

**Knots** The knots of a :ref:`polynomial B-spline<def-b_spline>` :math:`\beta^{n}` are those arguments :math:`x` at which :math:`\frac{{\mathrm{d}}^{n+1}\beta^{n}(x)}{{\mathrm{d}}x^{n+1}}\not\in
{\mathbb{R}}.` They are a finite subset of the :ref:`uniform split<def-uniform_split>` :math:`{\mathbb{S}}(\frac{n+1}{2})` associated to the B-spline.

.. _def-m_scale_relation:

**M-Scale Relation** The M-scale relation expresses a :ref:`B-spline<def-b_spline>` of :ref:`nonnegative<def-negative>` integer degree as a sum of translated and rescaled (minified) B-splines of same degree.

.. _def-iverson:

**Iverson Bracket** The Iverson bracket :math:`{\mathbf{[\![}}P\,{\mathbf{]\!]}}` of the statement :math:`P` is the indicator function of the set of values for which the statement is true.