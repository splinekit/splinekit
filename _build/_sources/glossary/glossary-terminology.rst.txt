.. _glossary-terminology:

Glossary—Terminology
====================

.. _def-even_symmetry:

**Even Symmetry** A real function :math:`f:{\mathbb{R}}\rightarrow{\mathbb{R}}` is said to be pointwise even-symmetric when :math:`\forall x\in{\mathbb{R}}:f(x)=f(-x).`

.. _def-odd_symmetry:

**Odd Symmetry** A real function :math:`f:{\mathbb{R}}\rightarrow{\mathbb{R}}` is said to be pointwise odd-symmetric when :math:`\forall x\in{\mathbb{R}}:f(x)=-f(-x).`

.. _def-support:

**Support** The support of a real function is the set of points of its domain where the function does not vanish.

.. _def-partition_of_unity:

**Partition of Unity** A real function :math:`f:{\mathbb{R}}\rightarrow{\mathbb{R}}` is said to satisfy the pointwise partition-of-unity condition when it is such that :math:`\forall x\in{\mathbb{R}}:\sum_{k\in{\mathbb{Z}}}\,f(x-k)=1.`

.. _def-interpolating_function:

**Interpolating Function** A real function :math:`f:{\mathbb{R}}\rightarrow{\mathbb{R}}` is said to be interpolating when its restriction to the integers is such that :math:`\forall k\in{\mathbb{Z}}:f(k)={\mathbf{[\![}}k=0\,{\mathbf{]\!]}},` where the notation :math:`{\mathbf{[\![}}P\,{\mathbf{]\!]}}` is that of the :ref:`Iverson bracket<def-iverson>`.

.. _def-integer-shift_orthogonal:

**Integer-Shift Orthogonality** The two functions :math:`f:{\mathbb{R}}\rightarrow{\mathbb{C}}` and  :math:`g:{\mathbb{R}}\rightarrow{\mathbb{C}}` are said to be mutually integer-shift-orthogonal in terms of the integer shift :math:`k` if :math:`\forall k\in{\mathbb{Z}}\setminus\{0\}:0=\int_{-\infty}^{\infty}\,f^{*}(x)\,g(x+k)\,{\mathrm{d}}x.` As a special case, the zero function :math:`x\mapsto f(x)=0` is integer-shift-orthogonal to every function :math:`g.`

.. _def-integer-shift_orthonormal:

**Integer-Shift Orthonormality** The two functions :math:`f:{\mathbb{R}}\rightarrow{\mathbb{C}}` and  :math:`g:{\mathbb{R}}\rightarrow{\mathbb{C}}` are said to be mutually integer-shift-orthonormal if they are :ref:`integer-shift-orthogonal<def-integer-shift_orthogonal>` and if :math:`1=\int_{-\infty}^{\infty}\,f^{*}(x)\,g(x)\,{\mathrm{d}}x`.

.. _def-iverson:

**Iverson Bracket** The Iverson bracket :math:`{\mathbf{[\![}}P\,{\mathbf{]\!]}}` of the statement :math:`P` is the indicator function of the set of values for which the statement is true.

