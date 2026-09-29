Fourier Transform of Periodic Splines
=====================================

How to access the Fourier coefficients of a one-dimensional spline.

----

Preamble
--------

Functions
^^^^^^^^^

Let :math:`f:{\mathbb{X}}\rightarrow{\mathbb{Y}},x\mapsto f(x)` be some mapping from the *domain* :math:`{\mathbb{X}}` to the *codomain* :math:`{\mathbb{Y}}.` The domain is the set of all admissible elements being mapped, and the codomain is a superset of all the elements being mapped to. Typical relevant examples are the real functions :math:`f:{\mathbb{R}}\rightarrow{\mathbb{R}}` and the vector-to-vector mappings :math:`{\mathbf{f}}:{\mathbb{R}}^{K}\rightarrow{\mathbb{R}}^{K}`.

It is sometimes convenient to describe the mapping by a computational recipe. For instance, a quadratic monomial would be :math:`f:{\mathbb{R}}\rightarrow{\mathbb{R}},x\mapsto f(x)=x^{2}.` Observe that the codomain is announced here to be the whole of :math:`{\mathbb{R}},` while indeed only nonnegative values are mapped to. The actual set of elements being mapped to is called the *image* of the mapping; it is a subset of the codomain. For a quadratic monomial, this image would be :math:`{\mathbb{R}}_{\geq0}.`

Transforms
^^^^^^^^^^

With real functions, the domain and the codomain consist of real numbers. But nothing prevents us to let the elements of the domain and the codomain be themselves mappings. This would lead to

..  math::
    {\mathcal{T}}:\left({\mathbb{X}}_{1}\rightarrow{\mathbb{Y}}_{1}\right)\rightarrow\left({\mathbb{X}}_{2}\rightarrow{\mathbb{Y}}_{2}\right),

with domain :math:`{\mathbb{X}}=\left({\mathbb{X}}_{1}\rightarrow{\mathbb{Y}}_{1}\right)` and codomain :math:`{\mathbb{Y}}=\left({\mathbb{X}}_{2}\rightarrow{\mathbb{Y}}_{2}\right).` In such cases, :math:`{\mathcal{T}}` is called a *transform*. The cases of interest to us are those transforms that map complex functions :math:`\left({\mathbb{R}}\rightarrow{\mathbb{C}}\right)` to complex functions :math:`\left({\mathbb{R}}\rightarrow{\mathbb{C}}\right)`.

Four Brands of Fourier Transforms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Continuous-Time
"""""""""""""""
A transform that got a lot of traction is the *continuous-time Fourier transform* described by

..  math::
    {\mathcal{F}}:\left({\mathbb{R}}\rightarrow{\mathbb{C}}\right)\rightarrow\left({\mathbb{R}}\rightarrow{\mathbb{C}}\right),f\mapsto{\mathcal{F}}\{f\}=\int_{{\mathbb{R}}}\,f(x)\,{\mathrm{e}}^{-{\mathrm{j}}\,\left(\cdot\right)\,x}\,{\mathrm{d}}x.

It proves to be an invaluable tool to make theoretical forays into the analysis of signals and systems, into statistics, and into many other disciplines.

Discrete
""""""""

By nature, however, :math:`{\mathcal{F}}` is handling objects defined in the continuum, whereas experimental data are always discrete and finitely supported. Consequently, another tool has been designed to retain the flavor of the continuous-time Fourier transform, but in the finite-dimensional discrete world; it is called the *discrete Fourier transform*, even if this discrete version is not a transform but is in fact just a classic vector-to-vector function. Letting :math:`{\mathbf{x}}\in{\mathbb{R}}^{K}` be the vector of :math:`K` data samples, this vector-to-vector function is expressed as

..  math::
    {\mathbf{F}}:{\mathbb{C}}^{K}\rightarrow{\mathbb{C}}^{K},{\mathbf{x}}\mapsto{\mathbf{F}}({\mathbf{x}})=\left(\sum_{k=0}^{K-1}\,x[k]\,{\mathrm{e}}^{-{\mathrm{j}}\,\nu\,\frac{2\,\pi}{K}\,k}\right)_{\nu=0}^{K-1}

which, for suitable :math:`{\mathbf{W}}\in{\mathbb{C}}^{K\times K},` is nothing but the matrix multiplication :math:`{\mathbf{F}}({\mathbf{x}})={\mathbf{W}}\,{\mathbf{x}}` in disguise. A great part of the success of the discrete Fourier transform is due to the fact that :math:`{\mathbf{W}}` happens to be structured in such a way that the matrix-vector multiplication can be computed in :math:`{\mathcal{O}}(K\,\log K)` operations instead of the :math:`{\mathcal{O}}(K^{2})` operations that an unstructured matrix would otherwise require.

Series
""""""

It is also possible to mix the discrete and the continuous world. A third brand of Fourier transform is called a *Fourier series*. It maps a complex periodic function of positive integer period :math:`K\in{\mathbb{N}}+1` to a sequence of complex coefficients, according to

..  math::
    F:\left({\mathbb{R}}\rightarrow{\mathbb{C}}\right)\rightarrow\left({\mathbb{Z}}\rightarrow{\mathbb{C}}\right),f\mapsto F\{f\}=\left(F[\nu]\right)_{\nu\in{\mathbb{Z}}}=\left(\frac{1}{K}\,\int_{0}^{K}\,f(x)\,{\mathrm{e}}^{-{\mathrm{j}}\,\nu\,\frac{2\,\pi}{K}\,x}\,{\mathrm{d}}x\right)_{\nu\in{\mathbb{Z}}}.

Discrete-Time
"""""""""""""

Finally, a fourth brand of Fourier transform is the so-called *discrete-time Fourier transform*—not to be confused with the discrete Fourier transform. Letting :math:`T\in{\mathbb{R}}_{>0}` be some sampling step, the transform is defined as

..  math::
    {\mathcal{F}}_{1/T}:\left({\mathbb{R}}\rightarrow{\mathbb{C}}\right)\rightarrow\left({\mathbb{R}}\rightarrow{\mathbb{C}}\right),f\mapsto{\mathcal{F}}_{1/T}\{f\}=T\,\sum_{k\in{\mathbb{Z}}}\,f(k\,T)\,{\mathrm{e}}^{-{\mathrm{j}}\,\left(\cdot\right)\,T\,k}.

Inverses
^^^^^^^^

The first three versions of the Fourier transform (continuous-time, discrete, and series) are invertible, with respective inverses

..  math::
    {\mathcal{F}}^{-1}:\left({\mathbb{R}}\rightarrow{\mathbb{C}}\right)\rightarrow\left({\mathbb{R}}\rightarrow{\mathbb{C}}\right),\hat{f}\mapsto{\mathcal{F}}^{-1}\{\hat{f}\}=\frac{1}{2\,\pi}\,\int_{{\mathbb{R}}}\,\hat{f}(\omega)\,{\mathrm{e}}^{{\mathrm{j}}\,\omega\,\left(\cdot\right)}\,{\mathrm{d}}\omega

..  math::
    {\mathbf{F}}^{-1}:{\mathbb{C}}^{K}\rightarrow{\mathbb{C}}^{K},\hat{{\mathbf{x}}}\mapsto{\mathbf{F}}^{-1}(\hat{{\mathbf{x}}})=\frac{1}{K}\,\left(\sum_{\nu=0}^{K-1}\,\hat{x}[\nu]\,{\mathrm{e}}^{{\mathrm{j}}\,\nu\,\frac{2\,\pi}{K}\,k}\right)_{k=0}^{K-1}

..  math::
    F^{-1}:\left({\mathbb{Z}}\rightarrow{\mathbb{C}}\right)\rightarrow\left({\mathbb{R}}\rightarrow{\mathbb{C}}\right),\left(F[\nu]\right)_{\nu\in{\mathbb{Z}}}\mapsto\left(x\mapsto\sum_{\nu\in{\mathbb{Z}}}\,F[\nu]\,{\mathrm{e}}^{{\mathrm{j}}\,\nu\,\frac{2\,\pi}{K}\,x}\right).

Meanwhile, the inspection of the definition of the discrete-time Fourier transform reveals that only samples of the continuously defined function :math:`f` being transformed are taken into account. Consequently, in the absence of restrictions on :math:`f,` no inverse can be found that would recreate the whole of :math:`f:{\mathbb{R}}\rightarrow{\mathbb{C}}.` However, its samples can still be recovered from :math:`{\mathcal{F}}_{1/T}\{f\}` as

..  math::
    \left(f(k\,T)\right)_{k\in{\mathbb{Z}}}=\left(\frac{1}{2\,\pi}\,\int_{-\pi/T}^{\pi/T}\,{\mathcal{F}}_{1/T}\{f\}(\omega)\,{\mathrm{e}}^{{\mathrm{j}}\,\omega\,T\,k}\,{\mathrm{d}}\omega\right)_{k\in{\mathbb{Z}}}.

Convergence
^^^^^^^^^^^

The expressions of the direct or inverse transforms sometimes involve either infinite integrals or infinite sums, which call for mathematical care about the proper treatment of convergence. Even in the absence of convergence in the classic sense, there are advanced mathematical theories that typically allow one to handle cases where :math:`\int_{{\mathbb{R}}}\,\left|{\mathcal{F}}^{-1}\{{\mathcal{F}}\{f\}\}(x)-f(x)\right|^{2}\,{\mathrm{d}}x=0` even though it may happen that :math:`{\mathcal{F}}^{-1}\{{\mathcal{F}}\{f\}\}(x)\ne f(x)` for some :math:`x\in{\mathbb{R}}.`

----

Truncated Fourier Series
------------------------

The ``splinekit.PeriodicSpline1D`` class maintains periodic functions. Thus, the most appropriate Fourier tool is the Fourier series, and the library gives access to coefficients of any index :math:`\nu\in{\mathbb{Z}}.` These coefficients are ordered by level of detail, with coefficients of low absolute index providing coarse contributions and coefficients of high absolute index carrying the details of the function.

Instead of considering the infinite sequence of Fourier coefficients :math:`F` to get the full signal recovery :math:`x\mapsto\sum_{\nu\in{\mathbb{Z}}}\,F[\nu]\,{\mathrm{e}}^{{\mathrm{j}}\,\nu\,\frac{2\,\pi}{K}\,x},` we can approximate a periodic function as the partial reconstruction :math:`x\mapsto\sum_{\nu=-N}^{N}\,F[\nu]\,{\mathrm{e}}^{{\mathrm{j}}\,\nu\,\frac{2\,\pi}{K}\,x}` obtained over a finite sum of :math:`2\,N+1` terms, with :math:`N\in{\mathbb{N}}.` This reconstruction will capture the overall shape of a periodic function when :math:`N` is small, and additional details will emerge when :math:`N` increases. Moreover, when the fonction being transformed is real, it turns out that its Fourier-series coefficients, although complex, are organized in such a way that the proposed partial reconstruction, with symmetric upper and lower limits, is also real.

We give now a piece of code that illustrates how a truncated Fourier-series reconstruction approximates a random periodic spline of specified period, degree, and delay.

..  admonition:: Jupyter Lab notebook

    `Truncated Fourier series <https://splinekit.github.io/splinekit-jupyterlite/lab/?path=periodic-spline/fourier/spline_fourier.ipynb&mode=single-document>`_

----

Fourier Smoothness
------------------

It is easy to verify that the partial reconstruction obtained with :math:`N=0` is nothing but a constant-valued function that takes as value the average of the continuously defined periodic function. Then, for larger :math:`N,` the partial reconstruction :math:`\tilde{f}_{N-1}` with :math:`\left(N-1\right)` terms can be refined one term at a time to obtain :math:`\tilde{f}_{N}.` Because :math:`F[N]=\left(F[-N]\right)^{*}` for the real :math:`f` that we consider here, such a refinement can be expressed as

..  math::
    \tilde{f}_{N}(x)-\tilde{f}_{N-1}(x)=2\,\Re(F[N])\,\cos(N\,\frac{2\,\pi}{K}\,x)-2\,\Im(F[N])\,\sin(N\,\frac{2\,\pi}{K}\,x).

This refinement term is a continuously defined periodic function in :math:`x,` with rational period :math:`K/N` that dwindles as :math:`N` increases. Thus, the more Fourier terms are taken into account in the partial reconstruction, the more agitated the reconstruction becomes. The phase and the amplitude of the refining terms is governed jointly by the real and imaginary parts of the corresponding Fourier-series coefficient.

It follows that a function that is very agitated will have a Fourier-series representation with coefficients (at high absolute indices) that will tend to be large. A smoother function will have tamer coefficients at the same high absolute indices. For instance, a spline of degree :math:`0` is discontinuous and much rougher than a spline of higher degree.

We give now a piece of code that illustrates this behavior. We synthesize a set of random splines of a specified degree, period, and delay. The randomness is achieved by letting the spline coefficients be independent and identically distributed realizations of a random variable that follows a normal Gaussian probability density function. Then, we compute a few terms of the Fourier series of these random splines and establish per-Fourier-coefficient empirical statistics. We verify visually that, at high absolute Fourier-series indices, rough splines (low degree) need larger contributions than smooth splines (high degree).

..  admonition:: Jupyter Lab notebook

    `Fourier smoothness of periodic splines <https://splinekit.github.io/splinekit-jupyterlite/lab/?path=periodic-spline/fourier/spline_smoothness.ipynb&mode=single-document>`_

----

Fourier Filters
---------------

Assume now that it is a uniform periodic piecewise-polynomial spline

..  math::
    f:{\mathbb{R}}\rightarrow{\mathbb{R}},x\mapsto f(x)=\sum_{k\in{\mathbb{Z}}}\,c[{k\bmod K}]\,\beta^{n}(x-\delta x-k)

of nonnegative integer degree :math:`n\in{\mathbb{N}}` and delay :math:`\delta x\in{\mathbb{R}}` that plays the role of the periodic function of positive integer period :math:`K\in{\mathbb{N}}+1` from which a Fourier series is computed. (There, :math:`c` represents the spline coefficients.) Because a Fourier series is invertible, the infinite-length sequence :math:`\left(F\{f\}[\nu]\right)_{\nu\in{\mathbb{Z}}}=\left(F[\nu]\right)_{\nu\in{\mathbb{Z}}}` provides a faithful alternative representation of the spline.

Let :math:`\hat{{\mathbf{f}}}=\left(F[\nu]\right)_{\nu=0}^{\left\lfloor\frac{K}{2}\right\rfloor}\in{\mathbb{C}}^{\left\lfloor\frac{K}{2}\right\rfloor+1}` be a finite-dimensional vector given by a finite-length subsequence of the Fourier series of the spline. Now, it is possible to recover the spline exactly out of just :math:`\hat{{\mathbf{f}}},` under the knowledge that it is a spline of period :math:`K,` degree :math:`n,` and delay :math:`\delta x` that is being recovered. The vector :math:`\hat{{\mathbf{f}}}` is an admissible representation of the spline if :math:`\hat{f}[0]\in{\mathbb{R}}.` A second condition of admissibility arises when the period is even, in which case it must hold that :math:`\hat{f}[\frac{K}{2}]\,{\mathrm{e}}^{{\mathrm{j}}\,\pi\,\delta x}\in{\mathbb{R}},` too. The remaining components of :math:`\hat{{\mathbf{f}}}` are free of constraints.

In the piece of code below, we annihilate a range of components of the subsequence of the Fourier series of some random spline. We then synthesize a new spline from this manipulated subsequence. This operation is equivalent to the filtering of a spline. It differs from the truncated Fourier-series approach proposed in an earlier notebook because the truncated version does not return a spline while the filtering version does.

..  admonition:: Jupyter Lab notebook

    `Periodic Splines from Fourier Coefficients <https://splinekit.github.io/splinekit-jupyterlite/lab/?path=periodic-spline/fourier/spline_from_fourier.ipynb&mode=single-document>`_
