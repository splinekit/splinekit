:orphan:

..  role:: raw-html(raw)
    :format: html

Multiresolution
===============

How to manipulate the period of splines.

----

Upscaling
---------

We consider two approaches to upscale a spline by an arbitrary positive integer magnification :math:`M\in{\mathbb{N}}+1.` The first one builds an upscaled spline that is an exact replica of the spline at the nominal scale, except for being larger. The second one offers freedom in the choice of the degree and delay of the upscaled spline.

Exact
^^^^^

A polynomial B-spline :math:`\beta^{n_{0}}` of any nonnegative integer degree :math:`n_{0}\in{\mathbb{N}}` satisfies the M-scale equality, according to which its version at nominal scale can be expressed as a finite weighted sum of B-splines that are appropriately shifted and shrunk by the factor :math:`M.` (Equivalently, a B-spline enlarged by :math:`M` can be expressed through B-splines at their nominal scale.) The collection of weights depend on :math:`n_{0}` and :math:`M` and is called the M-scale filter :math:`h_{M}^{n_{0}}.`

Because a spline is itself a weighted sum of shifted B-splines, it benefits from the M-scale equality. Indeed, let a spline of positive integer period :math:`K_{0}\in{\mathbb{N}}+1` and delay :math:`\delta x_{0}\in{\mathbb{R}}` be defined at nominal scale as

..  math::
    f_{0}:{\mathbb{R}}\rightarrow{\mathbb{R}},x\mapsto f_{0}(x)=\sum_{k\in{\mathbb{Z}}}\,c_{0}[{k\bmod K_{0}}]\,\beta^{n_{0}}(x-\delta x_{0}-k),

and let its :math:`M`-enlarged version (with the same, non-enlarged delay) be

..  math::
    f_{{\color{blue}{\uparrow M}}}:{\mathbb{R}}\rightarrow{\mathbb{R}},x\mapsto f_{{\color{blue}{\uparrow M}}}(x)=f_{0}(\frac{x}{{\color{blue}{M}}})=\sum_{k\in{\mathbb{Z}}}\,c_{0}[{k\bmod K_{0}}]\,\beta^{n_{0}}(\frac{x}{{\color{blue}{M}}}-\delta x_{0}-k).

There, the B-splines that take part in the sum are not at their nominal scale. However, for all :math:`x\in{\mathbb{R}},` the M-scale equality implies that

..  math::
    f_{\uparrow M}(x)=\sum_{k\in{\mathbb{Z}}}\,c_{\uparrow M}^{n_{0}}[{k\bmod K}]\,\beta^{n_{0}}(x-\delta x_{\uparrow M}^{n_{0}}-k)

also holds true, where it is more immediately apparent that :math:`f_{\uparrow M}` is not only :math:`K`-periodic with :math:`K=M\,K_{0},` but also a weighted sum of integer-shifted B-splines at nominal scale, with the vector of weights being :math:`{\mathbf{c}}_{\uparrow M}^{n_{0}}=\left(c_{\uparrow M}^{n_{0}}[k]\right)_{k=0}^{K-1}` and the delay being :math:`\delta x_{\uparrow M}^{n_{0}}.` More precisely, the equality between :math:`f_{\uparrow M}` and the enlarged :math:`f_{0}` is achieved for

..  math::
    {\mathbf{c}}_{\uparrow M}^{n_{0}}=\left(\frac{1}{M^{n_{0}}}\,\sum_{q=\left\lceil\frac{k-\left(M-1\right)\,\left(n+1\right)}{M}\right\rceil}^{\left\lfloor\frac{k}{M}\right\rfloor}\,c_{0}[{q\bmod K_{0}}]\,h_{M}^{n_{0}}[k-M\,q]\right)_{k=0}^{K-1}

and

..  math::
    \delta x_{\uparrow M}^{n_{0}}=M\,\delta x_{0}-\frac{\left(M-1\right)\,\left(n_{0}+1\right)}{2}.

We now propose a few lines of code that first create a random spline of specified period, degree, and delay, and then enlarge it by a factor :math:`M.` We display a stack where the top figure contains the spline at its nominal size and the bottom figure contains the enlarged spline. A pair of synchronized sliders allows one to explore the values taken by the two functions and to conclude that, up to change of scale, the two versions are identical.

..  admonition:: Jupyter Lab notebook

    `Upscaling of a spline <https://splinekit.github.io/splinekit-jupyterlite/lab/?path=periodic-spline/multiresolution/spline_upscaling.ipynb&mode=single-document>`_

----

Projected
^^^^^^^^^

We want now to determine which spline :math:`\tilde{f}_{\uparrow M}` of arbitrary degree :math:`n\in{\mathbb{N}}` and arbitrary delay :math:`\delta x\in{\mathbb{R}}` best represents the magnified spline :math:`f_{\uparrow M}` with, as before, a positive integer magnification :math:`M\in{\mathbb{N}}+1` and

..  math::
    f_{\uparrow M}:{\mathbb{R}}\rightarrow{\mathbb{R}},x\mapsto f_{\uparrow M}(x)=f_{0}(\frac{x}{M})=\sum_{k\in{\mathbb{Z}}}\,c_{0}[{k\bmod K_{0}}]\,\beta^{n_{0}}(\frac{x}{M}-\delta x_{0}-k),

where the spline of positive integer period :math:`K_{0}\in{\mathbb{N}}+1` at nominal scale is

..  math::
    f_{0}:{\mathbb{R}}\rightarrow{\mathbb{R}},x\mapsto f_{0}(x)=\sum_{k\in{\mathbb{Z}}}\,c_{0}[{k\bmod K_{0}}]\,\beta^{n_{0}}(x-\delta x_{0}-k).

More precisely, for :math:`K=M\,K_{0},` we want to establish the value of the spline coefficients :math:`\tilde{c}_{\uparrow M}^{n}` that parameterize the :math:`K`-periodic spline

..  math::
    \tilde{f}_{\uparrow M}:{\mathbb{R}}\rightarrow{\mathbb{R}},x\mapsto \tilde{f}_{\uparrow M}(x)=\sum_{k\in{\mathbb{Z}}}\,\tilde{c}_{\uparrow M}^{n}[{k\bmod K}]\,\beta^{n}(x-\delta x-k)

and are such that the least-squares criterion

..  math::
    J=\frac{1}{2}\,\int_{0}^{K}\,\left(\tilde{f}_{\uparrow M}(x)-f_{\uparrow M}(x)\right)^{2}\,{\mathrm{d}}x

is minimized. While the desired spline :math:`\tilde{f}_{\uparrow M}` could be directly obtained as ``f0.upscaled(m).projected(degree = n, delay = dx)``, we propose here a combined approach. Letting :math:`{\mathbf{[\![}}\cdot\,{\mathbf{]\!]}}` be the notation for the Iverson bracket, we observe that, :math:`\forall q\in[0\ldots K],`

..  math::
    \frac{\partial J}{\partial \tilde{c}_{\uparrow M}^{n}[q]}=\int_{0}^{K}\,\sum_{k\in{\mathbb{Z}}}\,{\mathbf{[\![}}q={k\bmod K}\,{\mathbf{]\!]}}\,\beta^{n}(x-\delta x-k)\,\left(\tilde{f}_{\uparrow M}(x)-f_{\uparrow M}(x)\right)\,{\mathrm{d}}x.

When the coefficients are optimal, :math:`\frac{\partial J}{\partial \tilde{c}_{\uparrow M}^{n}[q]}` vanishes. Now, the multiplication of this zero value by the quantity :math:`\tilde{c}_{\uparrow M}^{n}[q]` is still zero, and so is the sum over all indices :math:`q.` This leads to

..  math::
    \begin{array}{rcl}
    0&=&\sum_{q=0}^{K-1}\,\tilde{c}_{\uparrow M}^{n}[q]\,\frac{\partial J}{\partial \tilde{c}_{\uparrow M}^{n}[q]}\\
    &=&\int_{0}^{K}\,\tilde{f}_{\uparrow M}(x)\,\left(\tilde{f}_{\uparrow M}(x)-f_{\uparrow M}(x)\right)\,{\mathrm{d}}x\\
    &=&\left(\tilde{f}_{\uparrow M}^{\vee}*\tilde{f}_{\uparrow M}\right)(0)-\left(\tilde{f}_{\uparrow M}^{\vee}*f_{\uparrow M}\right)(0),
    \end{array}

where the last equality involves periodic convolutions and mirrored versions :math:`\tilde{f}_{\uparrow M}^{\vee}` of :math:`\tilde{f}_{\uparrow M}.` The solution of this equation in terms of the vector :math:`\tilde{{\mathbf{c}}}_{\uparrow M}^{n}=\left(\tilde{c}_{\uparrow M}^{n}[q]\right)_{q=0}^{K-1}` is obtained in the three successive steps

..  math::
    \left(\tilde{{\mathbf{c}}}_{\uparrow M}^{n}\right)'=\left(\frac{1}{M^{n}}\,\sum_{q=\left\lceil\frac{k-\left(M-1\right)\,\left(n+1\right)}{M}\right\rceil}^{\left\lfloor\frac{k}{M}\right\rfloor}\,c_{0}[{q\bmod K_{0}}]\,h_{M}^{n}[k-M\,q]\right)_{k=0}^{K-1}

..  math::
    \left(\tilde{{\mathbf{c}}}_{\uparrow M}^{n}\right)''=\left(\left(\left(b^{2\,n+1}\right)^{-1}*\left(\tilde{c}_{\uparrow M}^{n}\right)'\right)[k]\right)_{k=0}^{K-1}

..  math::
    \tilde{{\mathbf{c}}}_{\uparrow M}^{n}=\left(\sum_{q=\left\lceil-x_{0}-\frac{n_{0}+n+2}{2}\right\rceil}^{\left\lfloor-x_{0}+\frac{n_{0}+n+2}{2}\right\rfloor}\,\beta^{n_{0}+n+1}(q+x_{0})\,\left(\tilde{c}_{\uparrow M}^{n}\right)''[{\left(k-q\right)\bmod K}]\right)_{k=0}^{K-1},

where :math:`x_{0}=\left(\delta x-M\,\delta x_{0}+\frac{\left(M-1\right)\,\left(n_{0}+1\right)}{2}\right)` and where :math:`\left(b^{2n+1}\right)^{-1}` represents a B-spline inverse sequence.

We now propose a few lines of code that first create a random spline :math:`f_{0}` of specified period :math:`K_{0},` degree :math:`n_{0},` and delay :math:`\delta x_{0},` and then display its :math:`M`-magnified version :math:`f_{\uparrow M}.` The spline :math:`\tilde{f}_{\uparrow M}` of arbitrary degree :math:`n` and arbitrary delay :math:`\delta x` that best represents :math:`f_{\uparrow M}` is then determined and displayed. We validate optimality by verifying that a quantity that vanishes in theory does so numerically, too, first through the explicit numerical estimate of an integral, then with the help of convolutions.

..  admonition:: Jupyter Lab notebook

    `Upscaling and projection of a spline <https://splinekit.github.io/splinekit-jupyterlite/lab/?path=periodic-spline/multiresolution/spline_up_proj.ipynb&mode=single-document>`_

----

Downscaling
-----------

Let :math:`m\in{\mathbb{N}}+1` be a positive integer minification factor. The goal now is to create an :math:`m`-coarser version :math:`f_{K_{0}\downarrow m}` of the :math:`K_{0}`-periodic spline :math:`f_{0}` at nominal scale defined, as before, as

..  math::
    f_{0}:{\mathbb{R}}\rightarrow{\mathbb{R}},x\mapsto f_{0}(x)=\sum_{k\in{\mathbb{Z}}}\,c_{0}[{k\bmod K_{0}}]\,\beta^{n_{0}}(x-\delta x_{0}-k).

The coarser version :math:`f_{K_{0}\downarrow m}` is assumed to be a periodic spline of arbitrary nonnegative integer degree :math:`n\in{\mathbb{N}},` arbitrary delay :math:`\delta x\in{\mathbb{R}},` and positive integer period :math:`K=\frac{K_{0}}{\gcd(K_{0},m)}\in{\mathbb{N}}+1.` This period is shorter than that of :math:`f_{0}` when :math:`m` and :math:`K_{0}` have multiplicative factors in common; otherwise, :math:`f_{K_{0}\downarrow m}` is still coarser than the spline :math:`f_{0}` in terms of data details but nevertheless retains its periodicity.

Here is a table that gives :math:`K` in terms of :math:`K_{0}` and :math:`m` for a few examples. The blue entries highlight those cases where the period at nominal scale can be entirely divided by the minification factor. In all other cases, just sufficiently many periods of the :math:`K_{0}`-periodic :math:`f_{0}` are concatenated to make an extended spline whose overall period is entirely divisible both by :math:`K_{0}` and by :math:`m.`

    :raw-html:`<TABLE border="1" frame="hsides" rules="groups" align="center">
    <CAPTION><i>K</i></CAPTION>
    <COLGROUP span="2">
    <TR align="right"><TH><TH><i>m</i>&#160;<TH>&#160;1<TH>&#160;2<TH>&#160;3<TH>&#160;4<TH>&#160;5<TH>&#160;6
    <TBODY>
    <TR align="right"><TD><i>K</i><sub>0</sub><TD>1&#160;<TD>&#160;<FONT color="#0343df"><B>1</B><TD>&#160;1<TD>&#160;1<TD>&#160;1<TD>&#160;1<TD>&#160;1
    <TR align="right"><TD><TD>2&#160;<TD>&#160;<FONT color="#0343df"><B>2</B><TD>&#160;<FONT color="#0343df"><B>1</B><TD>&#160;2<TD>&#160;1<TD>&#160;2<TD>&#160;1
    <TR align="right"><TD><TD>3&#160;<TD>&#160;<FONT color="#0343df"><B>3</B><TD>&#160;3<TD>&#160;<FONT color="#0343df"><B>1</B><TD>&#160;3<TD>&#160;3<TD>&#160;1
    <TR align="right"><TD><TD>4&#160;<TD>&#160;<FONT color="#0343df"><B>4</B><TD>&#160;<FONT color="#0343df"><B>2</B><TD>&#160;4<TD>&#160;<FONT color="#0343df"><B>1</B><TD>&#160;4<TD>&#160;2
    <TR align="right"><TD><TD>5&#160;<TD>&#160;<FONT color="#0343df"><B>5</B><TD>&#160;5<TD>&#160;5<TD>&#160;5<TD>&#160;<FONT color="#0343df"><B>1</B><TD>&#160;5
    <TR align="right"><TD><TD>6&#160;<TD>&#160;<FONT color="#0343df"><B>6</B><TD>&#160;<FONT color="#0343df"><B>3</B><TD>&#160;<FONT color="#0343df"><B>2</B><TD>&#160;3<TD>&#160;6<TD>&#160;<FONT color="#0343df"><B>1</B>
    <TR align="right"><TD><TD>7&#160;<TD>&#160;<FONT color="#0343df"><B>7</B><TD>&#160;7<TD>&#160;7<TD>&#160;7<TD>&#160;7<TD>&#160;7
    <TR align="right"><TD><TD>8&#160;<TD>&#160;<FONT color="#0343df"><B>8</B><TD>&#160;<FONT color="#0343df"><B>4</B><TD>&#160;8<TD>&#160;<FONT color="#0343df"><B>2</B><TD>&#160;8<TD>&#160;4
    <TR align="right"><TD><TD>9&#160;<TD>&#160;<FONT color="#0343df"><B>9</B><TD>&#160;9<TD>&#160;<FONT color="#0343df"><B>3</B><TD>&#160;9<TD>&#160;9<TD>&#160;3
    <TR align="right"><TD><TD>10&#160;<TD>&#160;<FONT color="#0343df"><B>10</B><TD>&#160;<FONT color="#0343df"><B>5</B><TD>&#160;10<TD>&#160;5<TD>&#160;<FONT color="#0343df"><B>2</B><TD>&#160;5
    <TR align="right"><TD><TD>11&#160;<TD>&#160;<FONT color="#0343df"><B>11</B><TD>&#160;11<TD>&#160;11<TD>&#160;11<TD>&#160;11<TD>&#160;11
    <TR align="right"><TD><TD>12&#160;<TD>&#160;<FONT color="#0343df"><B>12</B><TD>&#160;<FONT color="#0343df"><B>6</B><TD>&#160;<FONT color="#0343df"><B>4</B><TD>&#160;<FONT color="#0343df"><B>3</B><TD>&#160;12<TD>&#160;<FONT color="#0343df"><B>2</B>
    <TR align="right"><TD><TD>13&#160;<TD>&#160;<FONT color="#0343df"><B>13</B><TD>&#160;13<TD>&#160;13<TD>&#160;13<TD>&#160;13<TD>&#160;13
    <TR align="right"><TD><TD>14&#160;<TD>&#160;<FONT color="#0343df"><B>14</B><TD>&#160;<FONT color="#0343df"><B>7</B><TD>&#160;14<TD>&#160;7<TD>&#160;14<TD>&#160;7
    <TR align="right"><TD><TD>15&#160;<TD>&#160;<FONT color="#0343df"><B>15</B><TD>&#160;15<TD>&#160;<FONT color="#0343df"><B>5</B><TD>&#160;15<TD>&#160;<FONT color="#0343df"><B>3</B><TD>&#160;5
    </TABLE>`

From now on, we assume for simplicity the generic case where :math:`K_{0}=m\,K` and let the desired minified spline be

..  math::
    f_{K_{0}\downarrow m}:{\mathbb{R}}\rightarrow{\mathbb{R}},x\mapsto f_{K_{0}\downarrow m}(x)=\sum_{k\in{\mathbb{Z}}}\,c_{K_{0}\downarrow m}^{n}[{k\bmod K}]\,\beta^{n}(x-\delta x-k).

Our goal is to determine the spline coefficients :math:`c_{K_{0}\downarrow m}^{n}` such that the least-squares criterion

..  math::
    J=\frac{1}{2}\,\int_{0}^{K_{0}}\,\left(f_{K_{0}\downarrow m}(\frac{x}{m})-f_{0}(x)\right)^{2}\,{\mathrm{d}}x

is minimized. For reasons that are similar to those developed in the projected-upscaling case, it turns out that

..  math::
    \begin{array}{rcl}
    0&=&\sum_{q=0}^{K-1}\,c_{K_{0}\downarrow m}^{n}[q]\,\frac{\partial J}{\partial c_{K_{0}\downarrow m}^{n}[q]}\\
    &=&\int_{0}^{K_{0}}\,f_{\left(K_{0}\downarrow m\right)\uparrow m}(x)\,\left(f_{\left(K_{0}\downarrow m\right)\uparrow m}(x)-f_{0}(x)\right)\,{\mathrm{d}}x\\
    &=&\left(f_{\left(K_{0}\downarrow m\right)\uparrow m}^{\vee}*f_{\left(K_{0}\downarrow m\right)\uparrow m}\right)(0)-\left(f_{\left(K_{0}\downarrow m\right)\uparrow m}^{\vee}*f_{0}\right)(0),
    \end{array}

where :math:`f_{\left(K_{0}\downarrow m\right)\uparrow m}` is the exact :math:`m`-upscaled version of :math:`f_{K_{0}\downarrow m},` with :math:`f_{\left(K_{0}\downarrow m\right)\uparrow m}(x)=f_{K_{0}\downarrow m}(\frac{x}{m})` for all :math:`x\in{\mathbb{R}}.` The solution of this equation in terms of the vector :math:`{\mathbf{c}}_{K_{0}\downarrow m}^{n}=\left(c_{K_{0}\downarrow m}^{n}[q]\right)_{q=0}^{K-1}` is obtained in the three successive steps

..  math::
    \left({\mathbf{c}}_{0}\right)'=\left(\frac{1}{m^{n+1}}\,\sum_{q=0}^{\left(m-1\right)\,\left(n+1\right)}\,h_{m}^{n}[q]\,c_{0}[{\left(k-q\right)\bmod K_{0}}]\right)_{k=0}^{K_{0}-1}

..  math::
    \left({\mathbf{c}}_{K_{0}\downarrow m}^{n}\right)'=\left(\sum_{q=0}^{n_{0}+n+1}\,\beta^{n_{0}+n+1}(q+k_{0}-x_{0})\,\left(c_{0}\right)'[{\left(m\,k-k_{0}-q\right)\bmod K_{0}}]\right)_{k=0}^{K-1}

..  math::
    {\mathbf{c}}_{K_{0}\downarrow m}^{n}=\left(\left(\left(b^{2\,n+1}\right)^{-1}*\left({\mathbf{c}}_{K_{0}\downarrow m}^{n}\right)'\right)[k]\right)_{k=0}^{K-1},

where :math:`x_{0}=\left(\delta x_{0}-m\,\delta x-\frac{\left(m-1\right)\,\left(n+1\right)}{2}\right)` and :math:`k_{0}=\left\lfloor x_{0}-\frac{n_{0}+n}{2}\right\rfloor.`

We now propose a few lines of code that first create a random spline :math:`f_{0}` of fixed period :math:`K_{0}` and specified degree :math:`n_{0}` and delay :math:`\delta x_{0},` and then determine and display its optimal :math:`m`-minified version :math:`f_{K_{0}\downarrow m}` of arbitrary degree :math:`n` and arbitrary delay :math:`\delta x.` We validate optimality by verifying that a quantity that vanishes in theory does so numerically, too, first through the explicit numerical estimate of an integral, then with the help of convolutions.

..  admonition:: Jupyter Lab notebook

    `Downscaling of a spline <https://splinekit.github.io/splinekit-jupyterlite/lab/?path=periodic-spline/multiresolution/spline_down_proj.ipynb&mode=single-document>`_

----

Rescaling
---------

TODO