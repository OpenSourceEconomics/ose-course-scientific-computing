########
Lectures
########

We provide a set of lectures as `Jupyter Notebooks <https://jupyter.org>`_.


=======
Tooling
=======

We showcase the basics of Python programming and point students to useful resources to study
further. There are numerous excellent introductory lectures on Python programming in economics
available online. Among them is `Python programming for economics and finance
<https://python-programming.quantecon.org/intro.html>`_ and we will sample some of their material
for our lecture.

================
Linear equations
================

We explore different solution algorithms to solve linear equations. We look at direct methods
building on L-U decomposition as well as iterative methods. We study the impact of ill-conditioned
matrices on the performance of algorithms. In the process, we learn some basic ideas behind testing
and benchmarking numerical routines.

.. toctree::
   :maxdepth: 1

   linear_equations/notebook.ipynb
   linear_equations/algorithms.rst

===================
Nonlinear equations
===================

We explore different solution algorithms to solve nonlinear equations. We start with the bisection
method. We then turn to function iteration before exploring Newton's method for nonlinear equations.
Finally, we look at Quasi-Newton methods and benchmark their performance in solving a standard
Cournot problem. We briefly discuss some criterion to choose the right algorithm for the problem at
hand.

.. toctree::
   :maxdepth: 1

   nonlinear_equations/notebook.ipynb
   nonlinear_equations/algorithms.rst

============
Optimization
============

We discuss the key attributes of optimization algorithms that determine the choice of a suitable
optimization algorithm. We explore the role of noise in the criterion function and ill-conditioning
for different groups of optimizers: local vs. global, derivative-based vs. derivative-free. We
conclude with some programming exercises for nonlinear least squares problems and implement a simple
maximum likelihood estimation.


.. toctree::
   :maxdepth: 1

   optimization/notebook.ipynb
   optimization/algorithms.rst

===========
Integration
===========

We examine different strategies for the numerical integration of functions. We discuss rules
based on Newton-Cotes quadrature formulas, Gaussian quadrature, and Monte Carlo methods in the
uni-dimensional and multi-dimensional case. We conclude by looking comparing the performance of
the different approaches under different scenarios.


.. toctree::
   :maxdepth: 1

   integration/notebook.ipynb
   integration/algorithms.rst

=============
Approximation
=============

We study the function approximation using polynomials. We combine different strategies for the
interpolation nodes and basis functions to study how they interact to determine the approximation's
overall quality. We use this as an opportunity to iteratively develop a function that allows to
combine the different ingredients to set up an interpolator. Finally, we extend the ideas to the
case of multivariate interpolation.


.. toctree::
   :maxdepth: 1

   approximation/notebook.ipynb
   approximation/algorithms.rst
