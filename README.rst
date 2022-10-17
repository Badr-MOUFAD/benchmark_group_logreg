Benchmark group logistic regression
===================================
|Python 3.9+|

Benchopt is a package to simplify and make more transparent and
reproducible the comparisons of optimization algorithms.
This benchmark is dedicated to solver of the group logistic regression problem:


$$\\min_{w} \\sum_{i=1}^n \\log(1 + e^{-(Xw)_i}) + \\sum_{g \\in \\mathcal{G}} \\lVert w_g \\rVert_2 $$


where $n$ (or ``n_samples``) stands for the number of samples, $p$ (or ``n_features``) stands for the number of features,
$\\mathcal{G}$ the considered groups, and


$$X \\in \\mathbb{R}^{n \\times p} \\ , \\quad w \\in \\mathbb{R}^p$$


Install
-------

This benchmark can be run using the following commands:

.. code-block::

   $ pip install -U benchopt
   $ git clone https://github.com/#ORG/#BENCHMARK_NAME
   $ benchopt run benchmark_group_logreg

Apart from the problem, options can be passed to ``benchopt run``, to restrict the benchmarks to some solvers or datasets, e.g.:

.. code-block::

	$ benchopt run benchmark_group_logreg -s solver1 -d dataset2 --max-runs 10 --n-repetitions 10


Use ``benchopt run -h`` for more details about these options, or visit https://benchopt.github.io/api.html.

.. |Build Template| image:: https://github.com/benchopt/template_benchmark/workflows/Tests/badge.svg
   :target: https://github.com/benchopt/template_benchmark/actions
.. |Build Status| image:: https://github.com/Badr-MOUFAD/benchmark_group_logreg/workflows/Tests/badge.svg
   :target: https://github.com/Badr-MOUFAD/benchmark_group_logreg/actions
.. |Python 3.6+| image:: https://img.shields.io/badge/python-3.6%2B-blue
   :target: https://www.python.org/downloads/release/python-360/
