"""Bounds of the Python tests, each with its derivation.

LIBM: Difference between rsdag and numpy for an expression with elementary
functions, relative to 1 + |value|. numpy's float64 functions are within
4 ulp (the bound Intel states for SVML, which numpy uses), rsdag's within
1 ulp (exp 0.51 and ln 0.63 ulp, rsdag::constants::exp_ln), and each rounded
operation adds 0.5 ulp. At the points of the tests the error count of both
sides times the condition of the expression is at most 30 ulp: two numpy
functions and four operations without cancellation (14 ulp), or one function
and four operations at the condition 4.8 of the tanh gradient (29 ulp).

ROUND_OFF: Difference between two evaluations of an arithmetic expression of
at most four rounded operations each, in a different order or with fused
multiply-adds, relative to 1 + |value|. At the points of the tests the
condition is at most 2, so the evaluations differ by 2 * 4 * 2 u = 8 eps.

SUM_ORDER: Difference of two sums of the same three terms in different
orders, relative to 1 + |sum|. To first order the orders differ by
2 gamma_2 sum |t| = 4 u sum |t|, and at the points of the test
sum |t| <= 1.35 (1 + |sum|), so the bound is 5.4 u < 3 eps.

FINITE_DIFFERENCE: Error of a central difference with h = 1e-6 of the Lorenz
right-hand side at x = (1, 2, 3), relative to 1 + |J|. The function is
quadratic, so the truncation error is zero. Each component takes at most
three operations on terms of magnitude below 30, so an evaluation is within
3 u 30 = 1e-14, which the division by 2h turns into 1e-8. The roundings of
x + h and x - h move the step by at most ulp(4) / 2h = 4.4e-10 of itself.
The bound is 2e-8.

LINEAR_ALGEBRA: Error of a dense solve and a matrix product against numpy or
the exact solution, relative to 1 + |value|. The product of A (8 x 6) and
B (6 x 3) in two orders differs by 2 gamma_6 |A| |B|, at most
1.2e-14 (1 + |AB|) for the matrices of the test. Solving A x = y for y = A x
with n = 10 and partial pivoting has the backward error gamma_3n |L| |U|
(Higham, Accuracy and Stability of Numerical Algorithms, 2002, thm. 9.4) and
y the error gamma_n |A| |x|, so x is within
40 u || |A^-1| |L| |U| |x| || = 40 u 3.15 = 1.4e-14 of the exact solution.
The bound is 1.5e-14.
"""

import numpy as np

EPS = np.finfo(float).eps

LIBM = 32 * EPS

ROUND_OFF = 8 * EPS

SUM_ORDER = 3 * EPS

FINITE_DIFFERENCE = 2e-8

LINEAR_ALGEBRA = 1.5e-14
