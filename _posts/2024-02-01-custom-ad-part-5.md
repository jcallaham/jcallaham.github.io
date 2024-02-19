---
layout: post
title: "Custom autodiff part 5: a final example"
date: 2024-02-01T05:30:30-04:00
categories:
  - blog
tags:
  - automatic-differentiation
  - machine-learning
  - scientific-computing
  - python
---

This post is Part 5/5 in a series on how to derive custom rules for differentiating through functions in autodiff frameworks like PyTorch, JAX, or Flux.
In [Part 1]({% link _posts/2024-01-04-custom-ad-part-1.md %}) we went over the basic idea of automatic differentiation as applied to functions with a single vector-valued input and output.
In [Part 2]({% link _posts/2024-01-11-custom-ad-part-2.md %}) we took a step back to go over linear maps, their adjoints, and how this relates to automatic differentiation.
In [Part 3]({% link _posts/2024-01-18-custom-ad-part-3.md %}) we learned how to derive custom rules for forward mode automatic differentiation, and then walked through a number of examples of doing so.
In [Part 4]({% link _posts/2024-01-25-custom-ad-part-4.md %}) we did the same for reverse-mode autodiff.  Here, we'll end with one last example worked end-to-end to see the full process: differentiating through an optimization problem.

As a reminder, here's the outline of this series:

* [**Part 1:**]({% link _posts/2024-01-04-custom-ad-part-1.md %}) Autodiff basics: forward- and reverse-mode
* [**Part 2:**]({% link _posts/2024-01-11-custom-ad-part-2.md %}) Linear maps and adjoints
* [**Part 3:**]({% link _posts/2024-01-18-custom-ad-part-3.md %}) Deriving forward-mode AD rules (w/ examples)
* [**Part 4:**]({% link _posts/2024-01-25-custom-ad-part-4.md %}) Deriving reverse-mode AD rules (w/ examples)
* **Part 5:** Putting it all together: differentiating constrained optimization

For the code, I'll once again be using [my fork of "micrograd"](https://github.com/jcallaham/micrograd) to demonstrate implementations of the examples.
Micrograd is a minimal autodiff engine developed by Andrej Karpathy as a hands-on explanation of how reverse-mode automatic differentiation works.
It's not intended for any heavy-duty machine learning or scientific computing, but it's easy to understand and work with, and the basic mechanics are very similar to what you would need in order to implement one of these rules yourself in something like PyTorch or JAX.

### Equality-constrained quadratic program

The example problem in this post is an equality-constrained quadratic program (QP), defined mathematically as

$$
\min_x \frac{1}{2} x^T Q x + c^T x, \qquad \textrm{subject to } \qquad Ax = b.
$$

Here $x, c \in \mathbb{R}^n$ and $b \in \mathbb{R}^m$, so $Q \in \mathbb{R}^{n \times n}$ (symmetric) and $A \in \mathbb{R}^{m \times n}$.
That is, there are $n$ "decision variables" and $m$ constraints.

As with the Lyapunov example, it's easiest to view the QP solve as a function defined implicitly by its optimality conditions.
These can be derived from the Lagrangian of the optimization problem.
Introducing the Lagrange multipliers $\lambda \in \mathbb{R}^m$, the Lagrangian is

$$
\mathcal{L}(x, \lambda) = \frac{1}{2} x^T Q x + c^T x + \lambda^T (Ax - b).
$$

The optimality conditions are determined by setting the variations of the Lagrangian with respect to $x$ and $\lambda$ equal to zero in turn.  For the variation with respect to $x$:

$$
0 = Qx + c + A^T \lambda,
$$

and the variation for $\lambda$ the result is just the constraint:

$$
0 = Ax - b.
$$

Note that in deriving the first equation I transposed the last term of the Lagrangian so that $\lambda$ wasn't transposed: $\lambda^T A x = x^T A^T \lambda$.

Combined, we can view these two as a linear system of equations implicitly defining $(x, \lambda)$ as a function of the various parameters $(Q, c, A, b)$:

$$
\begin{bmatrix}
Q & A^T \\
A & 0
\end{bmatrix}
\begin{bmatrix}
x \\ \lambda
\end{bmatrix}
+ \begin{bmatrix}
c \\
-b
\end{bmatrix} = 0.
$$

In contrast to inequality-constrained quadratic or more general nonlinear programs, the inequality-constrained QP can be solved in one shot by solving this symmetric linear system.
Even better, Q and A are often sparse, so it can be solved efficiently with iterative linear system solvers or sparse direct solvers.
Systems of this form are also often solved as subproblems of iterative solution methods for more general constrained optimization problems.

We _could_ continue working with this as a function with four inputs and two outputs, but now that we see the structure it is easier to just treat it as a linear system solve and re-use our results from [Part 3, example 2]({% link _posts/2024-01-18-custom-ad-part-3.md %}).
That is, let

$$
M = \begin{bmatrix}
Q & A^T \\
A & 0
\end{bmatrix},
\qquad z = 
\begin{bmatrix}
x \\ \lambda
\end{bmatrix},
\qquad p = \begin{bmatrix}
-c \\ b
\end{bmatrix},
$$

forming a square system of $m + n$ equations $M z = p$.

Then the pushforward is

$$
M \dot{z} = \dot{p} - \dot{M} z,
$$

which we can expand to the forward-mode rule

$$
\begin{bmatrix}
Q & A^T \\
A & 0
\end{bmatrix}
\begin{bmatrix}
\dot{x} \\ \dot{\lambda}
\end{bmatrix}
= 
-\begin{bmatrix}
\dot{Q} x + \dot{c} + \dot{A}^T \lambda \\
\dot{A} x - \dot{b}
\end{bmatrix}.
$$

Given the primals $(Q, c, A, b, x, \lambda)$ and tangent input values $(\dot{Q}, \dot{c}, \dot{A}, \dot{b})$, this tells us how to compute the tangent outputs $(\dot{x}, \dot{\lambda})$ by solving a (symmetric) linear system of equations.
In fact, comparing the structure of the tangent problem with the original optimality conditions, we can see that this is just another quadratic program with different values for the vectors $b$ and $c$!
As for the Lyapunov equations in Parts 3 and 4, we can re-use our original solver code and/or matrix factorizations to compute the pushforward.

The corresponding pullback could also be derived by reusing the results for the linear system solve.
However, this case does require a little special treatment due to the symmetric structure of $M$.
Because the matrix $Q$ is symmetric, the input space is not the space of $n \times n$ matrices but _symmetric_ $n \times n$ matrices.
As a result, the adjoint values $\bar{Q}$ must also be symmetric.

We can derive a pullback that enforces this by going back to the standard procedure of taking an inner product of the pushforward with an arbitrary adjoint value: in this case $w = (w_x, w_\lambda)$.
In matrix notation this looks like the following:

$$
\begin{bmatrix}
w_x^T & w_\lambda^T
\end{bmatrix}
\begin{bmatrix}
Q & A^T \\
A & 0
\end{bmatrix}
\begin{bmatrix}
\dot{x} \\ \dot{\lambda}
\end{bmatrix}
= 
-
\begin{bmatrix}
w_x^T & w_\lambda^T
\end{bmatrix}
\begin{bmatrix}
\dot{Q} x + \dot{c} + \dot{A}^T \lambda \\
\dot{A} x - \dot{b}
\end{bmatrix},
$$

but we could also write it using inner product notation:

$$
\langle w_x, Q\dot{x} + A^T \dot{\lambda} \rangle + \langle w_\lambda, A \dot{x} \rangle = 
- \langle w_x, \dot{Q} x + \dot{c} + \dot{A}^T \lambda \rangle
- \langle w_\lambda, \dot{A} x - \dot{b} \rangle.
$$

As usual, we want to manipulate these inner products to isolate the tangent values (i.e. isolate everything with a "dot" superscript in the second operand).
The algebra here isn't anything new, so let's skip to the result:

$$
\langle Q w_x + A^T w_\lambda, \dot{x} \rangle + \langle A w_x, \dot{\lambda} \rangle
=
- \langle w_x x^T, \dot{Q} \rangle
- \langle w_x, \dot{c} \rangle
- \langle w_\lambda x^T + \lambda w_x^T, \dot{A} \rangle
+ \langle w_\lambda, \dot{b} \rangle.
$$

From this we can read off equations for each of the adjoint "bar" values:

$$
\begin{align}
\bar{x} &= Q w_x + A^T w_\lambda \\
\bar{\lambda} &= A w_x \\
\bar{Q} &= -w_x x^T\\
\bar{c} &= -w_x \\
\bar{A} &= - (w_\lambda x^T + \lambda w_x^T)\\
\bar{b} &= w_\lambda
\end{align}.
$$

What do we do with this?
So far, we've seen many cases where the adjoints $w$ end up used as an intermediate stage of the calculation: we solve some "adjoint problem" first to determine $w$ and then use these to determine the result of the pullback.
In this case, we will be given primals $(Q, c, A, b, x, \lambda)$ and adjoint outputs $(\bar{x}, \bar{\lambda})$ and will need to compute the adjoint inputs $(\bar{Q}, \bar{c}, \bar{A}, \bar{b})$.

We can rewrite this set of equations in the two-step form by first writing a system of equations for the unknowns $(w_x, w_\lambda)$ in terms of the inputs to the pullback:

$$
\begin{bmatrix}
\bar{x} \\ \bar{\lambda}
\end{bmatrix}
=
\begin{bmatrix}
Q & A^T \\ A & 0
\end{bmatrix}
\begin{bmatrix}
w_x \\ w_\lambda
\end{bmatrix}.
$$

Comparing again with the optimality conditions we originally derived, we see this is a _third_ quadratic program!
In fact, it has the same $Q$ and $A$ matrices, but replaces the vectors $c \rightarrow - \bar{x}$ and $b \rightarrow \bar{\lambda}$.
Once again, this can be taken advantage of by reusing matrix factorizations.

From there we have everything we need to compute the adjoint inputs $(\bar{Q}, \bar{c}, \bar{A}, \bar{b})$.
The last wrinkle is the symmetry of $\bar{Q}$.
The formula above has $\bar{Q} = -w_x x^T$, but there is no guarantee that this rank-1 matrix will be symmetric.
We can instead _enforce_ that it is symmetric by taking its symmetric part: $\bar{Q} = -(w_x x^T + x^T w_x) / 2$.
This probably amounts to a projection onto the input space of symmetric $n \times n$ matrices, but honestly I don't know if that's technically true or not.
The final equations for the adjoint inputs given the intermediate adjoint values $(w_x, w_\lambda)$ are

$$
\begin{align}
\bar{Q} &= -\frac{1}{2}(w_x x^T + x w_x^T)\\
\bar{c} &= -w_x \\
\bar{A} &= - (w_\lambda x^T + \lambda w_x^T)\\
\bar{b} &= w_\lambda
\end{align}.
$$

Here's a simple implementation in micrograd using [OSQP](https://osqp.org/) that doesn't take advantage of sparse matrices, but does reuse matrix factorizations by keeping the same OSQP object:

```python
def solve_qp(Q, c, A, b, **settings):
    # Solve the equality-constrained quadratic program
    # min 0.5 x^T Q x + c^T x
    # s.t. A x = b

    # Initialize the OSQP solver
    solver = osqp.OSQP()
    P_sp = scipy_sparse.csc_matrix(Q.data)
    A_sp = scipy_sparse.csc_matrix(A.data)
    solver.setup(P=P_sp, q=c.data, A=A_sp, l=b.data, u=b.data, **settings)
    results = solver.solve()

    x = Array(results.x, (Q, c, A, b), 'solve_qp')  # Solution
    y = Array(results.y, (Q, c, A, b), 'solve_qp')  # Lagrange multipliers

    def _backward():
        x_bar, y_bar = x.grad, y.grad

        # Solve the adjoint system using the same OSQP solver
        solver.update(q=-x_bar, l=y_bar, u=y_bar)
        adj_results = solver.solve()

        w_x, w_y = adj_results.x, adj_results.y

        # Compute the adjoint inputs
        Q_bar = -0.5 * (w_x @ x.data.T + x.data @ w_x.T)
        c_bar = -w_x
        A_bar = -(np.outer(w_y, x.data) + np.outer(y.data, w_x))
        b_bar = w_y

        # Accumulate the adjoint inputs in the gradients
        Q.grad += Q_bar
        c.grad += c_bar
        A.grad += A_bar
        b.grad += b_bar

    x._backward = _backward
    y._backward = _backward
    
    return x, y
```

As usual, the naive mathematical derivation isn't always the most practically efficient.
For more on this example, see a [2017 paper from Amos & Kolter](https://arxiv.org/abs/1703.00443) that derives forward- and reverse-mode rules for the more general inequality-constrained quadratic program using the Karush-Kuhn-Tucker (KKT) conditions, including an efficient algorithm for the reverse-mode computation.
You can compare the equations here to Eq (8) in that paper (by taking the case where there are no inequality constraints).


### Final thoughts

At this point hopefully you've been able to work through these examples and build up some confidence in deriving custom rules for automatic differentiation.
Being able to do this helps release you from the bounds of "what has already been implemented" and also lets you look out for opportunities to do something more efficient numerically.
