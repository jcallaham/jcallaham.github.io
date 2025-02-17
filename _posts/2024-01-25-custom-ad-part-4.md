---
layout: post
title: "Custom autodiff part 4: reverse mode"
date: 2024-01-25T05:30:30-04:00
categories:
  - blog
tags:
  - automatic-differentiation
  - machine-learning
  - scientific-computing
  - python
---


This post is Part 4 in a series on how to derive custom rules for differentiating through functions in autodiff frameworks like PyTorch, JAX, or Flux.
In [Part 1]({% link _posts/2024-01-04-custom-ad-part-1.md %}) we went over the basic idea of automatic differentiation as applied to functions with a single vector-valued input and output.
In [Part 2]({% link _posts/2024-01-11-custom-ad-part-2.md %}) we took a step back to go over linear maps, their adjoints, and how this relates to automatic differentiation.
In [Part 3]({% link _posts/2024-01-18-custom-ad-part-3.md %}) we learned how to derive custom rules for forward mode automatic differentiation, and then walked through a number of examples of doing so.
This time, we'll get into reverse-mode autodiff rules and revisit the same set of examples from last time.

As a reminder, here's the outline of this series:

* [**Part 1:**]({% link _posts/2024-01-04-custom-ad-part-1.md %}) Autodiff basics: forward- and reverse-mode
* [**Part 2:**]({% link _posts/2024-01-11-custom-ad-part-2.md %}) Linear maps and adjoints
* [**Part 3:**]({% link _posts/2024-01-18-custom-ad-part-3.md %}) Deriving forward-mode AD rules (w/ examples)
* **Part 4:** Deriving reverse-mode AD rules (w/ examples)
* [**Part 5:**]({% link _posts/2024-02-01-custom-ad-part-5.md %}) Putting it all together: differentiating constrained optimization

We've already covered how to work with tangent linear maps and their adjoints (in part 2), as well as how to derive the tangent linear map, or pushforward (in part 3).
This time around we'll be deriving the pullback for reverse-mode autodiff, which will combine what we did for forward-mode with some manipulations related to adjoint maps.
The math to understand is the most difficult part of this whole series, but hopefully by breaking it down to a simple procedure and walking through some examples in code it will still be fairly approachable.

For the code, I'll be using [my fork of "micrograd"](https://github.com/jcallaham/micrograd) to demonstrate implementations of the examples.
Micrograd is a minimal autodiff engine developed by Andrej Karpathy as a hands-on explanation of how reverse-mode automatic differentiation works.
It's not intended for any heavy-duty machine learning or scientific computing, but it's easy to understand and work with, and the basic mechanics are very similar to what you would need in order to implement one of these rules yourself in something like PyTorch or JAX.

Before we get started, here is a quick reminder of some terminology and notation conventions I'm using for this series (discussed in more depth in the last post).

* For a function $y = f(x)$, $x$ and $y$ are the "nominal" or "primal" values, indicating that they're the values that are being "linearized about" when we are calculating derivatives.
* The Jacobian or will be denoted with the partial derivative symbol, so for a function $f(x)$, $\partial f(x)$ means "the Jacobian of $f$ evaluated at nominal inputs $x$".  The transpose notation $\partial f(x)^T$ indicates the adjoint.
* The "pushforward" $\partial f$ is the same thing as the "tangent linear map" or "Jacobian-vector product", which maps from input perturbations to output perturbations.
* The "pullback" $\partial f^T$ is the same thing as the "adjoint map" or "vector-Jacobian product", which maps from output perturbations to input perturbations.
* I'll call inputs to the pushforward (a.k.a. Jacobian-vector product) "tangents" or "tangent values" and inputs to the pullback (a.k.a. vector-Jacobian product) "adjoints" or "adjoint values".
* Tangent values will be denoted by a "dot" superscript, and adjoint values will be denoted by a "bar".  So for a function $y = f(x)$, the seed value in forward mode is $\dot{x}$ and the Jacobian-vector product calculates $\dot{y} = \partial f(x) \dot{x}$.  In reverse mode the seed value is $\bar{y}$ and the vector-Jacobian product calculates $\bar{x} = \partial f(x)^T \bar{y}$.
* I won't distinguish between input/output spaces and their tangents, cotangents, etc. All of these examples use "flat" spaces like $\mathbb{R}^n$ anyway.

---
### Deriving reverse-mode AD rules (pullback)

The general idea procedure is to start by deriving the pushforward (Jacobian-vector product), write an inner product, and then manipulate it until you have an expression that looks like the definition of the adjoint.
From there you can infer how to compute the pullback.

#### Explicit functions

To make that a little more concrete, let's say we have a function $y = f(x)$, for which we've already derived the pushforward (JVP) $\dot{y} = \partial f(x) \dot{x}$.
What we would like to compute is the pullback (VJP) $\bar{x} = \partial f(x)^T \bar{y}$ given some adjoint vector $\bar{y}$.

As a reminder, in forward mode we represented the tangent values with a dot, so the tangent corresponding to an input $x$ was $\dot{x}$.
For reverse-mode the adjoint values are often denoted with a bar, so the adjoint value corresponding to an output $y$ is $\bar{y}$.

We know from the definition of the adjoint that for any element $w$ of the output space,

$$
\langle w, \partial f(x) \dot{x} \rangle = \langle \partial f(x)^T w, \dot{x} \rangle.
$$

If we choose $w = \bar{y}$, then we have

$$
\langle \bar{y}, \partial f(x) \dot{x} \rangle = \langle \partial f(x)^T \bar{y}, \dot{x} \rangle.
$$

By definition of the pushforward $\dot{y} = \partial f(x) \dot{x}$, and likewise $\bar{x} = \partial f(x)^T \bar{y}$ by definition of the pullback.
Then the inner product simplifies to

$$
\langle \bar{y}, \dot{y} \rangle = \langle \bar{x}, \dot{x} \rangle.
$$

Our goal is to manipulate the inner product equation into a form like this.
Of course, $\bar{x}$ won't appear explicitly, but since we know this equation is true, we can equate _whatever_ appears as the factor with $\dot{x}$ with $\bar{x}$.

As usual, this will probably make more sense with examples.  In the next section we'll derive rules for all the forward-mode examples above, but let's just start with the simplest one to get the flavor: matrix-vector multiplication.

The matrix-vector multiplication function for a fixed matrix $A$ is $f(x) = A x$ with $x \in \mathbb{R}^n$ and $A \in \mathbb{R}^{m \times n}$.
Above we derived the Jacobian-vector product for a tangent vector $\dot{x} \in \mathbb{R}^n$:

$$
\dot{y} = \partial f(x) \dot{x} = A \dot{x}.
$$

To derive the adjoint, first we'll take the inner product of the pullback result with an arbitrary element of the output space $w \in \mathbb{R}^m$:

$$
\langle w, \dot{y} \rangle = \langle w, A \dot{x} \rangle = w^T A \dot{x}.
$$

We know that $w^T A = (A^T w)^T$, so we could also write this as $(A^T w)^T \dot{x}$.
But notice that this is the inner product between the vectors $\dot{x}$ and $A^T w$, so we could rewrite the inner product as

$$
\langle w, \dot{y} \rangle = \langle w, A \dot{x} \rangle = \langle A^T w, \dot{x} \rangle.
$$

If we specifically choose $w = \bar{y}$, then this is in the form $\langle \bar{y}, \dot{y} \rangle = \langle \bar{x}, \dot{x} \rangle$, provided

$$
\bar{x} = A^T \bar{y}.
$$

This is the pullback for matrix-vector multiplication.
This is a good sanity check -- the function was already a linear map defined by $A$, so the pullback (adjoint map) should just be the transpose $A^T$.

#### Implicit functions

Why use the intermediate value $w$ rather than just taking the inner product with $\bar{y}$ directly?
This extra step turns out to be useful for more complicated functions, for instance when the result $y$ is defined implicitly by $f(x, y) = 0$.
In this case the pushforward will look like

$$
\partial_x f(x, y) \dot{x} + \partial_y f(x, y) \dot{y} = 0,
$$

which defines a linear system of equations to be solved for $\dot{y}$ (see the [implicit function theorem](https://en.wikipedia.org/wiki/Implicit_function_theorem)).

If we want to derive the corresponding pullback, rather than inverting $\partial_y f$, we can just take the inner product with an arbitrary $w$:

$$
\langle w, \partial_x f(x, y) \dot{x} \rangle + \langle w, \partial_y f(x, y) \dot{y} \rangle = 0.
$$

Then we manipulate both inner products to isolate the tangent values $\dot{x}$ and $\dot{y}$ to get something defined in terms of the adjoint maps:

$$
\langle \partial_x f(x, y)^T w, \dot{x} \rangle + \langle \partial_y f(x, y)^T w, \dot{y} \rangle = 0.
$$

Again, we know from the basic definitions of the adjoint, pushforward, and pullback that $ \langle \bar{y}, \dot{y} \rangle = \langle \bar{x}, \dot{x} \rangle $.
What we need to do is manipulate this equation into that form.
Since $w$ is an _arbitrary_ element of the output space, we can do that by choosing $w$ to be the solution to the linear system $\partial_y f(x, y)^T w = -\bar{y}$, for the specific $\bar{y}$ handed to us by the autodiff system.
Is this the same thing as inverting the Jacobian $\partial_y f$?  Formally yes, but as we will see in the examples, sometimes breaking this into a separate calculation will lead to more straightforward implementations.

Then we have

$$
\langle \bar{y}, \dot{y} \rangle = \langle \partial_x f(x, y)^T w, \dot{x} \rangle,
$$

from which we can infer that $\bar{x} = \partial_x f(x, y)^T w$.

This gives us a two-step algorithm for the pullback: solve a linear system for the intermediate adjoint value $w$, then use that to compute the result of the pullback $\bar{x}$.
We'll see this pattern come up in a number of the examples below.


#### The general approach

Summarizing the process for deriving pullbacks,

1. Derive the Jacobian-vector product, defining the action of the tangent linear map $\partial f(x)$ on a tangent input $\dot{x}$.
2. Take the inner product of $\dot{y} = \partial f(x) \dot{x}$ with an arbitrary vector $w$ from the output space.
3. Manipulate the expression for the inner product into the form $\langle \bar{y}, \dot{y} \rangle = \langle \bar{x}, \dot{x} \rangle$. On the left hand, we want the inner product of $\dot{y}$ with _some linear function of_ $w$ which we associate with the adjoint value $\bar{y}$ and on the right side we want the inner product of $\dot{x}$ with _some other linear function of_ $w$, which we associate with the result $\bar{x}$.
4. Use this expression to derive a rule for computing $\bar{x}$.

It will always be _possible_ (though not always _easy_) to rework the inner product expression into one inner product with $\dot{x}$ and one with $\dot{y}$ as a consequence of the linearity of the Jacobian map.

In the case of a function with multiple inputs, say $z = f(x, y)$, we will have multiple tangent input values and the pullback will have to return a tuple of adjoint values $(\bar{x}, \bar{y})$.
In this case we will be looking for an equation like

$$
\langle \bar{z}, \dot{z} \rangle = \langle \bar{x}, \dot{x} \rangle + \langle \bar{y}, \dot{y} \rangle,
$$

where we each expression in the inner product with one of the tangent inputs tells us how to compute that adjoint value.
As for the pushforward, if the function has multiple _outputs_ it may (or may not) be easiest to derive the pullback as if it was two separate functions.

---
### Reverse mode examples

Now we're finally ready to work through the full derivation of reverse-mode autodiff rules.
To show how this works in a simple AD framework, I'll also show implementations of these functions in my [fork of micrograd](https://github.com/jcallaham/micrograd), which I extended to support vectors in a [previous post]({% link _posts/2023-06-22-micrograd.md %}).


#### Example 1: matrix-vector product

We already took the simple case of a matrix-vector product where the matrix $A$ was held fixed.
To see how the reverse-mode derivation works on a simple function with multiple inputs, let's look at the case where the matrix $A \in \mathbb{R}^{m \times n}$ is also considered an input:

$$
y = f(A, x) = A x.
$$

In the section on forward-mode examples we derived the following pushforward rule from tangent values $(\dot{A}, \dot{x})$ to $\dot{y}$:

$$
\dot{y} = A \dot{x} + \dot{A} x.
$$

To derive the pullback, we take the inner product of $\dot{y}$ with an adjoint value $w \in \mathbb{R}^m$ and then apply some basic properties of the inner product to try to isolate the tangent values $\dot{x}$ and $\dot{A}$ in the inner products:

$$
\begin{align}
\langle w, \dot{y} \rangle &= \langle w, A \dot{x} + \dot{A} x \rangle \\
&= \langle w, A \dot{x} \rangle + \langle w, \dot{A} x \rangle \\
&= \langle A^T w, \dot{x} \rangle + \langle w, \dot{A} x \rangle.
\end{align}
$$

That's almost it -- we have $\dot{y}$ and $\dot{x}$ isolated in the inner products.  What do we do about $\dot{A}$?
Since it's a matrix, it's actually easier to work with the trace inner product for this manipulation:

$$
\begin{align}
\langle w, \dot{A} x \rangle &= \mathrm{trace}(w x^T \dot{A}^T) \\
&= \langle w x^T, \dot{A} \rangle.
\end{align}
$$

Then we have it in the form we're looking for:

$$
\langle w, \dot{y} \rangle = \langle A^T w, \dot{x} \rangle + \langle w x^T, \dot{A} \rangle.
$$

Finally, we identify the factor of $\dot{y}$ (here, $w$) with the adjoint values $\bar{y}$, and the factors of $\dot{x}$ and $\dot{A}$ with the adjoint values $\bar{x}$ and $\bar{A}$:

$$
\begin{gather}
\bar{x} = A^T \bar{y} \\
\bar{A} = \bar{y} x^T.
\end{gather}
$$

The pullback function will be given $\bar{y}$ and the primal values $(A, x, y)$ and will need to return the tuple of adjoint values $(\bar{A}, \bar{x})$.

Here's an implementation for the "dunder" method `__matmul__` of `micrograd.Array`:

```python
class Array:
    # [...]

    def __matmul__(self, other):
        out = Array(self.data @ other.data, (self, other), '@')

        def _backward():
            _out_grad = out.grad.reshape(-1, 1) if out.grad.ndim == 1 else out.grad
            _self_data = self.data.reshape(-1, 1) if self.data.ndim == 1 else self.data
            _other_data = other.data.reshape(-1, 1) if other.data.ndim == 1 else other.data
        
            _self_grad = _out_grad @ _other_data.T  # Y_bar @ X.T
            _other_grad = _self_data.T @ _out_grad  # A.T @ Y_bar

            self.grad += _self_grad.reshape(self.grad.shape)
            other.grad += _other_grad.reshape(other.grad.shape)

        out._backward = _backward

        return out
```

#### Example 2: Linear system solve

The derivation for the linear system solve $x = f(A, b) = A^{-1} b$ with $A \in \mathbb{R}^{n \times n}$ works in much the same way.
We've already derived the pushforward for tangent values $(\dot{A}, \dot{b})$ to $\dot{x}$:

$$
A \dot{x} = \dot{b} - \dot{A} x.
$$

Again we take the inner product with an arbitrary element of the output space $w \in \mathbb{R}^n$ and manipulate it to isolate the tangent values in the inner products, which will work in the same way as the previous example.

$$
\begin{align}
\langle w, A \dot{x} \rangle &= \langle w, \dot{b} - \dot{A} x \rangle \\
&= \langle w, \dot{b} \rangle + \langle w, - \dot{A} x \rangle \\
\langle A^T w, \dot{x} \rangle &= \langle w, \dot{b} \rangle + \langle -w x^T, \dot{A} \rangle.
\end{align}
$$

From this we can identify $\bar{x} = A^T w$, $\bar{b} = w$, and $\bar{A} = -w x^T $.
Then we have the following equations determining $(\bar{A}, \bar{b})$ as a function of $\bar{x}$:

$$
\begin{gather}
\bar{x} = A^T w \\
\bar{A} = -w x^T \\
\bar{b} = w.
\end{gather}
$$

The procedure for calculating $(\bar{A}, \bar{b})$ is therefore to solve the first system of equations for $w$, and then use this result in the other equations for $\bar{A}$ and $\bar{b}$ (or just do the obvious elimination of $w = \bar{b}$ first).
Here's the code:

```python
def solve(A, b):
    # Solve the linear system `Ax = b`` for x
    # Both A and b are "dynamic" data, so we need to compute both their gradients.

    # Forward pass: compute the solution
    out = Array(linalg.solve(A.data, b.data), (A, b), 'solve')

    # Backward pass: compute the gradients with respect to A and b
    def _backward():
        # The adjoint outputs will be in `out.grad`
        x_bar = out.grad.reshape(-1, 1) if out.grad.ndim == 1 else out.grad
        x = out.data.reshape(-1, 1) if out.data.ndim == 1 else out.data

        # Solve the adjoint system A^T w = x_bar
        w = linalg.solve(A.data.T, x_bar)

        # Compute the adjoint inputs
        A_bar = -w @ x.T
        b_bar = w.reshape(b.grad.shape)

        # Accumulate the adjoint inputs in the gradients
        A.grad += A_bar
        b.grad += b_bar

    out._backward = _backward
    return out
```

#### Example 3: Matrix inversion

Next up is matrix inversion: $Y = f(X) = X^{-1}$.
This is where the adjoint machinery will start to be a lot more useful.
We've already derived the action of the Jacobian on a tangent input $\dot{X}$:

$$
\dot{Y} = \partial f(X) \dot{X} = -Y \dot{X} Y.
$$

The next step of the process requires taking the inner product with an element of the output space $W$.
However, this time we have to use an inner product over matrices - a natural choice is the trace:

$$
\langle A, B \rangle = \mathrm{trace}\left(A B^T\right) = \mathrm{trace}\left(A^T B\right).
$$

Taking the inner product of the JVP with the adjoint value $W$, we have

$$
\langle W, \dot{Y} \rangle = -\mathrm{trace}\left(W^T Y \dot{X} Y\right).
$$

To get anything useful out of this, we want to write it as an inner product of something with the tangent values $\dot{X}$.
This will require some properties of the trace, in particular the [cyclic property](https://en.wikipedia.org/wiki/Trace_(linear_algebra)#Cyclic_property):

$$
\mathrm{trace}\left(W^T Y \dot{X} Y\right) = \mathrm{trace}\left(Y W^T Y \dot{X}\right) = \langle Y^T W Y^T , \dot{X} \rangle.
$$

So the inner product equation we're looking for is

$$
\langle W, \dot{Y} \rangle = - \langle Y^T W Y^T, \dot{X} \rangle.
$$

Once again, we can equate the term in the inner product of $\dot{Y}$ with $\bar{Y}$, so $W = \bar{Y}$ and then equate the term in the inner product with $\dot{X}$ with $\bar{X}$.
We can infer from this that the action of the adjoint on $\bar{Y}$ is the pullback

$$
\bar{X} = \partial f(X)^T \bar{Y} = - Y^T \bar{Y} Y^T.
$$

Again, note that we don't have to write the adjoint operator $\partial f(X)^T$ explicitly as a matrix.
It is a linear map, and all we need to know is how to apply it given the primal and adjoint values.

Here's the micrograd code:

```python
def inv(X: Array):
    # Matrix inverse of X

    # Forward pass: compute the inverse
    out = Array(np.linalg.inv(X.data), (X,), 'inv')

    # Backward pass: compute the gradient with respect to X
    # using the reverse-mode rule `X_bar = -Y^T @ Y_bar @ Y^T`,
    # where Y = X^{-1} and Y_bar = out.grad
    def _backward():
        Y, Y_bar = out.data, out.grad
        X.grad += -Y.T @ Y_bar @ Y.T

    out._backward = _backward

    return out
```

#### Example 4: Lyapunov equation

Finally we're back to the Lyapunov equation.
Remember that we viewed this equation as defining the solution $P \in \mathbb{n \times n}$ as an implicit function of the square matrix $A \in \mathbb{R}^{n \times n}$.
That is, $P = f(A)$ is the solution to the Lyapunov equation

$$
A P + P A^T + Q = 0,
$$

where $Q \in \mathbb{R}^{n \times n}$ is a symmetric matrix we will consider to be fixed, and the result $P$ is also symmetric.
This has the consequence that the output space is not just the space of $n \times n$ matrices, but _symmetric_ $n \times n$ matrices.
This means that tangent and adjoint values like $\dot{P}$ and $\bar{P}$ also must be symmetric.

Earlier we derived that result of the pushforward (Jacobian-vector product) $\dot{P} = \partial f(A) \dot{A}$ is the solution to the "tangent" Lyapunov equation

$$
A \dot{P} + \dot{P} A^T + \tilde{Q} = 0, \qquad \tilde{Q} = \dot{A} P + P \dot{A}^T.
$$

So far we've been taking the inner product of an adjoint value $W$ directly with the tangent values like $\langle W, \dot{P} \rangle$.
However, in this case we don't really know how to write an explicit function for $\dot{P}$.
Instead, let's take the inner product with the whole tangent equation:

$$
\langle W, A \dot{P} + \dot{P} A^T \rangle + \langle W, \dot{A} P + P \dot{A}^T \rangle = 0.
$$

Remember that what we're trying to do here is manipulate this equation into a form like $\langle \bar{P}, \dot{P} \rangle = \langle \bar{A}, \dot{A} \rangle$.
To get there, we need to end up with two inner product expressions, one which isolates $\dot{P}$ and one which isolates $\dot{A}$.

As in the matrix inversion case, we're using the trace inner product, so $\langle X, Y\rangle = \mathrm{trace}(X Y^T)$.
We will also need a couple of properties of this inner product, which are easy to prove using the permutation property of the trace and rules for matrix transposition (again, I'm only giving these for real-valued matrices, but there are complex equivalents):

1. Association: $\langle A + B, C + D \rangle = \langle A, C \rangle + \langle A, D \rangle + \langle B, C \rangle + \langle B, D \rangle$
2. Permutation: $\langle A, BC \rangle = \langle B^T A, C \rangle = \langle A C^T, B \rangle$
3. Conjugation: $\langle A, B \rangle = \langle B, A \rangle$

First, expanding the inner products with the associativity rule,

$$
\langle W, A \dot{P} \rangle + \langle W, \dot{P} A^T \rangle = -\left[ \langle W, \dot{A} P \rangle + \langle W, P \dot{A}^T \rangle \right].
$$

Now we will apply the properties of the trace inner product.
For each term that contains $\dot{P}$, we will try to get it to a form like $\langle \cdot, \dot{P} \rangle$, and for each term that contains $\dot{A}$, we will try to get it to a form like $\langle \cdot, \dot{A} \rangle$.
I'll skip the algebra, but here's the result (which also uses the symmetry of $W$ and $P$):

$$
\langle A^T W, \dot{P} \rangle + \langle W A, \dot{P} \rangle = - \langle 2 W P, \dot{A} \rangle.
$$

Using the associativity rule, this simplifies to

$$
-\langle A^T W + W A, \dot{P} \rangle = \langle 2 W P, \dot{A} \rangle.
$$

This is not as simple to make sense of as the previous examples, but it is in the form we're looking for.
Let's try applying the procedure as written and see what we get.
To do that, we equate the product of $\dot{P}$ with the adjoint values $\bar{P}$, and the product of $\dot{A}$ with $\bar{A}$:

$$
\bar{P} = -(A^T W + W A) \\
\bar{A} = 2 W P.
$$

But the first equation is _another_ Lyapunov equation for $W$ based on the transpose of the original matrix $A$!
Hence, we can use this pair of equations to derive a two-step rule for computing $\bar{A}$ as a function of $\bar{P}$ (and the primal values $(A, P)$).
First, we solve the adjoint Lyapunov equation for the intermediate value $W$:

$$
A^T W + W A + \bar{P} = 0,
$$

then we use this result to compute $\bar{A} = 2 W P$.

This example illustrates a case where it's useful to let $W$ be an _arbitrary_ element of the output space, rather than assuming it corresponds directly to the adjoint values.
Because the pushforward was only defined implicitly in terms of the solution to the tangent Lyapunov equation, this gives us the flexibility to find another equation that defines the pullback implicitly as well.

Here's a simple implementation in micrograd:

```python
def solve_continuous_lyapunov(A, Q):
    # Solve the Lyapunov equation A P + P A^T + Q = 0 for X
    P = linalg.solve_continuous_lyapunov(A.data, -Q.data)  # Primal solution
    out = Array(P, (A,), 'solve_continuous_lyapunov')

    def _backward():
        # Solve the adjoint equation A^T W + W A + P_bar = 0
        P_bar = out.grad
        W = linalg.solve_continuous_lyapunov(A.data.T, -P_bar)

        # Compute the gradient using the reverse-mode rule
        # and the solution to the adjoint Lyapunov equation
        A_bar = 2 * W @ P

        # Accumulate the gradient to A
        A.grad += A_bar

    out._backward = _backward

    return out
```

However, this is a case where a smarter implementation could be much more computationally efficient.
The Bartels-Stewart algorithm used to solve the Lyapunov equation relies on Schur decompositions of $A$ and $A^T$, which could be reused in the adjoint Lyapunov equation.
It's always worth looking for places where computations like matrix factorizations can be reused.


---
### Looking ahead

Actually, we're almost done!  So far, we've covered the basic mechanics of automatic differentiation in Part 1, some more advanced math background in Part 2, and learned how to derive custom autodiff rules in forward mode (Part 3) and reverse mode (Part 4).

At this point hopefully you've been able to work through these examples and build up some confidence in deriving custom rules for automatic differentiation.
Being able to do this helps release you from the bounds of "what has already been implemented" and also lets you look out for opportunities to do something more efficient numerically.

I just want to close with a couple of assorted general tips for deriving these autodiff rules:

* **Don't be afraid of implicit functions.** For many complex functions it turns out that there is a relatively simple equation that defines the output _implicitly_, or states a condition under which the result is valid.  This can be much easier to work with than trying to figure out how to differentiate a function directly.  For example, differentiating an eigenvalue decomposition hardly makes sense when you think about it as an explicit function $V, \Lambda = f(A)$, and differentiating through the decomposition algorithm would probably be a nightmare.  On the other hand, the results $(V, \Lambda)$ are defined implicitly by the simple matrix equation $A V = V \Lambda$, which is no harder to work with than the examples above.
* **Trust the process.** Personally, I'm never really comfortable deriving things when I don't have a good intuition for what the symbols mean.  This can be tough when thinking about adjoints and pullbacks, but I've found that there is often just a "dark forest" phase of the derivation where you're blindly manipulating inner products... and then on the far side you can reconstruct an adjoint equation that makes sense.
* **Look for structure.**  Repeatedly in these examples we end up with a pushforward and pullback that sort of mirrors the original problem: the linear system solve, Lyapunov equations, the quadratic program, and so on.  There is a reason for that -- the pushforward is just a linearization of the original problem, and the pullback is its adjoint -- but in terms of deriving useful rules, just look for similar structure (or "transposed" structure, in the case of the pullback).  You might be able to reuse a lot of the "machinery" you already have for the primal problem, or come up with tricks for reusing things like matrix factorizations.
* **Browse Wikipedia.**  Useful in general, but also specifically when doing this, looking for possibly unfamiliar properties of the operations you're working with (like the trace).  You never know when you'll stumble across some helpful identity that makes everything simple all of a sudden.

With that, I'll leave off for now.
Next time we'll have the final part of the series, in which we'll work through the full process of deriving and implementing an automatic differentiation rule for a constrained optimization problem.