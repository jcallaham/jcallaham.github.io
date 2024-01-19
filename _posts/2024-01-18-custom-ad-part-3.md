---
layout: post
title: "Custom autodiff part 3: forward mode"
date: 2024-01-11T05:30:30-04:00
categories:
  - blog
tags:
  - automatic-differentiation
  - machine-learning
  - scientific-computing
  - python
---

This post is Part 3 in a series on how to derive custom rules for differentiating through functions in autodiff frameworks like PyTorch, JAX, or Flux.
In [Part 1]({% link _posts/2024-01-04-custom-ad-part-1.md %}) we went over the basic idea of automatic differentiation as applied to functions with a single vector-valued input and output.
In [Part 2]({% link _posts/2024-01-11-custom-ad-part-2.md %}) we took a step back to learn about linear maps, their adjoints, and how this relates to automatic differentiation.
This time we'll finally learn how to derive custom rules for automatic differentiation starting with forward mode, and then walk through a number of examples of doing so.

As a reminder, here's the outline of this series:

* [**Part 1:**]({% link _posts/2024-01-04-custom-ad-part-1.md %}) Autodiff basics: forward- and reverse-mode
* [**Part 2:**]({% link _posts/2024-01-11-custom-ad-part-2.md %}) Linear maps and adjoints
* **Part 3:** Deriving forward-mode AD rules (w/ examples)
* **Part 4:** Deriving reverse-mode AD rules (w/ examples)
* **Part 5:** Putting it all together: differentiating constrained optimization

In the rest of this series, I'm using a [fork of the micrograd autodiff engine](https://github.com/jcallaham/micrograd) to demonstrate how to implement these rules in code.
However, since that doesn't support forward mode AD, I can't code up these examples in that framework.
Fortunately, the implementation of forward-mode rules is usually pretty straightforward and intuitive, but maybe I'll revisit this post if I can come up with a simple way to demo forward mode autodiff.

As we've seen in the previous posts, forward-mode autodiff essentially amounts to decomposing a complex Jacobian-vector product (or pushforward, or tangent linear map) into a number of simpler Jacobian-vector products using the chain rule.
If we want to implement a custom AD rule for forward mode, then we just have to derive and implement its "pushforward", which basically amounts to linearizing the function.
This is usually easier than deriving reverse-mode rules, and is actually a key step in that process, but we'll come back to that in Part 4.

Before we get started, here is a quick reminder of some terminology and notation conventions I'm using for this series (discussed in more depth in the last post).

* For a function $y = f(x)$, $x$ and $y$ are the "nominal" or "primal" values, indicating that they're the values that are being "linearized about" when we are calculating derivatives.
* The Jacobian or will be denoted with the partial derivative symbol, so for a function $f(x)$, $\partial f(x)$ means "the Jacobian of $f$ evaluated at nominal inputs $x$".  The transpose notation $\partial f(x)^T$ indicates the adjoint.
* The "pushforward" $\partial f$ is the same thing as the "tangent linear map" or "Jacobian-vector product", which maps from input perturbations to output perturbations.
* The "pullback" $\partial f^T$ is the same thing as the "adjoint map" or "vector-Jacobian product", which maps from output perturbations to input perturbations.
* I'll call inputs to the pushforward (a.k.a. Jacobian-vector product) "tangents" or "tangent values" and inputs to the pullback (a.k.a. vector-Jacobian product) "adjoints" or "adjoint values".
* Tangent values will be denoted by a "dot" superscript, and adjoint values will be denoted by a "bar".  So for a function $y = f(x)$, the seed value in forward mode is $\dot{x}$ and the Jacobian-vector product calculates $\dot{y} = \partial f(x) \dot{x}$.  In reverse mode the seed value is $\bar{y}$ and the vector-Jacobian product calculates $\bar{x} = \partial f(x)^T \bar{y}$.
* I won't distinguish between input/output spaces and their tangents, cotangents, etc. All of these examples use "flat" spaces like $\mathbb{R}^n$ anyway.

---
### Deriving the pushforward

In mathematical terms, let's say our custom function is $f(x): \mathbb{R}^n \rightarrow \mathbb{R}^m$.
The input is a vector $x \in \mathbb{R}^n$ and the output is a vector $y = f(x) \in \mathbb{R}^m$.
The Jacobian of $f$ evaluated at a primal value $x$ is a linear map $\partial f(x): \mathbb{R}^n \rightarrow \mathbb{R}^m$, which for vector-valued inputs and outputs could also be represented as a matrix $\mathbb{R}^{m \times n}$.

<!-- See https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html#jacobian-vector-products-jvps-aka-forward-mode-autodiff -->
What we typically have to implement to have $f$ work in a forward-mode AD system is a function that maps from a (primal, tangent) input pair $(x, \dot{x})$, where $x, \dot{x} \in \mathbb{R}^n$ to the (primal, tangent) output pair $(y, \dot{y})$, where $y = f(x)$ is the result of the primal calculation, $\dot{y} = \partial f(x) \dot{x}$ is the tangent output, and $y, \dot{y} \in \mathbb{R}^m$ are elements of the output space.
Since we presumably know how to calculate $y = f(x)$, we'll focus on the pushforward (Jacobian-vector product) $ \dot{y} = \partial f(x) \dot{x} $.
For now, think of this as a matrix-vector product with the Jacobian matrix $\partial f(x)$.
With that in mind, the only difficulty is deriving the Jacobian: linearizing the function.

Here's a simple process that should work out in most cases, including the more complicated ones we'll get to later (like functions with multiple arguments, matrix-valued functions, and implicit functions):

* Start with $y = f(x)$ evaluated at the nominal input values $x$.
* Perturb both the inputs and the outputs: $y + \delta y = f(x + \delta x)$.
* Expand and rearrange until $f(x + \delta x)$ is in the form of a Taylor series:

$$
y + \delta y = f(x + \delta x) = (\textrm{terms that don't depend on } \delta x) + (\textrm{terms linear in } \delta x) + (\textrm{higher-order terms})
$$

* Since we are linearizing $f$, it's safe to discard the higher-order terms
* As usual in the Taylor series, we expect that the "terms that don't depend on $\delta x$" are identical to $f(x)$ (if this isn't the case, double check your math).  Then we can cancel these with $y$ on the right-hand side of the equation, leaving

$$
\delta y = (\textrm{terms linear in } \delta x).
$$

This is the equation for the Jacobian-vector product, if we replace the input perturbation $\delta x$ with the tangent input value $\dot{x}$ and the output perturbation $\delta y$ with the tangent output $\dot{y}$.

**This same approach works for inputs and outputs that are not vectors, provided we view the result of the linearization as a general tangent linear map rather than an $m \times n$ Jacobian matrix**, as we discussed in Part 2.
We can even use this in the case where $y$ is defined as a function of $x$ _implicitly_ by the solution of some equation $f(x, y) = 0$. Again, we will perturb both $x$ and $y$, discard higher-order terms in $\delta$, and then manipulate it to "Taylor series" form.
We'll see both matrix functions and implicit functions in the examples section.

---
### Multiple inputs and outputs

So far we've only looked at functions with one input and one output, but really the more general case is not much more difficult.
Let's take the case of multiple inputs first.
In normal multivariate calculus, we expand multi-input functions with the total derivative. For $z = f(x, y)$, this looks like

$$
\delta z = \frac{\partial f}{\partial x} \delta x + \frac{\partial f}{\partial y} \delta y.
$$

Similarly, for a function with two inputs, we can think of the Jacobian as consisting of two maps, one for each input $\rightarrow$ output pathway.
The pushforward will need to accept one tangent value for each input, so if $v = (\dot{x}, \dot{y})$, then we could write
<!-- To keep track of various tangent inputs, sometimes these are denoted by a "dot" in the Jacobian-vector product notation, so that $v = (\dot{x}, \dot{y})$ and we can write the expanded JVP like this (it is _not_ a time derivative): -->

$$
\dot{z} = \partial f(x, y) v = \partial_x f(x, y) \dot{x} + \partial_y f(x, y) \dot{y}.
$$

Now we have to derive each map $\partial_x f$ and $\partial_y f$ in a similar manner as we did for the single-input case, but otherwise everything works more or less the same.

For functions with multiple outputs, it's sometimes easiest to think of each output as the result of a different function and derive the pushforward separately for each.
For instance, if a function $f(x)$ returns a tuple $(y, z)$, the pushforward for $f$ can be derived by thinking of it as two stacked functions $y = g(x)$ and $z = h(x)$:

$$
\begin{bmatrix}
y \\ z
\end{bmatrix} =
f(x) \equiv
\begin{bmatrix}
g(x) \\ h(x)
\end{bmatrix}.
$$

Then you can derive the rules for $g$ and $h$ as usual and just return a tuple of the tangent outputs.
Of course, it will commonly be the case that $g$ and $h$ might share some computations, so actually _implementing_ the JVPs as two separate functions might not be very efficient, but sometimes it's at least a convenient way to derive them.

Other times, it's more straightforward to keep the outputs together.
This will be the case for instance in the equality-constrained quadratic programming example in Part 5.

---
### Forward-mode examples

#### Example 1: matrix-vector product

To illustrate this process, let's look at a simple function: a matrix-vector multiplication $y = f(A, x) = A x$ with $x \in \mathbb{R}^n$ and $A \in \mathbb{R}^{m \times n}$.
First we perturb the inputs and the output:

$$
y + \delta y = (A + \delta A) (x + \delta x)
$$

Then expand everything:

$$
y + \delta y = Ax + A \delta x + \delta A x + \delta A \delta x
$$

Since we know $y = Ax$ from the original equation, we can cancel those terms.
We can also discard the last term, since it's a product of two perturbations and we're only keeping first-order terms.
We're left with

$$
\delta y = A \delta x + \delta A x,
$$

which tells us that the pushforward can be calculated for tangent inputs $v = (\dot{A}, \dot{x})$ with

$$
\dot{y} = \partial f(A, x) v = A \dot{x} + \dot{A} x.
$$

-->

#### Example 2: linear system solve

A slightly more difficult example is solving the linear system of equations $A x = b$ for $x$.
This can be written explicitly as $f(A, b) = A^{-1} b$, but deriving the pushforward is easiest in the implicit form $Ax = b$.
Perturbing the inputs and outputs,

$$
(A + \delta A) (x + \delta x) = b + \delta b.
$$

As usual, we cancel out the nominal solution, discard any higher-order terms, and interpret the result as the definition of the Jacobian-vector product, leaving

$$
A \dot{x} + \dot{A} x = \dot{b}.
$$

This tells us that we can compute the tangent outputs $\dot{x}$ by a second linear system solve:

$$
A \dot{x} = \dot{b} - \dot{A} x.
$$

Note that we can reuse our existing linear system solving machinery, including any matrix factorizations, to calculate the Jacobian-vector product.

#### Example 3: matrix inversion

Next let's derive the pushforward for the matrix inversion operation $Y = f(X) = X^{-1}$.
This is a matrix-valued function, so it's an example of a case where it will be easier to think in terms of linear maps and adjoints than Jacobians and transposes.
We can derive the pushforward using the definition of the inverse: $X Y = I$, where $I$ is the identity matrix, as an implicit definition of $Y$ as a function of $X$.
We can follow the same procedure as before, beginning by perturbing $X \rightarrow X + \delta X$ and $Y \rightarrow Y + \delta Y$:

$$
(X + \delta X) (Y + \delta Y) = I.
$$

Since $X Y = I$ and we can discard the product of perturbations $\delta X \delta Y$ when linearizing, we are left with

$$
X \delta Y + \delta X Y = 0,
$$

or after a little algebra,

$$
\delta Y = - X^{-1} \delta X Y = - Y \delta X Y.
$$

Hence the pushforward is

$$
\dot{Y} = - Y \dot{X} Y.
$$

This is basically the expression for the "derivative" of the matrix inversion operation.
This equation determines the action of the Jacobian, producing the tangent outputs $\dot{Y}$ as a function of the tangent inputs $\dot{X}$ and the primal values $(X, Y)$.

Note that this could also be derived a little more succintly using the matrix product rule (shown for example in the [ChainRulesCore.jl docs](https://juliadiff.org/ChainRulesCore.jl/stable/maths/arrays.html#Matrix-inversion)), but personally I think the perturbation strategy is easier to reuse in more cases.

#### Example 4: Lyapunov equation

Let's try something a little more ambitious.
The Lyapunov equation is a matrix equation that frequently appears in linear systems and control theory.
Viewed as a matrix-valued function, a Lyapunov equation solver takes as input a matrix $A$ and returns as output the symmetric matrix $P$ to the equation

$$
A P + P A^T + Q = 0,
$$

where $Q$ is a symmetric, positive definite matrix.
We can see this as a function $P = f(A)$, where $f: \mathbb{R}^{n \times n} \rightarrow \mathbb{R}^{n \times n}$, although in this case we cannot write $f$ explicitly.
As a result, deriving an explicit tensor representation for the Jacobian $\partial f: \mathbb{R}^{n \times n} \rightarrow \mathbb{R}^{n \times n}$ would also be a challenge.

On the other hand, we can easily derive the Jacobian-vector product as a linear map.
We can derive the action of this map in the same way as before, by perturbing the inputs and outputs and identifying the leading-order term in the series expansion.

First, perturb the inputs $A \rightarrow A + \delta A$ and the outputs $P \rightarrow P + \delta P$:

$$
(A + \delta A) (P + \delta P) + (P + \delta P)(A + \delta A)^T + Q = 0.
$$

Next, rearrange into Taylor-series form:

$$
\left( A P + P A^T \right) + \left( \delta A P + A \delta P + \delta P A^T + P \delta A ^T \right) + \mathcal{Q}(\delta^2) + Q = 0.
$$

From the original Lyapunov equation, we know that $A P + P A^T = -Q$, so we can cancel the first term with $Q$.
The final set of terms is quadratic in the perturbations, so we can discard it as well, leaving

$$
\delta A P + A \delta P + \delta P A^T + P \delta A^T = 0.
$$

Again, this determines the action of the Jacobian on the input perturbation $\delta A$.
We derive the pushforward (Jacobian-vector product) by replacing the input perturbation $\delta A$ with the tangent input $\dot{A}$ and the output perturbation $\delta P$ with the tangent output $\dot{P}$:

$$
\dot{A} P + A \dot{P} + \dot{P} A^T + P \dot{A}^T = 0.
$$

But this isn't an explicit function; how are we supposed to calculate the output $\dot{P}$?
We have to get a little creative.

First, let's rearrange this to group the input and output perturbations:

$$
A \dot{P} + \dot{P} A^T + \left(\dot{A} P  + P \dot{A}^T \right)  = 0.
$$

This _almost_ looks like a new Lyapunov equation, if the terms in the parentheses were a symmetric matrix...

But remember that for any square matrix $M$, the sum $M + M^T$ is symmetric, so $(\dot{A} P) + (\dot{A} P)^T$ is symmetric.
But also we know that the nominal solution to $P$ is already symmetric, so $(\dot{A}P)^T = P\dot{A}^T$ and the terms in the parentheses _do_ make up a symmetric matrix!
Let's call this $\tilde{Q} = \dot{A} P  + P \dot{A}^T$, and note that this only depends on the tangent inputs and the nominal outputs.
In other words, we can calculate the pushforward with tangent inputs $\dot{A}$ by solving a second Lyapunov equation

$$
A \dot{P} + \dot{P} A^T + \tilde{Q} = 0, \qquad \tilde{Q} = \dot{A} P + P \dot{A}^T.
$$

This might seem like a weird coincidence, but actually it often turns out that the pushforward calculation shares a lot of structure with the original function, so it's not uncommon that we can reuse some of the same "machinery" for autodiff that we're using anyway for the primal computation -- in fact, we already saw this for the linear system solve in Example 2.
The same will be true of the pullbacks, as we'll see.

---
### Looking ahead

Finally, we've gotten to the heart of this series: deriving custom automatic differentiation rules.
In this post we covered forward-mode autodiff (deriving pushforwards), which is only half the story, but it's a good place to start and we'll need to understand this pretty well to be able to tackle reverse-mode rules (pullbacks) in the next post.

In Part 4 we'll jump right back in with an approach to deriving pullbacks.
We'll need to know how to derive the tangent linear map (which we covered this time), and we'll need to know how to work with their adjoints as well, which we introduced in a pretty abstract way in [Part 2]({% link _posts/2024-01-11-custom-ad-part-2.md %}).
Once we get through the basic process we'll revisit all the examples derived here, this time with straightforward Python implementations along with the math.