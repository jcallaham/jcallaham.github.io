---
layout: post
title: "Custom autodiff part 2: linear maps"
date: 2024-01-11T05:30:30-04:00
categories:
  - blog
tags:
  - automatic-differentiation
  - machine-learning
  - scientific-computing
  - python
---

This post is Part 2 in a series on how to derive custom rules for differentiating through functions in autodiff frameworks like PyTorch, JAX, or Flux.
In [Part 1]({% link _posts/2024-01-04-custom-ad-part-1.md %}) we went over the basic idea of automatic differentiation as applied to functions with a single vector-valued input and output.
This time we'll take a step back and go over some relatively abstract math that will actually turn out to be the key to deriving our autodiff rules.
In [Part 3]({% link _posts/2024-01-18-custom-ad-part-3.md %}) we'll actually get started on some examples.

As a reminder, here's the outline of this series:

* [**Part 1:**]({% link _posts/2024-01-04-custom-ad-part-1.md %}) Autodiff basics: forward- and reverse-mode
* **Part 2:** Linear maps and adjoints
* [**Part 3:**]({% link _posts/2024-01-18-custom-ad-part-3.md %}) Deriving forward-mode AD rules (w/ examples)
* [**Part 4:**]({% link _posts/2024-01-25-custom-ad-part-4.md %}) Deriving reverse-mode AD rules (w/ examples)
* **Part 5:** Putting it all together: differentiating constrained optimization

As we've seen, a "pass" through an autodiff framework is basically made up of a sequence of linear operations, even when the function being differentiated is nonlinear.
There's nothing magic about that; differentiation is linear, the chain rule is linear, and automatic differentiation is really just an algorithm for computing the chain rule.
So in order to really understand AD, we need to understand two things: linear maps, and their adjoints.

Honestly, for a long time I thought that a linear operator was basically the same thing as a matrix, an adjoint was the same thing as a matrix transpose, and these distinctions were just mathematicians being pedantic.
Now I've come around, and I hope I can make the case for why these concepts are practically useful.

---
### Linear maps

Let's start with an $m \times n$ matrix $A$. We might say that by virtue of matrix-vector multiplication this matrix defines a linear map from $\mathbb{R}^n$ to $\mathbb{R}^m$. Its "adjoint" defines another linear map from $\mathbb{R}^m$ to $\mathbb{R}^n$.
Consider the dot product of an arbitrary vector $w$ in the output space $\mathbb{R}^m$ with the vector $y = A x$, also in the output space.
Using basic properties of the transpose we could equivalently write this as

$$
\begin{align}
w \cdot (Ax) &= w^T (A x) \\
&= (A^T w)^T x \\
&= (A^T w) \cdot x.
\end{align}
$$

The left-hand side is the dot product of $y$ with $Ax$ and the right-hand side is the dot product of the vector $A^T y$ with $x$.
In "bra-ket" inner product notation, we could write this equivalence as

$$
\langle w, Ax \rangle = \langle A^\dagger w, x \rangle,
$$

where $A^\dagger$ is the adjoint, which in this case is just the matrix transpose $A^T$.
In fact this is the _definition_ of the adjoint, which will turn out to be more general than a simple identity relating to the matrix transpose, but we'll come back to that.
For now, let's make a couple of observations about this definition of the adjoint:

1. The output space of $A^\dagger$ is the input space of $A$, and vice versa. That is, the adjoint maps from the output space of the original matrix to its input space.
2. The left-hand side of the equation is an inner product in the output space (in this case a dot product between two vectors in $\mathbb{R}^m$) and the right-hand side is an inner product in the input space (a dot product between two vectors in $\mathbb{R}^n$).
3. The adjoint is only defined with respect to a particular inner product. If we were to start from this definition we would find that the adjoint and the transpose of a matrix are only the same when the inner product is specifically the dot product $\langle u, v \rangle = u^T v$. For instance, if $m=n$ you could choose a weighted inner product $\langle u, v \rangle = u^T M v$ with symmetric positive definite $n \times n$ matrix $M$ and you would find that the adjoint is $A^\dagger = M^{-1} A^T M$ rather than just $A^T$ (try it!).

<!-- From these observations we can try to make an intuitive statement about what an adjoint "is": **the adjoint of a linear operator is a "dual" linear operator which maps from the output space of the original operator to its input space and is defined by a particular equivalence between inner products in the input and output spaces**. -->

This is all fine so far, and for our purposes that's all we need for functions with vector inputs and outputs. But what about functions of tensors? Or functions with multiple inputs and outputs? What if it's really expensive or inconvenient to actually represent our linear operation as a matrix?

This is where the more general idea of a "linear map" comes in. If we have a linear operator $\mathcal{A}$ (in slightly curly font to distinguish it from a normal matrix $A$) that maps from input space $\mathcal{X}$ to output space $\mathcal{Y}$, then it has the general linearity property that for any two scalars $\alpha$ and $\beta$ and elements of the input space $u, v \in \mathcal{X}$, then $\mathcal{A}(\alpha u + \beta v) = \alpha \mathcal{A} u + \beta \mathcal{A} v$.
So far this still looks like we might just be describing a matrix, except that elements of $\mathcal{X}$ and $\mathcal{Y}$ _don't need to be vectors_.
They could be matrices, tensors, L2-integrable functions, or any number of other things.

From a computational point of view, for a linear operator $\mathcal{A}$ all we need to know is how to compute its "action".
That is, given $x \in \mathcal{X}$, we have to be able to return $y = \mathcal{A} x$.
Again, this looks like matrix multiplication, but it might be better to think of $\mathcal{A}$ as a function that happens to be linear.
A matrix is one example of a linear map where the "action" is defined by matrix-vector multiplication.

A simple example of a linear map is matrix symmetrization. For the input space of square $n \times n$ matrices, we can define the linear map $\mathcal{A}: \mathbb{R}^{n \times n} \rightarrow \mathbb{R}^{n \times n}$ as $\mathcal{A} X =(X + X^T)/2$. This is clearly linear, but it's not straightforward to find a matrix or tensor representation for the operator $\mathcal{A}$: should it be an $n \times n \times n \times n$ tensor, an $n^2 \times n^2$ matrix, or something else? It's much easier (and memory-efficient!) to just think of it as a linear map and leave it at that.

---
### Adjoints

Given a linear map $\mathcal{A}: \mathcal{X} \rightarrow \mathcal{Y}$ and inner products defined on both $\mathcal{X}$ (the input space) and $\mathcal{Y}$ (the output space), then as in the matrix case we can define the _adjoint_ operator $\mathcal{A}^\dagger: \mathcal{Y} \rightarrow \mathcal{X}$ with

$$
\langle w, \mathcal{A} x \rangle = \langle \mathcal{A}^\dagger w, x \rangle,
$$

where $w \in \mathcal{Y}$ is an arbitrary element of the output space.
As before, notice that the left-hand side is an inner product in output space $\mathcal{Y}$, while the right-hand side is in input space $\mathcal{X}$.

We don't need to get too technical about inner products here; the most basic case is that $x$ and $y$ are just vectors, so that a natural inner product is $\langle u, v\rangle = u^T v$.
However, the more general inner product definition opens up some additional possibilities.
Here are a couple of useful inner products (assuming everything is real -- you can look up the complex versions if you need to):

* Vectors: $\langle u, v\rangle = u^T v$
* Matrices: $\langle A, B \rangle = \mathrm{trace}\left(A B^T\right)$
* Continuous functions: $\langle f, g \rangle = \int f(x) g(x) ~ dx$.

<!-- The one that might be surprising there is the continuous functions.
This can be useful for deriving various reverse-mode AD rules in scientific computing, where we often use things like piecewise polynomials to approximate continuous functions. -->

<!-- As a hopefully-not-too-confusing terminology reminder, I'm also calling the vectors that the adjoint map _acts on_ "adjoint values".
These are the ones with the bar superscript, as opposed to the "tangent values" which have the dot superscript.
I believe a more correct term would be something like "covariant" or "cotangent" vector, but I've seen "adjoint values" used quite a bit in more applied contexts, and anyway I think those other terms are a little obscure. -->

<!-- 

$$
\langle w, \mathcal{A} x \rangle = \langle \mathcal{A}^T w, x \rangle.
$$

Note that the adjoint maps from the _output_ space to the _input_ space (as does a matrix transpose), so the first inner product is between elements of the output space $\mathcal{Y}$, while the second is between elements of the input space $\mathcal{X}$. -->

Given a linear map and an appropriate inner product, we can derive the adjoint by taking the inner product of $y = \mathcal{A} x$ with an arbitrary element of the output space $w$, so $\langle w, \mathcal{A} x \rangle$, and then manipulating it to the form $\langle A^\dagger w, x \rangle$. From that we can infer the action of the adjoint $\mathcal{A}^\dagger$ on $w$.

In the matrix symmetrization example, we could take the inner product of $\mathcal{A}X = (X + X^T)/2$ with an arbitrary $n \times n$ matrix $W$.
Using the trace inner product, we get:

$$
\begin{align}
\langle W, (X + X^T) / 2 \rangle &= \mathrm{trace}(W (X^T + X)) / 2 \\
&= \left( \mathrm{trace}(W X^T) + \mathrm{trace}(W^T X^T) \right) / 2 \\
&= \mathrm{trace}((W + W^T) X^T) / 2 \\
&= \langle (W + W^T)/2, X \rangle.
\end{align}
$$

This is now in the form of the adjoint inner product: $\langle \mathcal{A}^\dagger W, X \rangle$ if the action of the adjoint operator is defined to be $\mathcal{A}^\dagger W = (W + W^T)/2$.  Again, we don't need a matrix representation for the adjoint operator.
In this particular example the operator is self-adjoint, so $\mathcal{A} = \mathcal{A}^\dagger$.
Also, we could have just transposed everything in the original linear map to get the right answer.
That's not exactly a coincidence, but we won't want to rely on that; it's much safer to just go through the manipulations with the inner products.

---
### Sidebar: terminology and notation

Variable terminology is part of what makes learning about the details of automatic differentiation a bit difficult.
My goal here is to use language that makes sense and I'm fine with that being at the expense of a little rigor.

As we've said, the tangent linear map is also sometimes called the "pushforward" and its adjoint is sometimes called the "pullback", borrowing terminology from differential geometry.
These terms are sometimes used interchangeably with "Jacobian-vector product" and "vector-Jacobian product"
Although Jacobians and vectors are more intuitive to me than "pullback" and "pushforward", I'll mostly use the latter terms instead.
Partly this is because I like that the "map" terminology doesn't imply literal matrix-vector products in the way that "Jacobian-vector product" or "vector-Jacobian product" does, and partly this is to avoid having the confusingly similar acronyms "JVP" and "VJP" all over the place.
Also, "pushforward" and "pullback" correspond neatly to their use in autodiff systems: "pushforward" is what's used in forward-mode AD, and "pullback" is what's needed for reverse-mode AD (or backpropagation).

Here are the main points:

* For a function $y = f(x)$, $x$ and $y$ are the "nominal" or "primal" values, indicating that they're the values that are being "linearized about" when we are calculating derivatives.
* The Jacobian or will be denoted with the partial derivative symbol, so for a function $f(x)$, $\partial f(x)$ means "the Jacobian of $f$ evaluated at nominal inputs $x$".  The transpose notation $\partial f(x)^T$ indicates the adjoint.
* The "pushforward" $\partial f$ is the same thing as the "tangent linear map" or "Jacobian-vector product", which maps from input perturbations to output perturbations.
* The "pullback" $\partial f^T$ is the same thing as the "adjoint map" or "vector-Jacobian product", which maps from output perturbations to input perturbations.
* I'll call inputs to the pushforward (a.k.a. Jacobian-vector product) "tangents" or "tangent values" and inputs to the pullback (a.k.a. vector-Jacobian product) "adjoints" or "adjoint values".  This is probably also not technically correct, but you'll see it used sometimes and I think it gets the idea across.
* Tangent values will be denoted by a "dot" superscript, and adjoint values will be denoted by a "bar".  So for a function $y = f(x)$, the seed value in forward mode is $\dot{x}$ and the Jacobian-vector product calculates $\dot{y} = \partial f(x) \dot{x}$.  In reverse mode the seed value is $\bar{y}$ and the vector-Jacobian product calculates $\bar{x} = \partial f(x)^T \bar{y}$. Personally I don't love this notation (especially when working with ODEs where $\dot{x}$ is usually $dx/dt$), but it's relatively common and I couldn't think of anything better.
* I won't distinguish between input/output spaces and their tangents, cotangents, etc. This is non-rigorous at best, but it makes the terminology a lot simpler. Also, all of these examples use "flat" spaces like $\mathbb{R}^n$ where the distinction is not that important (in my opinion as someone without a background in advanced geomtery).


---
### Looking ahead

With the more general concepts of the linear map and its adjoint in hand, we'll now be able to deal with linear operators that operate on and return all kinds of mathematical objects.
But many functions are inherently nonlinear: why is the idea of a linear map and its adjoint useful in automatic differentiation?
For functions where the input and/or output are not 1D vectors, the generalization of the Jacobian is the _tangent linear map_, or _pushforward_. This is the result of linearizing a nonlinear map about some nominal input values.

While forward-mode autodiff for "vector functions" boils down to a series of Jacobian-vector products, for more general functions the chain rule results in a series of applications of the tangent linear maps (pushforwards).  Likewise, reverse-mode autodiff for "vector functions" is a sequence of vector-Jacobian products, but the generalization is a series of applications of the _adjoint linear maps_, or _pullbacks_.

We can write the math for forward- and reverse-mode in basically the same notation as before, except that we will no longer think of the "Jacobian" in the JVP as a literal matrix, but as a more general linear map from the input space to the output space (though see the caveat about tangent and cotangent spaces below). Similarly, we will think of the VJP as an application of the adjoint map, not literally a transposed-matrix-vector product.
For a function $f$, we'll denote the tangent linear map by $\partial f$, and its adjoint by $\partial f^T$, but keep in mind that the "T" doesn't necessarily indicate a literal matrix transposition.

At this point these ideas might still seem fairly abstract, but they make more sense (and start to feel more useful) once you have a little experience working with them.  In the next post we'll see how to derive a tangent linear map, and hence how to derive custom rules for forward-mode automatic differentiation.
