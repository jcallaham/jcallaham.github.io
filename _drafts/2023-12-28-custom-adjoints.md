---
layout: post
title: "Deriving automatic differentiation rules"
date: 2023-12-28T05:30:30-04:00
categories:
  - blog
tags:
  - python
  - autodiff
---

<!-- NOTE: This was the full draft - since been split into multiple parts. -->

If you're working heavily with an automatic differentiation system like PyTorch, JAX, or Flux.jl, a lot of the time you don't have to think too much about how the magic happens. But every so often you might find yourself doing something that the framework doesn't know how to differentiate properly. It's also common that a custom autodiff "rule" will be more efficient when a function output is the result of an algorithm: matrix inversion, linear system solving, root-finding, etc.

The documentation for autodiff frameworks usually shows some way of implementing these kinds of custom "rules" (see for example the relevant pages for [PyTorch](https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html), [JAX](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html), and [Zygote.jl](https://fluxml.ai/Zygote.jl/dev/adjoints/)), but these pages typically don't go into depth about how to derive them. And while there are some great resources for learning about how AD works in general, they usually stop short of explanations for what to do in these oddball cases. The goal of this post is to write the tutorial I was looking for when I first ran into this.

I'm going to build on my [last post]({% link _posts/2023-06-22-micrograd.md %}), which walked through extending Andrej Karpathy's micrograd to vector math, so I'll illustrate simple implementations of some custom reverse-mode rules using [my fork of micrograd](https://github.com/jcallaham/micrograd).
It might be helpful to look at that first, though I'll try to make this pretty self-contained.

One last note before we get started: my intent here is not to write a math treatise (nor am I really qualified to).
Basically, I'm deliberately sacrificing some mathematical precision with the goal of writing an approachable tutorial for those of us whose eyes glaze over at phrases like "cotangent bundle".
If that troubles you, I'm guessing you may not really need a tutorial on this topic.
That said, the [documentation for ChainRulesCore.jl](https://juliadiff.org/ChainRulesCore.jl/stable/maths/propagators.html) has a great writeup on this topic for anyone looking for a little more rigor (although they also caveat their own lack of rigor...), whether or not you're actually using Julia.
You could also read this first and then go check out their documentation.

Here's what we'll cover:

1. **Autodiff basics**
2. **Linear maps and adjoints**
3. **Deriving forward-mode AD rules**
4. **Forward-mode examples**
5. **Deriving reverse-mode AD rules**
6. **Reverse mode examples**
7. **Putting it all together: equality-constrained quadratic program**
8. **Final thoughts**

---
### Autodiff basics

Let's start with the simple case of a function $y = f(x)$ that takes as input one vector $x \in \mathbb{R}^n$ and returns another $y \in \mathbb{R}^m$, so $f: \mathbb{R}^n \rightarrow \mathbb{R}^m$. This actually won't really cover a lot of practical cases, but hopefully it'll be a good way to get a feel for the basic mechanics. In the next section we'll start to see how to extend the basic version to more general cases.

The purpose of automatic differentiation (AD) is computing the derivative of the result of some numerical computation with respect to the inputs.
AD systems assume that even highly complex functions can be decomposed into a sequence (or graph) of relatively simple primitive operations, each of which can be analytically differentiated in isolation.
Under that assumption, the derivative of complex functions can be decomposed into a second graph of the derivatives of the primitive operations using the chain rule from calculus.

The way that the analytic derivatives are implemented is in terms of rules for Jacobian-vector products (in forward mode) and vector-tranpose-Jacobian products (in reverse mode).
So for $f: \mathbb{R}^n \rightarrow \mathbb{R}^m$, we can think of its derivative at some nominal input values $x \in \mathbb{R}^n$ is the Jacobian matrix $\partial f(x) \equiv df/dx \in \mathbb{R}^{m \times n}$.
The Jacobian-vector product (JVP) applied to a "tangent" vector $v \in \mathbb{R}^n$ essentially implements the matrix-vector multiplication

$$
\partial f(x) v.
$$

Why is this useful?
The key thing is that JVP rule can usually evaluate this product _without_ explicitly constructing the Jacobian.

If the input is scalar-valued, then $v \in \mathbb{R}$ and this JVP with $v = 1$ will return the derivative of $f$ at $x$, giving the sensitivity of the outputs to the inputs near the nominal inputs $x$.
On the other hand, if the input is vector-valued, reconstructing the full Jacobian would require evaluating the JVP $n$ times, for instance using the unit basis vectors as tangent vectors $v$.
When $n$ is large this becomes very expensive, but fortunately it is rarely necessary to explicitly form the full Jacobian; many, many algorithms boil down to a series of linear system solves, and iterative linear solvers like CG and GMRES only really need to evaluate the matrix-vector product, which is exactly what the JVP provides.

As we'll see, the Jacobian-vector product (also known as the pushforward) is the building block of forward-mode autodiff.
Similarly, the vector-Jacobian product (also known as the pullback) is the main building block of reverse-mode autodiff.
For the same function $f: \mathbb{R}^n \rightarrow \mathbb{R}^m$ above, the vector-Jacobian product (VJP) applied to an  "adjoint" (or "covariant", or "dual") vector $w \in \mathbb{R}^m$ implements the vector-transpose-matrix multiplication

$$
w^T \partial f(x_0).
$$

This can also be viewed as a matrix-vector multiplication with the transpose or _adjoint_ of the Jacobian $\partial f(x)^T \in \mathbb{R}{n \times m}$:

$$
\partial f(x)^T w.
$$

Note that while the seed of the Jacobian-vector product is an element of the _input_ space, the seed for the vector-Jacobian product is an element of the _output_ space (technically this is not quite true, but it's close enough -- see the note on "terminology" below).
In a sense, these "adjoint" vectors represent perturbations of the function outputs, and reverse-mode autodiff is responsible for computing corresponding perturbations of the function inputs.

![Propagators](/assets/images/autodiff/propagators.jpeg){: width="400" style="display: block; margin: 0 auto" }

This might be a little counterintuitive: we need Jacobians for a lot of different numerical methods and sensitivity analyses in scientific computing, so it might make sense that an efficient Jacobian-vector product would be useful, but when do we ever need the _transpose_ of the Jacobian?
Much less, what good is the product of the transpose with some particular vector?
There's one context in particular that's very important and where it turns out that the vector-Jacobian product is exactly what we need: optimization.

Let's say we have an unconstrained optimization problem with decision variables (parameters) $\theta \in \mathbb{R}^n$.
The optimization problem for a scalar-valued cost function $J(\theta)$ is

$$
\min_\theta J(\theta).
$$

In order to solve this problem, a key piece of information is the gradient of $J$ evaluated at some particular set of parameters $\theta$.
Since the cost function maps from $\mathbb{R}^n$ to $\mathbb{R}$, we usually think of the gradient as a vector $\nabla J(\theta) \in \mathbb{R}^n$.

How do we get this gradient using automatic differentiation?
The Jacobian for a function $f: \mathbb{R}^n \rightarrow \mathbb{R}$ is a matrix (basically a row vector) $\partial J(\theta) \in \mathbb{R}^{1 \times n}$.
Hence, the gradient is effectively the transpose of the Jacobian: $\nabla J(\theta) = \partial J(\theta)^T$.

In forward-mode autodiff what we get is the Jacobian-vector product with some seed vector $v \in \mathbb{R}^n$, which amounts to a dot product with the gradient: $\partial J(\theta) v = \nabla J(\theta) \cdot v$.
So in order to evaluate the full gradient using forward-mode autodiff, we would need to compute $n$ Jacobian-vector products, using as seed vector each of the $n$ standard basis vectors like $e_1 = \begin{bmatrix} 1 & 0 & 0 & \cdots \end{bmatrix}^T$.

On the other hand, in reverse-mode autodiff what we get is the vector-Jacobian product with a seed value $w \in \mathbb{R}$, which is essentially a rescaled gradient: $\partial J(\theta)^T w = w\nabla J(\theta)$.
If we happen to choose $w=1.0$ as the seed value, then we immediately get the gradient $\nabla J(\theta)$ using only a _single_ vector-Jacobian product.

This is what makes reverse-mode AD (a.k.a. backpropagation) so powerful in the context of optimization.
It only requires a single backwards pass with about the same computational complexity as the forwards pass to compute the full gradient, no matter how many input variables there are.
For the same reason, reverse-mode AD is much more efficient at computing full Jacobians for functions with many more inputs than outputs; again, you just have to do a VJP seeded with a standard basis vector for each row of the Jacobian.
There's a very naive (but hopefully easy to follow) implementation of this in [micrograd.functional](https://github.com/jcallaham/micrograd/blob/master/micrograd/functional.py).


#### Static vs dynamic data

We'll often write and/or implement functions using a number of variables, not all of which are really "inputs".
For example, if we write a function $f(x) = ax$, then the variable $a$ is neither an input not an output to the function.
In a sense it just parameterizes the function.

Variables like this are sometimes called "static" data, as opposed to "dynamic" data that can change depending on the inputs.
We don't have to account for these in our derivations of the autodiff rules, and we should be careful to implement the functions in such a way that it is clear what data is static and what is dynamic.
However, you always _can_ make them dynamic by treating them as additional arguments to the function and deriving autodiff rules for them as well.
In general it might be helpful to treat as many variables as possible as dynamic -- you never know what you might want to optimize.

For instance, if we have function representing a matrix-vector product $f(x) = A x$, then the matrix $A$ is static data and the Jacobian is $\partial_x f = A$.
On the other hand, if we wrote the function as $f(A, x) = A x$, then both $A$ and $x$ are dynamic and the Jacobian has two parts: $\partial_x f = A$ and $\partial_A f = x$.
The derivative with respect to $x$ is obviously unchanged, but now we also have to account for variations in $A$.

This may seem obvious but sometimes it can be a bit tricky to keep track of, particularly in machine learning applications.
For example, suppose we have a neural network $f_\theta(x)$, where $\theta$ are the various weights and biases in the network.
In the context of training this network the inputs to a neural network are samples $x$ from the training data, and the outputs are the predictions $y = f_\theta(x)$.... right?

But really, the training task is to minimize some loss function over a training set $\{ (x_i, y_i) \}_{i=1}^N$:

$$
\min_\theta \frac{1}{N} \sum_{i=1}^N \mathcal{L}(\hat{y}_i, y_i), \qquad \hat{y}_i = f_\theta(x_i).
$$

From this point of view the function we actually want to differentiate is

$$
J(\theta) = \frac{1}{N} \sum_{i=1}^N \mathcal{L}(f_\theta(x_i), y_i).
$$

The real inputs are the weights and biases $\theta$!
The training data are not inputs, but _static_ data.
That is, we don't need to calculate the derivative of anything with respect to the data itself in the training process.
That's not to say you never need the Jacobian $\partial_x f$ for _anything_, but you don't need it for training; all you need is the Jacobian $\partial_\theta f$.

This can be hard to keep track of, especially since $\theta$ are often called parameters and $x$ and $y$ are inputs and outputs.
Just try to think carefully about what you actually need to compute the sensitivity of, and with respect to what inputs.

#### Composing functions

If $f$ represents a complex calculation -- like a partial differential equation solve or prediction from a machine learning model -- it becomes more and more difficult to compute the linearization $\partial f (x)$ analytically.
The key insight of AD systems is that often $f$ is composed of a large number of simpler functions, in which case the chain rule says that the Jacobian is a series of matrix-vector products.

For instance, if $f(x) = g(h(x))$, where we know how to calculate the Jacobian-vector product (pushforward) for both $g$ and $h$, then

$$
\partial f(x) v = \partial g(h(x)) \partial h(x) v.
$$

Note that the nominal inputs to $g$ are the outputs from $h$ at its nominal inputs $x$, and the tangent vector for the JVP for $g$ is the output of the JVP for $h$.
We can calculate the full JVP by propagating both the nominal (or "primal") and tangent values through the computational graph in parallel:

![Pushforward](/assets/images/autodiff/pushforward.jpeg){: width="400" style="display: block; margin: 0 auto" }

1. Begin with the nominal input value $x_1$ and seed vector $\dot{x}_1$.
2. Compute the nominal output $x_2 = g(x_1)$ and pushforward $\dot{x}_2 = \partial g(x_1) \dot{x}_1$.
3. Compute the nominal output $y = h(x_2)$ and pushforward $\dot{y} = \partial h(x_2) \dot{x}_2$.
4. Finally, the nominal output is $y$ and the result of the JVP is $\dot{y}$.

All information required to compute each component Jacobian-vector product is passed via the primal and tangent vectors.
This means that the forward-mode rule for some primitive function can be implemented in complete isolation. Basically, your function will be somewhere in this chain.

**As the implementer of the pushforward (Jacobian-vector product), you can expect to be handed a seed vector (an element of the input space) along with the nominal input values and your job is to calculate the pushforward (returning an element of the output space)**.

Here's a simple example borrowed from the [JAX documentation](https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html#custom-jvps-with-jax-custom-jvp) deriving the pushforward (JVP) for the function $f(x, y) = \sin(x) + y$.
Since this is basically a scalar function (you don't have to worry too much about broadcasting, ufuncs, etc in JAX), it's easy to calculate the derivatives with basic calculus.
Note that the JVP is basically the total derivative: if $z = f(x, y)$, the value `tangent_out` is $\dot{z}$, where

$$
\dot{z} = \frac{\partial f}{\partial x} \dot{x} + \frac{\partial f}{\partial y} \dot{y}.
$$

We'll explain that and build on it further later, for now just take a quick look at the code to get a feel for the structure -- maybe you can recognize some of the pieces we've already talked about.

```python
import jax.numpy as jnp
from jax import custom_jvp

@custom_jvp
def f(x, y):
  return jnp.sin(x) * y

@f.defjvp
def f_jvp(primals, tangents):
  x, y = primals
  x_dot, y_dot = tangents
  primal_out = f(x, y)
  tangent_out = jnp.cos(x) * x_dot * y + jnp.sin(x) * y_dot
  return primal_out, tangent_out
```

In a sense, the vector-Jacobian product (pullback) just performs the products in the chain rule in the opposite order.
The pullback for $f(x) = g(h(x))$ expands to

$$
w^T \partial f(x) = w^T \partial g(h(x)) \partial h(x).
$$

Calculating this presents a bit of a puzzle.
The vector-transpose-matrix products propagate from left to right, but $g$ still requires as its inputs the nominal outputs $h(x)$, implying that the primal calculation has to be completed before the VJP.
This is the root of the "forward/backwards" passes in the reverse-mode algorithm.
The primal calculation is fully completed _first_, with all intermediate values stored.
Then the pullback can be calculated by propagating the adjoint values _backwards_ through the computational graph:

![Pullback](/assets/images/autodiff/pullback.jpeg){: width="340" style="display: block; margin: 0 auto" }

1. Begin with the nominal input value $x$ and seed vector $\bar{y}$.
2. Compute the intermediate primal output $x_2 = g(x_1)$.  Save this value for later.
3. Compute the final primal output $y = h(x_2)$.
4. Compute the VJP for the second function: $\bar{x}_2 = \partial h(x_2)^T \bar{y}$.
5. Compute the VJP for the first function: $\bar{x}_1 = \partial g(x)^T \bar{x}_2 \equiv \bar{x}$.
6. The full VJP $\partial f(x)^T \bar{y}$ is $\bar{x}$.

Note that the full "forward pass" happens _first_, storing intermediate results, and _then_ the "backwards pass" happens, using stored primal values from the forward pass.

Again, your custom function will happen somewhere in this process. **As the implementer of the pullback (vector-Jacobian product), you will implement one stage of the backwards pass: you can expect to be handed an adjoint vector (an element of the output space) along with the nominal input/output values and your job is to calculate the pullback (returning an element of the input space)**.

Here's the same example as before borrowed from the [JAX docs](https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html#custom-vjps-with-jax-custom-vjp), but this time implementing the VJP.
As before, it's easy to calculate these derivatives by hand.
But this time note that what gets returned is the gradient of the output evaluated at the nominal values, scaled by the adjoint value `g`:

```python
from jax import custom_vjp

@custom_vjp
def f(x, y):
  return jnp.sin(x) * y

def f_fwd(x, y):
# Returns primal output and residuals to be used in backward pass by f_bwd.
  return f(x, y), (jnp.cos(x), jnp.sin(x), y)

def f_bwd(res, g):
  cos_x, sin_x, y = res # Gets residuals computed in f_fwd
  return (cos_x * g * y, sin_x * g)

f.defvjp(f_fwd, f_bwd)
```


---
### Linear maps and adjoints

Now that we've covered the basics, let's zoom out a bit.
As we've seen, a "pass" through an autodiff framework is basically made up of a sequence of linear operations, even when the function being differentiated is nonlinear.
There's nothing magic about that; differentiation is linear, the chain rule is linear, and automatic differentiation is really just an algorithm for computing the chain rule.
So in order to really understand AD, we need to understand two things: linear maps, and their adjoints.

Honestly, for a long time I thought that a linear operator was basically the same thing as a matrix, an adjoint was the same thing as a matrix transpose, and these distinctions were just mathematicians being pedantic.
Now I've come around, and I hope I can make the case here for why these concepts are practically useful.

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

With the more general concepts of the linear map and its adjoint in hand, we'll now be able to deal with linear operators that operate on and return all kinds of mathematical objects.
But many functions are inherently nonlinear: why is the idea of a linear map and its adjoint useful in automatic differentiation?
For functions where the input and/or output are not 1D vectors, the generalization of the Jacobian is the _tangent linear map_, or _pushforward_. This is the result of linearizing a nonlinear map about some nominal input values.

While forward-mode autodiff for "vector functions" boils down to a series of Jacobian-vector products, for more general functions the chain rule results in a series of applications of the tangent linear maps (pushforwards).  Likewise, reverse-mode autodiff for "vector functions" is a sequence of vector-Jacobian products, but the generalization is a series of applications of the adjoints of the tangent linear maps (pullbacks).

We can write the math for forward- and reverse-mode in basically the same notation as before, except that we will no longer think of the "Jacobian" in the JVP as a literal matrix, but as a more general linear map from the input space to the output space (though see the caveat about tangent and cotangent spaces below). Similarly, we will think of the VJP as an application of the adjoint map, not literally a transposed-matrix-vector product.
For a function $f$, we'll denote the tangent linear map by $\partial f$, and its adjoint by $\partial f^T$, but keep in mind that the "T" doesn't necessarily indicate a literal matrix transposition.

Now we've gone over the key concepts of the linear map and its adjoint.
At this point these ideas might still seem fairly abstract, but they make more sense (and start to feel more useful) once you have a little experience working with them.

#### Sidebar: terminology and notation

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


<!--
The differences between forward and reverse mode have important ramifications for computational and memory complexity, which are explained in depth in most discussions of automatic differentiation.
 -->

---
### Deriving forward-mode AD rules (pushforward)

Let's start with _forward-mode_ autodiff, which as we've seen essentially amounts to decomposing a complex Jacobian-vector product (or pushforward, or tangent linear map) into a number of simpler pushforwards using the chain rule.
If we want to implement a custom AD rule for forward mode, then we just have to derive and implement its pushforward, which basically amounts to linearizing the function.
This is usually easier than deriving reverse-mode rules, and often is actually a key step in that process, but we'll come back to that.

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

This same approach works for inputs and outputs that are not vectors, provided we view the result of the linearization as a general tangent linear map rather than an $m \times n$ Jacobian matrix.
We can even use this in the case where $y$ is defined as a function of $x$ _implicitly_ by the solution of some equation $f(x, y) = 0$. Again, we will perturb both $x$ and $y$, discard higher-order terms in $\delta$, and then manipulate it to "Taylor series" form.

#### Multiple inputs and outputs


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
Of course, it will commonly be the case that $g$ and $h$ might share some computations, so actually _implementing_ the JVPs as two separate functions might not be very efficient, but it's at least a convenient way to derive them.

Other times, it's more straightforward to keep the outputs together.
This will be the case for instance in the equality-constrained quadratic programming example at the end.

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

<!-- 

#### Example: quadratic form

Let's take a slightly less trivial example: the quadratic form

$$
f(x) = \frac{1}{2} x^T Q x,
$$

where $x \in \mathbb{R}^n$ and $Q \in \mathbb{R}^{n \times n}$, so the output is a scalar $y \in \mathbb{R}$.
Hence, we know that the Jacobian is a map $\partial f(x): \mathbb{R}^n \rightarrow \mathbb{R}$, which we will also be able to express as a row vector $\in \mathbb{R}^{1 \times n}$.

Often this kind of function arises in optimization application where $Q$ is symmetric; we don't need this but it'll make the final result simpler, so let's assume $Q = Q^T$.

Again, perturb both the input and output:

$$
y + \delta y = \frac{1}{2} (x + \delta x)^T Q (x + \delta x)
$$

Then rearrange to Taylor-series form:

$$
y + \delta y = \left( \frac{1}{2} x^T Q x \right) + \left( \frac{1}{2} \delta x^T Q x + \frac{1}{2} x^T Q \delta x \right) + \left( \frac{1}{2} \delta x^T Q \delta x \right).
$$

The final term is quadratic in $\delta x$, so we can discard it since we're looking for the leading-order terms in $\delta x$.
The first term on the right-hand side is just the nominal output $f(x)$, so since $y = f(x)$ we can also cancel that, leaving

$$
\delta y = \frac{1}{2} \delta x^T Q x + \frac{1}{2} x^T Q \delta x,
$$

Finally, we can simplify this with the symmetry of Q, since $\delta x^T Q x$ = $x^T Q^T \delta x$ and $Q = Q^T$.

$$
\delta y = x^T Q \delta x,
$$

which tells us that the Jacobian is the row vector $\partial f(x) = x^T Q$ and the pushforward (JVP) can be calculated with

$$
\dot{y} = \partial f(x) \dot{x} = x^T Q \dot{x}.
$$ 
-->

#### Example 2: linear system solve

A slightly less obvious example is solving the linear system of equations $A x = b$ for $x$.
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

As in the Lyapunov equation example, we can reuse our existing linear system solving machinery to calculate the Jacobian-vector product.

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

This might seem like a weird coincidence, but actually it often turns out that the pushforward calculation shares a lot of structure with the original function, so it's not uncommon that we can reuse some of the same "machinery" for autodiff that we're using anyway for the primal computation.
The same will be true of the pullbacks, as we'll see.

<!-- 
#### Example: rotation matrix
A 2D vector can be rotated counter-clockwise by an angle $\theta$ using the rotation matrix

$$
R(\theta) = \begin{bmatrix}
\cos \theta & - \sin \theta \\ \sin \theta & \cos \theta
\end{bmatrix}
$$

The construction of this rotation matrix is a matrix-valued function $R: \mathbb{R} \rightarrow \mathbb{R}^{2 \times 2}$, so its Jacobian is a linear map $\partial R: \mathbb{R} \rightarrow \mathbb{R}^{2 \times 2}$.
We can derive the action of this map in the same way as before, by perturbing the inputs and identifying the leading-order term in the series expansion:

$$
R(\theta + \delta \theta) &= \begin{bmatrix}
\cos (\theta + \delta \theta) & - \sin (\theta + \delta \theta) \\
\sin (\theta + \delta \theta) & \cos (\theta + \delta \theta)
\end{bmatrix}
$$

Next we can expand each entry of the matrix using the Taylor series expansions for sine and cosine.
For instance, $\cos(\theta + \delta theta) = \cos (\theta) - \delta \theta \sin \theta + \mathcal{O} (\delta \theta^2)$.
Applying this to the entire rotation matrix,

$$
&= \begin{bmatrix}
\cos (\theta) - \delta \theta \sin \theta + \mathcal{O} (\delta \theta^2)
\end{bmatrix}
\end{aligned}
$$ -->


<!-- 

### The implicit function theorem

One last trick that can be useful in deriving autodiff rules is the implicit function theorem.
Basically, if we have a function defined as $f(x, y) = 0$, under some assumptions on $f$ we could look at this as an implicit definition of $y$ as a function of $x$.
We can write this as $y = g(x)$, recognizing that we may have no explicit way or writing an actual expression for $g$.
Let's say that $x \in \mathbb{R}^n$ and $y \in \mathbb{R}^m$, so that $f: \mathbb{R}^{n + m} \rightarrow \mathbb{R}^m$ and $g: \mathbb{R}^n \rightarrow \mathbb{R}^m$.

To give the implicit function theorem, we will need the Jacobian of $f$.
As seen in the "multiple inputs and multiple outputs" section above, this has two parts corresponding to the Jacobians with respect to $x$ and with respect to $y$.  Let's write this as

$$
\partial f(x, y) = \begin{bmatrix} \partial_x f(x, y) & \partial_y f(x, y) \end{bmatrix}.
$$

If $x \in \mathbb{R}^n$ and $y \in \mathbb{R}^m$, then $\partial_x f \in \mathbb{m \times n}$ and $\partial_y f \in \mathbb{m \times m}$.
What we want to do is derive an expression for the Jacobian-vector product $\partial g(x) v$, _without_ needing an explicit expression for either $g$ or its derivative.

According to the implicit function theorem, provided $\partial_y f$ is invertible at the nominal values $(x, y)$, then

$$
\partial g(x) = -[\partial_x f(x, g(x))]^{-1} [\partial_y f(x, g(x))].
$$

The Jacobian-vector product for $g$ can therefore be defined in terms of the Jacobians of $f$ alone.

Once again, the usefulness of this is probably best seen by example.

#### Example: root-finding

A root of a function $f(x, p)$ with inputs $x$ and parameters $p$ is defined as a point $x_0$ where $f(x_0, p) = 0$.
We might think of the call to a "root-finding" function like `scipy.optimize.root` as an implicit function $x = g(p)$.
That is, for some parameters $p$ the function $g$ returns a value $x$ satisfying $f(x, p) = 0$, although the function $g$ itself is only really explicitly defined in terms of the root-finding algorithm.

Since the result of the root-finding call satisfies $f(x, p) = 0$, we can derive the Jacobian-vector product rule for the root-finding function call (i.e. $g(p)$) using the implicit function theorem.
In particular,

$$
\partial g(p) = -\left[ \partial_x f(x, p) \right]^{-1} \left[ \partial_p f(x, p) \right],
$$

where $x$ is the nominal solution to the root-finding problem for parameters $p$.

This doesn't require being able to differentiate $g$ directly, although it does require the Jacobians of $f$ with respect to both arguments.
However, the JVP $\partial g(p) v$ can be evaluated only using the JVPs of the Jacobians of $f$, provided an iterative linear solver is used for the inverse.
-->

---
### Deriving reverse-mode AD rules (pushforward)

Now that we've seen how to derive a tangent linear map (pushforward), which we need for forward-mode AD, the next step is to look at deriving the pullback for reverse-mode.
The general idea is to start by deriving the pushforward (Jacobian-vector product), write an inner product, and then manipulate it until you have an expression that looks like the definition of the adjoint.
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
But notice that this is the inner product between the vectors $v$ and $A^T w$, so we could rewrite the inner product as

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
Since $w$ is an _arbitrary_ element of the output space, we can do that by choosing $w$ to be the solution to the linear system $f(x, y)^T w = -\bar{y}$, for the specific $\bar{y}$ handed to us by the autodiff system.
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

<!-- 
Step 3 may seem more complicated than it needs to be; in simple cases we can just manipulate the inner product into the form $\langle w, \dot{y} \rangle$, so that it's clear that $w = \bar{y}$ and it seems like we could have skipped $w$ altogether and just started Step 2 by taking the inner product with $\bar{y}$.
However, this will help in more complicated derivations like the Lyapunov equation below - hopefully this will make sense when we get there.
-->

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

<!-- 
#### Quadratic form

A quadratic form with symmetric weight matrix $Q$ is defined by:

$$
y = f(x) = \frac{1}{2} x^T Q x,
$$

where $x \in \mathbb{R}^n$ and $y \in \mathbb{R}$.
Above we derived the pushforward (Jacobian-vector product) for tangent values $\dot{x}$:

$$
\dot{y} = \partial f(x) \dot{x} = x^T Q \dot{x}.
$$

Following the procedure above, first we'll take the inner product with an arbitrary adjoint value $w \in \mathbb{R}$:

$$
\langle w, \partial f(x) \dot{x} \rangle = w x^T Q \dot{x} = w \dot{y}.
$$

Since $w$ is a scalar, it is trivial to rewrite this to isolate either $\dot{x}$ or $w$ in the inner product:

$$
w x^T Q \dot{x} = \langle w, \dot{y}\rangle = \langle w Q x, \dot{x} \rangle.
$$

Following the procedure above, we can identify the term in the inner product with $\dot{y}$ as $\bar{y}$ and the term in the inner product with $\dot{x}$ as $\bar{x}$, which is what we're looking for.
That is, we can replace $w$ with $\bar{y}$ and then find the pullback (vector-Jacobian product) rule

$$
\bar{x} = \partial f(x)^T \bar{y} = \bar{y} Q x.
$$

How could we interpret this?
Remember that the pullback can be used to evaluate the gradient of a scalar-valued function by setting the "seed" to 1.
That is, the pullback tells us that

$$
\nabla f(x) = Qx,
$$

which can be confirmed by directly differentiating the original function.
In this case, directly computing the gradient would actually have been much easier, but hopefully going through the process will help with the more complicated examples.

Here's the micrograd implementation.
Note that because we've already implemented the transpose and matrix multiplication, we could just compute the result as `0.5 * x.T @ Q @ x` without implementing this function.
But let's do it anyway because it gives a simple example with the basic mechanics.

```python
def quadratic_form(Q, x: Array):
    # Compute quadratic form (1/2) x^T Q x, where Q is a square matrix and x is a vector
    # Q is considered static data (constant) and x is considered dynamic data (variable)
    
    # Since Q is static, we don't need to worry about its gradient.
    Q = Q.data if isinstance(Q, Array) else Q

    # Forward pass: compute the quadratic form
    out = Array(0.5 * x.data.T @ Q.data @ x.data, (x,), 'quadratic_form')

    # Backward pass: compute the gradient with respect to x
    # using the reverse-mode rule `x_bar = y_bar * (Q @ x)`
    def _backward():
        # The adjoint outputs will be in `out.grad`
        y_bar = out.grad

        # Compute the adjoint inputs
        x_bar = y_bar * (Q @ x.data)

        # Accumulate the adjoint inputs in the gradient
        x.grad += x_bar

    out._backward = _backward

    return out
```
-->


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

<!-- 
#### Root finding

The last forward-mode example we looked at above was finding the root $x$ of a parametric function $f(x, p)$.
We did this by viewing the condition $f(x, p) = 0$ as defining an implicit function $x = g(p)$, whose pushforward we derived using the implicit function theorem.
The tangent outputs $\dot{x}$ are determined from the tangent inputs $\dot{p}$ by solving the linear system

$$
\partial_x f(x, p) \dot{x} = -\partial_p f(x, p) \dot{p},
$$

defined in terms of the Jacobians of $f$ with respect to $x$ and $p$ (which can presumably also be calculated via autodiff).
These Jacobians are just matrices, so to make the notation clearer let's define $J_x = \partial_x f(x, p)$ and $J_p = \partial_p f(x, p)$, remembering that these are both matrix-valued functions of the nominal values $(p, x)$.

$$
\begin{align}
\langle w, J_x \dot{x} \rangle &= \langle w, -J_p \dot{p} \rangle \\
\langle J_x^T w, \dot{x} \rangle &= \langle -J_p^T w, \dot{p} \rangle.
\end{align}
$$

From this we can read off the equations for the adjoint values:

$$
\begin{gather}
\bar{x} = J_x^T w \\
\bar{p} = -J_p^T w.
\end{gather}
$$

To make sense of this, recall that the function $x = g(p)$ defined implicitly by the root condition returns the root $x$ as a function of the parameters $p$.
Hence, the pullback will be given adjoint values corresponding to the outputs ($\bar{x}$) and will have to calculate the adjoint values for the inputs ($\bar{p}$).

We can do this in two stages: solve $\bar{x} = J_x^T w$ for the intermediate values $w$, and then compute $\bar{p} = -J_p^T w$ via matrix-vector multipliciation.
In more compact notation,

$$
\bar{p} = -J_p^T J_x^{-T} \bar{x}.
$$
-->

### Putting it all together: equality-constrained quadratic program

I want to end with one last example worked end-to-end to see the full process: differentiating through an optimization problem.
Specifically we'll look at an equality-constrained quadratic program (QP), defined mathematically as

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

We _could_ continue working with this as a function with four inputs and two outputs, but now that we see the structure it is easier to just treat it as a linear system solve and re-use our results from before.
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
Once again we can re-use our original solver code and/or matrix factorizations to compute the pushforward.

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

I just want to close with a couple of assorted general tips for deriving these autodiff rules:

* **Don't be afraid of implicit functions.** For many complex functions it turns out that there is a relatively simple equation that defines the output _implicitly_, or states a condition under which the result is valid.  This can be much easier to work with than trying to figure out how to differentiate a function directly.  For example, differentiating an eigenvalue decomposition hardly makes sense when you think about it as an explicit function $V, \Lambda = f(A)$, and differentiating through the decomposition algorithm would probably be a nightmare.  On the other hand, the results $(V, \Lambda)$ are defined implicitly by the simple matrix equation $A V = V \Lambda$, which is no harder to work with than the examples above.
* **Trust the process.** Personally, I'm never really comfortable deriving things when I don't have a good intuition for what the symbols mean.  This can be tough when thinking about adjoints and pullbacks, but I've found that there is often just a "dark forest" phase of the derivation where you're blindly manipulating inner products... and then on the far side you can reconstruct an adjoint equation that makes sense.
* **Look for structure.**  Repeatedly in these examples we end up with a pushforward and pullback that sort of mirrors the original problem: the linear system solve, Lyapunov equations, the quadratic program, and so on.  There is a reason for that -- the pushforward is just a linearization of the original problem, and the pullback is its adjoint -- but in terms of deriving useful rules, just look for similar structure (or "transposed" structure, in the case of the pullback).  You might be able to reuse a lot of the "machinery" you already have for the primal problem, or come up with tricks for reusing things like matrix factorizations.
* **Browse Wikipedia.**  Useful in general, but also specifically when doing this, looking for possibly unfamiliar properties of the operations you're working with (like the trace).  You never know when you'll stumble across some helpful identity that makes everything simple all of a sudden.

With that, I'll leave off for now.
Next I'm hoping to go beyond tensors and demonstrate using these ideas to differentiate through continuous dynamical systems using the calculus of variations: "backpropagation through time".