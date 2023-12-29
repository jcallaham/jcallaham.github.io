---
layout: post
title: "Deriving automatic differentiation rules"
date: 2023-12-12T05:30:30-04:00
categories:
  - blog
tags:
  - python
  - autodiff
---

If you're working heavily with an automatic differentiation system like PyTorch, JAX, or Flux.jl, a lot of the time you don't have to think too much about how the magic happens. But every so often you might find yourself doing something that the framework doesn't know how to differentiate properly (or maybe, efficiently). I came across this when using LogSumExp in a cost function in Firedrake for minimizing peak von Mises stress in structural design optimization. It also came up when differentiating through mechanical collisions in a JAX-based multibody dynamics simulation using saltation matrices (which is an interesting topic that probably merits its own post).

The documentation for autodiff frameworks usually shows some way of implementing these kinds of custom "rules" (see for example the relevant pages for [PyTorch](https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html), [JAX](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html), and [Zygote.jl](https://fluxml.ai/Zygote.jl/dev/adjoints/)), but these pages typically don't go into depth about how to derive them. And while there are some great resources for learning about how AD works in general, they usually stop short of explanations for what to do in these oddball cases. The goal of this post is to write the tutorial I was looking for when I first ran into this. I'm going to build on my [last post]({% link _posts/2023-06-22-micrograd.md %}), which walked through extending Andrej Karpathy's micrograd to vector math.

One last note before we get started: my intent here is not to write a math treatise (nor am I really qualified to).
I'm just hoping to provide an approachable introduction to this problem for those of us whose eyes glaze over at phrases like "cotangent bundle".
That said, the [documentation for ChainRulesCore.jl](https://juliadiff.org/ChainRulesCore.jl/stable/maths/propagators.html) has a great writeup on this topic for anyone looking for a little more rigor (although they also caveat their own lack of rigor...), whether or not you're actually using Julia.
You could also read this first and then go check out their documentation.

<!-- TODO: Use the dot notation all the way through (or stick with deltas) -->
<!-- TODO: Make the point about primitive operations vs more rules... inversion or system solve examples -->
<!-- TODO: Terminology... use pushforward/pullback instead of JVP/VJP? -->
<!-- TODO: Introduce some code (JAX?) to illustrate implementing JVP earlier -->

#### Outline
<!-- TODO: Give a quick outline of the structure of the post -->

### Autodiff basics

The basic function of automatic differentiation (AD) is computing the derivative of the result of some numerical computation with respect to the inputs.
AD systems assume that even highly complex functions can be decomposed into a sequence (or graph) of relatively simple primitive operations, each of which can be analytically differentiated in isolation.
Under that assumption, the derivative of the large functions can be calculated using the chain rule from calculus.

The way that the analytic derivatives are implemented is in terms of rules for Jacobian-vector products (in forward mode) and vector-tranpose-Jacobian products (in reverse mode).
For example, if we have a vector-valued function $f: \mathbb{R}^n \rightarrow \mathbb{R}^m$, its derivative at some nominal input values $x_0 \in \mathbb{R}^n$ is the Jacobian $\partial f(x_0) = df/dx \in \mathbb{R}^{m \times n}$.
The Jacobian-vector product (JVP) applied to a "tangent" vector $v \in \mathbb{R}^n$ essentially implements the matrix-vector multiplication

$$
\partial f(x_0) v.
$$

Why is this useful?
The key thing is that JVP rule can usually evaluate this product _without_ explicitly constructing the Jacobian.

If the input is scalar-valued, then $v \in \mathbb{R}$ and this JVP with $v = 1$ will return the gradient of $f$ at $x_0$, giving the sensitivity of the outputs to the inputs near the nominal inputs $x_0$.
On the other hand, if the input is vector-valued, reconstructing the full Jacobian would require evaluating the JVP $n$ times, for instance using the unit basis vectors as tangent vectors $v$.
When $n$ and $m$ are large this becomes very expensive, but fortunately it is rarely necessary to explicitly form the full Jacobian; iterative linear solvers like CG and GMRES only really need to evaluate the matrix-vector product, which is exactly what the JVP provides.

As we'll see in the next section, the Jacobian-vector product is the building block of forward-mode autodiff.
Similarly, the vector-Jacobian product is the main building block of reverse-mode autodiff.
For the same function $f: \mathbb{R}^n \rightarrow \mathbb{R}^m$ above, the vector-Jacobian product (VJP) applied to an "cotangent" vector $w \in \mathbb{R}^m$ implements the vector-transpose-matrix multiplication

$$
w^T \partial f(x_0).
$$

This can also be viewed as a matrix-vector multiplication with the transpose or _adjoint_ of the Jacobian $\partial f(x_0)^T \in \mathbb{R}{n \times m}$:

$$
\partial f(x_0)^T w.
$$

Note that while the seed of the Jacobian-vector product is an element of the _input_ space, the seed for the vector-Jacobian product is an element of the _output_ space.
<!-- TODO: "cotangent" -> "adjoint"? -->
In a sense, these "cotangent" vectors represent perturbations of the function outputs, and reverse-mode autodiff is responsible for computing corresponding perturbations of the function inputs.

This might be a little counterintuitive, but it's very useful in the context of optimization, where the inputs to the objective are the decision variables and the outputs are the objective value we're trying to minimize.
In this case a single VJP returns the gradient with respect to _all_ decision variables, which is a much more efficient way of determining a descent direction compared to trying to map input perturbations to promising directions in output space with Jacobian-vector products.

#### Terminology

Variable terminology is part of what makes learning about the details of automatic differentiation a bit difficult.
My goal here is to use language that makes sense and I'm fine with that being at the expense of a little rigor.

The Jacobian-vector product is also sometimes called the "pushforward" and the vector-Jacobian product is sometimes called the "pullback", borrowing terminology from differential geometry.
The pushforward is also called the "tangent linear map" and the pullback the "adjoint map".
Although Jacobians and vectors are more intuitive to me than "pullback" and "pushforward", I'll mostly use the latter terms instead.
Partly this is because I like that the "map" terminology doesn't imply literal matrix-vector products in the way that "Jacobian-vector product" or "vector-Jacobian product" does, and partly this is to avoid having the confusingly similar acronyms "JVP" and "VJP" all over the place.
Also, "pushforward" and "pullback" correspond neatly to their use in autodiff systems: "pushforward" is what's used in forward-mode AD, and "pullback" is what's needed for reverse-mode AD (or backpropagation).

Summarizing:

* The "pushforward" is the same thing as the "tangent linear map" or "Jacobian-vector product", which maps from input perturbations to output perturbations.
* The "pullback" is the same thing as the "adjoint map" or "vector-Jacbian product", which maps from output perturbations to input perturbations.
* I won't distinguish between input and output spaces and their tangents, cotangents, etc. This is non-rigorous at best, but it makes the terminology a lot simpler. Also, all of these examples use "flat" spaces where the distinction is basically a technicality (in my opinion as someone without a background in advanced geomtery).
* For a function $y = f(x)$ I'll call $x$ and $y$ the "nominal" or "primal" values, indicating that they're the values that are being "linearized about" when we are calculating derivatives.
* I'll call inputs to the pushforward (a.k.a. Jacobian-vector product) "tangents" or "tangent values" and inputs to the pullback (a.k.a. vector-Jacobian product) "adjoints" or "adjoint values".  This is probably also not technically correct, but you'll see it used sometimes and I think it gets the idea across.

### Composing functions
<!-- TODO: add graphics for forward and reverse propagation -->
If $f$ represents a complex calculation -- like a partial differential equation solve or prediction from a machine learning model -- it becomes more and more difficult to compute the linearization $\partial f (x_0)$ analytically.
The key insight of AD systems is that often $f$ is composed of a large number of simpler functions, in which case the chain rule tells us that we could write the Jacobian as an expanded set of matrix-vector products.

#### Forward mode

For instance, if $f(x) = g(h(x))$, where we know how to calculate the JVP for both $g$ and $h$, then

$$
\partial f(x_0) v = \partial g(h(x_0)) \partial h(x_0) v.
$$

Note that the nominal inputs to $g$ are the outputs from $h$ at its nominal inputs $x_0$, and the tangent vector for the JVP for $g$ is the output from the JVP for $h$.
We can calculate the full JVP by propagating both the nominal (a.k.a. _primal_) and tangent values through the computational graph:

1. Begin with the nominal input value $x_0$ and seed vector $v$.
2. Compute the nominal output $y_h = h(x_0)$ and JVP $v_h = \partial h(x_0) v$.
3. Compute the nominal output $y_g = g(y_h)$ and JVP $v_g = \partial g(y_h) v_h$.
4. Finally, the nominal output is $y_g$ and the full JVP is $v_g$.

All information required to compute each component JVP is passed via the primal and tangent vectors.
This means that the JVP rule for some primitive function can be implemented in complete isolation.

Basically, your function will be somewhere in this chain. **As the implementer of the Jacobian-vector product, you can expect to be handed a seed vector (an element of the input space) along with the nominal input values and your job is to calculate the Jacobian-vector product (returning an element of the output space)**. Depending on the AD system, you may also need to return the nominal output values.

#### Reverse mode

In a sense, the vector-Jacobian product just performs the products in the chain rule in the opposite order.
The VJP for $f(x) = g(h(x))$ expands to

$$
w^T \partial f(x_0) = w^T \partial g(h(x_0)) \partial h(x_0).
$$

Calculating this presents a bit of a puzzle.
The vector-transpose-matrix products propagate from left to right, but $g$ still requires as its inputs the nominal outputs $h(x_0)$, implying that the primal calculation has to be completed before the VJP.
This is the root of the "forward/backwards" passes in the reverse-mode algorithm.
The primal calculation is fully completed _first_, with all intermediate values stored.
Then the VJP can be calculated by propagating the cotangent values _backwards_ through the computational graph:

1. Begin with the nominal input value $x_0$ and seed vector $w$.
2. Compute the intermediate primal output $y_h = h(x_0)$.  Save this value for later.
3. Compute the final primal output $y_g = g(x_0)$.
4. Compute the VJP for the second function: $w_g^T = w^T \partial g(h(x_0))$.
5. Compute the VJP for the first function: $w_h^T = w_g^T \partial h(x_0)$.
6. The full VJP $w^T \partial f(x_0)$ is $w_h$.

Note that the full "forward pass" happens _first_, storing intermediate results, and then the "backwards pass" happens.
<!-- pullback -->

Again, your custom function will happen somewhere in this process. **As the implementer of the vector-Jacobian product, you will implement one stage of the backwards pass: you can expect to be handed a seed vector (an element of the output space) along with the nominal input/output values and your job is to calculate the vector-Jacobian product (returning an element of the input space)**.

<!-- TODO: Link to something or expand on this -->
The differences between forward and reverse mode have important ramifications for computational and memory complexity, which are explained in depth in most discussions of automatic differentiation.


#### Static vs dynamic data

We'll often write and/or implement functions using a number of variables, not all of which are really "inputs" in the mathematical sense.
For example, if we write a function $f(x) = ax$, then the variable $a$ is neither an input not an output to the function.
In a sense it just parameterizes the function.

Variables like this are sometimes called "static" data, as opposed to "dynamic" data that can change depending on the inputs.
We don't have to account for these in our derivations of the autodiff rules, and we should be careful to implement the functions in such a way that it is clear what data is static and what is dynamic.
However, you always _can_ treat them as dynamic, and depending on the autodiff system it might be helpful to treat as many variables as possible as dynamic -- you never know what you might want to optimize.

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

Really, from the point of view of autodiff the inputs are the weights and biases $\theta$!
The training data are not inputs, but _static_ data.
That is, we don't need to calculate the derivative of anything with respect to the data itself in the training process.
That's not to say you never need the Jacobian $\partial_x f$ for anything, but you don't need it for training; all you need is the Jacobian $\partial_\theta f$.

This can be hard to keep track of, especially since $\theta$ are often called parameters and $x$ and $y$ are inputs and outputs.
Just try to think carefully about what you actually need to compute the sensitivity of, and with respect to what inputs.

### Deriving forward-mode AD rules (JVP)

Let's start with _forward-mode_ autodiff, which as described in the previous section essentially amounts to decomposing a complex Jacobian-vector product into a number of simpler JVPs using the chain rule.
If we want to implement a custom AD rule for forward mode, then, we just have to derive and implement its Jacobian-vector product.
This is usually easier than deriving reverse-mode rules, and often is actually a key step in that process, but we'll come back to that.

In mathematical terms, let's say our custom function is $f(x): \mathbb{R}^n \rightarrow \mathbb{R}^m$.
The input is a vector $x \in \mathbb{R}^n$ and the output is a vector $y = f(x) \in \mathbb{R}^m$.
The Jacobian of $f$ evaluated at a primal value $x$ is a linear map $\partial f(x): \mathbb{R}^n \rightarrow \mathbb{R}^m$, which we could also think of as a matrix $\mathbb{R}^{m \times n}$.

<!-- Clean this up: hard to read -->
<!-- See https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html#jacobian-vector-products-jvps-aka-forward-mode-autodiff -->
What we typically have to implement to have $f$ work in a forward-mode AD system is a function that maps from a (primal, tangent) input pair $(x, v)$, where $x, v \in \mathbb{R}^n$ to the (primal, tangent) output pair $(y, w)$, where $y = f(x)$, $w = \partial f(x) v$, and $y, w \in \mathbb{R}^m$.
Since we presumably know how to calculate $y = f(x)$, we'll focus on the Jacobian-vector product $ w = \partial f(x) v $.
For now, think of this as a matrix-vector product with the Jacobian matrix $\partial f(x)$.
With that in mind, the only difficulty is deriving the Jacobian.

<!-- Technically the seed vectors in forward mode are elements of the _tangent_ input space and the JVP results are likewise elements of the _tangent_ output space. Practically though, we mostly end up dealing with "flat" Euclidean spaces like R^n, so this distinction doesn't really make a difference and I'll ignore it completely here. -->

Here's a simple process that should work out in most cases:

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

This is the equation for the Jacobian-vector product, if we replace the input perturbation $\delta x$ with $v$ and the output perturbation $\delta y$ with $w$.

#### Example: matrix-vector product

To illustrate this process, let's look at a very simple function, for which we can just read off the Jacobian right away: a matrix-vector multiply $f(x) = A x$ with $x \in \mathbb{R}^n$ and $A \in \mathbb{R}^{m \times n}$.
Of course, since this is a linear function, we can immediately see that the Jacobian is just $A$, but let's go through the process anyway.
First we perturb both the input and the output:

$$
y + \delta y = A (x + \delta x)
$$

Then expand to Taylor-series form:

$$
y + \delta y = Ax + A \delta x
$$

Since we know $y = Ax$ from the original equation, we can cancel those terms, leaving

$$
\delta y = A \delta x,
$$

which tells us that the JVP can be calculated with

$$
\partial f(x) v = Av.
$$

#### Example: quadratic form

Let's take a slightly more difficult example: the quadratic form

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

The final term is quadratic in $\delta x$, so let's discard it since we're looking for the leading-order terms in $\delta x$.
The first term on the right-hand side is just the nominal output $f(x)$, so since $y = f(x)$ we can also cancel that, leaving

$$
\delta y = \frac{1}{2} \delta x^T Q x + \frac{1}{2} x^T Q \delta x,
$$

Finally, we can simplify this with the symmetry of Q, since $\delta x^T Q x$ = $x^T Q^T \delta x$ and $Q = Q^T$.

$$
\delta y = x^T Q \delta x,
$$

which tells us that the Jacobian is the row vector $\partial f(x) = x^T Q$ and the JVP can be calculated with

$$
\partial f(x) v = x^T Q v.
$$


### Beyond vectors

So far we'e only looked at functions that had vector inputs and vector outputs, so that the Jacobian could be expressed neatly as a matrix.
Not all functions can be written conveniently in this way, but these concepts can be expanded neatly to more general input and output spaces.

The simplest of these are functions with tensor inputs and/or outputs.
Let's say we have a function $f: \mathbb{R}^{n \times m} \rightarrow \mathbb{R}^{p \times q \times r}$, which has a 2D tensor as input and returns a 3D tensor.
What is the Jacobian here?
It may be possible to write it as a 5D tensor $\in \mathbb{R}^{p \times q \times r \times n \times m}$ where the Jacobian-vector product is a tensor contraction over the last two dimensions... or if the inputs and outputs are flattened maybe there is a matrix representation $\in \mathbb{R}^{pqr \times nm}$, but neither of these is very convenient.

Instead, it is often much easier to view the Jacobian simply as a linear _map_, which can be conveniently expressed as a matrix when the inputs and outputs are both vectors.
<!-- TODO: Introduce "pushforward" earlier -->
In this context, what we've been calling the Jacobian is sometimes also called the "pushforward" or "tangent linear map".
The relations in terms of $\delta y$ and $\delta x$ we were just deriving define the "action" of the linear map on an element of the input space, which is all you need for the JVP.
Then we don't need to worry about writing any kind of tensor representation for it.

It's probably easiest to see this by example.

#### Example: matrix inversion

Suppose $Y = f(X) = X^{-1}$.
We can derive the pushforward using the definition of the inverse: $X Y = I$, where $I$ is the identity matrix.
We can follow the same procedure as before, beginning by perturbing $X \rightarrow X + \delta X$ and $Y \rightarrow Y + \delta Y$:

$$
(X + \delta X) (Y + \delta Y) = I.
$$

Since $X Y = I$ and we can discard the product of perturbations $\delta X \delta Y$ in the linearization, we are left with

$$
X \delta Y + \delta X Y = 0,
$$

or after a little algebra,

$$
\delta Y = - - X^{-1} \delta X Y = - Y \delta X Y.
$$

This is basically the expression for the "derivative" of the matrix inversion operation.
This equation determines the action of the Jacobian, producing the tangent outputs $\delta Y$ as a function of the tangent inputs $\delta X$ and the primal values $(X, Y)$.
For an arbitrary tangent input $V$ we can then calculate the JVP result $W$ as $W = -Y V Y$.

Note that this could also be derived a little more succintly using the matrix product rule (shown for example in the [ChainRulesCore.jl docs](https://juliadiff.org/ChainRulesCore.jl/stable/maths/arrays.html#Matrix-inversion)), but personally I think the perturbation strategy is easier to reuse in more cases.

#### Example: Lyapunov equation

The Lyapunov equation is a matrix equation that frequently appears in linear systems and control theory.
Viewed as a matrix-valued function, the Lyapunov equation takes as input a matrix $A$ and returns as output the symmetric matrix $P$ to the equation

$$
A P + P A^T + Q = 0,
$$

where $Q$ is a symmetric, positive definite matrix.
We can see this as a function $P = f(A)$, where $f: \mathbb{R}^{n \times n} \rightarrow \mathbb{R}^{n \times n}$, although in this case we cannot write $f$ explicitly.
As a result, deriving an explicit tensor representation for the Jacobian $\partial f: \mathbb{R}^{n \times n} \rightarrow \mathbb{R}^{n \times n}$ would also be quite a challenge.

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
We derive Jacobian-vector product by replacing the input perturbation $\delta A$ with the seed $V$ and the output perturbation $\delta P$ with the result $W$:

$$
V P + A W + W A^T + P V^T = 0.
$$

But this isn't an explicit function; how are we supposed to calculate the output $W$?
We have to get a little creative.

First, let's rearrange this to group the input and output perturbations:

$$
A W + W A^T + \left(V P  + P V^T \right)  = 0.
$$

This _almost_ looks like a new Lyapunov equation, if the terms in the parentheses were a symmetric matrix...

But remember that for any square matrix $M$, the sum $M + M^T$ is symmetric, so $(VP) + (VP)^T$ is symmetric.
But also we know that the nominal solution to $P$ is already symmetric, so $(VP)^T = PV^T$ and the terms in the parentheses _do_ make up a symmetric matrix!
Let's call this $\tilde{Q} = V P  + P V^T$ (note that it only depends on the tangent inputs and the nominal outputs).

In other words, it turns out that we can calculate the Jacobian-vector product with tangent inputs $V$ by solving a second Lyapunov equation

$$
A W + W A^T + \tilde{Q} = 0, \qquad \tilde{Q} = V P + P V^T.
$$

This might seem like a weird one-off case, but actually it often turns out that the JVP calculation shares a lot of structure with the original function, so it's not uncommon that we can reuse some of the same "machinery" for autodiff that we're using anyway for the primal computation.

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



### Multiple inputs and outputs

So far we've only looked at functions with one input and one output, but really the more general case is not much more difficult.
Let's take the case of multiple inputs first.
In normal multivariate calculus, we expand multi-input functions with the total derivative. For $f(x, y)$, this looks like

$$
\delta f = \frac{\partial f}{\partial x} \delta x + \frac{\partial f}{\partial y} \delta y.
$$

Similarly, for a function with two inputs, we can think of the Jacobian as consisting of two maps, one for each input $\rightarrow$ output pathway.
The Jacobian-vector product will need to accept one tangent value for each input.
To keep track of various tangent inputs, sometimes these are denoted by a "dot" in the Jacobian-vector product notation, so that $v = (\dot{x}, \dot{y})$ and we can write the expanded JVP like this (it is _not_ a time derivative):

$$
\partial f(x, y) v = \partial_x f(x, y) \dot{x} + \partial_y f(x, y) \dot{y}.
$$

Now we have to derive each map in a similar manner as we did for the single-input case, but otherwise everything works more or less the same.

For functions with multiple outputs, it's probably easiest to think of each output as the result of a different function and derive the JVP rules separately for each.
For instance, if a function $f(x)$ returns a tuple $(y, z)$, the JVP for $f$ can be derived by thinking of it as two stacked functions $y = g(x)$ and $z = h(x)$:

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

#### Example: matrix-vector product

To see how this works on a simple example, let's revisit the matrix-vector product, but this time viewing it as a two-input, one-output function: $y = f(A, x) = A x$.
As before $x \in \mathbb{R}^n$ and $A \in \mathbb{R}^{m \times n}$.
I'll skip some of the algebra here, but following the same procedure as before we get the following linearization:

$$
\delta y = A \delta x + \delta A x,
$$

so with tangent inputs $v = (\dot{A}, \dot{x})$, the JVP is

$$
\partial f(A, x) v = A \dot{x} + \dot{A} x.
$$

#### Example: linear system solve

A slightly less obvious example is solving the linear system of equations $A x = b$ for $x$.
This can be written explicitly as $f(A, b) = A^{-1} b$, but deriving the JVP is easiest in the original form.
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


### The implicit function theorem

One last trick that can be useful in deriving autodiff rules is the implicit function theorem.
Basically, if we have a function defined as $f(x, y) = 0$, under some assumptions on $f$ we could look at this as an implicit definition of $y$ as a function of $x$.
We can write this as $y = g(x)$, recognizing that we may have no explicit way or writing an actual expression for $g$.
Let's say that $x \in \mathbb{R}^n$ and $y \in \mathbb{R}^m$, so that $f: \mathbb{R}^{n + m} \rightarrow \mathbb{R}^m$ and $g: \mathbb{R}^n \rightarrow \mathbb{R}^m$.
<!-- Also, the choice of $y$ as a function of $x$ here is arbitrary; we could just as well write $x$ as a function of $y$, but in practice the inputs and outputs will probably be obvious from context. -->

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

### Adjoint maps and the vector-Jacobian product

So far we've only gone over the Jacobian-vector product, which is the main building block of forward-mode autodiff.
But this is only half the story; to take full advantage of autodiff systems we also need to be able to derive reverse-mode rules: vector-Jacobian products.
This requires a little more math compared to forward-mode, but we can approach it methodically.
One mathematical concept that will be very useful is the _adjoint linear map_ (in this context also sometimes called the "pullback").

If you know only one thing about adjoints, it's probably that the adjoint of a (real-valued) matrix is the same thing as its transpose.
This is true, and it's a good place to start.
Consider the bilinear form $y^T A x$ with $x \in \mathbb{R}^n$, $y \in \mathbb{R}^m$, and $A \in \mathbb{R}^{m \times n}$.
Using the properties of the transpose we could equivalently write this as

$$
y^T (A x) = (A^T y)^T x.
$$

The first one is basically the dot product of $y$ with the vector $Ax$, and the second is the dot product of the vector $A^T y$ with $x$.
The adjoint is basically just the generalization of this simple equivalence to more general linear maps and "inner product spaces".
Suppose we have a linear operator $A$ that maps from input space $\mathcal{X}$ (for instance $\mathbb{R}^n$) to output space $\mathcal{Y}$ (for instance $\mathbb{R}^m$).
If we have an inner product $\langle u, v \rangle$ defined for any two elements $u, v$ in the input or output spaces, then the adjoint $A^T$ is defined as

$$
\langle y, Ax \rangle = \langle A^T y, x \rangle.
$$

We don't need to get too technical about inner products; probably the most common case is that $x$ and $y$ are just vectors, so that the natural inner product is $\langle u, v\rangle = u^T v$.
In this case, then the adjoint definition is really saying the same thing as the previous equation in terms of matrix transposes.
However, the more general inner product definition opens up some additional possibilities.
Here are a couple of useful inner products (assuming everything is real -- you can look up the complex versions if you need to):

* Vectors: $\langle u, v\rangle = u^T v$
* Matrices: $\langle A, B \rangle = \mathrm{trace}\left(A B^T\right)$
* Continuous functions: $\langle f, g \rangle = \int f(x) g(x) ~ dx$.

The one that might be surprising there is the continuous functions.
This can be useful for deriving reverse-mode AD rules in scientific computing, where we often use things like piecewise polynomials to approximate continuous functions.

At this point the idea of an adjoint might seem fairly abstract, but it's something that makes more sense (and starts to feel more useful) once you have a little experience working with them.

### Deriving reverse-mode AD rules (VJP)

The definition of the adjoint is often the key to deriving vector-Jacobian products.
The general idea is to start by deriving the Jacobian-vector product, write an inner product, and then manipulate it until you have something that looks like the definition of the adjoint.
From there you can infer how to compute the adjoint-vector product.

As usual, this will probably make more sense with examples.  In the next section we'll derive rules for _all_ the forward-mode examples above, but let's just start with the simplest one to get the flavor: matrix-vector multiplication.

One bit of notation before we get started: in forward mode we represented the tangent values with a dot, so the tangent corresponding to an input $x$ was $\dot{x}$.
For reverse-mode the cotangent values are often denoted with a bar, so the cotangent corresponding to an output $y$ is $\bar{y}$.

The matrix-vector multiplication function for a fixed matrix $A$ is $f(x) = A x$ with $x \in \mathbb{R}^n$ and $A \in \mathbb{R}^{m \times n}$.
Above we derived the Jacobian-vector product for a tangent vector $v \in \mathbb{R}^n$:

$$
\partial f(x) v = A v.
$$

To derive the adjoint, first we'll take the inner product of the JVP result with an arbitrary element of the output space $w \in \mathbb{R}^m$:

$$
\langle w, A v \rangle = w^T A v.
$$

We know that $w^T A = (A^T w)^T$, so we could also write this as $(A^T w)^T v$.
But notice that this is the inner product between the vectors $v$ and $A^T w$, so we could rewrite the inner product as

$$
\langle w, A v \rangle = \langle A^T w, v \rangle.
$$

By comparing with the definition of the adjoint we can just read off that the adjoint-vector product for a cotangent vector $w$ is $A^T w$.
This is kind of a trivial example, but we can apply this basic process to much more complicated functions.
The key idea is that beginning with the JVP

$$
\dot{y} = \partial f(x) \dot{x},
$$

we want to derive the VJP

$$
\bar{x} = \partial f(x)^T \bar{y}
$$

using the definition of the adjoint

$$
\langle w, \partial f(x) \dot{x} \rangle = \langle \partial f(x)^T w, \dot{x} \rangle,
$$

which holds for any element $w$ of the cotangent (output) space.
Specifically, if $w = \bar{y}$, then the JVP and VJP definitions tell us that

$$
\langle \bar{y}, \dot{y} \rangle = \langle \bar{x}, \dot{x} \rangle.
$$


1. Derive the Jacobian-vector product, defining the action of the tangent linear map $\partial f(x)$ on an tangent inputs $\dot{x}$.
2. Take the inner product of $\partial f(x) \dot{x}$ with an arbitrary cotangent vector $w$ from the output space.
3. Manipulate the expression for the inner product into the form $\langle \bar{y}, \dot{y} \rangle = \langle \bar{x}, \dot{x} \rangle$. On the left hand, we want the inner product of $\dot{y}$ with _some function of_ $w$ which we associate with the cotangent $\bar{y}$ and on the right side we want the inner product of $\dot{x}$ with _some other function of_ $w$, which we associate with the result $\bar{x}$.
4. Use this expression to derive a rule for computing $\bar{x}$.


Step 3 may seem more complicated than it needs to be; in simple cases we can just manipulate the inner product into the form $\langle w, \dot{y} \rangle$, so that it's clear that $w = \bar{y}$ and it seems like we could have skipped $w$ altogether and just started Step 2 by taking the inner product with $\bar{y}$.
However, this will help in more complicated derivations like the Lyapunov equation below - hopefully this will make sense when we get there.

It will always be _possible_ (though not always _easy_) to rework the inner product expression into one inner product with $\dot{x}$ and one with $\dot{y}$ as a consequence of the linearity of the Jacobian map.
In the case of a function with multiple inputs, say $z = f(x, y)$, we will have multiple tangent input values and the VJP will have to return a tuple of adjoint values $(\bar{x}, \bar{y})$.
In this case we will be looking for an equation like

$$
\langle \bar{z}, \dot{z} \rangle = \langle \bar{x}, \dot{x} \rangle + \langle \bar{y}, \dot{y} \rangle,
$$

where we each expression in the inner product with one of the tangent inputs tells us how to compute that adjoint value.
As for the JVP, if the function has multiple _outputs_ it will probably be easiest to derive the VJP rules as if it was two separate functions.

### Reverse mode examples

Now we're finally ready to work through the full derivation of reverse-mode autodiff rules.
To show how this works in a simple AD framework, I'll also show implementations of these functions in my [fork of micrograd](https://github.com/jcallaham/micrograd), which I extended to support vectors in a [previous post]({% link _posts/2023-06-22-micrograd.md %}).

#### Quadratic form

A quadratic form with symmetric weight matrix $Q$ is defined by:

$$
y = f(x) = \frac{1}{2} x^T Q x,
$$

where $x \in \mathbb{R}^n$ and $y \in \mathbb{R}$.
Above we derived the Jacobian-vector product for tangent values $\dot{x}$:

$$
\dot{y} = \partial f(x) \dot{x} = x^T Q \dot{x}.
$$

Following the procedure above, first we'll take the inner product with a cotangent vector $w \in \mathbb{R}$:

$$
\langle w, \partial f(x) \dot{x} \rangle = w x^T Q \dot{x} = w \dot{y}.
$$

Since $w$ is a scalar, it is trivial to rewrite this to isolate either $\dot{x}$ or $w$ in the inner product:

$$
w x^T Q \dot{x} = \langle w, \dot{y}\rangle = \langle w Q x, \dot{x} \rangle.
$$

Following the procedure above, we can identify the term in the inner product with $\dot{y}$ as $\bar{y}$ and the term in the inner product with $\dot{x}$ as $\bar{x}$, which is what we're looking for.
That is, we can replace $w$ with $\bar{y}$ and then find the VJP rule

$$
\bar{x} = \partial f(x)^T \bar{y} = \bar{y} Q x.
$$

How could we interpret this?
Remember that the VJP rule can be used to evaluate the gradient of a scalar-valued function by setting the "seed" to 1.
That is, the VJP tells us that

$$
\nabla f(x) = Qx,
$$

which can be confirmed by directly differentiating the original function.
In this case, directly computing the gradient would actually have been much easier, but hopefully going through the process will help with the more complicated examples.

Here's the micrograd implementation.
Note that because we've already implemented the transpose and matrix multiplication, we could just compute the result as `0.5 * x.T @ Q @ x` without implementing this function.
However, let's do it anyway because it gives a simple example with the basic mechanics.

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

#### Matrix inversion

The next example was matrix inversion: $Y = f(X) = X^{-1}$, which is a bit more complicated.
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

Taking the inner product of the JVP with the cotangent value $W$, we have

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

Once again, we can equate the term in the inner product with $\dot{Y}$ with $\bar{Y}$, so $W = \bar{Y}$ and then equate the term in the inner product with $\dot{X}$ with $\bar{X}$.
We can infer from this that the action of the adjoint on the cotangent values $\bar{Y}$ is

$$
\bar{X} = \partial f(X)^T \bar{Y} = - Y^T \bar{Y} Y^T.
$$

Again, note that we don't have to write the adjoint operator $\partial f(X)^T$ explicitly as a matrix.
It is a linear map, and all we need to know is how to apply it given the primal and cotangent values.

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

#### Lyapunov equation

Next up is the Lyapunov equation: probably the most complicated of these.
Remember that we viewed this equation as defining the solution $P \in \mathbb{n \times n}$ as an implicit function of the square matrix $A \in \mathbb{n \times n}$.
That is, $P = f(A)$ is the solution to the Lyapunov equation

$$
A P + P A^T + Q = 0,
$$

where $Q \in \mathbb{R}^{n \times n}$ is a symmetric matrix we will consider to be fixed, and the result $P$ is also symmetric.
This has the consequence that the output space is not just the space of $n \times n$ matrices, but _symmetric_ $n \times n$ matrices.
This means that tangent and adjoint values like $\dot{P}$ and $\bar{P}$ also must be symmetric.

<!-- TODO: Write as pushforward -->
Earlier we derived that result of Jacobian-vector product $\dot{P} = \partial f(A) \dot{A}$ is the solution to the "tangent" Lyapunov equation

$$
A \dot{P} + \dot{P} A^T + \tilde{Q} = 0, \qquad \tilde{Q} = \dot{A} P + P \dot{A}^T.
$$

So far we've been taking the inner product of an adjoint value $W$ directly with the tangent values like $\langle W, \dot{P} \rangle$.
However, in this case we don't really know how to write $\dot{P}$ explicitly.
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

However, this is a case where a smarter implementation could go a long way.
The Bartels-Stewart algorithm used to solve the Lyapunov equation relies on Schur decompositions of $A$ and $A^T$, which could be reused in the adjoint Lyapunov equation.
It's always worth looking for places where compuations like matrix factorizations can be reused.

#### Matrix-vector product revisited

We already took the simple case of a matrix-vector product where the matrix $A$ was held fixed.
To see how the reverse-mode derivation works on a simple function with multiple inputs, let's look at the case where the matrix $A \in \mathbb{R}^{m \times n}$ is also considered an input:

$$
y = f(A, x) = A x.
$$

We derived the following forward-mode rule for the pushforward from tangent values $(\dot{A}, \dot{x})$ to $\dot{y}$:

$$
\dot{y} = A \dot{x} + \dot{A} x.
$$

To derive the pullback, we take the inner product of $\dot{y}$ with an element $w \in \mathbb{R}^m$ and then apply some basic properties of the inner product to try to isolate the tangent values $\dot{x}$ and $\dot{A}$ in the inner products:

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

#### Linear system solve

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
Eliminating $w$ from the equation, we have the following equations determining $(\bar{A}, \bar{b})$ as a function of $\bar{x}$:

$$
\begin{gather}
\bar{x} = A^T \bar{b} \\
\bar{A} = -\bar{b} x^T.
\end{gather}
$$

The procedure for calculating $(\bar{A}, \bar{b})$ is therefore to solve the first system of equations for $\bar{b}$, and then use this result in the second equation for $\bar{A}$.
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

### A final example: equality-constrained quadratic program

I want to end with one last example worked end-to-end: differentiating through an equality-constrained quadratic program (QP), defined mathematically as

$$
\min_x \frac{1}{2} x^T Q x + c^T x, \qquad \textrm{subject to } Ax = b.
$$

Here $x, c \in \mathbb{R}^n$ and $b \in \mathbb{R}^m$, so $Q \in \mathbb{R}^{n \times n}$ (symmetric) and $A \in \mathbb{R}^{m \times n}$.
That is, there are $n$ "decision variables" and $m$ constraints.
As with the root-finding and Lyapunov examples, it's easiest to view the QP solve as a function defined implicitly by its optimality conditions.

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
In this case, we will be given primals $(Q, c, A, b, x, \lambda)$ and cotangent outputs $(\bar{x}, \bar{\lambda})$ and will need to compute the cotangent inputs $(\bar{Q}, \bar{c}, \bar{A}, \bar{b})$.

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

Comparing with the optimality conditions we originally derived, we see this is just another QP!
In fact, it has the same $Q$ and $A$ matrices, but replaces the vectors $c \rightarrow - \bar{x}$ and $b \rightarrow \bar{\lambda}$.
As with the Lyapunov equation, this can be taken advantage of by reusing matrix factorizations.

From there we have everything we need to compute the cotangent inputs $(\bar{Q}, \bar{c}, \bar{A}, \bar{b})$.
The last wrinkle is the symmetry of $\bar{Q}$.
The formula above has $\bar{Q} = -w_x x^T$, but there is no guarantee that this rank-1 matrix will be symmetric.
We can instead _enforce_ that it is symmetric by taking its symmetric part: $\bar{Q} = -(w_x x^T + x^T w_x) / 2$.
This probably amounts to a projection onto the input space of symmetric $n \times n$ matrices, but I don't know if that's technically true or not.
The final equations for the cotangent inputs given the intermediate adjoint values $(w_x, w_\lambda)$ are

$$
\begin{align}
\bar{Q} &= -\frac{1}{2}(w_x x^T + x w_x^T)\\
\bar{c} &= -w_x \\
\bar{A} &= - (w_\lambda x^T + \lambda w_x^T)\\
\bar{b} &= w_\lambda
\end{align}.
$$

Here's a simple implementation in micrograd (that doesn't take advantage of sparse matrices):

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
For more on this example, see a [2017 paper from Amos & Kolter](https://arxiv.org/abs/1703.00443) that derives forward- and reverse-mode rules for the more general inequality-constrained quadratic program using the Karush-Kuhn-Tucker (KKT) conditions, including a very efficient algorithm for the reverse-mode computation.
You can compare the equations here to Eq (8) in that paper (by taking the case where there are no inequality constraints).


### Final thoughts

At this point hopefully you've been able to work through these examples and build up some confidence in deriving custom rules for automatic differentiation.
Being able to do this helps release you from the bounds of "what has already been implemented" and also lets you look out for opportunities to do something more efficient numerically.

I just want to close with a couple of assorted general tips for deriving these autodiff rules:

* **Don't be afraid of implicit functions.** For many complex functions it turns out that there is a relatively simple equation that defines the output _implicitly_, or states a condition under which the result is valid.  This can be much easier to work with than trying to figure out how to differentiate a function directly.  For example, differentiating an eigenvalue decomposition hardly makes sense when you think about it as an explicit function $V, \Lambda = f(A)$.  On the other hand, the results $(V, \Lambda)$ are defined implicitly by the simple matrix equation $A V = V \Lambda$, which is no harder to work with than the examples above.
* **Trust the process.** Personally, I'm never really comfortable deriving things when I don't have a good intuition for what the symbols mean.  This can be tough when thinking about adjoints and pullbacks, but I've found that there is often just a "dark forest" phase of the derivation where you're blindly manipulating inner products... and then on the far side you can reconstruct an adjoint equation that makes sense.
* **Look for structure.**  Repeatedly in these examples we end up with a pushforward and pullback that sort of mirrors the original problem: the linear system solve, Lyapunov equations, the quadratic program, and so on.  There is a reason for that -- the pushforward is just a linearization of the original problem, and the pullback is its adjoint -- but in terms of deriving useful rules, just look for similar structure (or "transposed" structure, in the case of the pullback).  You might be able to reuse a lot of the "machinery" you already have for the primal problem, or come up with tricks for reusing things like matrix factorizations.
* **Browse Wikipedia.**  Useful in general, but also specifically when doing this, looking for possibly unfamiliar properties of the operations you're working with (like the trace).  You never know when you'll stumble across some helpful identity that makes everything simple all of a sudden.

With that, I'll leave off for now.
Next I'm hoping to go beyond tensors and demonstrate using these ideas to differentiate through continuous dynamical systems using the calculus of variations: "backpropagation through time".