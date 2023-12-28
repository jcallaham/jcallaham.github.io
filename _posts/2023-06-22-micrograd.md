---
layout: post
title: "Beefing up Micrograd"
date: 2023-06-22T03:00:30-04:00
categories:
  - blog
tags:
  - python
  - autodiff
---

I'm working on a couple of posts about differentiable simulation and optimal control, and I wanted to use Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd) implementation of reverse-mode automatic differentiation as a starting point. However, I realized there were a couple of things I'd need to add:

1. A vector data structure and support for AD with vector-valued functions
2. A functional interface similar to JAX and functorch
3. Support for a few more basic math operations


I decided to write this up as a separate post because I thought that it sheds a little more light on some reverse-mode AD mechanics that are useful for applications beyond standard machine learning training.

I'm not going to derive "backprop" or reverse-mode AD from scratch here; if you haven't seen the [video](https://youtu.be/VMj-3S1tku0) or looked through the [code](https://github.com/karpathy/micrograd), I highly recommend it.  [Here's](https://thenumb.at/Autodiff/) another great intro blog post on automatic differentiation.  For background on the JAX functional interface, check out the "[autodiff cookbook](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html#how-it-s-made-two-foundational-autodiff-functions)" page of their documentation.

The code is available on [my fork](https://github.com/jcallaham/micrograd) of micrograd (which currently doesn't have the `nn` module, but I'll add that back when I have time).

### What is micrograd and why am I using it?

Micrograd is a minimal implementation of reverse-mode automatic differentiation, using an interface inspired by PyTorch.  There is a scalar `Value` data structure that can be differentiated over a small number of math operations, and a `micrograd.nn` module that has some basic tools for building neural nets.  

As far as I know, it was never intended to be used for any actual machine learning. PyTorch, Tensorflow, JAX, and other industrial-grade libraries are very highly optimized and there's no point in trying to build your own from scratch.  But the tradeoff is that it's not always easy to look at the source code and understand the principles behind it.  The beauty of micrograd is that it is as simple as possible so that you can really understand how backpropagation works by just reading the source code.  Personally, I learned a lot from it and I think it's a great way to show how something works in a way that also inspires hacking on it.

<!-- That's why I wanted to use it for this project.  While there are certainly more powerful libraries out there, it's hard to beat as a tool for explanation, just because of how easy it is to see exactly what is happening in the code. -->

<!-- As such, it doesn't have the features I listed above because (1) they would add lines of code and (2) they aren't necessary to show how backprop works.  I wanted to use it for this project because  -->

The heart of micrograd is the `Value` class, which is essentially a scalar value with some overloaded math operators that can construct the computation graph necessary for reverse-mode autodiff.  

Here's a stripped-down version of the original `Value` source:

```python
class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

      def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()
```

Each `Value` has four main attributes:
1. `data`: the scalar primal value of the `Value`
2. `grad`: the gradient data computed as part of the backwards pass
3. `_prev`: pointers to the "previous" nodes (subexpressions) in the computation graph
4. `_backward`: Definition of how to compute the gradient in the backwards pass

The `backward` method recursively computes the gradients of all its children, which are subsequently accessible via their `.grad` attribute. Each math operation has three main pieces:

1. Construct a new `Value`
2. Define a `_backward` function that computes the "pullback" map
3. Assign `_backward` to the new `Value` and return the completed object

<!-- The `_backward` function is the key piece of this.  This function essentially computes one stage of the chain rule for that particular operation, evaluated at the primal values of that function.  This process of breaking the gradient of an arbitrary calculation into small pieces is what allows AD frameworks to be modular and composable.  When a new function `f(x)` is defined, it can automatically be used with the rest of the AD pieces: we then know how to calculate `grad(f(x))`, and that's all we need to be able to calculate `grad(f(x)**2 + 4*y)`.

To see what's happening in `_backward` in a simple example, `__mul__` defines a mathematical function $ f(x, y) = x y $.  Let's say we build up a computational graph where the output is $z = g(f(x_1, x_2))$ for some other function $g$. We want to evaluate the gradients $\partial z / \partial x$ and $\partial z / \partial y$.

First, just to make the notation clear let's define an intermediate value $x_3 = f(x_1, x_2)$, so that $z = g(x_2)$.  Using the chain rule,

$$
\frac{\partial z}{\partial x_1} = \frac{\partial z}{\partial x_3} \frac{\partial x_3}{\partial x_1}
$$

This is the product of two nearly independent calculations: the derivative of $g$ (which is $\frac{\partial z}{\partial x_3}$) and the derivative of $f$ (which is $\frac{\partial x_3}{\partial x_1}$).  

Look at the definition of `_backward` for `__mul__`, for instance:

```python
def _backward():
    self.grad += other.data * out.grad
    other.grad += self.data * out.grad
``` -->

It's easy to add new functions with that interface, as long as we know how to differentiate them. For instance, we could add a tanh activation function like this:

```python
import numpy as np

def tanh(x: Value):
    # Step 1: construct the new Value
    out = Value(np.tanh(x.data), (self,), 'tanh')

    # Step 2: define the backwards pass
    def _backward():
        out.grad += (1 / np.cosh(x.data)**2) * out.grad

    # Step 3: save the _backward definition and return the completed output
    out._backward = _backward
    return out
```

That's pretty much all there is to it. But again, I strongly recommend looking through the original source and/or watching the video to understand it fully before continuing.  It's easy enough to understand how to define new `_backward` functions, for instance, but personally it took me longer to understand exactly what is happening there and how it relates to the rest of the autodiff system.

### Reverse-mode autodiff with vector-valued functions

Before launching into the modifications to the code, there's one important piece of mathematical background.  The standard case in machine learning is that we have some large number of trainable parameters (weights/biases) and one scalar output (loss).  Specifically,

$$
y = f(\mathbf{x}), \qquad \mathbf{x} \in \mathbb{R}^n, ~ y \in \mathbb{R},
$$

where $\mathbf{x}$ are our trainable parameters and $y$ is the loss function.  Here $f$ might include an evaluation over some number of training points, but from an optimization perspective it's really a function of the parameters.

In order to minimize the loss, we use some variation of gradient descent.  At each step the current parameters $\mathbf{x}_0$ are updated by taking a step in the direction of the negative gradient of the loss with respect to the parameters, evaluated at the current set of parameters.  This is written $ \nabla f(\mathbf{x}_0)$, which is an $n$-dimensional vector.

The "forward" approach to estimating this gradient using either finite differencing or forward-mode AD involves perturbing all $n$ inputs independently.  On the other hand, reverse-mode AD is very efficient in this setting: as you evaluate $f(\mathbf{x})$ you build up a "computational graph" representing the operations involved in evaluating the function.  Then a _single_ "backward" pass can build up the gradient with respect to all $n$ inputs. Again, there are great resources that explain this in more detail, so I won't do it here.

The important thing here is what happens in the case where the output isn't a scalar, but a vector $\mathbf{y}\in \mathbb{R}^m$. Just like forward-mode AD acts like a perturbation to the input, reverse-mode AD acts like a perturbation to the _output_.  This is very convenient in the machine learning training context, but the difference in efficieny tends to evaporate when $n$ and $m$ are comparable in size.

The generalization of the gradient to vector-valued functions is the $m \times n$ _Jacobian_ matrix $J(\mathbf{x}_0)$:

$$
J(\mathbf{x}_0) = \left.  \frac{dy}{dx}\right|_{\mathbf{x}=\mathbf{x}_0}
=  \left. \begin{bmatrix}
\partial y_0 / \partial x_0 & \partial y_0 / \partial x_1 &  \\
\partial y_1 / \partial x_0 & \partial y_1 / \partial x_1 & \cdots \\
& \vdots &
\end{bmatrix} \right|_{\mathbf{x}=\mathbf{x}_0}
$$

If the output is a scalar, then $J(\mathbf{x}_0) = \nabla f(\mathbf{x}_0)^T$: the first row of the Jacobian is the gradient.  With this in mind, reverse-mode AD can be viewed as computing a row of the Jacobian in each pass (but we'll see what it's really doing in a minute).  On the other hand, forward-mode AD can be seen as computing a column of the Jacobian in each pass.

As a simple example, let's say $\mathbf{f}$ is a vector-valued function given by

$$
\mathbf{f}(\mathbf{x}) = \begin{bmatrix}
f_1(\mathbf{x}) \\
f_2(\mathbf{x})
\end{bmatrix} = \begin{bmatrix}
x_1 + 4 x_2 \\
10 x_2^2 + \sin x_3
\end{bmatrix}.
$$

Then the Jacobian is

$$
J(\mathbf{x}) = \begin{bmatrix}
1 & 4 & 0 \\
0 & 20 x_2 & \cos x_3
\end{bmatrix}
$$

Note that the Jacobian of a vector-valued function is really a _matrix_-valued function: it still takes $\mathbf{x}$ as an input, but now produces an $m \times n$ matrix as its output.

### The vector-Jacobian product

How does this work numerically?  Notice in the micrograd code that the backward pass is "seeded" by [initializing the `grad` variable to 1](https://github.com/karpathy/micrograd/blob/c911406e5ace8742e5841a7e0df113ecb5d54685/micrograd/engine.py#L68).
If the function is vector-valued, so too is this `grad` variable, and we still have to seed it with some vector $\mathbf{v} \in \mathbb{R}^m$.  In PyTorch, this is what's happening when you pass the `gradient` kwarg to `Tensor.backward`, which is required when the `Tensor` isn't scalar-valued.  So, how do we choose $\mathbf{v}$?

The simplest case is if we use a "basis" vector $\mathbf{e}_k$ that is zero except in the $k$-th entry.  For instance, if $k=1$,

$$
\mathbf{e}_1 = \begin{bmatrix} 1 & 0 & 0 & \cdots \end{bmatrix}^T.
$$

If we do this, we are essentially discarding all information about $\mathbf{f}$ and its gradients _except_ for the $k$-th entry.  In other words, its as if we're taking the gradient of the scalar-valued function $f_k$, which is equivalent to the $k$-th row of the Jacobian.  To build the full Jacobian, we have to repeat this $m$ times, each time using a new seed vector $\mathbf{e}_k$, $k=1, 2, \dots, m$.

In the example above, we would do this twice.  First, we use $\mathbf{e}_1$ and get

$$
\nabla f_1(\mathbf{x}_0) = \begin{bmatrix}
1 & 4 & 0
\end{bmatrix}
$$

and then use $\mathbf{e}_2$ and get back

$$
\nabla f_2(\mathbf{x}_0) = \begin{bmatrix}
0 & 20 x_2 & \cos x_3
\end{bmatrix}
$$

Note that here the reverse-mode AD algorithm is not returning a matrix-valued _function_ of the nominal inputs, but is returning the _rows of that matrix_ evaluated at the nominal (forward) inputs.

So, when we evaluate the backward pass using the $k$-th basis vector $\mathbf{e}_k$ as a seed, we get the $k$-th row of the Jacobian evaluated at $\mathbf{x}_0$.  This is exactly what we would expect from a vector-transpose-matrix product!

$$
\mathbf{e}_k^T J(\mathbf{x}_0) = \nabla f_k(\mathbf{x}_0)
$$

With that in mind, what happens if we use an arbitrary seed vector $\mathbf{v} = v_1 \mathbf{e}_1 + v_2 \mathbf{e}_2$? Since the differentiation is linear, we should get back a linear combination of the results of using each basis vector as the seed.  In other words, we will be evaluating the vector-Jacobian product

$$
\mathbf{v}^T J(\mathbf{x}_0) = v_1 \mathbf{e}_1^T J(\mathbf{x}_0) + v_2 \mathbf{e}_2^T J(\mathbf{x}_0)
$$

In other words, a single backwards pass with seed vector $\mathbf{v}$ computes this vector-Jacobian product, without ever having to build the full Jacobian.  If $m$ and/or $n$ are large, this can be a huge computational savings.

Still, at this point it's probably not obvious why this would be useful.  Under what circumstances do we need to take derivatives of vector-valued functions?  And even then, why would we only want to compute a vector-matrix (not even matrix-vector!) product?  I'll get into this more in future posts on adjoint methods.  For now I just wanted to show enough of the mechanics of reverse-mode AD with vector-valued functions to get into the code.

<!-- Let me give a slightly more mathematical interpretation of this.  The Jacobian evaluated at a point $\mathbf{x}_0$ is a linear operator -->

### 1. Array data structure

Now that we know how to generalize backpropagation to vector-valued functions, we're ready to start hacking on micrograd.  Actually, there are surprisingly few modifications necessary to make this work.  First, let's replace `Value` with a new `Array` class:

```python
import numpy as np

class Array:
    def __init__(self, data, _children=(), _op=''):
        self.data = np.array(data, dtype=np.float64)
        self.grad = np.zeros_like(data, dtype=np.float64)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
```

So far this is almost identical to the original, but uses NumPy arrays to store the `data` and `grad` information.  The overloaded operators also needed minimal changes.  For instance, here's `__add__`:

```python
    def __add__(self, other):
        other_is_array = isinstance(other, type(self))
        _other = other if other_is_array else Array(other)
        out = Array(self.data + _other.data, (self, _other), '+')

        def _backward():
            self.grad = self.grad + out.grad
            if other_is_array:
                other.grad += out.grad
        out._backward = _backward

        return out
```

I did slightly change how the case of `other` not being an array is handled.  In the original micrograd, everything is converted to a `Value` and then gradients are accumulated to all `Value`s.  This doesn't quite work with implicit array broadcasting, for instance if we wanted to do something like `Array([1.0, 2.0]) + 3.0`.  We can (and do) convert `3.0` into an array, but it and its `grad` are scalar-valued, while `self` and `out` are both length-2 vectors.  If we tried to accumulate to `other.grad` we'd get a shape mismatch.  Instead, the overloaded operators here support broadcasting by accumulating gradients to `other` if and only if it is an `Array`.

Since we're working with arrays, we should also support indexing.  On the "get" side this is easy:

```python
    def __getitem__(self, idx):
        out = Array(self.data[idx], (self,), f'getitem[{idx}]')
        def _backward():
            self.grad[idx] += out.grad
        out._backward = _backward
        return out
```

If you want to derive the backwards function for this operation yourself, an easy way to think about it is that `x[k]` is essentially the inner product $\mathbf{e}_k^T \mathbf{x}$, where $\mathbf{e}_k$ is the $k$-th basis vector, as before.

The "set" side is not straightforward.  I think in general it's fair to say that mutable arrays and reverse-mode AD don't play well together.  If you implement it by changing the `x.data` array, you might get into trouble if there are other nodes in the graph whose `_prev` attribute is storing a reference to `x`.  In the backward pass, that node thinks that it is pointing to the _original_ `x`, but what it gets is the _mutated_ `x`.

There are ways to work around this.  For instance, JAX defines an [`.at`]((https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ndarray.at.html)) property that can be used for getting/setting ops that don't actually mess with the computational graph.  PyTorch, on the other hand, will throw

```python
RuntimeError: a view of a leaf Variable that requires grad is being used in an in-place operation.
```

For this project, I don't really need `__setitem__`, so I'm going to skip it and just raise an error:

```python
    def __setitem__(self, idx, value):
        raise NotImplementedError("In-place operations on arrays are not supported")
```

The last method we have to deal with is `backward`.  But once again, this is fairly straightforward.  We can just add a PyTorch-like `gradient` kwarg and make sure to handle the seeding correctly:

```python
    def backward(self, gradient=None):
        # topological order all of the children in the graph
        #   (unchanged from original Value.backward)
        topo = []
        visited = set()
        def build_topo(v):
            v.grad = np.zeros_like(v.data)
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # Seed the gradient with the "vec" part of the vector-Jacobian product
        #  (this is new)
        if gradient is None:
            assert self.data.shape == (), "must pass a gradient for non-scalar arrays"
            gradient = np.array(1.0)
        self.grad = gradient

        # go one variable at a time and apply the chain rule to get its gradient
        #   (unchanged from original Value.backward)
        for v in reversed(topo):
            v._backward()
```

With a little extra work, we could also have the `.grad` attribute be an `Array`, which would track its own gradients and allow higher-order derivatives (i.e. Hessians). It's enough of a change that I'm not going to do it here, but see [this discussion](https://github.com/karpathy/micrograd/pull/8) on the micrograd repo.

Finally, it would be nice to have a convenient way to create `Array` objects from different data structures, just as `numpy.array` creates `ndarray` objects and `torch.tensor` creates `Tensor` objects.  So let's define an `array(x)` function that creates a new `Array` depending on what `x` is.  There are really only three important cases here:

1. `x` is an Array: don't need to create anything
2. `x` is an list or tuple.  One of those elements may itself be an `Array`, in which case we want to treat the `Array` creation itself as a differentiable operation with respect to those arrays.
3. Base case: assume that whatever `x` is can be handled by the `np.array` call in `Array.__init__`.

A clean way to handle these cases is with the [`singledispatch`](https://docs.python.org/3/library/functools.html#functools.singledispatch) decorator:

```python
from typing import Union
from functools import singledispatch

# Case 3: base case
@singledispatch
def array(x):
    return Array(x)

# Case 1: input is already an Array
@array.register
def _(x: Array):
    return x

# Case 2: input is a list/tuple that may itself contain Arrays
@array.register
def _(x: Union[list, tuple]):
    # Create new Arrays for elements that aren't already Arrays (existing Arrays won't change)
    _x = [array(xi) for xi in x]  
    out = Array([xi.data for xi in _x], _x, f'array({x})')

    # The Array creation should be a differentiable operation with respect to `x`
    def _backward():
        for i in range(len(x)):
            _x[i].grad += out.grad[i]

    out._backward = _backward
    return out
```

That's it!  With these minor changes we have support for arrays in micrograd.

### 2. Functional interface

With the understanding of reverse-mode AD as a vector-Jacobian product, we can easily build up an approximation of some of the functional programming features of JAX and PyTorch.

Let's start with the basic [`vjp`](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html#vector-jacobian-products-vjps-aka-reverse-mode-autodiff).  We already know that we can calculate the vector-Jacobian product by calling `.backward()` with a seed vector `v`.  Assuming the function we want to differentiate has a single input and single output, `vjp` can be implemented in just a few lines:

```python
def vjp(f, x, v):
    """Vector-Jacobian product for vector-valued function"""
    y = f(x)
    y.backward(gradient=v)
    return x.grad
```

To compute the Jacobian, we can just repeat this process for each output to build up the Jacobian row-by-row.

```python
def jac(f):
    """Jacobian of a vector-valued function"""
    def _jac(*args, **kwargs):
        assert len(args) == 1 and isinstance(args[0], Array), "Only single-input functions supported"
        x = args[0]

        y = f(*args, **kwargs) # forward pass (return an array)
        assert isinstance(y, Array), "Only single-output functions supported"

        J = np.zeros((len(y.data), len(x.data)))
        for k in range(len(y.data)):
            # For each output, do a backward pass

            e = np.zeros_like(y.data)
            e[k] = 1.0  # Unit basis vector 

            y.zero_grad()  # Reset the gradients to do a new backwards pass

            y.backward(gradient=e)  # Seed the backwards pass with the basis vector
            J[k, :] = x.grad
        return J
    _jac.__name__ = f"grad({f.__name__})"
    return _jac
```

Note that for functions with large inputs and/or outputs, this becomes very expensive, both in storage and computation.  If possible, it's much better to use the direct evaluations like `vjp` (and `jvp`, if we had forward-mode autodiff).  With those two functions you can also create objects like [`LinearOperator`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.html) in SciPy that act like matrices for all intents and purposes but never actually explicitly compute and store the full representation.  [Matrix-free linear algebra](https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html) like Krylov methods will never know the difference.

### 3. More math

Finally, let's add a few more math functions.  I think a thorough explanation of how to compute reverse-mode AD rules for more complicated functions is worth its own post (see [this StackExchange question](https://stats.stackexchange.com/questions/504997/reverse-mode-automatic-differentiation-with-respect-to-a-matrix-how-to-matrix/619361#619361) in the meantime), so here I'll just do a few basic operations like NumPy ufuncs to give the flavor.

```python
def relu(x):
    out = Array(np.where(x.data < 0, 0, x.data), (x,), 'ReLU')

    def _backward():
        x.grad += (out.data > 0) * out.grad
    out._backward = _backward

    return out

def tanh(x):
    out = Array(np.tanh(x.data), (x,), 'tanh')

    def _backward():
        x.grad += (1 - out.data**2) * out.grad
    out._backward = _backward

    return out

def sin(x):
    out = Array(np.sin(x.data), (x,), 'sin')

    def _backward():
        x.grad += np.cos(x.data) * out.grad
    out._backward = _backward

    return out

def cos(x):
    out = Array(np.cos(x.data), (x,), 'cos')

    def _backward():
        x.grad += -np.sin(x.data) * out.grad
    out._backward = _backward

    return out

def tan(x):
    out = Array(np.tan(x.data), (x,), 'tan')

    def _backward():
        x.grad += (1 / np.cos(x.data)**2) * out.grad
    out._backward = _backward

    return out

def exp(x):
    out = Array(np.exp(x.data), (x,), 'exp')

    def _backward():
        x.grad += np.exp(x.data) * out.grad
    out._backward = _backward

    return out

def log(x):
    out = Array(np.log(x.data), (x,), 'log')

    def _backward():
        x.grad += (1 / x.data) * out.grad
    out._backward = _backward

    return out

def dot(x, y):
    out = Array(np.dot(x.data, y.data), (x, y), 'dot')

    def _backward():
        x.grad += np.dot(out.grad, y.data.T)
        y.grad += np.dot(x.data.T, out.grad)
    out._backward = _backward

    return out
```

### Summary

Now we have support for vector-valued functions in micrograd!  It did take adding a bit of complexity to the code and going a little deeper into the math of reverse-mode autodiff, but my hope is that working through that and seeing it in code will help build some intuition for what these systems are doing, and how to use them efficiently.

Of course, there are still a million things you could add into this code.  If you feel like exploring for learning purposes, I hope you [pull the code](https://github.com/jcallaham/micrograd/tree/master) and find it a useful starting point.  But as with the original micrograd, if you're actually using autodiff for something you should use a real library - my favorites are PyTorch, JAX, and CasADi.