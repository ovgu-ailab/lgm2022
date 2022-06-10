---
layout: default
title: Assignment 9
id: ass9
---


# Assignment 9: Flow
**Discussion: June 17th**  
**Deadline: June 16th 23:59**

In this assignment, we want to implement some simple flow models on toy 
datasets, as well as attempt to fit simple "real" datasets like MNIST.


## NICE

The OG flow model is the NICE model from 
[this paper](https://arxiv.org/pdf/1410.8516.pdf). It is also a very simple,
making it a good candidate for first experiences with these models. Recall that
in [one of the readings from the lecture](https://blog.evjang.com/2018/01/nf1.html),
code examples for how to implement flows in Tensorflow Probability are given.
However,
- they are outdated,
- they did not work when we tried to adapt them to TF 2.0 (no gradients),
- to understand every detail of training flow models, we want to implement
everything ourselves.

But first, a note on terminology: In principle, it doesn't matter which direction
of the flow you call forward or backward, in which direction a function `f` is 
applied and in which its inverse, etc. However, it's easy to get confused here
because people use different conventions. I will strictly stick to the convention
from the NICE paper, which is:
- `forward` goes from the data (complex distribution) to the simple distribution.
This is the function `f` ("encoder" if you will).
- `backward` goes from simple distribution to complex (data). This is `f^-1`,
the inverse of `f` ("decoder").

You might want to proceed as follows:
- Implement a single NICE layer. Note that this is _not_ a single neural network
layer (But inheriting  from `tf.keras.layers.Layer` might still be useful)!
  - NICE uses a very simple additive coupling layer. Check the paper, equations
  3 and 4 and/or section 3.2. Your layer needs a network to implement the 
  transformation function `m`. You can use a Keras model for this. This should
  take a _d_-dimensional input and also produce a _d_-dimensional input (what
  _d_ should be is explained further below). The network itself can be arbitrarily
  complex (this also means hidden layers can be much larger or smaller).
  - Implement forward and backward operations for the layer. These are very
  simple: Split the input into two parts, and take `y1 = x1` and `y2 = x2 + m(x1)`,
  where `m` is the network defined above. For the backward layer, subtract
  the shift instead.
    - How to split the input? In principle, there are many ways of doing it. Most
      likely, you want to split it into two parts of the same size.
      In this case, `d` above would be half the input dimensionality.
      A very simple one would be to simply split it in the middle (e.g. `tf.split`).  You could also try
    other schemes, e.g. all even dimensions go into part 1 and all odd dimensions
    into part 2, or...
- The full NICE model simply stacks an arbitrary number of such coupling layers.
  - The forward call chains the forward calls of the individual layers. However, note
  that the layers pass `x1` through unchanged! The usual way to solve this issue
  is to alternate layers, such that one modifies the first half, the next one the
  second half, the next one the first half again, etc. There are different ways
  to implement this.
    - You could implement the layers such that they can either change the first
    or second half, and use these in alternation. This is probably the best way.
    - You could implement the layers such that they always change the same half,
    but inbetween layer calls, "swap" the data halves. E.g., `tf.split` followed
    by `tf.concat`, but the other way around. This may be more "elegant", but is
      error-prone.
  - The backward call simply chains the backward calls in the same way (once 
  again take care to swap the data halves appropriately).
  - The NICE paper also proposes a diagonal scaling layer at the end. This is
  because otherwise the model has no way of contracting or expanding the space.
  You can implement this via a `tf.Variable`, one number per data dimension,
  and simply multiply this with the output at the very end of `forward`.
  Accordingly, you need to invert this at the _start_ of `backward` (via 
  division).
  - The NICE paper most certainly has an error in section 5: All the equations
  show applying `m` to `x`, but it should most certainly be the respective `h`
    output of the previous layer.
- As a first sanity check, you should set up a very simple model on some toy data
and check that the `forward` and `backward` functions are actually inverses of
each other. That is, check that the difference between `data` and 
`backward(forward(data))` (and/or the other way around) is near 0 (small 
differences due to numerical reasons are okay).

That takes care of the model itself. Once this works, setting up training is
very simple!
- The training criterion for flow models is maximum likelihood. As such, we
have to add functions to compute this. The general equation for flow models is:
`log(p(data)) = log(p(forward(data))) + log(det(forward_jacobian(data)))`. The
ingredients are:
  - Passing data through `forward` is simple, if you sanity-checked that your
  implementation works! Now we need a "simple" probability distribution for the
  "latent" space. In principle, this can be anything that is easy to evaluate.
  As is often the case, the Normal distribution is a popular choice, although
    the paper proposes using a logistic distribution instead. You should
  use `tensorflow_probability` to create a distribution and use the `log_prob`
  function to evaluate probabilities.
  - NICE with only additive coupling layers actually has a determinant of `1`
  for the Jacobian, meaning that the term disappears completely. If you use a
  scaling layer at the end (recommended) the determinant is simply 
  `prod(scale_factors)` (or `sum(log(scale_factors))` for the log deteminant). See section 3.3 in the paper!
  
With all this taken care of, your model is ready to train. First, try it on
simple toy data. See the notebook for a sample dataset (parabola). Training
proceeds as usual, by gradient descent. You can use the negative log likelihood
as a loss function and use standard TF optimizers. Feel free to try other toy
datasets as well. Make sure you can successfully fit such datasets before moving
on! If training fails, there are likely problems with your implementation.
- The paper advocates using four coupling layers. You can also try more; less
is a bad idea because it makes it difficult to model dependencies between the
two "splits".
- Give it some time -- flows seem to require more training steps to start
producing results (could be several thousands even on toy data).


## Moving on

Building a functional NICE model from the ground up is already quite an
achievement. If you want to move beyond this, try the following:

- A bigger dataset, e.g. MNIST. In principle, this works exactly the same way
as the toy data. You can check section 5 of the paper for the hyperparameters
they use. Note that they use quite large networks for each coupling layer --
flows are notoriously parameter-hungry. Even then, the results are somewhat
underwhelming.
- The successor to NICE, [Real NVP](https://arxiv.org/pdf/1605.08803.pdf). This
adds a _scaling_ operation to each coupling layer. This just means adding a
second network to each layer (the two nets could also share parameters, with
only the final layers being different). Also, the likelihood computation becomes
more complex since now _every_ coupling layer contributes to the Jacobian
determinant through the scaling function. The paper also describes other
improvements such as how to use CNNs for the coupling layers.
