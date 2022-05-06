---
layout: default
title: Assignment 4
id: ass4
---


# Assignment 4: Variational Autoencoders
**Discussion: May 13th**  
**Deadline: May 12th, 23:59**

In this assignment, we will implement a variational autoencoder (VAE).


## Implementing a VAE

From an implementation standpoint, a VAE is pretty much just an autoencoder with
a stochastic encoder and a regularized latent space. As such, you might want to
proceed as follows:
- Build an autoencoder. Use any dataset/model of your choice.
- Add stochasticity in the last encoder layer. With the common choice of a
Gaussian distribution, this just means splitting the layer into two parts, one
of which generates means, the other variances. Then you use these values to take
Gaussian samples. You can use `tf.random.normal`
for this -- take samples from a standard normal distribution, multiply
with the standard deviation and add the mean. Be careful with the layer that
generates stds/variances; think about what value range these can be in and what
value range your layer returns. That is, choose a sensible activation function!
- Add a regularizer term to the reconstruction loss, corresponding to the KL-divergence.
The exact form of this for the Gaussian case can be found in many available
  tutorials, implementations as well as [the original paper](https://arxiv.org/pdf/1312.6114.pdf).

You will likely find many VAE implementations around the web. Feel free to use
these for "inspiration", but make sure you understand what you are doing! In particular,
here are some technicalities to pay special attention to:

- Choose your reconstruction loss carefully. Recall the discussion from the first
exercise (second part of Assignment 0). The loss needs to correspond to the negative
log-likelihood of the data conditioned on the latents, which will depend on how you choose to parameterize it.
If you just pick "some" loss, you might not have an actual variational autoencoder.
  - One particularly devious issue comes up when using Keras losses like `BinaryCrossentropy`
    or `MeanSquaredError`.
 By default, this will compute a per-pixel loss and then _average_ over all pixels.
 However, in this case we should _sum_ over pixels since:
    - We make the assumption that pixels are independent.
    - With independence, the image probability becomes the product of pixel probabilities.
    - Since we use log probabilities, this becomes the sum of log probabilities.
  - Normally it doesn't matter much if we sum or average over pixels, since if the number of
   pixels is always the same, this is a constant factor. However in this case it matters _a lot_,
    since averaging over pixels would make the loss much smaller relative to the KL
    divergence term, and this will screw up the learning.
- Regarding the above issue, depending on the details of the data, loss function etc.
you might need to scale down the regularizer _significantly_ (by multiplying
with a number much smaller than 1) to achieve any learning at all. A typical
sign of "overregularization" is when all reconstructions look the same (often like
  some kind of average of all images).
  On the other hand, scaling it _up_ can also help to achieve better samples.
  Strictly speaking, adding a scaling parameter implements a beta-VAE.
- In your KL loss (and throughout the rest of your program), pay special 
  attention to where you need the variance, the log variance,
standard deviation, log standard deviation etc. Depending on how you parameterize
  it with your model, you will need different kinds of conversions here.

Train your VAE and generate some samples, perhaps trying out multiple
architectures and datasets. Think about the following issues:
- In case you ran into the aforementioned problem and your VAE refused to
reconstruct anything, forcing you to tone down the regularizer: Why do you think
this happens? Even if you don't run into this issue, think about why the VAE
regularization might be a particularly troublesome one.
- How can you check whether the regularization was "successful"? Try your method
of choice on your own model(s).
- Compare VAE reconstructions with those of a normal autoencoder. They will
likely be significantly more blurry. Why does this happen? Aside from that, why
does blurriness tend to already be an issue in "normal" AEs?
  