---
layout: default
title: Assignment 5
id: ass5
---


# Assignment 5: Density Estimation Using Parzen Windows
**Discussion: May 20th**  
**Deadline: May 19th, 23:59**


Sometimes you might be wondering how to evaluate a generative model. Evaluating
whatever loss function you trained with might be a start, but it is often not
very informative. Also, this precludes comparison between different frameworks
(e.g. RBMs vs VAEs). Looking at samples is cumbersome and highly subjective.
An objective measure of the "goodness of fit" of the model would be nice. One
attempt at such a measure uses _kernel density estimation_, in particular the
method of Parzen Windows. Work through 
[this interactive article](https://milania.de/blog/Introduction_to_kernel_density_estimation_%28Parzen_window_method%29)
for a good explanation.
**Then implement Parzen windows and use the method to evaluate your models as follows:**

- The quantity we want to estimate is the likelihood of the test set given the
model. The higher this is, the better we assume our model to be.
- Take an arbitrary number of samples from your model (a common number is 10,000,
actually recommended is much more; start with less for debugging).
- Choose some kernel. A Gaussian kernel is the standard choice. You can start
with a uniform one (a "hypercube"), but you are likely to get rather useless
results, in particular using MNIST. Why do you think this is?
- For each element of the test set (of whichever dataset you are using), compute
the Parzen estimate of the probability, using the aforementioned samples to
provide the kernels. To do this, compute the value of the kernel function (appropriately 
    normalized) between the test set element and each sample, then average those.
  - Since the data is very high-dimensional, you will likely
run into numerical issues already when trying to compute a single probability.
In this case, use log probabilities instead (keep in mind `tf.reduce_logsumexp`
  or the `logprob` methods of `tfp.distributions` classes).
  - Iterating over samples and computing each kernel function one after the other
    is very slow. Think about how you can optimize this process. In principle,
    you can compute the kernel between a test sample and all model samples at the
    same time, or even the kernel between _all_ test samples and all model samples,
    through appropriate tensor operations. In practice, this will take too much
    memory, however (generally there is a time-space tradeoff),
    so you will need to find some middle ground.
- The test set likelihood would be the product of all individual probabilities
  of the test set elements.
Since this will _definitely_ cause numerical issues, it is customary to compute
the log-likelihood instead by summing over all the log probabilities.
- You will need to choose some width parameter for the kernel (e.g. variance
for the Gaussian kernel). This is usually chosen such that the log-likelihood is
maximized. If you want to do this properly, you should put aside a separate
validation set and choose the variance to maximize the likelihood on here, then
report the value on the test set with this variance. If you're short on time,
just try to maximize it on the test set directly, but don't tell anyone you did
this.
- **Evaluate at least two different models and compare results.** These could be
two VAEs, or an RBM and a VAE, or any other generative model. You could also
construct one of the models to perform badly on purpose and see whether the
Parzen estimate agrees.
- Have a look at [this paper](https://arxiv.org/pdf/1511.01844.pdf) and
 immediately forget all of this. In case you are short on time, some takeaways are:
  - Likelihood and sample quality are more or less independent,
  - Parzen windows are very bad at estimating likelihoods,
  - In general, the evaluation method needs to be chosen appropriately depending
    on how the model will be used.
