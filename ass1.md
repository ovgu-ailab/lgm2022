---
layout: default
title: Assignment 1
id: ass1
---


# Assignment 1: Probability Review
**Discussion: April 22nd**  
**Deadline: April 21nd, 23:59**

NOTE the changed deadline!! To give Jens a chance to look at the submissions before
the exercise, the deadline will now be the evening before the exercise!

## Testing Illnesses
Consider a dangerous and/or common illness that people are being tested for to
recognize it early (e.g. cancer) and/or prevent its spread (e.g. COVID). The test
is either positive or negative. We make the following assumptions:

- About 1% of the population has the illness. That is, any given person has a
1% "a priori" probability of being sick.
- If a sick person is tested, the test returns a positive result 99.9% of the 
  time.
- If a healthy person is tested, the test still returns a positive result 1% of the 
  time.

You take part in a study where a random, representative sample of the population
is tested for the illness. Your test result is positive. What is the probability
that you have the illness?
1. **(Submission)** Solve this via simulation. 
   1. Take a "population sample" of a specific size
   (experiment with different sizes!) where every "generated person" has 1% 
    chance of turning out sick.  
   2. Test your "people" -- if they are sick, the test should have a 99.9% chance
    of returning a positive result; if they are healthy, it should be 1%.
   3. Out of all people that have been _tested sick_, get the proportion of people
    that _are actually sick_.
2. **(Submission)** Solve this via mathematics. This requires a basic grasp of marginal and conditional 
   probabilities as well as Bayes' theorem. These are fundamental concepts without which
   you will be lost in this class! [The corresponding wiki article](https://en.wikipedia.org/wiki/Bayes%27_theorem)
   should be sufficient.
   
Next (mathematical solution is sufficient, no need for more simulation):
1. **(Submission)** Conversely, assume the test result is negative. What is the probability that
you have the illness anyway?
2. **(Submission)** To bullet-proof the results of their study, the researchers decide to 
   administer _two_ tests to each participant. The second test has the following
   properties:
   - If a sick person is tested, the test returns a positive result 96% of the
    time.
   - If a healthy person is tested, the test still returns a positive result 2%
    of the time.  
     
   As we can see, the second test is much more prone to errors than the first.
   However, assume that the results of the second test are _independent_ of the
   first. That is, whether the second test makes an error does not depend
   on whether there is an error on the first test and vice versa.  
   Now, _both_ of your tests come back positive. Given this information,
   what is the probability that you are indeed sick?
   
## More on Conditional Distributions
So far we have mainly looked at conditional distributions of the kind `p(x|y)`,
i.e. "probability of data given label" or something similar. There is a different
kind, however: Consider high-dimensional data `x = (x1, x2, ..., xn)`. Often we need
to ask (and answer): What is `p(x1 | x2, ..., xn)` (or any other combination)?
- **(Submission)** Consider the general case, where you have a model p(x) and want to find the
conditional distribution of some subset of variables given the remaining ones.
  Why is this difficult (often impossible) to compute? Make use of the definition
  of conditional probabilities as well as the 
  [law of total probability](https://en.wikipedia.org/wiki/Law_of_total_probability)
  (or consider [marginal distributions](https://en.wikipedia.org/wiki/Marginal_distribution)).
  
Let us then consider a simpler special case: A mixture of Gaussians. You are 
familiar with the basic multidimensional Gaussian distribution; a mixture of
Gaussians is simply a weighted sum of different Gaussians, each with its own
mean and covariance matrix.

- To keep things simple, we will look at what is probably the simplest special case:
Two-dimensional data (variables `x1, x2`) and _isotropic_ Gaussians -- here, the
different dimensions are independent and have the same variance. Technically
speaking, the covariance matrix is a constant times the identity matrix.
- Now we want to look at the distribution `p(x1|x2)`. Read 
  [this Stackexchange answer](https://stats.stackexchange.com/questions/348941/general-conditional-distributions-for-multivariate-gaussian-mixtures)
  on the topic.
- **(Submission)** First consider a "mixture" with a single component. This is just a Gaussian
distribution. For this, you only need the first part of the answer linked above
  (up to the conditional mean and covariance matrix).
  - Given that we are looking at an isotropic Gaussian, what is `p(x1|x2)`? Give
    a general answer (formula) that is simplified as much as possible.
  - Assume `mu = (1, -1)` are the per-dimension means, and `sigma = 1` is the 
    standard deviation.  
    What is `p(x1 = 0.5 | x2 = -0.3)`?  
    What about
    `p(x1 = 1.5 | x2 = -0.3)`?
- **(Submission)** Next, consider a mixture of _two_ Gaussians. For this, the remaining part of
the answer linked above becomes relevant, i.e. the derivation of the conditional
  mixture components.
  - Assume the first mixture component has `mu1 = (1, 1)` and the second one has
    `mu2 = (-1, -1)`. Both components have `sigma = 0.5`. The _prior_ mixture weights are
    0.5, i.e. both components are weighted equally.  
    Now what is
    `p(x1 = 0.5 | x2 = -0.3)`?  
    What about
    `p(x1 = 1.5 | x2 = -0.3)`?  
    Finally, what is `p(x1 = -0.5 | x2 = -0.3)`?
    
Note: To compute the actual probabilities of Gaussians, you can of course use
software such as `scipy`. [See here](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html)
 -- the `pdf` function should be useful.

## Submission note
For the mathematics parts of this assignment, you can hand them in a separate
document (i.e. not a Jupyter notebook). A photograph of an on-paper solution is
okay, but it has to be **high quality and clearly readable**. Better alternatives
would be a document with a hand-written solution on something like a tablet, or,
best, using a typesetting program like LaTeX capable of producing mathematical
formulas.
